import pytest
import torch
import torch.nn as nn
import math
import time

from x_transformers import TransformerWrapper, Decoder

from easy_moe import (
    MoETransformerWrapper,
    MoEFFN,
    TopKGate,
    ExpertChoiceGate,
    HCA,
    CSA,
    DS4AttentionLayer,
    HybridAttentionBlock,
    Muon,
    MuonWithAdamW,
    configure_muon_optimizer,
    replace_ffn_with_moe,
    collect_moe_aux_loss,
    reset_moe_aux_loss,
    set_aux_loss_compute,
    enable_gradient_checkpointing,
    DataPrefetcher,
    ThroughputLogger,
    CUDAGraphCapturer,
    get_linear_warmup_cosine_scheduler,
    get_warmup_cosine_scheduler_for_muon,
)


def _make_model(dim=64, depth=2, heads=4, num_experts=4, top_k=2, batched=False, **kwargs):
    decoder = Decoder(dim=dim, depth=depth, heads=heads, ff_glu=True, rotary_pos_emb=True)
    transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=num_experts,
        expert_top_k=top_k,
        glu=True,
        mult=4,
        no_bias=True,
        batched_experts=batched,
        **kwargs,
    )
    return model


class TestOptimizedTopKDispatch:
    def test_optimized_forward_matches_original_shape(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_optimized_backward(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_optimized_aux_loss(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 32))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0
        model.reset_moe_aux_loss()
        assert model.moe_aux_loss.item() == 0.0

    def test_output_consistency_optimized_vs_original(self):
        model1 = _make_model()
        model2 = _make_model()
        model2.load_state_dict(model1.state_dict())
        x = torch.randint(0, 100, (2, 16))
        torch.manual_seed(42)
        loss1 = model1(x).item()
        torch.manual_seed(42)
        loss2 = model2(x).item()
        assert abs(loss1 - loss2) < 0.01


class TestBatchedExperts:
    def test_batched_forward_shape(self):
        model = _make_model(batched=True)
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_batched_backward(self):
        model = _make_model(batched=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_batched_aux_loss(self):
        model = _make_model(batched=True)
        x = torch.randint(0, 100, (2, 32))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0
        model.reset_moe_aux_loss()
        assert model.moe_aux_loss.item() == 0.0

    def test_batched_training_step(self):
        model = _make_model(batched=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        for _ in range(3):
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_batched_expert_choice(self):
        model = _make_model(batched=False, routing_strategy="expert_choice", capacity_factor=1.0)
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert not torch.isnan(loss)


class TestLazyAuxLoss:
    def test_disable_aux_loss(self):
        model = _make_model()
        model.set_aux_loss_compute(False)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        aux = model.moe_aux_loss
        assert aux.item() == 0.0
        model.reset_moe_aux_loss()

    def test_reenable_aux_loss(self):
        model = _make_model()
        model.set_aux_loss_compute(False)
        x = torch.randint(0, 100, (2, 16))
        loss1 = model(x)
        model.set_aux_loss_compute(True)
        loss2 = model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0

    def test_set_aux_loss_via_function(self):
        model = _make_model()
        set_aux_loss_compute(model, False)
        x = torch.randint(0, 100, (2, 16))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() == 0.0


class TestMixedPrecisionTraining:
    def test_autocast_forward(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        model = _make_model().cuda()
        x = torch.randint(0, 100, (2, 16)).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(x)
        assert not torch.isnan(loss)
        loss.backward()

    def test_bfloat16_autocast(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported")
        model = _make_model().cuda()
        x = torch.randint(0, 100, (2, 16)).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x)
        assert not torch.isnan(loss)


class TestWarmupScheduler:
    def test_warmup_scheduler_increases_lr(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_linear_warmup_cosine_scheduler(opt, warmup_steps=10, total_steps=100, eta_min=0.1)
        lrs = []
        for _ in range(10):
            lrs.append(opt.param_groups[0]["lr"])
            scheduler.step()
        assert lrs[-1] > lrs[0], "LR should increase during warmup"

    def test_warmup_scheduler_for_muon(self):
        model = _make_model()
        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3, adamw_lr=3e-4)
        muon_s, adamw_s = get_warmup_cosine_scheduler_for_muon(
            muon_opt, adamw_opt, warmup_steps=5, total_steps=50, muon_lr=1e-3, adamw_lr=3e-4
        )
        muon_lrs = []
        adamw_lrs = []
        for _ in range(10):
            muon_lrs.append(muon_opt.param_groups[0]["lr"])
            adamw_lrs.append(adamw_opt.param_groups[0]["lr"])
            muon_s.step()
            adamw_s.step()
        assert muon_lrs[-1] > muon_lrs[0]
        assert adamw_lrs[-1] > adamw_lrs[0]

    def test_warmup_then_decay(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = get_linear_warmup_cosine_scheduler(opt, warmup_steps=5, total_steps=50, eta_min=0.1)
        lrs = []
        for _ in range(50):
            lrs.append(opt.param_groups[0]["lr"])
            scheduler.step()
        peak_idx = max(range(len(lrs)), key=lambda i: lrs[i])
        assert peak_idx >= 4, "Peak should be at or after warmup"
        assert lrs[-1] < lrs[peak_idx], "Should decay after warmup"


class TestThroughputLogger:
    def test_logger_step(self):
        logger = ThroughputLogger(log_interval=5, window=5)
        logger.start_epoch()
        logger._start_time = time.time() - 0.01
        logger.log_step(num_tokens=100, step=0)
        assert logger._total_tokens == 100
        assert logger._total_steps == 1

    def test_epoch_summary(self):
        logger = ThroughputLogger(log_interval=5, window=5)
        logger.start_epoch()
        for i in range(10):
            logger._start_time = time.time() - 0.001
            logger.log_step(num_tokens=100, step=i)
        summary = logger.epoch_summary()
        assert "tokens_per_sec" in summary
        assert summary["total_tokens"] == 1000


class TestDataPrefetcher:
    def test_prefetcher_iteration(self):
        dataset = torch.utils.data.TensorDataset(torch.randn(20, 10))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        device = torch.device("cpu")
        prefetcher = DataPrefetcher(loader, device)
        batches = list(prefetcher)
        assert len(batches) == 5

    def test_prefetcher_half(self):
        dataset = torch.utils.data.TensorDataset(torch.randn(20, 10))
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        device = torch.device("cpu")
        prefetcher = DataPrefetcher(loader, device, half=False)
        batches = list(prefetcher)
        assert len(batches) == 5


class TestGradientCheckpointing:
    def test_enable_gradient_checkpointing(self):
        model = _make_model()
        count = model.enable_gradient_checkpointing()
        assert count > 0, "Should enable gradient checkpointing on at least 1 MoE layer"

    def test_gradient_checkpointing_forward(self):
        model = _make_model()
        model.enable_gradient_checkpointing()
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_gradient_checkpointing_backward(self):
        model = _make_model()
        model.enable_gradient_checkpointing()
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestCUDAGraphCapturer:
    def test_capturer_init(self):
        model = nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters())
        capturer = CUDAGraphCapturer(model, opt)
        assert not capturer.captured

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_capture_and_replay(self):
        model = nn.Linear(10, 10).cuda()
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(2, 10, device="cuda")
        capturer = CUDAGraphCapturer(model, opt)
        capturer.capture(x)
        assert capturer.captured
        loss = capturer.replay(x)
        assert loss is not None


class TestFlashAttention:
    def test_flash_attention_flag(self):
        decoder = Decoder(
            dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True,
            attn_flash=True,
        )
        transformer = TransformerWrapper(
            num_tokens=100, max_seq_len=64, attn_layers=decoder,
        )
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)


class TestScatterGatherDispatch:
    def test_scatter_add_correctness(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 16))
        torch.manual_seed(0)
        loss1 = model(x).item()
        torch.manual_seed(0)
        loss2 = model(x).item()
        assert abs(loss1 - loss2) < 1e-5

    def test_expert_choice_scatter_add(self):
        model = _make_model(routing_strategy="expert_choice", capacity_factor=1.0)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


class TestFusedOptimizer:
    def test_fused_adamw(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        model = _make_model().cuda()
        muon_opt, adamw_opt = configure_muon_optimizer(
            model, lr=1e-3, adamw_lr=3e-4, fused=True
        )
        combo = MuonWithAdamW(muon_opt, adamw_opt)
        x = torch.randint(0, 100, (2, 16)).cuda()
        loss = model(x)
        loss.backward()
        combo.step()
        combo.zero_grad()


class TestAccelerateIntegration:
    def test_accelerate_basic(self):
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision=None)
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model, opt = accelerator.prepare(model, opt)
        x = torch.randint(0, 100, (2, 16), device=accelerator.device)
        with accelerator.autocast():
            loss = model(x)
        accelerator.backward(loss)
        opt.step()
        opt.zero_grad()

    def test_accelerate_gradient_accumulation(self):
        from accelerate import Accelerator
        accelerator = Accelerator(mixed_precision=None, gradient_accumulation_steps=2)
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model, opt = accelerator.prepare(model, opt)
        x = torch.randint(0, 100, (2, 16), device=accelerator.device)
        with accelerator.autocast():
            loss = model(x)
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()


class TestTorchCompile:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for compile test")
    def test_compile_forward_backward(self):
        model = _make_model().cuda()
        model = torch.compile(model, backend="inductor", dynamic=True)
        x = torch.randint(0, 100, (2, 16)).cuda()
        loss = model(x)
        assert not torch.isnan(loss)
        loss.backward()


class TestFullTrainingLoopWithImprovements:
    def test_training_loop_with_all_improvements_cpu(self):
        model = _make_model()
        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-4, adamw_lr=1e-4)
        combo = MuonWithAdamW(muon_opt, adamw_opt)

        muon_scheduler = get_linear_warmup_cosine_scheduler(
            muon_opt, warmup_steps=2, total_steps=10, eta_min=0.1
        )
        adamw_scheduler = get_linear_warmup_cosine_scheduler(
            adamw_opt, warmup_steps=2, total_steps=10, eta_min=0.1
        )

        model.enable_gradient_checkpointing()
        x = torch.randint(0, 100, (2, 16))
        initial_loss = model(x).item()

        for step in range(5):
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            combo.step()
            muon_scheduler.step()
            adamw_scheduler.step()
            combo.zero_grad()

        final_loss = model(x).item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_batched_expert_training(self):
        model = _make_model(batched=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        initial_loss = model(x).item()

        for _ in range(5):
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            opt.step()
            opt.zero_grad()

        final_loss = model(x).item()
        assert final_loss < initial_loss

    def test_lazy_aux_loss_training(self):
        model = _make_model()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))

        for step in range(5):
            if step % 2 == 0:
                model.set_aux_loss_compute(True)
            else:
                model.set_aux_loss_compute(False)
            loss = model(x)
            model.reset_moe_aux_loss()
            loss.backward()
            opt.step()
            opt.zero_grad()

        final_loss = model(x).item()
        assert not math.isnan(final_loss)

    def test_autocast_training_step(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        model = _make_model().cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16)).cuda()
        for _ in range(3):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()


class TestExistingFunctionality:
    def test_basic_forward(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_generate(self):
        model = _make_model()
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_moe_layers_param(self):
        model = _make_model(depth=4, moe_layers=[0, 3])
        moe_count = sum(1 for m in model.modules() if isinstance(m, MoEFFN))
        assert moe_count == 2

    def test_expert_choice(self):
        model = _make_model(routing_strategy="expert_choice", capacity_factor=1.0)
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_model_param_count(self):
        model = _make_model()
        assert model.num_params > 0
        assert model.num_trainable_params == model.num_params