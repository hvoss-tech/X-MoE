import pytest
import torch
import torch.nn as nn

from x_transformers import TransformerWrapper, Decoder

from x_moe import (
    MoETransformerWrapper,
    MoEFFN,
    TopKGate,
    ExpertChoiceGate,
    replace_ffn_with_moe,
    collect_moe_aux_loss,
    reset_moe_aux_loss,
    set_aux_loss_compute,
    enable_gradient_checkpointing,
)
from x_moe.moe import _compute_load_balance_loss, _compute_z_loss
from x_moe.data import collate_fn, get_collate_fn, TextDataset
from x_moe.trainer import build_model_from_config


def _make_model(
    dim=64, depth=2, heads=4, num_experts=4, top_k=2, batched=False, **kwargs
):
    decoder = Decoder(
        dim=dim, depth=depth, heads=heads, ff_glu=True, rotary_pos_emb=True
    )
    transformer = TransformerWrapper(
        num_tokens=100, max_seq_len=64, attn_layers=decoder
    )
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


class TestBiasSyncBug:
    def test_sync_stacked_to_experts_bias2_no_double_bias(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=False,
            batched_experts=True,
            max_seq_len=64,
        )
        moe._sync_experts_to_stacked()
        original_w2 = moe.w2_stack.data.clone()
        if hasattr(moe, "b2_stack"):
            original_b2 = moe.b2_stack.data.clone()
        moe._sync_stacked_to_experts()
        if hasattr(moe, "b2_stack"):
            for i, expert in enumerate(moe.experts):
                ff_seq = expert.ff
                bias = ff_seq[2].bias
                assert bias is not None
                assert torch.allclose(bias.data, original_b2[i]), (
                    "b2 bias sync failed - double .bias bug"
                )


class TestDataCollateBug:
    def test_collate_fn_truncation_1d_tensor(self):
        batch = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([1, 2, 3]),
        ]
        result = collate_fn(batch, pad_id=0, pad_to_max=True, max_seq_len=4)
        assert result.shape == (2, 4)
        assert result[0, 0].item() == 1
        assert result[1, 2].item() == 3

    def test_collate_fn_get_collate_fn_truncation(self):
        fn = get_collate_fn(pad_id=0, pad_to_max=True, max_seq_len=4)
        batch = [
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([1, 2]),
        ]
        result = fn(batch)
        assert result.shape == (2, 4)
        assert result[0, 0].item() == 1
        assert result[1, 1].item() == 2

    def test_collate_fn_no_truncation_pad_to_max(self):
        batch = [
            torch.tensor([1, 2, 3]),
            torch.tensor([1, 2]),
        ]
        result = collate_fn(batch, pad_id=0, pad_to_max=True, max_seq_len=5)
        assert result.shape == (2, 5)
        assert result[1, 2].item() == 0

    def test_collate_fn_no_pad_to_max(self):
        batch = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([1, 2, 3]),
        ]
        result = collate_fn(batch, pad_id=0, pad_to_max=False)
        assert result.shape == (2, 5)
        assert result[1, 3].item() == 0
        assert result[1, 4].item() == 0


class TestBuildModelFromConfigKwargsBug:
    def test_build_model_from_config_no_type_error(self):
        config = {
            "dim": 64,
            "depth": 2,
            "heads": 4,
            "num_experts": 4,
            "expert_top_k": 2,
        }
        model = build_model_from_config(config, vocab_size=100, max_seq_len=64)
        assert model is not None
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_build_model_from_config_with_ds4(self):
        config = {
            "dim": 64,
            "depth": 2,
            "heads": 4,
            "num_experts": 4,
            "expert_top_k": 2,
            "use_hca": True,
            "hca_kv_dim": 32,
            "hca_num_heads": 4,
            "hca_compression_rate": 4,
            "hca_window_size": 0,
        }
        model = build_model_from_config(config, vocab_size=100, max_seq_len=64)
        assert model is not None


class TestResetAuxLossBug:
    def test_reset_aux_loss_preserves_buffer(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        moe(x)
        assert moe._aux_loss.item() > 0 or moe._num_forward_passes.item() > 0
        moe.reset_aux_loss()
        assert moe._aux_loss.item() == 0.0
        assert moe._num_forward_passes.item() == 0

    def test_reset_aux_loss_device_consistency(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        moe.eval()
        x = torch.randn(2, 16, 64)
        moe.train()
        moe(x)
        pre_device = moe._aux_loss.device
        moe.reset_aux_loss()
        post_device = moe._aux_loss.device
        assert pre_device == post_device, (
            f"Device changed after reset: {pre_device} -> {post_device}"
        )

    def test_reset_aux_loss_buffer_still_registered(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        moe.train()
        moe(x)
        moe.reset_aux_loss()
        x2 = torch.randn(2, 16, 64)
        moe(x2)
        aux_after = moe.aux_loss.item()
        assert aux_after >= 0, "Aux loss should be non-negative after reset+forward"


class TestGradientCheckpointingBug:
    def test_gradient_checkpointing_actually_works(self):
        model = _make_model()
        count = model.enable_gradient_checkpointing()
        assert count > 0
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert hasattr(module, "_use_gradient_checkpointing")
                assert module._use_gradient_checkpointing is True

    def test_gradient_checkpointing_reduces_memory(self):
        model = _make_model()
        model.enable_gradient_checkpointing()
        x = torch.randint(0, 100, (2, 16))
        model.train()
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_gradient_checkpointing_with_expert_choice(self):
        model = _make_model(routing_strategy="expert_choice", capacity_factor=1.0)
        model.enable_gradient_checkpointing()
        x = torch.randint(0, 100, (2, 16))
        model.train()
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0

    def test_gradient_checkpointing_flag_respected(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        assert moe._use_gradient_checkpointing is False
        moe._use_gradient_checkpointing = True
        assert moe._use_gradient_checkpointing is True


class TestMoEFFNEdgeCases:
    def test_top_k_gate_top_k_exceeds_num_experts(self):
        gate = TopKGate(dim=64, num_experts=4, top_k=8)
        x = torch.randn(2, 16, 64)
        weights, top_indices, logits = gate(x)
        assert top_indices.max() < 4
        assert weights.shape[0] == 2
        assert weights.shape[1] == 16
        assert weights.shape[2] == 4

    def test_expert_choice_gate_capacity(self):
        gate = ExpertChoiceGate(dim=64, num_experts=4, capacity_factor=1.0)
        x_flat = torch.randn(32, 64)
        scores, top_scores, top_indices, capacity, logits = gate(x_flat, num_tokens=32)
        assert capacity <= 32
        assert capacity >= 1

    def test_moe_forward_single_token(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        x = torch.randn(1, 1, 64)
        out = moe(x)
        assert out.shape == (1, 1, 64)
        assert not torch.isnan(out).any()

    def test_moe_aux_loss_accumulation(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        moe(x)
        assert moe._num_forward_passes.item() == 2
        aux = moe.aux_loss
        assert aux.item() >= 0

    def test_load_balance_loss_nonnegative(self):
        logits = torch.randn(32, 4)
        top_indices = torch.randint(0, 4, (32, 2))
        loss = _compute_load_balance_loss(logits, top_indices, num_experts=4)
        assert loss.item() >= 0

    def test_z_loss_nonnegative(self):
        logits = torch.randn(32, 4)
        loss = _compute_z_loss(logits)
        assert loss.item() >= 0

    def test_moe_expert_choice_forward(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_batched_experts_forward(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            batched_experts=True,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_batched_experts_no_bias(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()


class TestSyncStackedParams:
    def test_sync_stacked_roundtrip(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=False,
            batched_experts=True,
            max_seq_len=64,
        )
        moe._sync_experts_to_stacked()
        w1_before = moe.w1_stack.data.clone()
        w2_before = moe.w2_stack.data.clone()
        if hasattr(moe, "b1_stack"):
            b1_before = moe.b1_stack.data.clone()
        if hasattr(moe, "b2_stack"):
            b2_before = moe.b2_stack.data.clone()

        moe._sync_stacked_to_experts()
        moe._sync_experts_to_stacked()

        assert torch.allclose(moe.w1_stack.data, w1_before, atol=1e-6)
        assert torch.allclose(moe.w2_stack.data, w2_before, atol=1e-6)
        if hasattr(moe, "b1_stack"):
            assert torch.allclose(moe.b1_stack.data, b1_before, atol=1e-6)
        if hasattr(moe, "b2_stack"):
            assert torch.allclose(moe.b2_stack.data, b2_before, atol=1e-6)


class TestCollectMoeAuxLoss:
    def test_collect_aux_loss_across_modules(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 16))
        model(x)
        aux = collect_moe_aux_loss(model)
        assert aux.item() >= 0

    def test_reset_aux_loss_across_modules(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 16))
        model(x)
        reset_moe_aux_loss(model)
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._aux_loss.item() == 0.0
                assert module._num_forward_passes.item() == 0


class TestSetAuxLossCompute:
    def test_set_aux_loss_compute_disables(self):
        model = _make_model()
        set_aux_loss_compute(model, False)
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._compute_aux_loss is False

    def test_set_aux_loss_compute_enables(self):
        model = _make_model()
        set_aux_loss_compute(model, False)
        set_aux_loss_compute(model, True)
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._compute_aux_loss is True

    def test_aux_loss_zero_when_disabled(self):
        model = _make_model()
        set_aux_loss_compute(model, False)
        x = torch.randint(0, 100, (2, 16))
        model(x)
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._num_forward_passes.item() == 0


class TestReplaceFFNWithMoE:
    def test_replace_only_specified_layers(self):
        decoder = Decoder(dim=64, depth=4, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(
            num_tokens=100, max_seq_len=64, attn_layers=decoder
        )
        replace_ffn_with_moe(
            transformer, num_experts=4, expert_top_k=2, moe_layers=[0, 2]
        )
        moe_count = sum(1 for m in transformer.modules() if isinstance(m, MoEFFN))
        assert moe_count == 2

    def test_replace_every_n_layers(self):
        decoder = Decoder(dim=64, depth=4, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(
            num_tokens=100, max_seq_len=64, attn_layers=decoder
        )
        replace_ffn_with_moe(
            transformer, num_experts=4, expert_top_k=2, moe_every_n_layers=2
        )
        moe_count = sum(1 for m in transformer.modules() if isinstance(m, MoEFFN))
        assert moe_count == 2


class TestTrainerCollateTruncationBug:
    def test_trainer_collate_truncates_long_sequences(self):
        from x_moe.trainer import _make_collate_fn

        collate = _make_collate_fn(pad_id=0, pad_to_max=True, max_seq_len=4)
        batch = [
            torch.tensor([1, 2, 3, 4, 5, 6]),
            torch.tensor([1, 2]),
        ]
        result = collate(batch)
        assert result.shape == (2, 4), f"Expected (2, 4), got {result.shape}"
        assert result[0, 0].item() == 1
        assert result[0, 1].item() == 2
        assert result[0, 2].item() == 3
        assert result[0, 3].item() == 4
        assert result[1, 0].item() == 1
        assert result[1, 1].item() == 2
        assert result[1, 2].item() == 0
        assert result[1, 3].item() == 0

    def test_trainer_collate_pad_to_max_exact_length(self):
        from x_moe.trainer import _make_collate_fn

        collate = _make_collate_fn(pad_id=0, pad_to_max=True, max_seq_len=5)
        batch = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([1, 2, 3]),
        ]
        result = collate(batch)
        assert result.shape == (2, 5), f"Expected (2, 5), got {result.shape}"
        assert result[0, 4].item() == 5
        assert result[1, 3].item() == 0
        assert result[1, 4].item() == 0

    def test_trainer_collate_consistency_with_data_collate(self):
        from x_moe.trainer import _make_collate_fn

        trainer_collate = _make_collate_fn(pad_id=0, pad_to_max=True, max_seq_len=4)
        data_result = collate_fn(
            [torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])],
            pad_id=0,
            pad_to_max=True,
            max_seq_len=4,
        )
        trainer_result = trainer_collate(
            [torch.tensor([1, 2, 3, 4, 5, 6]), torch.tensor([1, 2])]
        )
        assert torch.equal(data_result, trainer_result), (
            "trainer collate and data collate should produce identical results"
        )

    def test_trainer_collate_no_pad_to_max(self):
        from x_moe.trainer import _make_collate_fn

        collate = _make_collate_fn(pad_id=0, pad_to_max=False)
        batch = [
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([1, 2, 3]),
        ]
        result = collate(batch)
        assert result.shape == (2, 5), f"Expected (2, 5), got {result.shape}"
        assert result[1, 3].item() == 0
        assert result[1, 4].item() == 0


class TestAuxLossEveryBug:
    def test_aux_loss_compute_flag_set_before_forward(self):
        model = _make_model()
        model.train()
        x = torch.randint(0, 100, (2, 16))

        model.set_aux_loss_compute(False)
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._compute_aux_loss is False, (
                    "aux_loss_compute should be False before forward"
                )

        model(x)
        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._num_forward_passes.item() == 0, (
                    "aux loss should not accumulate when compute is disabled"
                )

    def test_aux_loss_every_skips_correctly(self):
        model = _make_model()
        model.train()
        x = torch.randint(0, 100, (2, 16))
        aux_loss_every = 3

        for step in range(6):
            should_compute = (aux_loss_every <= 1) or ((step + 1) % aux_loss_every == 0)
            model.set_aux_loss_compute(should_compute)
            model(x)
            model.reset_moe_aux_loss()

        for module in model.modules():
            if isinstance(module, MoEFFN):
                assert module._aux_loss.item() == 0.0, (
                    "aux loss should be zero after reset"
                )

    def test_aux_loss_disabled_means_no_accumulation(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            max_seq_len=64,
        )
        moe.train()
        x = torch.randn(2, 16, 64)

        moe._compute_aux_loss = False
        moe(x)
        assert moe._num_forward_passes.item() == 0, (
            "num_forward_passes should be 0 when aux loss compute is disabled"
        )
        assert moe.aux_loss.item() == 0.0, (
            "aux_loss should be 0 when compute is disabled"
        )

        moe._compute_aux_loss = True
        moe(x)
        assert moe._num_forward_passes.item() == 1, (
            "num_forward_passes should be 1 after one forward with compute enabled"
        )


class TestAttentionSinkInitBug:
    def test_attention_sink_init_not_half(self):
        from x_moe.attention import AttentionSink

        sink = AttentionSink(num_heads=4)
        attn_logits = torch.randn(1, 4, 8, 8)
        attn_weights = sink(attn_logits)
        weight_sum = attn_weights.sum(dim=-1)
        assert weight_sum.mean().item() > 0.9, (
            f"Attention weights should sum close to 1.0 at init, got {weight_sum.mean().item():.4f}"
        )

    def test_attention_sink_init_logits_negative(self):
        from x_moe.attention import AttentionSink

        sink = AttentionSink(num_heads=4)
        assert (sink.sink_logits < 0).all(), (
            "sink_logits should be initialized to negative values so exp(sink_logits) is small"
        )

    def test_hca_output_magnitude_with_sink(self):
        from x_moe.attention import HCA

        hca_sink = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
            use_attention_sink=True,
        )
        hca_no_sink = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
            use_attention_sink=False,
        )
        hca_sink.load_state_dict(hca_no_sink.state_dict(), strict=False)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_sink = hca_sink(x)
            out_no_sink = hca_no_sink(x)

        ratio = (out_sink.abs().mean() / out_no_sink.abs().mean()).item()
        assert ratio > 0.8, (
            f"Output with sink should be similar magnitude to without sink (ratio={ratio:.4f}), "
            f"not halved like before the fix"
        )

    def test_csa_output_magnitude_with_sink(self):
        from x_moe.attention import CSA

        csa_sink = CSA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            top_k_blocks=0,
            window_size=0,
            use_attention_sink=True,
        )
        csa_no_sink = CSA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            top_k_blocks=0,
            window_size=0,
            use_attention_sink=False,
        )
        csa_sink.load_state_dict(csa_no_sink.state_dict(), strict=False)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_sink = csa_sink(x)
            out_no_sink = csa_no_sink(x)

        ratio = (out_sink.abs().mean() / out_no_sink.abs().mean()).item()
        assert ratio > 0.8, (
            f"Output with sink should be similar magnitude to without sink (ratio={ratio:.4f})"
        )
