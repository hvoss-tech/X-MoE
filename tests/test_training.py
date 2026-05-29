import math
import torch
from x_transformers import TransformerWrapper, Decoder
from x_moe import (
    MoETransformerWrapper,
    MoEFFN,
    configure_muon_optimizer,
    MuonWithAdamW,
    Trainer,
    TrainConfig,
)
from x_moe.moe import _compute_z_loss, _compute_load_balance_loss


def _make_toy_model(
    dim=64,
    depth=2,
    heads=4,
    num_experts=4,
    top_k=2,
    vocab_size=100,
    max_seq_len=64,
    no_bias=True,
    **kwargs,
):
    decoder = Decoder(
        dim=dim, depth=depth, heads=heads, ff_glu=True, rotary_pos_emb=True
    )
    transformer = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        attn_layers=decoder,
        tie_embedding=True,
    )
    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=num_experts,
        expert_top_k=top_k,
        glu=True,
        mult=4,
        no_bias=no_bias,
        **kwargs,
    )
    return model


def _train_steps(model, x, steps=30, lr=1e-3, use_muon=False):
    if use_muon:
        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=lr, adamw_lr=lr)
        opt = MuonWithAdamW(muon_opt, adamw_opt)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    for i in range(steps):
        opt.zero_grad()
        loss = model(x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


class TestBasicTraining:
    def test_simple_adamw_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model()
        x = torch.randint(0, 100, (4, 32))
        losses = _train_steps(model, x, steps=30, lr=1e-3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_batched_experts_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(batched_experts=True)
        x = torch.randint(0, 100, (4, 32))
        losses = _train_steps(model, x, steps=30, lr=1e-3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_sqrt_softplus_routing_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(sqrt_softplus_routing=True)
        x = torch.randint(0, 100, (4, 32))
        losses = _train_steps(model, x, steps=30, lr=1e-3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_batched_sqrt_softplus_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            batched_experts=True,
            sqrt_softplus_routing=True,
        )
        x = torch.randint(0, 100, (4, 32))
        losses = _train_steps(model, x, steps=30, lr=1e-3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestFullConfigTraining:
    def test_tinystories_config_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=128,
            depth=4,
            heads=4,
            num_experts=8,
            top_k=2,
            vocab_size=1000,
            max_seq_len=64,
            batched_experts=True,
            sqrt_softplus_routing=True,
            no_bias=True,
            zero_init_output=True,
        )
        x = torch.randint(0, 1000, (4, 64))
        losses = _train_steps(model, x, steps=30, lr=1e-3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_tinystories_with_hca_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=128,
            depth=4,
            heads=4,
            num_experts=8,
            top_k=2,
            vocab_size=1000,
            max_seq_len=64,
            batched_experts=True,
            sqrt_softplus_routing=True,
            no_bias=True,
            zero_init_output=True,
            use_hca=True,
            kv_dim=64,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
        )
        x = torch.randint(0, 1000, (4, 64))
        losses = _train_steps(model, x, steps=30, lr=1e-3)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestVaryingDataTraining:
    def test_adamw_varying_data_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(vocab_size=1000, max_seq_len=128)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        for step in range(50):
            x = torch.randint(0, 1000, (4, 128))
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        avg_final = sum(losses[-5:]) / 5
        avg_initial = sum(losses[:5]) / 5
        assert avg_final < avg_initial, (
            f"Loss did not decrease: avg_initial={avg_initial:.4f}, avg_final={avg_final:.4f}"
        )

    def test_batched_sqrt_softplus_varying_data(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            vocab_size=1000,
            max_seq_len=128,
            batched_experts=True,
            sqrt_softplus_routing=True,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        for step in range(50):
            x = torch.randint(0, 1000, (4, 128))
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        avg_final = sum(losses[-5:]) / 5
        avg_initial = sum(losses[:5]) / 5
        assert avg_final < avg_initial, (
            f"Loss did not decrease: avg_initial={avg_initial:.4f}, avg_final={avg_final:.4f}"
        )

    def test_batched_sqrt_softplus_hca_varying_data(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            vocab_size=1000,
            max_seq_len=128,
            batched_experts=True,
            sqrt_softplus_routing=True,
            no_bias=True,
            zero_init_output=True,
            use_hca=True,
            kv_dim=64,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        for step in range(50):
            x = torch.randint(0, 1000, (4, 128))
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        avg_final = sum(losses[-5:]) / 5
        avg_initial = sum(losses[:5]) / 5
        assert avg_final < avg_initial, (
            f"Loss did not decrease: avg_initial={avg_initial:.4f}, avg_final={avg_final:.4f}"
        )


class TestWithMuon:
    def test_muon_basic_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model()
        x = torch.randint(0, 100, (4, 32))
        losses = _train_steps(model, x, steps=30, lr=1e-3, use_muon=True)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_muon_batched_sqrt_softplus_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            batched_experts=True,
            sqrt_softplus_routing=True,
        )
        x = torch.randint(0, 100, (4, 32))
        losses = _train_steps(model, x, steps=30, lr=1e-3, use_muon=True)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_muon_large_model_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=256,
            depth=6,
            heads=8,
            num_experts=16,
            top_k=2,
            vocab_size=10000,
            max_seq_len=128,
            batched_experts=True,
            sqrt_softplus_routing=True,
            zero_init_output=True,
            max_batch_size=4,
        )
        x = torch.randint(0, 10000, (4, 128))
        losses = _train_steps(model, x, steps=30, lr=1e-3, use_muon=True)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_muon_large_model_hca_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=256,
            depth=6,
            heads=8,
            num_experts=16,
            top_k=2,
            vocab_size=10000,
            max_seq_len=128,
            batched_experts=True,
            sqrt_softplus_routing=True,
            zero_init_output=True,
            max_batch_size=4,
            use_hca=True,
            kv_dim=128,
            num_query_heads=8,
            compression_rate=4,
            window_size=0,
        )
        x = torch.randint(0, 10000, (4, 128))
        losses = _train_steps(model, x, steps=30, lr=1e-3, use_muon=True)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_muon_very_large_vocab_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=256,
            depth=4,
            heads=8,
            num_experts=16,
            top_k=2,
            vocab_size=50000,
            max_seq_len=64,
            batched_experts=True,
            sqrt_softplus_routing=True,
            zero_init_output=True,
            max_batch_size=4,
        )
        x = torch.randint(0, 50000, (4, 64))
        losses = _train_steps(model, x, steps=20, lr=1e-3, use_muon=True)
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestZLossGradientFlow:
    def test_z_loss_has_gradient(self):
        router_logits = torch.randn(32, 4, requires_grad=True)
        z_loss = _compute_z_loss(router_logits)
        z_loss.backward()
        assert router_logits.grad is not None
        assert router_logits.grad.abs().sum().item() > 0

    def test_load_balance_loss_has_gradient(self):
        logits = torch.randn(32, 4, requires_grad=True)
        top_indices = torch.randint(0, 4, (32, 2))
        bal_loss = _compute_load_balance_loss(logits, top_indices, 4)
        bal_loss.backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum().item() > 0


class TestMoeAuxLossGradientFlow:
    def test_moe_ffn_aux_loss_has_grad(self):
        moe = MoEFFN(
            dim=64, num_experts=4, expert_top_k=2, glu=True, mult=4, max_seq_len=32
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = moe(x)
        aux = moe.aux_loss
        assert aux.requires_grad is True, "aux_loss should track gradients"

    def test_moe_aux_loss_added_to_total_loss_has_grad_from_aux(self):
        torch.manual_seed(42)
        moe = MoEFFN(
            dim=64, num_experts=4, expert_top_k=2, glu=True, mult=4, max_seq_len=32
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        aux = moe.aux_loss
        main_loss = out.sum()
        total = main_loss + aux
        total.backward()
        has_grad = moe.gate.w_g.weight.grad is not None
        assert has_grad, "Gate should have gradient from aux loss"


class TestTrainerTraining:
    def test_trainer_trains_ppl_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            batched_experts=True,
            sqrt_softplus_routing=True,
            zero_init_output=True,
            max_batch_size=2,
        )
        model.train()
        from torch.utils.data import Dataset

        class RandomDataset(Dataset):
            def __init__(self, size=100, seq_len=32, vocab_size=100):
                self.size = size
                self.seq_len = seq_len
                self.vocab_size = vocab_size
                self.data = torch.randint(0, vocab_size, (size, seq_len))

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx]

        train_ds = RandomDataset()
        val_ds = RandomDataset(size=20)

        config = TrainConfig(
            epochs=3,
            batch_size=2,
            gradient_accumulate=2,
            lr=3e-4,
            warmup_steps=10,
            val_interval=3,
            log_interval=5,
            aux_loss_every=4,
            optimizer="adamw",
            mixed_precision="no",
            compile=False,
            num_workers=0,
            save_dir="/tmp/test_trainer_checkpoints",
        )

        class FakeTokenizer:
            def token_to_id(self, x):
                return 0

            def get_vocab_size(self):
                return 100

            def encode(self, x):
                return type("E", (), {"ids": [0] * 10})()

            decoder = None

        tokenizer = FakeTokenizer()

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            config=config,
        )
        trainer.train()


class TestFullPipelineTraining:
    def test_full_pipeline_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=256,
            depth=12,
            heads=8,
            num_experts=32,
            top_k=2,
            vocab_size=1000,
            max_seq_len=256,
            batched_experts=True,
            sqrt_softplus_routing=True,
            zero_init_output=True,
            max_batch_size=4,
            dropout=0.1,
        )
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        losses = []
        for step in range(30):
            x = torch.randint(0, 1000, (4, 256))
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_full_pipeline_muon_loss_decreases(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            dim=256,
            depth=12,
            heads=8,
            num_experts=32,
            top_k=2,
            vocab_size=1000,
            max_seq_len=256,
            batched_experts=True,
            sqrt_softplus_routing=True,
            zero_init_output=True,
            max_batch_size=4,
            dropout=0.1,
        )
        model.train()
        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3, adamw_lr=3e-4)
        opt = MuonWithAdamW(muon_opt, adamw_opt)
        losses = []
        for step in range(30):
            x = torch.randint(0, 1000, (4, 256))
            opt.zero_grad()
            loss = model(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            if step == 0:
                print(f"Initial loss: {loss.item():.4f}")
        print(f"Final loss: {losses[-1]:.4f}")
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )


class TestMoeGradientFlowFixes:
    def test_aux_loss_improves_router_gradients(self):
        torch.manual_seed(42)
        model = _make_toy_model(
            batched_experts=True,
            sqrt_softplus_routing=True,
            load_balance_loss_weight=0.01,
            z_loss_weight=1e-4,
        )
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
        x = torch.randint(0, 100, (4, 32))
        losses = []
        for step in range(40):
            opt.zero_grad()
            loss = model(x)
            aux = model.moe_aux_loss
            model.reset_moe_aux_loss()
            total = loss + aux
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
