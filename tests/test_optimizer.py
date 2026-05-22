import pytest
import torch
import torch.nn as nn

from x_moe.optimizer import Muon, HybridNewtonSchulz, MuonWithAdamW, configure_muon_optimizer
from x_moe.moe import MoEFFN


class TestHybridNewtonSchulz:
    def test_orthogonalization_shape(self):
        ns = HybridNewtonSchulz()
        M = torch.randn(32, 16)
        out = ns.orthogonalize(M)
        assert out.shape == M.shape

    def test_orthogonality(self):
        ns = HybridNewtonSchulz()
        torch.manual_seed(42)
        M = torch.randn(32, 16)
        out = ns.orthogonalize(M)
        MtM = out.T @ out
        eye = torch.eye(16)
        diff = (MtM - eye).abs().max().item()
        assert diff < 0.3, f"Column orthogonality check failed: max diff = {diff}"

    def test_row_orthogonality_wide(self):
        ns = HybridNewtonSchulz()
        torch.manual_seed(42)
        M = torch.randn(8, 32)
        out = ns.orthogonalize(M)
        MMt = out @ out.T
        eye = torch.eye(8)
        diff = (MMt - eye).abs().max().item()
        assert diff < 0.3, f"Row orthogonality check failed: max diff = {diff}"

    def test_square_matrix(self):
        ns = HybridNewtonSchulz()
        M = torch.randn(16, 16)
        out = ns.orthogonalize(M)
        assert out.shape == (16, 16)

    def test_wide_matrix(self):
        ns = HybridNewtonSchulz()
        M = torch.randn(8, 32)
        out = ns.orthogonalize(M)
        assert out.shape == (8, 32)

    def test_gradient_flow(self):
        ns = HybridNewtonSchulz()
        M = torch.randn(16, 8, requires_grad=True)
        out = ns.orthogonalize(M)
        loss = out.sum()
        loss.backward()
        assert M.grad is not None


class TestMuon:
    def test_basic_step(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        opt = Muon(model.parameters(), lr=1e-3)
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    def test_momentum(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.Linear(8, 4))
        opt = Muon(model.parameters(), lr=1e-3, momentum=0.95)
        x = torch.randn(2, 16)
        for _ in range(3):
            loss = model(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

    def test_weight_decay(self):
        model = nn.Sequential(nn.Linear(16, 4))
        opt = Muon(model.parameters(), lr=1e-3, weight_decay=0.01)
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_nesterov(self):
        model = nn.Sequential(nn.Linear(16, 4))
        opt = Muon(model.parameters(), lr=1e-3, nesterov=True)
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_with_1d_params(self):
        model = nn.Sequential(nn.Linear(16, 4))
        params = list(model.parameters())
        assert any(p.dim() < 2 for p in params)
        opt = Muon(params, lr=1e-3)
        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        opt.step()


class TestMuonWithAdamW:
    def test_combined_step(self):
        model = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))
        muon_params = list(model.parameters())
        adamw_params = [model[0].bias]
        muon_opt = Muon([p for p in muon_params if p.dim() >= 2], lr=1e-3)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=3e-4)
        combo = MuonWithAdamW(muon_opt, adamw_opt)

        x = torch.randn(2, 16)
        loss = model(x).sum()
        loss.backward()
        combo.step()
        combo.zero_grad()

    def test_state_dict(self):
        model = nn.Sequential(nn.Linear(16, 4))
        muon_opt = Muon(model.parameters(), lr=1e-3)
        adamw_opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        combo = MuonWithAdamW(muon_opt, adamw_opt)
        sd = combo.state_dict()
        assert "muon" in sd
        assert "adamw" in sd


class TestConfigureMuonOptimizer:
    def test_configure(self):
        from x_transformers import TransformerWrapper, Decoder
        from x_moe.wrapper import MoETransformerWrapper

        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )

        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3, adamw_lr=3e-4)
        assert len(list(muon_opt.param_groups)) > 0
        assert len(list(adamw_opt.param_groups)) > 0

    def test_params_dont_overlap(self):
        from x_transformers import TransformerWrapper, Decoder
        from x_moe.wrapper import MoETransformerWrapper

        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )

        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3, adamw_lr=3e-4)
        muon_ids = set()
        for group in muon_opt.param_groups:
            for p in group["params"]:
                muon_ids.add(id(p))
        adamw_ids = set()
        for group in adamw_opt.param_groups:
            for p in group["params"]:
                adamw_ids.add(id(p))
        overlap = muon_ids & adamw_ids
        assert len(overlap) == 0, f"Params found in both optimizers: {len(overlap)}"

    def test_all_params_covered(self):
        from x_transformers import TransformerWrapper, Decoder
        from x_moe.wrapper import MoETransformerWrapper

        decoder = Decoder(dim=64, depth=2, heads=4, ff_glu=True, rotary_pos_emb=True)
        transformer = TransformerWrapper(num_tokens=100, max_seq_len=64, attn_layers=decoder)
        model = MoETransformerWrapper(
            transformer=transformer, num_experts=4, expert_top_k=2,
            glu=True, mult=4, no_bias=True,
        )

        muon_opt, adamw_opt = configure_muon_optimizer(model, lr=1e-3, adamw_lr=3e-4)
        all_opt_ids = set()
        for group in muon_opt.param_groups:
            for p in group["params"]:
                all_opt_ids.add(id(p))
        for group in adamw_opt.param_groups:
            for p in group["params"]:
                all_opt_ids.add(id(p))
        model_ids = {id(p) for p in model.parameters()}
        missing = model_ids - all_opt_ids
        assert len(missing) == 0, f"Params not covered by any optimizer: {len(missing)}"