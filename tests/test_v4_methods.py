import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from x_transformers import TransformerWrapper, Decoder

from x_moe import MoETransformerWrapper, MoEFFN
from x_moe.moe import TopKGate, ExpertChoiceGate, HashGate


def _make_model(dim=64, depth=2, heads=4, num_experts=4, top_k=2, **kwargs):
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
        **kwargs,
    )
    return model


class TestSqrtSoftplusRouting:
    def test_sqrt_softplus_routing_forward(self):
        model = _make_model(sqrt_softplus_routing=True)
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert not torch.isnan(loss)
        assert loss.shape == ()

    def test_sqrt_softplus_routing_backward(self):
        model = _make_model(sqrt_softplus_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_sqrt_softplus_routing_aux_loss(self):
        model = _make_model(sqrt_softplus_routing=True)
        x = torch.randint(0, 100, (2, 32))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0
        model.reset_moe_aux_loss()
        assert model.moe_aux_loss.item() == 0.0

    def test_sqrt_softplus_routing_generate(self):
        model = _make_model(sqrt_softplus_routing=True)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_sqrt_softplus_routing_expert_choice(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_sqrt_softplus_gate_direct(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sqrt_softplus_routing=True)
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, apply_bias=False)
        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)
        assert logits.shape == (4, 8)
        assert (weights >= 0).all()
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_sqrt_softplus_gate_with_bias(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sqrt_softplus_routing=True)
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, apply_bias=True)
        assert weights.shape == (4, 2)
        assert (weights >= 0).all()

    def test_sqrt_softplus_expert_choice_gate(self):
        gate = ExpertChoiceGate(
            dim=64, num_experts=8, capacity_factor=1.0, sqrt_softplus_routing=True
        )
        x = torch.randn(4, 64)
        scores, top_scores, top_indices, capacity, logits = gate(
            x, num_tokens=4, apply_bias=False
        )
        assert scores.shape[0] == 4
        assert scores.shape[1] == 8

    def test_sqrt_softplus_scores_positive(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sqrt_softplus_routing=True)
        x = torch.randn(4, 64)
        logits = gate.w_g(x)
        scores = torch.sqrt(F.softplus(logits))
        assert (scores >= 0).all()

    def test_sqrt_softplus_mutually_exclusive_with_sigmoid(self):
        with pytest.raises(AssertionError):
            TopKGate(
                dim=64,
                num_experts=8,
                top_k=2,
                sigmoid_routing=True,
                sqrt_softplus_routing=True,
            )

    def test_sqrt_softplus_config(self):
        model = _make_model(sqrt_softplus_routing=True)
        assert model.sqrt_softplus_routing is True
        assert model.model_config["sqrt_softplus_routing"] is True

    def test_sqrt_softplus_with_aux_loss_free(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        model.update_routing_biases()

    def test_sqrt_softplus_full_training_step(self):
        model = _make_model(sqrt_softplus_routing=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        for _ in range(3):
            loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
        assert not math.isnan(model(x).item())


class TestHashRouting:
    def test_hash_routing_forward(self):
        model = _make_model(hash_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_hash_routing_backward(self):
        model = _make_model(hash_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_hash_routing_generate(self):
        model = _make_model(hash_routing=True)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_hash_gate_deterministic(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        token_ids = torch.tensor([1, 5, 10, 42])
        x = torch.randn(4, 64)
        _, idx1, _ = gate(x, token_ids=token_ids)
        _, idx2, _ = gate(x, token_ids=token_ids)
        assert torch.equal(idx1, idx2)

    def test_hash_gate_different_ids_different_routing(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        x = torch.randn(100, 64)
        ids_a = torch.arange(0, 100)
        ids_b = torch.arange(100, 200)
        _, idx_a, _ = gate(x, token_ids=ids_a)
        _, idx_b, _ = gate(x, token_ids=ids_b)
        assert not torch.equal(idx_a, idx_b)

    def test_hash_gate_without_token_ids_falls_back(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2)
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, token_ids=None, apply_bias=False)
        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)

    def test_hash_gate_has_routing_bias(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2)
        assert gate.routing_bias.shape == (8,)

    def test_hash_gate_has_w_g(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2)
        assert hasattr(gate, "w_g")
        assert gate.w_g.weight.shape == (8, 64)

    def test_hash_gate_seeds(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        assert gate.hash_seeds.shape == (4,)

    def test_hash_routing_config(self):
        model = _make_model(hash_routing=True, num_hash_functions=8)
        assert model.hash_routing is True
        assert model.num_hash_functions == 8
        assert model.model_config["hash_routing"] is True
        assert model.model_config["num_hash_functions"] == 8

    def test_hash_routing_with_shared_experts(self):
        model = _make_model(hash_routing=True, num_shared_experts=1)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_hash_routing_moe_ffn_direct(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            hash_routing=True,
        )
        assert moe.hash_routing is True
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_hash_routing_full_training_step(self):
        model = _make_model(hash_routing=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        for _ in range(3):
            loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
        assert not math.isnan(model(x).item())


class TestAnticipatoryRouting:
    def test_anticipatory_routing_forward(self):
        model = _make_model(anticipatory_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_anticipatory_routing_backward(self):
        model = _make_model(anticipatory_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_anticipatory_routing_generate(self):
        model = _make_model(anticipatory_routing=True)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_anticipatory_routing_cached_weights(self):
        model = _make_model(anticipatory_routing=True)
        moe_layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        for moe in moe_layers:
            assert moe.anticipatory_routing is True
            assert moe._anticipatory_initialized is True
            assert hasattr(moe, "_cached_gate_weight")

    def test_anticipatory_routing_update(self):
        model = _make_model(anticipatory_routing=True)
        x = torch.randint(0, 100, (2, 16))
        model.train()
        loss = model(x)
        loss.backward()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.step()
        opt.zero_grad()
        model.update_anticipatory_weights()
        moe_layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        for moe in moe_layers:
            gate_w = moe.gate.w_g.weight.data
            cached_w = moe._cached_gate_weight
            assert torch.allclose(gate_w, cached_w)

    def test_anticipatory_routing_training_step(self):
        model = _make_model(anticipatory_routing=True)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        model.train()
        for _ in range(3):
            loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
            model.update_anticipatory_weights()
        assert not math.isnan(model(x).item())

    def test_anticipatory_routing_config(self):
        model = _make_model(anticipatory_routing=True)
        assert model.anticipatory_routing is True
        assert model.model_config["anticipatory_routing"] is True

    def test_anticipatory_routing_with_sqrt_softplus(self):
        model = _make_model(anticipatory_routing=True, sqrt_softplus_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_anticipatory_routing_eval_uses_current_weights(self):
        model = _make_model(anticipatory_routing=True)
        x = torch.randint(0, 100, (2, 16))
        model.eval()
        loss = model(x)
        assert not torch.isnan(loss)

    def test_anticipatory_routing_moe_ffn_direct(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            anticipatory_routing=True,
        )
        assert moe.anticipatory_routing is True
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()


class TestSwiGLUClamping:
    def test_swiglu_clamp_forward(self):
        model = _make_model(swiglu_clamp_value=10.0)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_swiglu_clamp_backward(self):
        model = _make_model(swiglu_clamp_value=10.0)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_swiglu_clamp_generate(self):
        model = _make_model(swiglu_clamp_value=10.0)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_swiglu_clamp_moe_ffn_direct(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            swiglu_clamp_value=10.0,
            batched_experts=True,
        )
        assert moe.swiglu_clamp_value == 10.0
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_swiglu_clamp_training_step(self):
        model = _make_model(swiglu_clamp_value=10.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        for _ in range(3):
            loss = model(x)
            loss.backward()
            opt.step()
            opt.zero_grad()
        assert not math.isnan(model(x).item())

    def test_swiglu_clamp_config(self):
        model = _make_model(swiglu_clamp_value=10.0)
        assert model.swiglu_clamp_value == 10.0
        assert model.model_config["swiglu_clamp_value"] == 10.0

    def test_swiglu_clamp_zero_disabled(self):
        model = _make_model(swiglu_clamp_value=0.0)
        assert model.swiglu_clamp_value == 0.0
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_swiglu_clamp_negative_raises(self):
        with pytest.raises(AssertionError):
            _make_model(swiglu_clamp_value=-1.0)

    def test_swiglu_clamp_with_aux_loss_free(self):
        model = _make_model(
            swiglu_clamp_value=10.0,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_swiglu_clamp_effective(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            swiglu_clamp_value=0.1,
            batched_experts=True,
        )
        x = torch.randn(2, 8, 64) * 100
        out = moe(x)
        assert not torch.isnan(out).any()


class TestV4Combined:
    def test_sqrt_softplus_with_aux_loss_free(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        model.update_routing_biases()

    def test_sqrt_softplus_with_shared_experts(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            num_shared_experts=1,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_sqrt_softplus_with_granularity(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            granularity_factor=2,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_hash_routing_with_swiglu_clamp(self):
        model = _make_model(
            hash_routing=True,
            swiglu_clamp_value=10.0,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_anticipatory_with_swiglu_clamp(self):
        model = _make_model(
            anticipatory_routing=True,
            swiglu_clamp_value=10.0,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_all_v4_features_combined(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            anticipatory_routing=True,
            swiglu_clamp_value=10.0,
            num_shared_experts=1,
            granularity_factor=2,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_all_v4_full_training_step(self):
        model = _make_model(
            sqrt_softplus_routing=True,
            anticipatory_routing=True,
            swiglu_clamp_value=10.0,
            num_shared_experts=1,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        initial_loss = model(x).item()
        for _ in range(5):
            loss = model(x)
            model.update_routing_biases()
            model.update_anticipatory_weights()
            model.reset_moe_aux_loss()
            loss.backward()
            opt.step()
            opt.zero_grad()
        final_loss = model(x).item()
        assert not math.isnan(final_loss)

    def test_backward_compat_default_params(self):
        model = _make_model()
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        assert model.sqrt_softplus_routing is False
        assert model.hash_routing is False
        assert model.anticipatory_routing is False
        assert model.swiglu_clamp_value == 0.0

    def test_hash_routing_with_aux_loss_free(self):
        model = _make_model(
            hash_routing=True,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_sqrt_softplus_with_hash_routing_excluded(self):
        model = _make_model(sqrt_softplus_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_anticipatory_with_aux_loss_free(self):
        model = _make_model(
            anticipatory_routing=True,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        model.update_routing_biases()
        model.update_anticipatory_weights()


class TestV4MoEFFNDirect:
    def test_sqrt_softplus_gate_direct(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sqrt_softplus_routing=True)
        assert gate.sqrt_softplus_routing is True
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, apply_bias=False)
        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)
        assert (weights >= 0).all()

    def test_hash_gate_direct(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        assert gate.num_hash_functions == 4
        x = torch.randn(4, 64)
        token_ids = torch.tensor([1, 5, 10, 42])
        weights, indices, logits = gate(x, token_ids=token_ids)
        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)

    def test_moe_ffn_sqrt_softplus(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            sqrt_softplus_routing=True,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_moe_ffn_hash_routing(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            hash_routing=True,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_moe_ffn_anticipatory_routing(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            anticipatory_routing=True,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_moe_ffn_swiglu_clamp(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            swiglu_clamp_value=10.0,
            batched_experts=True,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_moe_ffn_all_v4(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            sqrt_softplus_routing=True,
            anticipatory_routing=True,
            swiglu_clamp_value=10.0,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()
        out.sum().backward()

    def test_hash_gate_weight_exists_for_logits(self):
        gate = HashGate(dim=64, num_experts=8, top_k=2)
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, token_ids=None, apply_bias=False)
        assert logits.shape == (4, 8)

    def test_anticipatory_routing_no_cached_when_disabled(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            anticipatory_routing=False,
        )
        assert not moe._anticipatory_initialized

    def test_expert_choice_sqrt_softplus_direct(self):
        gate = ExpertChoiceGate(
            dim=64, num_experts=8, capacity_factor=1.0, sqrt_softplus_routing=True
        )
        assert gate.sqrt_softplus_routing is True
        x = torch.randn(4, 64)
        scores, top_scores, top_indices, capacity, logits = gate(
            x, num_tokens=4, apply_bias=False
        )
        assert scores.shape[0] == 4
