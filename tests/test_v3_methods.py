import pytest
import torch
import torch.nn as nn
import math

from x_transformers import TransformerWrapper, Decoder

from x_moe import MoETransformerWrapper, MoEFFN
from x_moe.moe import TopKGate, ExpertChoiceGate


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


class TestSigmoidRouting:
    def test_sigmoid_routing_forward(self):
        model = _make_model(sigmoid_routing=True)
        x = torch.randint(0, 100, (2, 32))
        loss = model(x)
        assert not torch.isnan(loss)
        assert loss.shape == ()

    def test_sigmoid_routing_backward(self):
        model = _make_model(sigmoid_routing=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_sigmoid_routing_aux_loss(self):
        model = _make_model(sigmoid_routing=True)
        x = torch.randint(0, 100, (2, 32))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0
        model.reset_moe_aux_loss()
        assert model.moe_aux_loss.item() == 0.0

    def test_sigmoid_routing_generate(self):
        model = _make_model(sigmoid_routing=True)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_sigmoid_routing_expert_choice(self):
        model = _make_model(
            sigmoid_routing=True,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_sigmoid_gate_has_routing_bias(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sigmoid_routing=True)
        assert gate.routing_bias.shape == (8,)
        assert gate.sigmoid_routing is True

    def test_sigmoid_routing_config(self):
        model = _make_model(sigmoid_routing=True)
        assert model.sigmoid_routing is True
        assert model.model_config["sigmoid_routing"] is True


class TestSharedExperts:
    def test_shared_experts_forward(self):
        model = _make_model(num_shared_experts=1)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_shared_experts_backward(self):
        model = _make_model(num_shared_experts=1)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_shared_experts_more_params(self):
        model_no_shared = _make_model(num_shared_experts=0)
        model_shared = _make_model(num_shared_experts=1)
        assert model_shared.num_params > model_no_shared.num_params

    def test_shared_experts_generate(self):
        model = _make_model(num_shared_experts=1)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_shared_experts_config(self):
        model = _make_model(num_shared_experts=2)
        assert model.num_shared_experts == 2
        assert model.model_config["num_shared_experts"] == 2

    def test_shared_experts_with_expert_choice(self):
        model = _make_model(
            num_shared_experts=1,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)


class TestFineGrainedExperts:
    def test_granularity_forward(self):
        model = _make_model(granularity_factor=2)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_granularity_backward(self):
        model = _make_model(granularity_factor=2)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_granularity_more_routed_experts(self):
        model_g1 = _make_model(granularity_factor=1)
        model_g2 = _make_model(granularity_factor=2)
        moe_g1 = [m for m in model_g1.modules() if isinstance(m, MoEFFN)]
        moe_g2 = [m for m in model_g2.modules() if isinstance(m, MoEFFN)]
        assert len(moe_g1) > 0 and len(moe_g2) > 0
        assert moe_g2[0].num_routed_experts > moe_g1[0].num_routed_experts

    def test_granularity_generate(self):
        model = _make_model(granularity_factor=2)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_granularity_with_shared_experts(self):
        model = _make_model(granularity_factor=2, num_shared_experts=1)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_granularity_with_shared_backward(self):
        model = _make_model(granularity_factor=2, num_shared_experts=1)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_granularity_invalid(self):
        with pytest.raises(AssertionError):
            _make_model(granularity_factor=0)

    def test_granularity_config(self):
        model = _make_model(granularity_factor=2)
        assert model.granularity_factor == 2
        assert model.model_config["granularity_factor"] == 2


class TestAuxLossFree:
    def test_aux_loss_free_forward(self):
        model = _make_model(aux_loss_free=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_aux_loss_free_backward(self):
        model = _make_model(aux_loss_free=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_aux_loss_free_no_balance_loss(self):
        model = _make_model(
            aux_loss_free=True, load_balance_loss_weight=0.01, z_loss_weight=1e-4
        )
        x = torch.randint(0, 100, (2, 16))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() == 0.0

    def test_aux_loss_free_with_seq_balance(self):
        model = _make_model(
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() >= 0

    def test_aux_loss_free_bias_update(self):
        model = _make_model(aux_loss_free=True, bias_update_speed=0.01)
        x = torch.randint(0, 100, (2, 16))
        model(x)
        model.update_routing_biases()
        moe_layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        for moe in moe_layers:
            bias = moe.gate.routing_bias
            assert bias is not None
            assert bias.shape[0] == moe.num_routed_experts

    def test_aux_loss_free_generate(self):
        model = _make_model(aux_loss_free=True)
        model.eval()
        prompt = torch.randint(0, 100, (1, 5))
        out = model.generate(prompt, seq_len=10, temperature=0.8)
        assert out.shape == (1, 10)

    def test_aux_loss_free_config(self):
        model = _make_model(aux_loss_free=True, bias_update_speed=0.05)
        assert model.aux_loss_free is True
        assert model.bias_update_speed == 0.05
        assert model.model_config["aux_loss_free"] is True
        assert model.model_config["bias_update_speed"] == 0.05

    def test_aux_loss_free_with_expert_choice(self):
        model = _make_model(
            aux_loss_free=True,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)


class TestCombinedFeatures:
    def test_sigmoid_plus_shared_plus_granularity(self):
        model = _make_model(
            sigmoid_routing=True,
            num_shared_experts=1,
            granularity_factor=2,
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

    def test_sigmoid_plus_aux_loss_free(self):
        model = _make_model(
            sigmoid_routing=True,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        model.update_routing_biases()
        model.reset_moe_aux_loss()

    def test_all_features_combined(self):
        model = _make_model(
            sigmoid_routing=True,
            num_shared_experts=1,
            granularity_factor=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
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

    def test_full_training_step_all_features(self):
        model = _make_model(
            sigmoid_routing=True,
            num_shared_experts=1,
            granularity_factor=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randint(0, 100, (2, 16))
        initial_loss = model(x).item()
        for _ in range(5):
            loss = model(x)
            model.update_routing_biases()
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
        assert model.sigmoid_routing is False
        assert model.num_shared_experts == 0
        assert model.granularity_factor == 1
        assert model.aux_loss_free is False

    def test_shared_experts_with_granularity_and_sigmoid(self):
        model = _make_model(
            num_experts=4,
            top_k=2,
            sigmoid_routing=True,
            num_shared_experts=1,
            granularity_factor=2,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        moe_layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        assert moe_layers[0].num_routed_experts == 4 * 2
        assert moe_layers[0].routed_top_k == 2 * 2 - 1
        assert moe_layers[0].shared_experts is not None


class TestMoEFFNDirect:
    def test_sigmoid_gate_direct(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sigmoid_routing=True)
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, apply_bias=True)
        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)
        assert logits.shape == (4, 8)

    def test_sigmoid_gate_without_bias(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sigmoid_routing=True)
        x = torch.randn(4, 64)
        weights, indices, logits = gate(x, apply_bias=False)
        assert weights.shape == (4, 2)
        assert indices.shape == (4, 2)

    def test_expert_choice_gate_sigmoid(self):
        gate = ExpertChoiceGate(
            dim=64, num_experts=8, capacity_factor=1.0, sigmoid_routing=True
        )
        x = torch.randn(4, 64)
        scores, top_scores, top_indices, capacity, logits = gate(
            x, num_tokens=4, apply_bias=True
        )
        assert scores.shape[0] == 4
        assert scores.shape[1] == 8

    def test_routing_bias_initialized_zero(self):
        gate = TopKGate(dim=64, num_experts=8, top_k=2, sigmoid_routing=True)
        assert torch.all(gate.routing_bias == 0.0)

    def test_moe_ffn_shared_experts(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            num_shared_experts=1,
        )
        assert moe.shared_experts is not None
        assert len(moe.shared_experts) == 1
        assert moe.num_routed_experts == 4

    def test_moe_ffn_granularity(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            granularity_factor=2,
        )
        assert moe.num_routed_experts == 8
        assert moe.routed_top_k == 4
        assert len(moe.routed_experts) == 8

    def test_moe_ffn_granularity_with_shared(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            granularity_factor=2,
            num_shared_experts=1,
        )
        assert moe.num_routed_experts == 8
        assert moe.routed_top_k == 3
        assert moe.shared_experts is not None

    def test_moe_ffn_aux_loss_free(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        assert moe.aux_loss_free is True
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()

    def test_moe_ffn_sigmoid_routing(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            sigmoid_routing=True,
        )
        x = torch.randn(2, 8, 64)
        out = moe(x)
        assert not torch.isnan(out).any()
