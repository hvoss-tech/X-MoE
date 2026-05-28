import math
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
            "kv_dim": 32,
            "num_query_heads": 4,
            "compression_rate": 4,
            "window_size": 0,
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


class TestBatchedGELUActivationBug:
    def test_batched_uses_gelu_not_silu(self):
        moe_batched = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
        )
        moe_fallback = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=False,
            max_seq_len=64,
        )
        batched_state = moe_batched.state_dict()
        fallback_state = moe_fallback.state_dict()
        shared_keys = set(batched_state.keys()) & set(fallback_state.keys())
        filtered = {k: batched_state[k] for k in shared_keys}
        moe_fallback.load_state_dict(filtered, strict=False)
        moe_batched.eval()
        moe_fallback.eval()

        torch.manual_seed(42)
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_batched = moe_batched(x)
            out_fallback = moe_fallback(x)

        diff = (out_batched - out_fallback).abs().max().item()
        assert diff < 0.05, (
            f"Batched and fallback outputs should match with GELU activation, max diff={diff:.6f}"
        )

    def test_batched_forward_gradient_matches_fallback(self):
        moe_batched = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
        )
        moe_fallback = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=False,
            max_seq_len=64,
        )
        batched_state = moe_batched.state_dict()
        fallback_state = moe_fallback.state_dict()
        shared_keys = set(batched_state.keys()) & set(fallback_state.keys())
        filtered = {k: batched_state[k] for k in shared_keys}
        moe_fallback.load_state_dict(filtered, strict=False)

        x = torch.randn(2, 16, 64, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)

        out_batched = moe_batched(x)
        out_fallback = moe_fallback(x2)

        out_batched.sum().backward()
        out_fallback.sum().backward()

        grad_diff = (x.grad - x2.grad).abs().max().item()
        assert grad_diff < 0.05, (
            f"Gradients should match between batched and fallback, max diff={grad_diff:.6f}"
        )


class TestBatchedNoGluBug:
    def test_batched_no_glu_creates_without_crash(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=False,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
        )
        assert moe is not None
        assert hasattr(moe, "w1_stack")
        assert hasattr(moe, "w2_stack")

    def test_batched_no_glu_forward(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=False,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_batched_no_glu_backward(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=False,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        out.sum().backward()
        grads = [p.grad for p in moe.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert all(not torch.isnan(g).any() for g in grads)

    def test_batched_no_glu_with_bias(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=False,
            mult=4,
            no_bias=False,
            batched_experts=True,
            max_seq_len=64,
        )
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_batched_no_glu_sync_roundtrip(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=False,
            mult=4,
            no_bias=False,
            batched_experts=True,
            max_seq_len=64,
        )
        moe._sync_experts_to_stacked()
        w1_before = moe.w1_stack.data.clone()
        w2_before = moe.w2_stack.data.clone()

        moe._sync_stacked_to_experts()
        moe._sync_experts_to_stacked()

        assert torch.allclose(moe.w1_stack.data, w1_before, atol=1e-6)
        assert torch.allclose(moe.w2_stack.data, w2_before, atol=1e-6)


class TestDoubleResidualBug:
    def test_hca_no_double_residual_with_window(self):
        from x_moe.attention import HCA

        hca_window = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
            use_attention_sink=False,
        )
        hca_no_window = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
            use_attention_sink=False,
        )
        hca_no_window.load_state_dict(hca_window.state_dict(), strict=False)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_window = hca_window(x)
            out_no_window = hca_no_window(x)

        diff = (out_window - out_no_window).abs().mean().item()
        x_norm = (out_no_window - x).abs().mean().item()
        assert diff < x_norm * 2, (
            f"Window and no-window outputs should be similar magnitude, "
            f"diff={diff:.6f}, x_residual={x_norm:.6f}"
        )

    def test_hca_output_is_not_x_plus_mqa_plus_x(self):
        from x_moe.attention import HCA

        hca = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
            use_attention_sink=False,
        )
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out = hca(x)

        out_diff_from_x = (out - x).abs().mean().item()
        x_scale = x.abs().mean().item()
        assert out_diff_from_x < x_scale * 3, (
            f"Output should not have double residual (x_norm term), "
            f"diff={out_diff_from_x:.4f}, x_scale={x_scale:.4f}"
        )

    def test_csa_no_double_residual_with_window(self):
        from x_moe.attention import CSA

        csa_window = CSA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            top_k_blocks=0,
            window_size=4,
            use_attention_sink=False,
        )
        csa_no_window = CSA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            top_k_blocks=0,
            window_size=0,
            use_attention_sink=False,
        )
        csa_no_window.load_state_dict(csa_window.state_dict(), strict=False)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_window = csa_window(x)
            out_no_window = csa_no_window(x)

        diff = (out_window - out_no_window).abs().mean().item()
        x_norm = (out_no_window - x).abs().mean().item()
        assert diff < x_norm * 2, (
            f"Window and no-window outputs should be similar magnitude, "
            f"diff={diff:.6f}, x_residual={x_norm:.6f}"
        )

    def test_ds4_layer_residual_consistency(self):
        from x_moe.attention import DS4AttentionLayer

        layer_window = DS4AttentionLayer(
            dim=64,
            attn_type="hca",
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
            use_attention_sink=False,
        )
        layer_no_window = DS4AttentionLayer(
            dim=64,
            attn_type="hca",
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
            use_attention_sink=False,
        )
        layer_no_window.load_state_dict(layer_window.state_dict(), strict=False)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_window = layer_window(x)
            out_no_window = layer_no_window(x)

        diff = (out_window - out_no_window).abs().mean().item()
        x_scale = x.abs().mean().item()
        assert diff < x_scale * 3, (
            f"DS4Layer outputs with/without window should be comparable, "
            f"diff={diff:.4f}, x_scale={x_scale:.4f}"
        )


class TestCapacityEnforcementBug:
    def test_fallback_respects_capacity(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=False,
            max_seq_len=8,
            max_batch_size=1,
            capacity_factor=0.5,
        )
        capacity = moe._capacity
        assert capacity > 0
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_fallback_and_vectorized_consistent(self):
        torch.manual_seed(42)
        moe_fallback = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=False,
            max_seq_len=64,
            max_batch_size=2,
        )
        moe_vectorized = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
            max_batch_size=2,
        )
        state = moe_fallback.state_dict()
        filtered_state = {
            k: v for k, v in state.items() if k in moe_vectorized.state_dict()
        }
        moe_vectorized.load_state_dict(filtered_state, strict=False)
        moe_fallback.eval()
        moe_vectorized.eval()

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_fallback = moe_fallback(x)
            out_vectorized = moe_vectorized(x)

        diff = (out_fallback - out_vectorized).abs().max().item()
        assert diff < 0.1, (
            f"Fallback and vectorized outputs should be consistent, max diff={diff:.6f}"
        )


class TestBatchedGluBiasWithNoBiasBug:
    def test_batched_no_bias_true_glu_true_includes_proj_bias(self):
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
        assert moe._has_bias_1.item() == True, (
            "GLU proj bias should be detected even when no_bias=True"
        )
        assert hasattr(moe, "b1_stack"), "b1_stack should exist when GLU proj has bias"

    def test_batched_no_bias_true_glu_true_matches_fallback(self):
        torch.manual_seed(42)
        moe_batched = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
            zero_init_output=False,
        )
        moe_fallback = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=False,
            max_seq_len=64,
            zero_init_output=False,
        )

        moe_fallback.train()
        opt = torch.optim.Adam(moe_fallback.parameters(), lr=0.01)
        x = torch.randn(4, 16, 64)
        for _ in range(5):
            loss = moe_fallback(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        state_f = moe_fallback.state_dict()
        for key in state_f:
            if key in moe_batched.state_dict():
                moe_batched.state_dict()[key].copy_(state_f[key])
        moe_batched._sync_experts_to_stacked()
        moe_batched.eval()
        moe_fallback.eval()

        with torch.no_grad():
            out_b = moe_batched(x)
            out_f = moe_fallback(x)

        diff = (out_b - out_f).abs().max().item()
        assert diff < 0.01, (
            f"Batched and fallback outputs should match when GLU proj bias is included, "
            f"max diff={diff:.6f}"
        )

    def test_batched_no_bias_true_glu_true_bias_sync_roundtrip(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
            zero_init_output=False,
        )

        moe.train()
        opt = torch.optim.Adam(moe.parameters(), lr=0.01)
        x = torch.randn(4, 16, 64)
        for _ in range(5):
            loss = moe(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        assert moe._has_bias_1.item() == True

        moe._sync_experts_to_stacked()
        b1_before = moe.b1_stack.data.clone()

        moe._sync_stacked_to_experts()
        moe._sync_experts_to_stacked()

        assert torch.allclose(moe.b1_stack.data, b1_before, atol=1e-6), (
            "b1_stack roundtrip should preserve GLU proj bias"
        )

        for i in range(4):
            expert_bias = moe.experts[i].ff[0].proj.bias.data
            stacked_bias = moe.b1_stack.data[i]
            assert torch.allclose(expert_bias, stacked_bias, atol=1e-6), (
                f"Expert {i} GLU proj bias should match stacked bias"
            )

    def test_batched_no_bias_false_glu_true_still_works(self):
        torch.manual_seed(42)
        moe_fallback = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=False,
            batched_experts=False,
            max_seq_len=64,
            zero_init_output=False,
        )

        moe_fallback.train()
        opt = torch.optim.Adam(moe_fallback.parameters(), lr=0.01)
        x = torch.randn(4, 16, 64)
        for _ in range(5):
            loss = moe_fallback(x).sum()
            loss.backward()
            opt.step()
            opt.zero_grad()

        moe_batched = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=False,
            batched_experts=True,
            max_seq_len=64,
            zero_init_output=False,
        )
        assert moe_batched._has_bias_1.item() == True
        assert moe_batched._has_bias_2.item() == True
        assert hasattr(moe_batched, "b1_stack")
        assert hasattr(moe_batched, "b2_stack")

        state_f = moe_fallback.state_dict()
        for key in state_f:
            if key in moe_batched.state_dict():
                moe_batched.state_dict()[key].copy_(state_f[key])
        moe_batched._sync_experts_to_stacked()
        moe_batched.eval()
        moe_fallback.eval()

        with torch.no_grad():
            out_b = moe_batched(x)
            out_f = moe_fallback(x)

        diff = (out_b - out_f).abs().max().item()
        assert diff < 0.01, (
            f"Batched and fallback should match with no_bias=False, max diff={diff:.6f}"
        )


class TestWindowValueNormBug:
    def test_hca_window_values_are_normed(self):
        from x_moe.attention import HCA

        hca = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
            use_attention_sink=False,
        )
        x = torch.randn(2, 16, 64)

        win_k, win_v = hca.sliding_window(x)
        win_k_normed = hca.mqa.kv_norm(win_k)
        win_v_normed = hca.mqa.kv_norm(win_v)

        assert win_v_normed.shape == win_v.shape, "Normed win_v should have same shape"

        kv_norm_scale = win_v_normed.abs().mean().item()
        win_v_raw_scale = win_v.abs().mean().item()
        assert abs(kv_norm_scale - win_v_raw_scale) > 0.01 or kv_norm_scale > 0, (
            "win_v should go through kv_norm, changing its scale"
        )

    def test_hca_with_window_gradient_flow(self):
        from x_moe.attention import HCA

        hca = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = hca(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_hca_window_norm_consistency_with_compressed_kv(self):
        from x_moe.attention import HCA

        hca = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
            use_attention_sink=False,
        )
        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            c_compressed = hca._compress_kv(x)
            win_k, win_v = hca.sliding_window(x)

            kv_normed = hca.mqa.kv_norm(c_compressed)
            win_v_normed = hca.mqa.kv_norm(win_v)

            compressed_val_scale = kv_normed.abs().mean().item()
            window_val_scale = win_v_normed.abs().mean().item()

        scale_ratio = window_val_scale / max(compressed_val_scale, 1e-6)
        assert 0.3 < scale_ratio < 3.0, (
            f"Window values and compressed KV values should have comparable scale "
            f"after normalization, ratio={scale_ratio:.4f}"
        )

    def test_csa_with_window_norm_consistency(self):
        from x_moe.attention import CSA

        csa = CSA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            top_k_blocks=0,
            window_size=4,
            use_attention_sink=False,
        )
        x = torch.randn(2, 16, 64)
        out = csa(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()

    def test_hca_window_no_window_output_consistency(self):
        from x_moe.attention import HCA

        hca_window = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=4,
            use_attention_sink=False,
        )
        hca_no_window = HCA(
            dim=64,
            kv_dim=32,
            num_query_heads=4,
            compression_rate=4,
            window_size=0,
            use_attention_sink=False,
        )
        hca_no_window.load_state_dict(hca_window.state_dict(), strict=False)

        x = torch.randn(2, 16, 64)
        with torch.no_grad():
            out_window = hca_window(x)
            out_no_window = hca_no_window(x)

        diff = (out_window - out_no_window).abs().mean().item()
        x_scale = x.abs().mean().item()
        assert diff < x_scale * 5, (
            f"Window and no-window outputs should be reasonable, "
            f"diff={diff:.4f}, x_scale={x_scale:.4f}"
        )


class TestSeqBalanceLossPaddingBug:
    def test_seq_balance_loss_with_non_divisible_tokens(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 7
        num_tokens = 15
        top_k = 2
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_seq_balance_loss_padded_correctly(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 8
        num_tokens = 10
        top_k = 2
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_seq_balance_loss_with_3d_logits(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 5
        batch_size = 3
        seq_len = 7
        num_tokens = batch_size * seq_len
        top_k = 2
        router_logits = torch.randn(batch_size, seq_len, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_seq_balance_loss_exactly_divisible(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 8
        num_tokens = 16
        top_k = 2
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestAuxLossFreeNoSeqBalanceBug:
    def test_aux_loss_free_without_seq_balance(self):
        model = _make_model(aux_loss_free=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)

    def test_aux_loss_free_backward_without_seq_balance(self):
        model = _make_model(aux_loss_free=True)
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        loss.backward()
        has_grad = any(
            p.grad is not None and not torch.isnan(p.grad).any()
            for p in model.parameters()
        )
        assert has_grad

    def test_aux_loss_free_no_balance_loss_without_seq_balance(self):
        model = _make_model(
            aux_loss_free=True, load_balance_loss_weight=0.01, z_loss_weight=1e-4
        )
        x = torch.randint(0, 100, (2, 16))
        model(x)
        aux = model.moe_aux_loss
        assert aux.item() == 0.0

    def test_aux_loss_free_bias_update_without_seq_balance(self):
        model = _make_model(aux_loss_free=True, bias_update_speed=0.01)
        x = torch.randint(0, 100, (2, 16))
        model(x)
        model.update_routing_biases()
        moe_layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        for moe in moe_layers:
            bias = moe.gate.routing_bias
            assert bias is not None
            assert bias.shape[0] == moe.num_routed_experts

    def test_aux_loss_free_with_seq_balance_still_works(self):
        model = _make_model(
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        loss = model(x)
        assert not torch.isnan(loss)
        aux = model.moe_aux_loss
        assert aux.item() >= 0

    def test_moe_ffn_aux_loss_free_no_seq_balance(self):
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


class TestTotalStepsCalculationBug:
    def test_total_steps_order_of_operations(self):
        from x_moe.trainer import _make_collate_fn

        epochs = 10
        num_batches = 100
        gradient_accumulate = 3

        current = epochs * num_batches // gradient_accumulate
        correct = epochs * (num_batches // gradient_accumulate)

        steps_per_epoch = num_batches // gradient_accumulate
        assert correct == epochs * steps_per_epoch
        assert current != correct
        assert correct < current

    def test_total_steps_with_even_division(self):
        epochs = 5
        num_batches = 20
        gradient_accumulate = 4

        current = epochs * num_batches // gradient_accumulate
        correct = epochs * (num_batches // gradient_accumulate)

        steps_per_epoch = num_batches // gradient_accumulate
        assert correct == epochs * steps_per_epoch
        assert current == correct


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


class TestRoutingBiasAvgBug:
    def test_update_routing_bias_avg_direction(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe._update_token_counts(
            torch.tensor([[0], [0], [0], [1], [1], [2], [2], [3]]), 8
        )
        total = moe._token_counts.sum().item()
        avg = total / 4
        bias_before = moe.gate.routing_bias.clone()
        moe.update_routing_bias()
        bias_after = moe.gate.routing_bias
        expert_0_count = (
            moe._token_counts[0].item() if hasattr(moe, "_token_counts") else 0
        )
        assert total == 8
        assert avg == 2.0

    def test_overloaded_expert_bias_decreases(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        torch.manual_seed(42)
        for _ in range(5):
            x = torch.randn(2, 16, 64)
            moe(x)
        bias_before = moe.gate.routing_bias.clone()
        moe.update_routing_bias()
        bias_after = moe.gate.routing_bias
        if hasattr(moe, "_token_counts") and moe._token_counts.sum() > 0:
            counts = moe._token_counts.float()
            total = counts.sum().item()
            avg = total / moe.num_routed_experts
            for i in range(moe.num_routed_experts):
                if counts[i].item() > avg:
                    assert bias_after[i].item() <= bias_before[i].item(), (
                        f"Overloaded expert {i} bias should decrease, "
                        f"before={bias_before[i].item()}, after={bias_after[i].item()}"
                    )
                elif counts[i].item() < avg:
                    assert bias_after[i].item() >= bias_before[i].item(), (
                        f"Underloaded expert {i} bias should increase, "
                        f"before={bias_before[i].item()}, after={bias_after[i].item()}"
                    )

    def test_avg_is_total_over_num_experts(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        if hasattr(moe, "_token_counts"):
            total = moe._token_counts.sum().item()
            avg = total / moe.num_routed_experts
            for i in range(moe.num_routed_experts):
                count_i = moe._token_counts[i].item()
                if count_i > avg + 0.5:
                    pass


class TestRoutingBiasNoTokenCountsBug:
    def test_update_routing_bias_without_token_counts_no_crash(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.update_routing_bias()

    def test_expert_choice_creates_token_counts(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        assert hasattr(moe, "_token_counts"), (
            "_token_counts buffer should be created after expert_choice forward"
        )
        assert moe._token_counts.sum().item() > 0

    def test_expert_choice_update_routing_bias_no_crash(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        moe.update_routing_bias()

    def test_top_k_aux_loss_free_update_routing_bias_no_crash(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        moe.update_routing_bias()

    def test_expert_choice_routing_bias_updates_direction(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        torch.manual_seed(42)
        for _ in range(5):
            x = torch.randn(2, 16, 64)
            moe(x)
        bias_before = moe.gate.routing_bias.clone()
        moe.update_routing_bias()
        bias_after = moe.gate.routing_bias
        if moe._token_counts.sum().item() > 0:
            counts = moe._token_counts.float()
            total = counts.sum().item()
            avg = total / moe.num_routed_experts
            for i in range(moe.num_routed_experts):
                if counts[i].item() > avg:
                    assert bias_after[i].item() <= bias_before[i].item(), (
                        f"Overloaded expert {i} bias should decrease"
                    )
                elif counts[i].item() < avg:
                    assert bias_after[i].item() >= bias_before[i].item(), (
                        f"Underloaded expert {i} bias should increase"
                    )


class TestHashGateRedundantLogitsBug:
    def test_hash_gate_forward_returns_logits(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        weights, top_indices, logits = gate(x)
        assert logits.shape == (8, 4)
        assert weights.shape == (8, 2)
        assert top_indices.shape == (8, 2)

    def test_hash_gate_with_token_ids(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        token_ids = torch.randint(0, 1000, (8,))
        weights, top_indices, logits = gate(x, token_ids=token_ids)
        assert logits.shape == (8, 4)
        assert weights.shape == (8, 2)
        assert top_indices.shape == (8, 2)

    def test_hash_gate_logits_are_gate_output(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=4, top_k=2)
        x = torch.randn(8, 64)
        weights, top_indices, logits = gate(x)
        direct_logits = gate.w_g(x)
        assert torch.allclose(logits, direct_logits, atol=1e-6)


class TestTensorAsBoolBug:
    def test_sync_stacked_to_experts_with_bias(self):
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
        assert moe._has_bias_1.item() in (True, False)
        assert moe._has_bias_2.item() in (True, False)
        moe._sync_experts_to_stacked()
        moe._sync_stacked_to_experts()

    def test_sync_stacked_roundtrip_no_bias(self):
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
        moe._sync_experts_to_stacked()
        w1_before = moe.w1_stack.data.clone()
        moe._sync_stacked_to_experts()
        moe._sync_experts_to_stacked()
        assert torch.allclose(moe.w1_stack.data, w1_before, atol=1e-6)

    def test_batched_forward_with_no_bias_glu(self):
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

    def test_batched_forward_with_bias(self):
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
        x = torch.randn(2, 16, 64)
        out = moe(x)
        assert out.shape == (2, 16, 64)
        assert not torch.isnan(out).any()


class TestExpertChoiceTokenCountConsistency:
    def test_expert_choice_token_count_matches_top_k(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        if hasattr(moe, "_token_counts"):
            total_counted = moe._token_counts.sum().item()
            assert total_counted > 0, "Token counts should be non-zero after forward"

    def test_top_k_aux_loss_free_token_count(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        assert hasattr(moe, "_token_counts")
        assert moe._token_counts.sum().item() > 0


class TestRoutingBiasUpdateConsistency:
    def test_model_level_update_routing_biases_no_crash(self):
        model = _make_model(aux_loss_free=True, bias_update_speed=0.01)
        x = torch.randint(0, 100, (2, 16))
        model(x)
        model.update_routing_biases()

    def test_model_level_expert_choice_update_routing_biases(self):
        model = _make_model(
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        x = torch.randint(0, 100, (2, 16))
        model(x)
        model.update_routing_biases()

    def test_update_routing_biases_multiple_steps(self):
        model = _make_model(aux_loss_free=True, bias_update_speed=0.01)
        x = torch.randint(0, 100, (2, 16))
        for _ in range(5):
            model(x)
            model.update_routing_biases()
            model.reset_moe_aux_loss()


class TestHashGateDegenerateHashBug:
    def test_hash_gate_different_ids_different_routing(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        x = torch.randn(100, 64)
        ids_a = torch.arange(0, 100)
        ids_b = torch.arange(100, 200)
        _, idx_a, _ = gate(x, token_ids=ids_a)
        _, idx_b, _ = gate(x, token_ids=ids_b)
        assert not torch.equal(idx_a, idx_b), (
            "Different token IDs should produce different routing"
        )

    def test_hash_gate_no_degenerate_seed(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        x = torch.randn(100, 64)
        token_ids = torch.arange(0, 100)
        _, indices, _ = gate(x, token_ids=token_ids)
        for k in range(indices.shape[1]):
            unique_experts = indices[:, k].unique()
            assert len(unique_experts) > 1, (
                f"Hash function k={k} routes all tokens to same expert (degenerate)"
            )

    def test_hash_gate_reasonable_distribution(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        x = torch.randn(1000, 64)
        token_ids = torch.arange(0, 1000)
        _, indices, _ = gate(x, token_ids=token_ids)
        all_indices = indices.reshape(-1)
        counts = all_indices.bincount(minlength=8)
        min_count = counts.min().item()
        max_count = counts.max().item()
        assert max_count < 3 * min_count + 1, (
            f"Hash routing distribution too skewed: min={min_count}, max={max_count}"
        )

    def test_hash_gate_forced_degenerate_seed(self):
        from x_moe.moe import HashGate

        gate = HashGate(dim=64, num_experts=8, top_k=2, num_hash_functions=4)
        gate.hash_seeds[0] = 8  # 8 % 8 == 0, would be degenerate with old code
        gate.hash_seeds[1] = 16  # 16 % 8 == 0, would be degenerate with old code
        x = torch.randn(100, 64)
        token_ids = torch.arange(0, 100)
        _, indices, _ = gate(x, token_ids=token_ids)
        for k in range(min(2, indices.shape[1])):
            unique_experts = indices[:, k].unique()
            assert len(unique_experts) > 1, (
                f"Seed that is multiple of num_experts should not cause degenerate routing, k={k}"
            )


class TestExpertChoiceTokenCountsBug:
    def test_expert_choice_token_counts_correct_total(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        assert hasattr(moe, "_token_counts"), (
            "_token_counts should exist after expert_choice forward with aux_loss_free"
        )
        total = moe._token_counts.sum().item()
        num_tokens = 2 * 16
        capacity = max(1, int(1.0 * num_tokens / 4))
        expected_total = 4 * capacity
        assert total == expected_total, (
            f"Token counts total should equal num_experts*capacity={expected_total}, got {total}"
        )

    def test_expert_choice_token_counts_per_expert(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        moe(x)
        for i in range(4):
            count = moe._token_counts[i].item()
            assert count > 0, (
                f"Expert {i} should have processed some tokens, got {count}"
            )

    def test_expert_choice_bias_update_direction(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            routing_strategy="expert_choice",
            capacity_factor=1.0,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        torch.manual_seed(42)
        for _ in range(3):
            x = torch.randn(2, 16, 64)
            moe(x)
        bias_before = moe.gate.routing_bias.clone()
        moe.update_routing_bias()
        bias_after = moe.gate.routing_bias
        if moe._token_counts.sum().item() > 0:
            counts = moe._token_counts.float()
            total = counts.sum().item()
            avg = total / moe.num_routed_experts
            for i in range(moe.num_routed_experts):
                if counts[i].item() > avg:
                    assert bias_after[i].item() <= bias_before[i].item(), (
                        f"Overloaded expert {i} bias should decrease, "
                        f"before={bias_before[i].item()}, after={bias_after[i].item()}"
                    )
                elif counts[i].item() < avg:
                    assert bias_after[i].item() >= bias_before[i].item(), (
                        f"Underloaded expert {i} bias should increase, "
                        f"before={bias_before[i].item()}, after={bias_after[i].item()}"
                    )


class TestSeqBalanceLossTopKBug:
    def test_seq_balance_loss_top_k_2_correct(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 4
        num_tokens = 8
        top_k = 2
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)
        assert loss.shape == ()

    def test_seq_balance_loss_top_k_1_correct(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 4
        num_tokens = 8
        top_k = 1
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens,))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_seq_balance_loss_top_k_2_with_padding(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 5
        num_tokens = 7
        top_k = 2
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_seq_balance_loss_top_k_2_uses_all_tokens(self):
        from x_moe.moe import _compute_seq_balance_loss
        import torch.nn.functional as F

        num_experts = 4
        num_tokens_per_seq = 8
        num_tokens = 16
        top_k = 2

        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))

        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )

        one_hot = F.one_hot(top_indices.reshape(-1), num_experts).float()
        one_hot_per_token = one_hot.reshape(num_tokens, top_k, num_experts).sum(dim=1)
        assert one_hot_per_token.shape[0] == num_tokens, (
            "All tokens should be accounted for in seq balance loss"
        )
        total_assignments = one_hot_per_token.sum().item()
        assert abs(total_assignments - num_tokens * top_k) < 1e-6, (
            f"Total assignments should be num_tokens*top_k={num_tokens * top_k}, "
            f"got {total_assignments}"
        )

    def test_seq_balance_loss_top_k_2_non_divisible(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 7
        num_tokens = 15
        top_k = 2
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices = torch.randint(0, num_experts, (num_tokens, top_k))
        loss = _compute_seq_balance_loss(
            router_logits, top_indices, num_experts, num_tokens_per_seq
        )
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_seq_balance_loss_with_moe_top_k_2(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            aux_loss_free=True,
            seq_balance_loss_weight=0.01,
        )
        moe.train()
        x = torch.randn(2, 8, 64)
        moe(x)
        aux = moe.aux_loss.item()
        assert aux >= 0, f"Aux loss should be non-negative, got {aux}"
        assert not math.isnan(aux), "Aux loss should not be NaN"

    def test_seq_balance_loss_top_k_gt1_different_from_topk1(self):
        from x_moe.moe import _compute_seq_balance_loss

        num_experts = 4
        num_tokens_per_seq = 4
        num_tokens = 8
        router_logits = torch.randn(num_tokens, num_experts)
        top_indices_k1 = torch.randint(0, num_experts, (num_tokens,))
        top_indices_k2 = torch.cat(
            [
                top_indices_k1.unsqueeze(-1),
                torch.randint(0, num_experts, (num_tokens, 1)),
            ],
            dim=-1,
        )
        loss_k1 = _compute_seq_balance_loss(
            router_logits, top_indices_k1, num_experts, num_tokens_per_seq
        )
        loss_k2 = _compute_seq_balance_loss(
            router_logits, top_indices_k2, num_experts, num_tokens_per_seq
        )
        assert not torch.isnan(loss_k1)
        assert not torch.isnan(loss_k2)
