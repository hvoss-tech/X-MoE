import contextlib

import pytest
import torch
import torch._dynamo as dynamo

from x_moe import MoEFFN


def _check_no_graph_breaks(model, x):
    """Compile model with torch.compile(fullgraph=True) and verify no graph breaks.

    fullgraph=True raises an error if any graph breaks are found.
    We also run a backward pass to ensure the full training step is compilable.
    """
    model.zero_grad()
    dynamo.reset()
    compiled = torch.compile(model, fullgraph=True)
    out = compiled(x)
    out.sum().backward()
    model.zero_grad()
    dynamo.reset()


class TestNoGraphBreaks:
    def test_batched_experts_top_k_no_graph_break(self):
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
        moe.train()
        x = torch.randn(2, 16, 64)
        _check_no_graph_breaks(moe, x)

    def test_batched_experts_with_bias_no_graph_break(self):
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
        moe.train()
        x = torch.randn(2, 16, 64)
        _check_no_graph_breaks(moe, x)

    def test_batched_experts_no_glu_no_graph_break(self):
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
        moe.train()
        x = torch.randn(2, 16, 64)
        _check_no_graph_breaks(moe, x)

    def test_batched_experts_aux_loss_free_no_graph_break(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
            aux_loss_free=True,
            bias_update_speed=0.01,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        _check_no_graph_breaks(moe, x)

    def test_batched_experts_sigmoid_routing_no_graph_break(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
            sigmoid_routing=True,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        _check_no_graph_breaks(moe, x)

    def test_batched_experts_swiglu_clamp_no_graph_break(self):
        moe = MoEFFN(
            dim=64,
            num_experts=4,
            expert_top_k=2,
            glu=True,
            mult=4,
            no_bias=True,
            batched_experts=True,
            max_seq_len=64,
            swiglu_clamp_value=2.0,
        )
        moe.train()
        x = torch.randn(2, 16, 64)
        _check_no_graph_breaks(moe, x)

    def test_batched_experts_forward_backward_correctness(self):
        """Verify compiled model produces same results as eager."""
        torch.manual_seed(42)
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
        moe.train()
        x = torch.randn(2, 16, 64, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)

        out_eager = moe(x)
        out_eager.sum().backward()

        dynamo.reset()
        compiled = torch.compile(moe, fullgraph=True)
        moe.zero_grad()
        out_compiled = compiled(x2)
        out_compiled.sum().backward()
        dynamo.reset()

        max_out_diff = (out_eager - out_compiled).abs().max().item()
        assert torch.allclose(out_eager, out_compiled, atol=1e-5), (
            f"Eager and compiled outputs differ: max diff={max_out_diff:.6f}"
        )
        assert x.grad is not None and x2.grad is not None
        max_grad_diff = (x.grad - x2.grad).abs().max().item()
        assert torch.allclose(x.grad, x2.grad, atol=1e-5), (
            f"Eager and compiled gradients differ: max diff={max_grad_diff:.6f}"
        )

    def test_batched_experts_no_bias_forward_backward_correctness(self):
        """Verify compiled model with no_bias=True produces same results as eager."""
        torch.manual_seed(42)
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
        moe.train()
        x = torch.randn(2, 16, 64, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)

        out_eager = moe(x)
        out_eager.sum().backward()

        dynamo.reset()
        compiled = torch.compile(moe, fullgraph=True)
        moe.zero_grad()
        out_compiled = compiled(x2)
        out_compiled.sum().backward()
        dynamo.reset()

        max_out_diff = (out_eager - out_compiled).abs().max().item()
        assert torch.allclose(out_eager, out_compiled, atol=1e-5), (
            f"Eager and compiled outputs differ (no_bias): max diff={max_out_diff:.6f}"
        )
