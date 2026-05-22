import pytest
import torch
import torch.nn as nn
import math

from x_moe.attention import (
    HCA,
    CSA,
    SharedKVMQA,
    AttentionSink,
    SlidingWindowKV,
    PartialRotaryEmbedding,
    DS4AttentionLayer,
    HybridAttentionBlock,
    _rotate_half,
    apply_partial_rope,
)


class TestPartialRotaryEmbedding:
    def test_output_shapes(self):
        rope = PartialRotaryEmbedding(dim=128, rot_dim=64)
        cos, sin = rope(seq_len=32, device=torch.device("cpu"), dtype=torch.float32)
        assert cos.shape == (32, 128)
        assert sin.shape == (32, 128)

    def test_rot_dim_clamp(self):
        rope = PartialRotaryEmbedding(dim=32, rot_dim=64)
        assert rope.rot_dim == 32

    def test_different_seq_lens(self):
        rope = PartialRotaryEmbedding(dim=64, rot_dim=32)
        for sl in [8, 16, 128]:
            cos, sin = rope(seq_len=sl, device=torch.device("cpu"), dtype=torch.float32)
            assert cos.shape == (sl, 64)


class TestRotateHalf:
    def test_rotate_half_shape(self):
        x = torch.randn(2, 8, 4)
        y = _rotate_half(x)
        assert y.shape == x.shape

    def test_rotate_half_values(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        y = _rotate_half(x)
        assert y.shape == (1, 1, 4)
        assert torch.allclose(y[0, 0, 0], torch.tensor(-3.0))
        assert torch.allclose(y[0, 0, 1], torch.tensor(-4.0))
        assert torch.allclose(y[0, 0, 2], torch.tensor(1.0))
        assert torch.allclose(y[0, 0, 3], torch.tensor(2.0))


class TestApplyPartialRoPE:
    def test_preserves_non_rotated_dims(self):
        t = torch.randn(2, 8, 64)
        rot_dim = 32
        cos = torch.randn(8, 64)
        sin = torch.randn(8, 64)
        out = apply_partial_rope(t, cos, sin, rot_dim)
        assert out.shape == t.shape
        assert torch.allclose(out[..., rot_dim:], t[..., rot_dim:], atol=1e-6)


class TestAttentionSink:
    def test_sink_output(self):
        sink = AttentionSink(num_heads=4)
        attn = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        out = sink(attn)
        assert out.shape == attn.shape
        assert (out >= 0).all()

    def test_sink_learnable(self):
        sink = AttentionSink(num_heads=2)
        assert sink.sink_logits.shape == (2,)
        assert sink.sink_logits.requires_grad


class TestSlidingWindowKV:
    def test_output_shapes(self):
        sw = SlidingWindowKV(dim=64, window_size=4, kv_dim=32)
        x = torch.randn(2, 16, 64)
        k, v = sw(x)
        assert k.shape == (2, 16, 32)
        assert v.shape == (2, 16, 32)


class TestSharedKVMQA:
    def test_output_shape(self):
        mqa = SharedKVMQA(dim=64, kv_dim=32, num_query_heads=4, num_groups=1)
        x = torch.randn(2, 8, 64)
        compressed_kv = torch.randn(2, 4, 32)
        out = mqa(x, compressed_kv)
        assert out.shape == (2, 8, 64)

    def test_grouped_output(self):
        mqa = SharedKVMQA(
            dim=64, kv_dim=16, num_query_heads=8, num_groups=2, group_out_dim=32
        )
        x = torch.randn(2, 8, 64)
        compressed_kv = torch.randn(2, 4, 16)
        out = mqa(x, compressed_kv)
        assert out.shape == (2, 8, 64)

    def test_with_sink(self):
        sink = AttentionSink(num_heads=4)
        mqa = SharedKVMQA(dim=64, kv_dim=32, num_query_heads=4, num_groups=1)
        x = torch.randn(2, 8, 64)
        compressed_kv = torch.randn(2, 4, 32)
        out = mqa(x, compressed_kv, sink=sink)
        assert out.shape == (2, 8, 64)

    def test_with_sliding_window(self):
        sw = SlidingWindowKV(dim=64, window_size=4, kv_dim=32)
        mqa = SharedKVMQA(dim=64, kv_dim=32, num_query_heads=4, num_groups=1)
        x = torch.randn(2, 8, 64)
        compressed_kv = torch.randn(2, 4, 32)
        win_k, win_v = sw(x)
        out = mqa(x, compressed_kv, win_k=win_k, win_v=win_v)
        assert out.shape == (2, 8, 64)


class TestHCA:
    def test_output_shape(self):
        hca = HCA(dim=64, kv_dim=32, num_query_heads=4, compression_rate=4, window_size=4)
        x = torch.randn(2, 16, 64)
        out = hca(x)
        assert out.shape == (2, 16, 64)

    def test_compression_rate(self):
        hca = HCA(dim=64, kv_dim=32, num_query_heads=4, compression_rate=8, window_size=0)
        x = torch.randn(2, 32, 64)
        out = hca(x)
        assert out.shape == (2, 32, 64)

    def test_no_window(self):
        hca = HCA(dim=64, kv_dim=32, num_query_heads=4, compression_rate=4, window_size=0)
        x = torch.randn(2, 16, 64)
        out = hca(x)
        assert out.shape == (2, 16, 64)

    def test_no_sink(self):
        hca = HCA(
            dim=64, kv_dim=32, num_query_heads=4, compression_rate=4,
            window_size=4, use_attention_sink=False
        )
        x = torch.randn(2, 16, 64)
        out = hca(x)
        assert out.shape == (2, 16, 64)

    def test_no_rope(self):
        hca = HCA(
            dim=64, kv_dim=32, num_query_heads=4, compression_rate=4,
            window_size=4, use_partial_rope=False
        )
        x = torch.randn(2, 16, 64)
        out = hca(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flow(self):
        hca = HCA(dim=64, kv_dim=32, num_query_heads=4, compression_rate=4, window_size=0)
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = hca(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_grouped_output(self):
        hca = HCA(
            dim=64, kv_dim=32, num_query_heads=8, compression_rate=4,
            num_groups=2, group_out_dim=32, window_size=0,
        )
        x = torch.randn(2, 16, 64)
        out = hca(x)
        assert out.shape == (2, 16, 64)


class TestCSA:
    def test_output_shape(self):
        csa = CSA(
            dim=64, kv_dim=32, num_query_heads=4, compression_rate=4,
            top_k_blocks=2, window_size=0,
        )
        x = torch.randn(2, 16, 64)
        out = csa(x)
        assert out.shape == (2, 16, 64)

    def test_with_topk(self):
        csa = CSA(
            dim=64, kv_dim=32, num_query_heads=4, compression_rate=4,
            top_k_blocks=2, window_size=4,
        )
        x = torch.randn(2, 32, 64)
        out = csa(x)
        assert out.shape == (2, 32, 64)

    def test_gradient_flow(self):
        csa = CSA(
            dim=64, kv_dim=32, num_query_heads=4, compression_rate=4,
            top_k_blocks=0, window_size=0,
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = csa(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestDS4AttentionLayer:
    def test_hca_layer(self):
        layer = DS4AttentionLayer(dim=64, attn_type="hca", kv_dim=32, num_query_heads=4, compression_rate=4, window_size=0)
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_csa_layer(self):
        layer = DS4AttentionLayer(
            dim=64, attn_type="csa", kv_dim=32, num_query_heads=4,
            compression_rate=4, top_k_blocks=0, window_size=0,
        )
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            DS4AttentionLayer(dim=64, attn_type="invalid")


class TestHybridAttentionBlock:
    def test_hca_only(self):
        block = HybridAttentionBlock(
            dim=64,
            hca_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "window_size": 0},
        )
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_csa_only(self):
        block = HybridAttentionBlock(
            dim=64,
            csa_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "top_k_blocks": 0, "window_size": 0},
        )
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_hybrid(self):
        block = HybridAttentionBlock(
            dim=64,
            hca_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "window_size": 0},
            csa_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "top_k_blocks": 0, "window_size": 0},
        )
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == (2, 16, 64)

    def test_gradient_flow(self):
        block = HybridAttentionBlock(
            dim=64,
            hca_config={"kv_dim": 32, "num_query_heads": 4, "compression_rate": 4, "window_size": 0},
        )
        x = torch.randn(2, 16, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None