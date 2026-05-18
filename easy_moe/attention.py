import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from x_transformers.x_transformers import RMSNorm


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(t: Tensor, freqs_cos: Tensor, freqs_sin: Tensor, rot_dim: int) -> Tensor:
    t_rot = t[..., :rot_dim]
    t_pass = t[..., rot_dim:]
    fc = freqs_cos[..., :rot_dim]
    fs = freqs_sin[..., :rot_dim]
    t_rot_out = t_rot * fc + _rotate_half(t_rot) * fs
    return torch.cat([t_rot_out, t_pass], dim=-1)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, rot_dim: int = 64, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.rot_dim = min(rot_dim, dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rot_dim, 2).float() / self.rot_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("n,j->nj", t, self.inv_freq.to(dtype))
        emb_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
        emb_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
        full_cos = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
        full_sin = torch.zeros(seq_len, self.dim, device=device, dtype=dtype)
        full_cos[:, : self.rot_dim] = emb_cos[:, : self.rot_dim]
        full_sin[:, : self.rot_dim] = emb_sin[:, : self.rot_dim]
        return full_cos, full_sin


class AttentionSink(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.sink_logits = nn.Parameter(torch.zeros(num_heads))

    def forward(self, attn_logits: Tensor) -> Tensor:
        sink_exp = torch.exp(self.sink_logits).view(1, self.num_heads, 1, 1)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + sink_exp)
        return attn_weights


class SlidingWindowKV(nn.Module):
    def __init__(self, dim: int, window_size: int, kv_dim: int):
        super().__init__()
        self.window_size = window_size
        self.w_k = nn.Linear(dim, kv_dim, bias=False)
        self.w_v = nn.Linear(dim, kv_dim, bias=False)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        return self.w_k(x), self.w_v(x)


class SharedKVMQA(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_query_heads: int,
        num_groups: int = 1,
        group_out_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim
        self.num_query_heads = num_query_heads
        self.num_groups = num_groups
        self.head_dim = kv_dim
        self.group_out_dim = group_out_dim if group_out_dim is not None else dim // num_groups

        self.w_dq = nn.Linear(dim, dim, bias=False)
        self.w_uq = nn.Linear(dim, kv_dim * num_query_heads, bias=False)

        if num_groups > 1:
            self.group_projections = nn.ModuleList(
                [nn.Linear(kv_dim * (num_query_heads // num_groups), self.group_out_dim, bias=False)
                 for _ in range(num_groups)]
            )
            self.output_proj = nn.Linear(self.group_out_dim * num_groups, dim, bias=False)
        else:
            self.output_proj = nn.Linear(kv_dim * num_query_heads, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.query_norm = RMSNorm(kv_dim)
        self.kv_norm = RMSNorm(kv_dim)

    def forward(
        self,
        x: Tensor,
        compressed_kv: Tensor,
        win_k: Tensor | None = None,
        win_v: Tensor | None = None,
        sink: AttentionSink | None = None,
    ) -> Tensor:
        b, n, _ = x.shape
        num_kv = compressed_kv.shape[1]

        c_q = self.w_dq(x)
        c_q = self.w_uq(c_q)
        q = c_q.view(b, n, self.num_query_heads, self.kv_dim)

        q = self.query_norm(q)
        kv = self.kv_norm(compressed_kv)

        keys = kv.unsqueeze(2).expand(-1, -1, self.num_query_heads, -1)
        values = keys.clone()

        if win_k is not None and win_v is not None:
            win_k_n = self.kv_norm(win_k)
            win_k_exp = win_k_n.unsqueeze(2).expand(-1, -1, self.num_query_heads, -1)
            win_v_exp = win_v.unsqueeze(2).expand(-1, -1, self.num_query_heads, -1)
            keys = torch.cat([keys, win_k_exp], dim=1)
            values = torch.cat([values, win_v_exp], dim=1)

        q_t = q.transpose(1, 2)
        k_t = keys.transpose(1, 2)
        v_t = values.transpose(1, 2)

        scale = self.kv_dim ** -0.5

        if sink is not None:
            attn_logits = torch.einsum("bhid,bhjd->bhij", q_t, k_t) * scale
            attn_weights = sink(attn_logits)
            out = torch.einsum("bhij,bhjd->bhid", attn_weights, v_t)
        else:
            out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=scale)

        out = out.transpose(1, 2).contiguous().view(b, n, self.num_query_heads * self.kv_dim)

        if self.num_groups > 1:
            heads_per_group = self.num_query_heads // self.num_groups
            group_outputs = []
            for g in range(self.num_groups):
                start = g * heads_per_group * self.kv_dim
                end = (g + 1) * heads_per_group * self.kv_dim
                group_out = out[:, :, start:end]
                group_out = self.group_projections[g](group_out)
                group_outputs.append(group_out)
            out = torch.cat(group_outputs, dim=-1)

        return self.output_proj(out)


class HCA(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: int = 128,
        num_query_heads: int = 8,
        compression_rate: int = 8,
        num_groups: int = 1,
        group_out_dim: int | None = None,
        window_size: int = 32,
        use_attention_sink: bool = True,
        use_partial_rope: bool = True,
        rope_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim
        self.compression_rate = compression_rate
        self.window_size = window_size
        self.use_partial_rope = use_partial_rope
        self.rope_dim = min(rope_dim, kv_dim)

        self.w_kv = nn.Linear(dim, kv_dim, bias=False)
        self.w_z = nn.Linear(dim, kv_dim, bias=False)
        self.pos_bias = nn.Parameter(torch.zeros(compression_rate, kv_dim))

        self.mqa = SharedKVMQA(
            dim=dim,
            kv_dim=kv_dim,
            num_query_heads=num_query_heads,
            num_groups=num_groups,
            group_out_dim=group_out_dim,
            dropout=dropout,
        )

        self.sliding_window = SlidingWindowKV(dim, window_size, kv_dim) if window_size > 0 else None
        self.sink = AttentionSink(num_query_heads) if use_attention_sink else None

    def _compress_kv(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        c = self.w_kv(x)
        z = self.w_z(x)

        pad_len = (self.compression_rate - n % self.compression_rate) % self.compression_rate
        if pad_len > 0:
            c = F.pad(c, (0, 0, 0, pad_len))
            z = F.pad(z, (0, 0, 0, pad_len))

        c = c.view(b, -1, self.compression_rate, self.kv_dim)
        z = z.view(b, -1, self.compression_rate, self.kv_dim)

        pos_bias = self.pos_bias.unsqueeze(0).unsqueeze(0)
        z_weights = F.softmax(z + pos_bias, dim=2)
        c_compressed = (z_weights * c).sum(dim=2)

        return c_compressed

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        b, n, _ = x.shape
        c_compressed = self._compress_kv(x)

        win_k, win_v = None, None
        if self.sliding_window is not None:
            win_k, win_v = self.sliding_window(x)

        out = self.mqa(
            x=x,
            compressed_kv=c_compressed,
            win_k=win_k,
            win_v=win_v,
            sink=self.sink,
        )

        if win_k is not None:
            out = out + x

        return out


class CSA(nn.Module):
    def __init__(
        self,
        dim: int,
        kv_dim: int = 128,
        num_query_heads: int = 8,
        compression_rate: int = 4,
        top_k_blocks: int = 32,
        num_groups: int = 1,
        group_out_dim: int | None = None,
        window_size: int = 32,
        use_attention_sink: bool = True,
        use_partial_rope: bool = True,
        rope_dim: int = 64,
        indexer_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.kv_dim = kv_dim
        self.compression_rate = compression_rate
        self.top_k_blocks = top_k_blocks
        self.window_size = window_size
        self.use_partial_rope = use_partial_rope
        self.rope_dim = min(rope_dim, kv_dim)
        self.indexer_dim = indexer_dim or dim // 4

        self.w_kv = nn.Linear(dim, kv_dim, bias=False)
        self.w_z = nn.Linear(dim, kv_dim, bias=False)
        self.pos_bias = nn.Parameter(torch.zeros(compression_rate, kv_dim))

        self.overlap = max(1, compression_rate // 2)

        self.w_idx_q = nn.Linear(dim, self.indexer_dim, bias=False)
        self.w_idx_k = nn.Linear(kv_dim, self.indexer_dim, bias=False)

        self.mqa = SharedKVMQA(
            dim=dim,
            kv_dim=kv_dim,
            num_query_heads=num_query_heads,
            num_groups=num_groups,
            group_out_dim=group_out_dim,
            dropout=dropout,
        )

        self.sliding_window = SlidingWindowKV(dim, window_size, kv_dim) if window_size > 0 else None
        self.sink = AttentionSink(num_query_heads) if use_attention_sink else None

    def _compress_kv_overlapped(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        c = self.w_kv(x)
        z = self.w_z(x)

        step = self.compression_rate - self.overlap
        if step <= 0:
            step = 1

        block_starts = list(range(0, max(n - self.compression_rate + 1, 1), step))
        if not block_starts:
            block_starts = [0]
        last_start = block_starts[-1]
        if last_start + self.compression_rate > n and last_start > 0:
            block_starts[-1] = max(n - self.compression_rate, 0)

        c_blocks = []
        z_blocks = []
        for s in block_starts:
            end = min(s + self.compression_rate, n)
            chunk_c = c[:, s:end]
            chunk_z = z[:, s:end]
            if chunk_c.shape[1] < self.compression_rate:
                pad_size = self.compression_rate - chunk_c.shape[1]
                chunk_c = F.pad(chunk_c, (0, 0, 0, pad_size))
                chunk_z = F.pad(chunk_z, (0, 0, 0, pad_size))
            c_blocks.append(chunk_c)
            z_blocks.append(chunk_z)

        c_stacked = torch.stack(c_blocks, dim=1)
        z_stacked = torch.stack(z_blocks, dim=1)

        pos_bias = self.pos_bias.unsqueeze(0).unsqueeze(0)
        z_weights = F.softmax(z_stacked + pos_bias, dim=2)
        c_compressed = (z_weights * c_stacked).sum(dim=2)

        return c_compressed

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        b, n, _ = x.shape
        c_compressed = self._compress_kv_overlapped(x)

        num_blocks = c_compressed.shape[1]
        if self.top_k_blocks > 0 and num_blocks > self.top_k_blocks:
            idx_q = self.w_idx_q(x)
            idx_k = self.w_idx_k(c_compressed)

            idx_q_pool = idx_q.mean(dim=1, keepdim=True)
            scores = torch.einsum("bnd,bmd->bnm", idx_q_pool, idx_k)
            _, topk_indices = scores.topk(self.top_k_blocks, dim=-1)
            topk_indices = topk_indices.squeeze(1)

            c_selected = torch.gather(
                c_compressed,
                1,
                topk_indices.unsqueeze(-1).expand(-1, self.top_k_blocks, self.kv_dim),
            )
            c_compressed = c_selected

        win_k, win_v = None, None
        if self.sliding_window is not None:
            win_k, win_v = self.sliding_window(x)

        out = self.mqa(
            x=x,
            compressed_kv=c_compressed,
            win_k=win_k,
            win_v=win_v,
            sink=self.sink,
        )

        if win_k is not None:
            out = out + x

        return out


class DS4AttentionLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_type: str = "hca",
        kv_dim: int = 128,
        num_query_heads: int = 8,
        compression_rate: int = 8,
        top_k_blocks: int = 32,
        num_groups: int = 1,
        group_out_dim: int | None = None,
        window_size: int = 32,
        use_attention_sink: bool = True,
        use_partial_rope: bool = True,
        rope_dim: int = 64,
        indexer_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "hca":
            self.attn = HCA(
                dim=dim,
                kv_dim=kv_dim,
                num_query_heads=num_query_heads,
                compression_rate=compression_rate,
                num_groups=num_groups,
                group_out_dim=group_out_dim,
                window_size=window_size,
                use_attention_sink=use_attention_sink,
                use_partial_rope=use_partial_rope,
                rope_dim=rope_dim,
                dropout=dropout,
            )
        elif attn_type == "csa":
            self.attn = CSA(
                dim=dim,
                kv_dim=kv_dim,
                num_query_heads=num_query_heads,
                compression_rate=compression_rate,
                top_k_blocks=top_k_blocks,
                num_groups=num_groups,
                group_out_dim=group_out_dim,
                window_size=window_size,
                use_attention_sink=use_attention_sink,
                use_partial_rope=use_partial_rope,
                rope_dim=rope_dim,
                indexer_dim=indexer_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}, use 'hca' or 'csa'")

        self.norm = RMSNorm(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return x + self.attn(self.norm(x), **kwargs)


class HybridAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hca_config: dict | None = None,
        csa_config: dict | None = None,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        layers = []
        if hca_config is not None:
            cfg = {"dim": dim, **hca_config}
            layers.append(DS4AttentionLayer(**cfg, attn_type="hca"))
        if csa_config is not None:
            cfg = {"dim": dim, **csa_config}
            layers.append(DS4AttentionLayer(**cfg, attn_type="csa"))
        self.layers = nn.ModuleList(layers)
        self.norm = RMSNorm(dim) if use_rmsnorm else nn.LayerNorm(dim)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        for layer in self.layers:
            x = layer(x, **kwargs)
        return self.norm(x)