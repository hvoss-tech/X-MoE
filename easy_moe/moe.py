import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from x_transformers.x_transformers import FeedForward


class TopKGate(nn.Module):
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.w_g = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: Tensor):
        logits = self.w_g(x)
        scores = F.softmax(logits, dim=-1)
        top_k = min(self.top_k, self.num_experts)
        top_scores, top_indices = scores.topk(top_k, dim=-1)
        weights = top_scores / top_scores.sum(dim=-1, keepdim=True)
        return weights, top_indices, logits


class ExpertChoiceGate(nn.Module):
    def __init__(self, dim: int, num_experts: int, capacity_factor: float = 1.0):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.w_g = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x_flat: Tensor, num_tokens: int):
        logits = self.w_g(x_flat)
        scores = F.softmax(logits, dim=-1)
        capacity = max(1, int(self.capacity_factor * num_tokens / self.num_experts))
        capacity = min(capacity, num_tokens)
        expert_scores = scores.t()
        top_scores, top_indices = expert_scores.topk(capacity, dim=-1)
        return scores, top_scores, top_indices, capacity, logits


def _compute_load_balance_loss(
    router_logits: Tensor,
    top_indices: Tensor,
    num_experts: int,
) -> Tensor:
    if router_logits.ndim == 3:
        router_logits = router_logits.reshape(-1, num_experts)
    num_tokens = router_logits.shape[0]
    top_k = top_indices.shape[-1] if top_indices.ndim > 1 else 1
    router_probs = F.softmax(router_logits, dim=-1)
    avg_probs = router_probs.mean(dim=0)
    with torch.no_grad():
        one_hot = F.one_hot(top_indices.reshape(-1), num_experts).float()
        tokens_per_expert = one_hot.sum(dim=0)
        fraction_per_expert = tokens_per_expert / max(num_tokens * top_k, 1)
    balance_loss = (fraction_per_expert * avg_probs).sum() * num_experts
    return balance_loss


def _compute_z_loss(router_logits: Tensor) -> Tensor:
    if router_logits.ndim == 3:
        router_logits = router_logits.reshape(-1, router_logits.shape[-1])
    z = router_logits.logsumexp(dim=-1)
    return z.pow(2).mean()


class MoEFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expert_top_k: int = 2,
        capacity_factor: float = 1.25,
        routing_strategy: str = "top_k",
        load_balance_loss_weight: float = 0.01,
        z_loss_weight: float = 1e-4,
        glu: bool = True,
        mult: int = 4,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = True,
        activation: Optional[nn.Module] = None,
        batched_experts: bool = False,
        max_seq_len: int = 256,
        max_batch_size: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.capacity_factor = capacity_factor
        self.routing_strategy = routing_strategy
        self.load_balance_loss_weight = load_balance_loss_weight
        self.z_loss_weight = z_loss_weight
        self.batched_experts = batched_experts
        self._glu = glu
        self._inner_dim = dim * mult
        self._no_bias = no_bias
        self._capacity = math.ceil(
            capacity_factor * max_seq_len * max_batch_size * expert_top_k / num_experts
        )

        self.experts = nn.ModuleList(
            [
                FeedForward(
                    dim=dim,
                    mult=mult,
                    glu=glu,
                    dropout=dropout,
                    no_bias=no_bias,
                    zero_init_output=zero_init_output,
                )
                for _ in range(num_experts)
            ]
        )

        if routing_strategy == "top_k":
            self.gate = TopKGate(dim, num_experts, top_k=expert_top_k)
        elif routing_strategy == "expert_choice":
            self.gate = ExpertChoiceGate(dim, num_experts, capacity_factor)
        else:
            raise ValueError(
                f"Unknown routing strategy: {routing_strategy}. "
                f"Use 'top_k' or 'expert_choice'."
            )

        self.register_buffer("_aux_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer(
            "_num_forward_passes", torch.tensor(0, dtype=torch.long), persistent=False
        )
        self._compute_aux_loss = True

        if batched_experts:
            self._init_stacked_params()

    def _init_stacked_params(self):
        proj_out_dim = self._inner_dim * (2 if self._glu else 1)
        self.register_buffer("_has_bias_1", torch.tensor(False))
        self.register_buffer("_has_bias_2", torch.tensor(False))

        self.w1_stack = nn.Parameter(
            torch.empty(self.num_experts, proj_out_dim, self.dim)
        )
        self.w2_stack = nn.Parameter(
            torch.empty(self.num_experts, self.dim, self._inner_dim)
        )

        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                ff_seq = expert.ff
                self.w1_stack.data[i] = ff_seq[0].proj.weight.data
                self.w2_stack.data[i] = ff_seq[2].weight.data

        if not self._no_bias:
            b1_exists = any(
                expert.ff[0].proj.bias is not None for expert in self.experts
            )
            b2_exists = any(expert.ff[2].bias is not None for expert in self.experts)
            if b1_exists:
                self.b1_stack = nn.Parameter(
                    torch.empty(self.num_experts, proj_out_dim)
                )
                self._has_bias_1.fill_(True)
                with torch.no_grad():
                    for i, expert in enumerate(self.experts):
                        if expert.ff[0].proj.bias is not None:
                            self.b1_stack.data[i] = expert.ff[0].proj.bias.data
            if b2_exists:
                self.b2_stack = nn.Parameter(torch.empty(self.num_experts, self.dim))
                self._has_bias_2.fill_(True)
                with torch.no_grad():
                    for i, expert in enumerate(self.experts):
                        if expert.ff[2].bias is not None:
                            self.b2_stack.data[i] = expert.ff[2].bias.data

        try:
            self._dropout_p = self.experts[0].ff[1].p
        except AttributeError:
            self._dropout_p = 0.0

    def _sync_stacked_to_experts(self):
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                ff_seq = expert.ff
                ff_seq[0].proj.weight.data.copy_(self.w1_stack.data[i])
                ff_seq[2].weight.data.copy_(self.w2_stack.data[i])
                if (
                    self._has_bias_1
                    and hasattr(self, "b1_stack")
                    and ff_seq[0].proj.bias is not None
                ):
                    ff_seq[0].proj.bias.data.copy_(self.b1_stack.data[i])
                if (
                    self._has_bias_2
                    and hasattr(self, "b2_stack")
                    and ff_seq[2].bias is not None
                ):
                    ff_seq[2].bias.bias.data.copy_(self.b2_stack.data[i])

    def _sync_experts_to_stacked(self):
        with torch.no_grad():
            for i, expert in enumerate(self.experts):
                ff_seq = expert.ff
                self.w1_stack.data[i] = ff_seq[0].proj.weight.data
                self.w2_stack.data[i] = ff_seq[2].weight.data
                if (
                    self._has_bias_1
                    and hasattr(self, "b1_stack")
                    and ff_seq[0].proj.bias is not None
                ):
                    self.b1_stack.data[i] = ff_seq[0].proj.bias.data
                if (
                    self._has_bias_2
                    and hasattr(self, "b2_stack")
                    and ff_seq[2].bias is not None
                ):
                    self.b2_stack.data[i] = ff_seq[2].bias.data

    @property
    def aux_loss(self):
        if self._num_forward_passes.item() > 0:
            return self._aux_loss / self._num_forward_passes.float()
        return torch.tensor(0.0, device=self._aux_loss.device)

    def reset_aux_loss(self):
        device = next(self.parameters()).device
        self._aux_loss = torch.tensor(0.0, device=device)
        self._num_forward_passes.fill_(0)

    def _accumulate_aux_loss(self, router_logits, top_indices):
        if not self._compute_aux_loss:
            return
        balance_loss = _compute_load_balance_loss(
            router_logits, top_indices, self.num_experts
        )
        z_loss = _compute_z_loss(router_logits)
        aux = self.load_balance_loss_weight * balance_loss + self.z_loss_weight * z_loss
        self._aux_loss = self._aux_loss + aux.detach()
        self._num_forward_passes.add_(1)

    def _forward_top_k_vectorized(self, x: Tensor, deep_embed=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]
        weights, top_indices, router_logits = self.gate(x_flat)
        top_k = top_indices.shape[-1]

        flat_expert_ids = top_indices.reshape(-1)
        flat_weights = weights.reshape(-1)
        flat_token_ids = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )

        sort_idx = flat_expert_ids.argsort(stable=True)
        sorted_expert_ids = flat_expert_ids[sort_idx]
        sorted_token_ids = flat_token_ids[sort_idx]
        sorted_weights = flat_weights[sort_idx]

        expert_counts = flat_expert_ids.bincount(minlength=self.num_experts)
        offsets = torch.zeros(self.num_experts + 1, device=x.device, dtype=torch.long)
        offsets[1:] = expert_counts.cumsum(0)
        local_positions = (
            torch.arange(sort_idx.shape[0], device=x.device, dtype=torch.long)
            - offsets[sorted_expert_ids]
        )

        capacity = self._capacity
        in_bounds = local_positions < capacity
        in_bounds_float = in_bounds.float()

        padded_input = torch.zeros(
            self.num_experts,
            capacity,
            self.dim,
            device=x.device,
            dtype=x.dtype,
        )
        padded_weights = torch.zeros(
            self.num_experts,
            capacity,
            device=x.device,
            dtype=x.dtype,
        )
        pad_mask = torch.zeros(
            self.num_experts,
            capacity,
            dtype=torch.bool,
            device=x.device,
        )

        sorted_x = x_flat[sorted_token_ids]
        sorted_in_bounds_x = sorted_x * in_bounds_float.unsqueeze(-1)
        sorted_in_bounds_w = sorted_weights * in_bounds_float

        flat_expert_idx = sorted_expert_ids * capacity + local_positions
        valid_flat_idx = flat_expert_idx.clamp(max=self.num_experts * capacity - 1)

        padded_input_flat = padded_input.reshape(-1, self.dim)
        padded_input_flat.scatter_add_(
            0,
            valid_flat_idx.unsqueeze(-1).expand_as(sorted_in_bounds_x),
            sorted_in_bounds_x,
        )

        padded_weights_flat = padded_weights.reshape(-1)
        padded_weights_flat.scatter_add_(0, valid_flat_idx, sorted_in_bounds_w)

        pad_mask_flat = pad_mask.reshape(-1).float()
        pad_mask_flat.scatter_add_(0, valid_flat_idx, in_bounds_float)
        pad_mask = pad_mask_flat.reshape(self.num_experts, capacity).clamp(max=1).bool()

        batched_out = self._batched_forward(padded_input, pad_mask)

        weighted_out = batched_out * padded_weights.unsqueeze(-1)
        weighted_out = weighted_out * pad_mask.unsqueeze(-1).float()

        output_flat = torch.zeros_like(x_flat)
        gathered = weighted_out[
            sorted_expert_ids, local_positions.clamp(max=capacity - 1)
        ]
        gathered = gathered * in_bounds_float.unsqueeze(-1)
        output_flat.index_add_(0, sorted_token_ids, gathered)

        self._accumulate_aux_loss(router_logits, top_indices)
        return output_flat.reshape(orig_shape)

    def _forward_top_k(self, x: Tensor, deep_embed=None):
        if deep_embed is not None:
            return self._forward_top_k_fallback(x, deep_embed=deep_embed)
        if self.batched_experts:
            return self._forward_top_k_vectorized(x)
        return self._forward_top_k_fallback(x)

    def _forward_top_k_fallback(self, x: Tensor, deep_embed=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]
        weights, top_indices, router_logits = self.gate(x_flat)
        top_k = top_indices.shape[-1]

        flat_indices = top_indices.reshape(-1)
        flat_weights = weights.reshape(-1, 1)
        token_expert_pairs = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )

        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            expert_mask = flat_indices == expert_idx
            if not expert_mask.any():
                continue
            selected_tokens = token_expert_pairs[expert_mask]
            selected_weights = flat_weights[expert_mask]
            expert_input = x_flat[selected_tokens]
            expert_out = self.experts[expert_idx](expert_input, deep_embed=deep_embed)
            weighted_out = selected_weights * expert_out
            output.scatter_add_(
                0, selected_tokens.unsqueeze(-1).expand_as(weighted_out), weighted_out
            )

        self._accumulate_aux_loss(router_logits, top_indices)
        return output.reshape(orig_shape)

    def _batched_forward(self, padded_input: Tensor, pad_mask: Tensor):
        if not self.batched_experts or not hasattr(self, "w1_stack"):
            out_all = torch.zeros_like(padded_input)
            for i in range(self.num_experts):
                count = pad_mask[i].sum().item()
                if count > 0:
                    out_all[i, :count] = self.experts[i](padded_input[i, :count])
            return out_all

        h1 = torch.einsum("esi,eoi->eso", padded_input, self.w1_stack)
        if hasattr(self, "b1_stack") and self._has_bias_1:
            h1 = h1 + self.b1_stack.unsqueeze(1)

        if self._glu:
            h1, h1_gate = h1.chunk(2, dim=-1)
            h1 = F.silu(h1_gate) * h1
        else:
            h1 = F.gelu(h1)

        if self._dropout_p > 0 and self.training:
            h1 = F.dropout(h1, p=self._dropout_p, training=True)

        h2 = torch.einsum("esi,eoi->eso", h1, self.w2_stack)
        if hasattr(self, "b2_stack") and self._has_bias_2:
            h2 = h2 + self.b2_stack.unsqueeze(1)

        h2 = h2 * pad_mask.unsqueeze(-1).float()
        return h2

    def _forward_expert_choice_optimized(self, x: Tensor, deep_embed=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]
        scores, top_scores, top_indices, capacity, router_logits = self.gate(
            x_flat, num_tokens
        )
        output = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            selected_indices = top_indices[expert_idx]
            if not torch.any(selected_indices >= 0):
                continue
            valid_mask = selected_indices < num_tokens
            selected_indices = selected_indices[valid_mask]
            if selected_indices.numel() == 0:
                continue
            expert_input = x_flat[selected_indices]
            expert_out = self.experts[expert_idx](expert_input, deep_embed=deep_embed)
            expert_weights = top_scores[expert_idx][valid_mask].unsqueeze(-1)
            weighted_out = expert_weights * expert_out
            output.scatter_add_(
                0, selected_indices.unsqueeze(-1).expand_as(weighted_out), weighted_out
            )

        if self._compute_aux_loss:
            if router_logits.ndim == 3:
                probs = F.softmax(router_logits.reshape(-1, self.num_experts), dim=-1)
            else:
                probs = F.softmax(router_logits, dim=-1)
            avg_probs = probs.mean(dim=0)
            uniform = torch.full_like(avg_probs, 1.0 / self.num_experts)
            balance_loss = (avg_probs - uniform).pow(2).sum() * self.num_experts
            z_loss = _compute_z_loss(router_logits)
            aux = (
                self.load_balance_loss_weight * balance_loss
                + self.z_loss_weight * z_loss
            )
            self._aux_loss = self._aux_loss + aux.detach()
            self._num_forward_passes.add_(1)
        return output.reshape(orig_shape)

    def forward(self, x: Tensor, deep_embed=None):
        if self.routing_strategy == "top_k":
            return self._forward_top_k(x, deep_embed=deep_embed)
        elif self.routing_strategy == "expert_choice":
            return self._forward_expert_choice_optimized(x, deep_embed=deep_embed)
        raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    def muon_parameters(self):
        weights = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weights.append(m.weight)
        return weights
