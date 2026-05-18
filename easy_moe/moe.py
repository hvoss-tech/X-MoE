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
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.capacity_factor = capacity_factor
        self.routing_strategy = routing_strategy
        self.load_balance_loss_weight = load_balance_loss_weight
        self.z_loss_weight = z_loss_weight

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

        self._aux_loss = torch.tensor(0.0)
        self._num_forward_passes = 0

    @property
    def aux_loss(self):
        if self._num_forward_passes > 0:
            return self._aux_loss / self._num_forward_passes
        return torch.tensor(0.0)

    def reset_aux_loss(self):
        self._aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        self._num_forward_passes = 0

    def _forward_top_k(self, x: Tensor, deep_embed=None):
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        weights, top_indices, router_logits = self.gate(x_flat)
        top_k = top_indices.shape[-1]
        output = torch.zeros_like(x_flat)
        for k in range(top_k):
            for expert_idx in range(self.num_experts):
                mask = top_indices[:, k] == expert_idx
                if not mask.any():
                    continue
                expert_input = x_flat[mask]
                expert_out = self.experts[expert_idx](
                    expert_input, deep_embed=deep_embed
                )
                output[mask] += weights[mask, k : k + 1] * expert_out
        balance_loss = _compute_load_balance_loss(
            router_logits, top_indices, self.num_experts
        )
        z_loss = _compute_z_loss(router_logits)
        aux = (
            self.load_balance_loss_weight * balance_loss
            + self.z_loss_weight * z_loss
        )
        self._aux_loss = self._aux_loss + aux.detach()
        self._num_forward_passes += 1
        return output.reshape(orig_shape)

    def _forward_expert_choice(self, x: Tensor, deep_embed=None):
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
            expert_out = self.experts[expert_idx](
                expert_input, deep_embed=deep_embed
            )
            expert_weights = top_scores[expert_idx][valid_mask].unsqueeze(-1)
            output[selected_indices] += expert_weights * expert_out
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
        self._num_forward_passes += 1
        return output.reshape(orig_shape)

    def forward(self, x: Tensor, deep_embed=None):
        if self.routing_strategy == "top_k":
            return self._forward_top_k(x, deep_embed=deep_embed)
        elif self.routing_strategy == "expert_choice":
            return self._forward_expert_choice(x, deep_embed=deep_embed)
        raise ValueError(f"Unknown routing strategy: {self.routing_strategy}")

    def muon_parameters(self):
        weights = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                weights.append(m.weight)
        return weights