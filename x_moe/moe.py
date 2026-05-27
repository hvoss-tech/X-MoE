import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from x_transformers.x_transformers import FeedForward


class TopKGate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        sigmoid_routing: bool = False,
    ):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.sigmoid_routing = sigmoid_routing
        self.w_g = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("routing_bias", torch.zeros(num_experts), persistent=True)

    def forward(self, x: Tensor, apply_bias: bool = False):
        logits = self.w_g(x)
        if self.sigmoid_routing:
            raw_scores = torch.sigmoid(logits)
        else:
            raw_scores = F.softmax(logits, dim=-1)

        if apply_bias:
            biased_logits = logits + self.routing_bias.unsqueeze(0)
            if self.sigmoid_routing:
                biased_scores = torch.sigmoid(biased_logits)
            else:
                biased_scores = F.softmax(biased_logits, dim=-1)
            top_k = min(self.top_k, self.num_experts)
            top_scores_biased, top_indices = biased_scores.topk(top_k, dim=-1)
            top_scores_raw = raw_scores.gather(-1, top_indices)
            weights = top_scores_raw / top_scores_raw.sum(dim=-1, keepdim=True).clamp(
                min=1e-9
            )
            return weights, top_indices, logits
        else:
            top_k = min(self.top_k, self.num_experts)
            top_scores, top_indices = raw_scores.topk(top_k, dim=-1)
            weights = top_scores / top_scores.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            return weights, top_indices, logits


class ExpertChoiceGate(nn.Module):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        sigmoid_routing: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.sigmoid_routing = sigmoid_routing
        self.w_g = nn.Linear(dim, num_experts, bias=False)
        self.register_buffer("routing_bias", torch.zeros(num_experts), persistent=True)

    def forward(self, x_flat: Tensor, num_tokens: int, apply_bias: bool = False):
        logits = self.w_g(x_flat)
        if self.sigmoid_routing:
            raw_scores = torch.sigmoid(logits)
        else:
            raw_scores = F.softmax(logits, dim=-1)

        capacity = max(1, int(self.capacity_factor * num_tokens / self.num_experts))
        capacity = min(capacity, num_tokens)

        if apply_bias:
            biased_logits = logits + self.routing_bias.unsqueeze(0)
            if self.sigmoid_routing:
                biased_scores = torch.sigmoid(biased_logits)
            else:
                biased_scores = F.softmax(biased_logits, dim=-1)
            expert_scores = biased_scores.t()
            top_scores, top_indices = expert_scores.topk(capacity, dim=-1)
            return raw_scores, top_scores, top_indices, capacity, logits
        else:
            expert_scores = raw_scores.t()
            top_scores, top_indices = expert_scores.topk(capacity, dim=-1)
            return raw_scores, top_scores, top_indices, capacity, logits


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


def _compute_seq_balance_loss(
    router_logits: Tensor,
    top_indices: Tensor,
    num_experts: int,
    num_tokens_per_seq: int,
) -> Tensor:
    if router_logits.ndim == 3:
        logits_2d = router_logits.reshape(-1, num_experts)
    else:
        logits_2d = router_logits
    num_tokens = logits_2d.shape[0]
    num_seqs = max(num_tokens // num_tokens_per_seq, 1)
    top_k = top_indices.shape[-1] if top_indices.ndim > 1 else 1
    router_probs = F.softmax(logits_2d, dim=-1)
    with torch.no_grad():
        one_hot = F.one_hot(top_indices.reshape(-1), num_experts).float()
    padded_tokens = num_seqs * num_tokens_per_seq
    if num_tokens < padded_tokens:
        one_hot = F.pad(one_hot, (0, 0, 0, padded_tokens - num_tokens))
        router_probs_padded = F.pad(router_probs, (0, 0, 0, padded_tokens - num_tokens))
    else:
        one_hot_reshaped = one_hot[:padded_tokens]
        router_probs_padded = router_probs[:padded_tokens]
    one_hot_reshaped = one_hot[:padded_tokens].reshape(
        num_seqs, num_tokens_per_seq, num_experts
    )
    router_probs_reshaped = router_probs[:padded_tokens].reshape(
        num_seqs, num_tokens_per_seq, num_experts
    )
    f_i = one_hot_reshaped.sum(dim=1) / max(num_tokens_per_seq * top_k, 1)
    s_prime = router_probs_reshaped.mean(dim=1)
    P_i = s_prime
    balance_loss = (f_i * P_i).sum(-1).mean() * num_experts
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
        sigmoid_routing: bool = False,
        num_shared_experts: int = 0,
        granularity_factor: int = 1,
        aux_loss_free: bool = False,
        bias_update_speed: float = 0.01,
        seq_balance_loss_weight: float = 0.0,
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
        self._no_bias = no_bias
        self.sigmoid_routing = sigmoid_routing
        self.num_shared_experts = num_shared_experts
        self.granularity_factor = granularity_factor
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed
        self.seq_balance_loss_weight = seq_balance_loss_weight

        if granularity_factor < 1:
            raise ValueError(
                f"granularity_factor must be >= 1, got {granularity_factor}"
            )
        if num_shared_experts < 0:
            raise ValueError(
                f"num_shared_experts must be >= 0, got {num_shared_experts}"
            )

        self.num_routed_experts = num_experts * granularity_factor
        effective_top_k = expert_top_k * granularity_factor
        routed_top_k = max(1, effective_top_k - num_shared_experts)

        if routed_top_k > self.num_routed_experts:
            raise ValueError(
                f"routed_top_k ({routed_top_k}) exceeds num_routed_experts "
                f"({self.num_routed_experts})"
            )

        self.effective_top_k = effective_top_k
        self.routed_top_k = routed_top_k

        routed_mult = mult / granularity_factor if granularity_factor > 1 else mult
        self._inner_dim = dim * mult
        self._routed_inner_dim = int(dim * routed_mult)
        self._capacity = math.ceil(
            capacity_factor
            * max_seq_len
            * max_batch_size
            * routed_top_k
            / self.num_routed_experts
        )

        self.routed_experts = nn.ModuleList(
            [
                FeedForward(
                    dim=dim,
                    mult=routed_mult,
                    glu=glu,
                    dropout=dropout,
                    no_bias=no_bias,
                    zero_init_output=zero_init_output,
                )
                for _ in range(self.num_routed_experts)
            ]
        )

        if num_shared_experts > 0:
            self.shared_experts = nn.ModuleList(
                [
                    FeedForward(
                        dim=dim,
                        mult=mult,
                        glu=glu,
                        dropout=dropout,
                        no_bias=no_bias,
                        zero_init_output=zero_init_output,
                    )
                    for _ in range(num_shared_experts)
                ]
            )
        else:
            self.shared_experts = None

        if routing_strategy == "top_k":
            self.gate = TopKGate(
                dim,
                self.num_routed_experts,
                top_k=routed_top_k,
                sigmoid_routing=sigmoid_routing,
            )
        elif routing_strategy == "expert_choice":
            self.gate = ExpertChoiceGate(
                dim,
                self.num_routed_experts,
                capacity_factor,
                sigmoid_routing=sigmoid_routing,
            )
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
        self._use_gradient_checkpointing = False

        if batched_experts:
            self._init_stacked_params()

    @property
    def experts(self):
        return self.routed_experts

    def _init_stacked_params(self):
        proj_out_dim = self._routed_inner_dim * (2 if self._glu else 1)
        self.register_buffer("_has_bias_1", torch.tensor(False))
        self.register_buffer("_has_bias_2", torch.tensor(False))

        self.w1_stack = nn.Parameter(
            torch.empty(self.num_routed_experts, proj_out_dim, self.dim)
        )
        self.w2_stack = nn.Parameter(
            torch.empty(self.num_routed_experts, self.dim, self._routed_inner_dim)
        )

        with torch.no_grad():
            for i, expert in enumerate(self.routed_experts):
                ff_seq = expert.ff
                if self._glu:
                    self.w1_stack.data[i] = ff_seq[0].proj.weight.data
                else:
                    self.w1_stack.data[i] = ff_seq[0][0].weight.data
                self.w2_stack.data[i] = ff_seq[2].weight.data

        if self._glu:
            b1_exists = any(
                expert.ff[0].proj.bias is not None for expert in self.routed_experts
            )
        else:
            b1_exists = any(
                expert.ff[0][0].bias is not None for expert in self.routed_experts
            )
        b2_exists = any(expert.ff[2].bias is not None for expert in self.routed_experts)
        if b1_exists:
            self.b1_stack = nn.Parameter(
                torch.empty(self.num_routed_experts, proj_out_dim)
            )
            self._has_bias_1.fill_(True)
            with torch.no_grad():
                for i, expert in enumerate(self.routed_experts):
                    if self._glu:
                        if expert.ff[0].proj.bias is not None:
                            self.b1_stack.data[i] = expert.ff[0].proj.bias.data
                    else:
                        if expert.ff[0][0].bias is not None:
                            self.b1_stack.data[i] = expert.ff[0][0].bias.data
        if b2_exists:
            self.b2_stack = nn.Parameter(torch.empty(self.num_routed_experts, self.dim))
            self._has_bias_2.fill_(True)
            with torch.no_grad():
                for i, expert in enumerate(self.routed_experts):
                    if expert.ff[2].bias is not None:
                        self.b2_stack.data[i] = expert.ff[2].bias.data

        try:
            self._dropout_p = self.routed_experts[0].ff[1].p
        except AttributeError:
            self._dropout_p = 0.0

    def _sync_stacked_to_experts(self):
        with torch.no_grad():
            for i, expert in enumerate(self.routed_experts):
                ff_seq = expert.ff
                if self._glu:
                    ff_seq[0].proj.weight.data.copy_(self.w1_stack.data[i])
                else:
                    ff_seq[0][0].weight.data.copy_(self.w1_stack.data[i])
                ff_seq[2].weight.data.copy_(self.w2_stack.data[i])
                if self._has_bias_1 and hasattr(self, "b1_stack"):
                    if self._glu:
                        if ff_seq[0].proj.bias is not None:
                            ff_seq[0].proj.bias.data.copy_(self.b1_stack.data[i])
                    else:
                        if ff_seq[0][0].bias is not None:
                            ff_seq[0][0].bias.data.copy_(self.b1_stack.data[i])
                if (
                    self._has_bias_2
                    and hasattr(self, "b2_stack")
                    and ff_seq[2].bias is not None
                ):
                    ff_seq[2].bias.data.copy_(self.b2_stack.data[i])

    def _sync_experts_to_stacked(self):
        with torch.no_grad():
            for i, expert in enumerate(self.routed_experts):
                ff_seq = expert.ff
                if self._glu:
                    self.w1_stack.data[i] = ff_seq[0].proj.weight.data
                else:
                    self.w1_stack.data[i] = ff_seq[0][0].weight.data
                self.w2_stack.data[i] = ff_seq[2].weight.data
                if self._has_bias_1 and hasattr(self, "b1_stack"):
                    if self._glu:
                        if ff_seq[0].proj.bias is not None:
                            self.b1_stack.data[i] = ff_seq[0].proj.bias.data
                    else:
                        if ff_seq[0][0].bias is not None:
                            self.b1_stack.data[i] = ff_seq[0][0].bias.data
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
        self._aux_loss.fill_(0.0)
        self._num_forward_passes.fill_(0)

    def _accumulate_aux_loss(self, router_logits, top_indices, seq_len: int = 0):
        if not self._compute_aux_loss:
            return
        if self.aux_loss_free:
            if self.seq_balance_loss_weight > 0 and seq_len > 0:
                seq_loss = _compute_seq_balance_loss(
                    router_logits, top_indices, self.num_routed_experts, seq_len
                )
                aux = self.seq_balance_loss_weight * seq_loss
                self._aux_loss = self._aux_loss + aux.detach()
                self._num_forward_passes.add_(1)
            return
        balance_loss = _compute_load_balance_loss(
            router_logits, top_indices, self.num_routed_experts
        )
        z_loss = _compute_z_loss(router_logits)
        aux = self.load_balance_loss_weight * balance_loss + self.z_loss_weight * z_loss
        self._aux_loss = self._aux_loss + aux.detach()
        self._num_forward_passes.add_(1)

    @torch.no_grad()
    def update_routing_bias(self):
        if not self.aux_loss_free:
            return
        bias = self.gate.routing_bias
        num_experts = self.num_routed_experts
        if hasattr(self, "_token_counts") and self._token_counts.sum() > 0:
            avg = num_experts / self._token_counts.sum().clamp(min=1)
            for i in range(num_experts):
                if self._token_counts[i] > avg:
                    bias[i] = bias[i] - self.bias_update_speed
                elif self._token_counts[i] < avg:
                    bias[i] = bias[i] + self.bias_update_speed
        else:
            ideal = 1.0 / num_experts
            for i in range(num_experts):
                if bias[i] > 0:
                    bias[i] = bias[i] - self.bias_update_speed * 0.1
                elif bias[i] < 0:
                    bias[i] = bias[i] + self.bias_update_speed * 0.1
        self._token_counts.zero_()

    def _forward_shared_experts(self, x: Tensor) -> Tensor:
        if self.shared_experts is None:
            return torch.zeros_like(x)
        output = torch.zeros_like(x)
        for expert in self.shared_experts:
            output = output + expert(x)
        return output

    def _forward_top_k_vectorized(self, x: Tensor, deep_embed=None):
        orig_shape = x.shape
        seq_len = orig_shape[1] if x.ndim > 1 else 0
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]
        apply_bias = self.aux_loss_free
        weights, top_indices, router_logits = self.gate(x_flat, apply_bias=apply_bias)
        top_k = top_indices.shape[-1]

        if apply_bias:
            self._update_token_counts(top_indices, num_tokens)

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

        num_routed = self.num_routed_experts
        expert_counts = flat_expert_ids.bincount(minlength=num_routed)
        offsets = torch.zeros(num_routed + 1, device=x.device, dtype=torch.long)
        offsets[1:] = expert_counts.cumsum(0)
        local_positions = (
            torch.arange(sort_idx.shape[0], device=x.device, dtype=torch.long)
            - offsets[sorted_expert_ids]
        )

        capacity = self._capacity
        in_bounds = local_positions < capacity
        in_bounds_float = in_bounds.float()

        padded_input = torch.zeros(
            num_routed,
            capacity,
            self.dim,
            device=x.device,
            dtype=x.dtype,
        )
        padded_weights = torch.zeros(
            num_routed,
            capacity,
            device=x.device,
            dtype=x.dtype,
        )
        pad_mask = torch.zeros(
            num_routed,
            capacity,
            dtype=torch.bool,
            device=x.device,
        )

        sorted_x = x_flat[sorted_token_ids]
        sorted_in_bounds_x = sorted_x * in_bounds_float.unsqueeze(-1)
        sorted_in_bounds_w = sorted_weights * in_bounds_float

        flat_expert_idx = sorted_expert_ids * capacity + local_positions
        valid_flat_idx = flat_expert_idx.clamp(max=num_routed * capacity - 1)

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
        pad_mask = pad_mask_flat.reshape(num_routed, capacity).clamp(max=1).bool()

        batched_out = self._batched_forward(padded_input, pad_mask)

        weighted_out = batched_out * padded_weights.unsqueeze(-1)
        weighted_out = weighted_out * pad_mask.unsqueeze(-1).float()

        output_flat = torch.zeros_like(x_flat)
        gathered = weighted_out[
            sorted_expert_ids, local_positions.clamp(max=capacity - 1)
        ]
        gathered = gathered * in_bounds_float.unsqueeze(-1)
        output_flat.index_add_(0, sorted_token_ids, gathered)

        if self.shared_experts is not None:
            output_flat = output_flat + self._forward_shared_experts(x_flat)

        self._accumulate_aux_loss(router_logits, top_indices, seq_len=seq_len)
        return output_flat.reshape(orig_shape)

    def _forward_top_k(self, x: Tensor, deep_embed=None):
        if deep_embed is not None:
            return self._forward_top_k_fallback(x, deep_embed=deep_embed)
        if self.batched_experts:
            return self._forward_top_k_vectorized(x)
        return self._forward_top_k_fallback(x)

    def _forward_top_k_fallback(self, x: Tensor, deep_embed=None):
        orig_shape = x.shape
        seq_len = orig_shape[1] if x.ndim > 1 else 0
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]
        apply_bias = self.aux_loss_free
        weights, top_indices, router_logits = self.gate(x_flat, apply_bias=apply_bias)
        top_k = top_indices.shape[-1]

        if apply_bias:
            self._update_token_counts(top_indices, num_tokens)

        flat_indices = top_indices.reshape(-1)
        flat_weights = weights.reshape(-1, 1)
        token_expert_pairs = (
            torch.arange(num_tokens, device=x.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )

        output = torch.zeros_like(x_flat)
        use_ckpt = getattr(self, "_use_gradient_checkpointing", False) and self.training
        capacity = self._capacity
        for expert_idx in range(self.num_routed_experts):
            expert_mask = flat_indices == expert_idx
            if not expert_mask.any():
                continue
            selected_tokens = token_expert_pairs[expert_mask]
            selected_weights = flat_weights[expert_mask]
            if selected_tokens.shape[0] > capacity:
                selected_tokens = selected_tokens[:capacity]
                selected_weights = selected_weights[:capacity]
            expert_input = x_flat[selected_tokens]
            if use_ckpt:
                expert_out = torch_checkpoint(
                    self.routed_experts[expert_idx],
                    expert_input,
                    deep_embed,
                    use_reentrant=False,
                )
            else:
                expert_out = self.routed_experts[expert_idx](
                    expert_input, deep_embed=deep_embed
                )
            weighted_out = selected_weights * expert_out
            output.scatter_add_(
                0, selected_tokens.unsqueeze(-1).expand_as(weighted_out), weighted_out
            )

        if self.shared_experts is not None:
            output = output + self._forward_shared_experts(x_flat)

        self._accumulate_aux_loss(router_logits, top_indices, seq_len=seq_len)
        return output.reshape(orig_shape)

    def _batched_forward(self, padded_input: Tensor, pad_mask: Tensor):
        use_ckpt = getattr(self, "_use_gradient_checkpointing", False) and self.training
        if not self.batched_experts or not hasattr(self, "w1_stack"):
            out_all = torch.zeros_like(padded_input)
            for i in range(self.num_routed_experts):
                count = pad_mask[i].sum().item()
                if count > 0:
                    if use_ckpt:
                        out_all[i, :count] = torch_checkpoint(
                            self.routed_experts[i],
                            padded_input[i, :count],
                            use_reentrant=False,
                        )
                    else:
                        out_all[i, :count] = self.routed_experts[i](
                            padded_input[i, :count]
                        )
            return out_all

        h1 = torch.einsum("esi,eoi->eso", padded_input, self.w1_stack)
        if hasattr(self, "b1_stack") and self._has_bias_1:
            h1 = h1 + self.b1_stack.unsqueeze(1)

        if self._glu:
            h1, h1_gate = h1.chunk(2, dim=-1)
            h1 = F.gelu(h1_gate) * h1
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
        seq_len = orig_shape[1] if x.ndim > 1 else 0
        x_flat = x.reshape(-1, self.dim)
        num_tokens = x_flat.shape[0]
        apply_bias = self.aux_loss_free
        scores, top_scores, top_indices, capacity, router_logits = self.gate(
            x_flat, num_tokens, apply_bias=apply_bias
        )

        if apply_bias:
            for expert_idx in range(self.num_routed_experts):
                selected = top_indices[expert_idx]
                valid = selected[selected >= 0]
                valid = valid[valid < num_tokens]
                if hasattr(self, "_token_counts"):
                    self._token_counts[expert_idx] += valid.numel()

        output = torch.zeros_like(x_flat)

        use_ckpt = getattr(self, "_use_gradient_checkpointing", False) and self.training
        for expert_idx in range(self.num_routed_experts):
            selected_indices = top_indices[expert_idx]
            if not torch.any(selected_indices >= 0):
                continue
            valid_mask = selected_indices < num_tokens
            selected_indices = selected_indices[valid_mask]
            if selected_indices.numel() == 0:
                continue
            expert_input = x_flat[selected_indices]
            if use_ckpt:
                expert_out = torch_checkpoint(
                    self.routed_experts[expert_idx],
                    expert_input,
                    deep_embed,
                    use_reentrant=False,
                )
            else:
                expert_out = self.routed_experts[expert_idx](
                    expert_input, deep_embed=deep_embed
                )
            expert_weights = top_scores[expert_idx][valid_mask].unsqueeze(-1)
            weighted_out = expert_weights * expert_out
            output.scatter_add_(
                0, selected_indices.unsqueeze(-1).expand_as(weighted_out), weighted_out
            )

        if self.shared_experts is not None:
            output = output + self._forward_shared_experts(x_flat)

        if self.aux_loss_free:
            if (
                self._compute_aux_loss
                and self.seq_balance_loss_weight > 0
                and seq_len > 0
            ):
                seq_loss = _compute_seq_balance_loss(
                    router_logits,
                    top_indices.reshape(-1)[: num_tokens * self.routed_top_k].reshape(
                        num_tokens, self.routed_top_k
                    )
                    if top_indices.ndim > 1
                    else top_indices[:num_tokens],
                    self.num_routed_experts,
                    seq_len,
                )
                aux = self.seq_balance_loss_weight * seq_loss
                self._aux_loss = self._aux_loss + aux.detach()
                self._num_forward_passes.add_(1)
        elif self._compute_aux_loss:
            if router_logits.ndim == 3:
                probs = F.softmax(
                    router_logits.reshape(-1, self.num_routed_experts), dim=-1
                )
            else:
                probs = F.softmax(router_logits, dim=-1)
            avg_probs = probs.mean(dim=0)
            uniform = torch.full_like(avg_probs, 1.0 / self.num_routed_experts)
            balance_loss = (avg_probs - uniform).pow(2).sum() * self.num_routed_experts
            z_loss = _compute_z_loss(router_logits)
            aux = (
                self.load_balance_loss_weight * balance_loss
                + self.z_loss_weight * z_loss
            )
            self._aux_loss = self._aux_loss + aux.detach()
            self._num_forward_passes.add_(1)
        return output.reshape(orig_shape)

    def _update_token_counts(self, top_indices: Tensor, num_tokens: int):
        if not hasattr(self, "_token_counts"):
            self.register_buffer(
                "_token_counts",
                torch.zeros(self.num_routed_experts, dtype=torch.long),
                persistent=False,
            )
            self._token_counts = self._token_counts.to(top_indices.device)
        flat = top_indices.reshape(-1)
        counts = flat.bincount(minlength=self.num_routed_experts)
        self._token_counts[: counts.shape[0]] += counts[: self.num_routed_experts]

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
