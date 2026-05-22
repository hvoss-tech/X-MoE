import torch
import torch.nn as nn
from typing import Optional
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from x_transformers.x_transformers import AttentionLayers

from x_moe.moe import MoEFFN
from x_moe.attention import HybridAttentionBlock


def replace_ffn_with_moe(
    model: nn.Module,
    num_experts: int = 8,
    expert_top_k: int = 2,
    capacity_factor: float = 1.25,
    routing_strategy: str = "top_k",
    load_balance_loss_weight: float = 0.01,
    z_loss_weight: float = 1e-4,
    moe_every_n_layers: int = 1,
    moe_layers: list | None = None,
    glu: bool = True,
    mult: int = 4,
    dropout: float = 0.0,
    no_bias: bool = False,
    zero_init_output: bool = True,
    batched_experts: bool = False,
    max_seq_len: int = 256,
    max_batch_size: int = 1,
) -> nn.Module:
    attn_layers = None
    if hasattr(model, "attn_layers"):
        attn_layers = model.attn_layers
    elif isinstance(model, AttentionLayers):
        attn_layers = model

    if attn_layers is None:
        raise ValueError(
            "Could not find attention layers in the model. "
            "Pass a TransformerWrapper or AttentionLayers module."
        )

    ffn_count = 0
    moe_layers_set = set(moe_layers) if moe_layers is not None else None

    for idx, (layer_type, (norms, block, residual_fn)) in enumerate(
        zip(attn_layers.layer_types, attn_layers.layers)
    ):
        if layer_type != "f":
            continue

        should_replace = False
        if moe_layers_set is not None:
            should_replace = ffn_count in moe_layers_set
        else:
            should_replace = (ffn_count % moe_every_n_layers) == 0

        if should_replace and not isinstance(block, MoEFFN):
            moe = MoEFFN(
                dim=attn_layers.dim,
                num_experts=num_experts,
                expert_top_k=expert_top_k,
                capacity_factor=capacity_factor,
                routing_strategy=routing_strategy,
                load_balance_loss_weight=load_balance_loss_weight,
                z_loss_weight=z_loss_weight,
                glu=glu,
                mult=mult,
                dropout=dropout,
                no_bias=no_bias,
                zero_init_output=zero_init_output,
                batched_experts=batched_experts,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            )
            attn_layers.layers[idx][1] = moe
        ffn_count += 1

    return model


def collect_moe_aux_loss(model: nn.Module) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for module in model.modules():
        if isinstance(module, MoEFFN):
            total_loss = total_loss + module.aux_loss
    return total_loss


def reset_moe_aux_loss(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module.reset_aux_loss()


def set_aux_loss_compute(model: nn.Module, compute: bool):
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module._compute_aux_loss = compute


def enable_gradient_checkpointing(model: nn.Module):
    count = 0
    for module in model.modules():
        if isinstance(module, MoEFFN):
            module._use_gradient_checkpointing = True
            count += 1
    return count


class MoETransformerWrapper(nn.Module):
    def __init__(
        self,
        transformer: TransformerWrapper,
        num_experts: int = 8,
        expert_top_k: int = 2,
        capacity_factor: float = 1.25,
        routing_strategy: str = "top_k",
        load_balance_loss_weight: float = 0.01,
        z_loss_weight: float = 1e-4,
        moe_every_n_layers: int = 1,
        moe_layers: Optional[list] = None,
        glu: bool = True,
        mult: int = 4,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = True,
        ds4_attention: Optional[HybridAttentionBlock] = None,
        batched_experts: bool = True,
        max_batch_size: int = 1,
        flash_attention: bool = True, #Flash attention is never saved in the decoder...
    ):
        super().__init__()

        max_seq_len = transformer.max_seq_len
        layer_dropout = transformer.attn_layers.layer_dropouts
        emb_dropout = transformer.emb_dropout

        self.transformer = replace_ffn_with_moe(
            transformer,
            num_experts=num_experts,
            expert_top_k=expert_top_k,
            capacity_factor=capacity_factor,
            routing_strategy=routing_strategy,
            load_balance_loss_weight=load_balance_loss_weight,
            z_loss_weight=z_loss_weight,
            moe_every_n_layers=moe_every_n_layers,
            moe_layers=moe_layers,
            glu=glu,
            mult=mult,
            dropout=dropout,
            no_bias=no_bias,
            zero_init_output=zero_init_output,
            batched_experts=batched_experts,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
        )

        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.routing_strategy = routing_strategy
        self._gradient_checkpointing = False

        attn_layers = (
            transformer.attn_layers
            if hasattr(transformer, "attn_layers")
            else transformer
        )
        has_rotary = bool(getattr(attn_layers, "rotary_pos_emb", False))
        disable_abs_pos_emb = getattr(
            attn_layers, "disable_abs_pos_emb", not has_rotary
        )

        self.model_config = {
            "dim": getattr(attn_layers, "dim", None),
            "depth": getattr(attn_layers, "depth", None),
            "heads": getattr(attn_layers, "attn_heads", None),
            "no_ff_glu": not glu,
            "ff_mult": mult,
            "ff_dropout": dropout,
            "ff_bias": not no_bias,
            "layer_dropout": layer_dropout,
            "no_rotary_pos_emb": not has_rotary,
            "flash_attention": flash_attention,
            "emb_dropout": emb_dropout,
            "use_abs_pos_emb": not disable_abs_pos_emb,
            "num_experts": num_experts,
            "expert_top_k": expert_top_k,
            "capacity_factor": capacity_factor,
            "routing_strategy": routing_strategy,
            "load_balance_loss_weight": load_balance_loss_weight,
            "z_loss_weight": z_loss_weight,
            "moe_every_n_layers": moe_every_n_layers,
            "moe_layers": moe_layers,
            "batched_experts": batched_experts,
            "max_batch_size": max_batch_size,
            "max_seq_len": max_seq_len,
            "vocab_size": getattr(transformer, "num_tokens", None),
        }



        self.ds4_attention = ds4_attention
        if self.ds4_attention is not None:
            if self.ds4_attention.config is not None:
                ds4_config = ds4_attention.config
                self.model_config.update(ds4_config)
            self.ds4_norm = nn.LayerNorm(
                transformer.emb_dim
                if hasattr(transformer, "emb_dim")
                else transformer.attn_layers.dim
            )

        self.autoregressive_wrapper = AutoregressiveWrapper(
            self.transformer,
        )

    @property
    def moe_aux_loss(self):
        return collect_moe_aux_loss(self)

    def reset_moe_aux_loss(self):
        reset_moe_aux_loss(self)

    def set_aux_loss_compute(self, compute: bool):
        set_aux_loss_compute(self, compute)

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True
        if hasattr(self.transformer, "attn_layers"):
            attn_layers = self.transformer.attn_layers
            if hasattr(attn_layers, "grad_checkpointing"):
                attn_layers.grad_checkpointing = True
        count = enable_gradient_checkpointing(self)
        return count

    def forward(self, x, **kwargs):
        return self.autoregressive_wrapper(x, **kwargs)

    @torch.no_grad()
    def generate(self, prompts, seq_len, **kwargs):
        return self.autoregressive_wrapper.generate(prompts, seq_len=seq_len, **kwargs)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
