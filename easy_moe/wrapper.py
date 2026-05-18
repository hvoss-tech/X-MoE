import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder, AutoregressiveWrapper
from x_transformers.x_transformers import AttentionLayers

from easy_moe.moe import MoEFFN
from easy_moe.attention import HCA, CSA, DS4AttentionLayer, HybridAttentionBlock


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
        moe_layers: list | None = None,
        glu: bool = True,
        mult: int = 4,
        dropout: float = 0.0,
        no_bias: bool = False,
        zero_init_output: bool = True,
        model_config: dict | None = None,
        ds4_attention: HybridAttentionBlock | None = None,
    ):
        super().__init__()

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
        )

        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.routing_strategy = routing_strategy
        self.model_config = model_config

        self.ds4_attention = ds4_attention
        if self.ds4_attention is not None:
            self.ds4_norm = nn.LayerNorm(transformer.emb_dim if hasattr(transformer, 'emb_dim') else transformer.attn_layers.dim)

        self.autoregressive_wrapper = AutoregressiveWrapper(
            self.transformer,
        )

    @property
    def moe_aux_loss(self):
        return collect_moe_aux_loss(self)

    def reset_moe_aux_loss(self):
        reset_moe_aux_loss(self)

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