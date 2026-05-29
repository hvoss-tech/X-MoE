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
    sigmoid_routing: bool = False,
    num_shared_experts: int = 0,
    granularity_factor: int = 1,
    aux_loss_free: bool = False,
    bias_update_speed: float = 0.01,
    seq_balance_loss_weight: float = 0.0,
    sqrt_softplus_routing: bool = False,
    hash_routing: bool = False,
    num_hash_functions: int = 4,
    anticipatory_routing: bool = False,
    swiglu_clamp_value: float = 0.0,
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
                sigmoid_routing=sigmoid_routing,
                num_shared_experts=num_shared_experts,
                granularity_factor=granularity_factor,
                aux_loss_free=aux_loss_free,
                bias_update_speed=bias_update_speed,
                seq_balance_loss_weight=seq_balance_loss_weight,
                sqrt_softplus_routing=sqrt_softplus_routing,
                hash_routing=hash_routing,
                num_hash_functions=num_hash_functions,
                anticipatory_routing=anticipatory_routing,
                swiglu_clamp_value=swiglu_clamp_value,
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
    """MoE Transformer model combining x-transformers with Mixture of Experts
    feedforward layers and optional HCA/CSA attention.

    This wrapper replaces feedforward layers in a transformer with MoE layers
    and optionally prepends HCA (Heavily Compressed Attention) and/or CSA
    (Compressed Sparse Attention) blocks for efficient long-sequence processing.

    Args:
        transformer: A TransformerWrapper instance (e.g. from x-transformers).
        num_experts: Number of experts in each MoE layer. Default: 8.
        expert_top_k: Number of experts to route to per token. Default: 2.
        capacity_factor: Capacity factor for expert routing. Default: 1.25.
        routing_strategy: Expert routing strategy, ``'top_k'`` or ``'expert_choice'``. Default: ``'top_k'``.
        load_balance_loss_weight: Weight for the load balance auxiliary loss. Default: 0.01.
        z_loss_weight: Weight for the z-loss auxiliary loss. Default: 1e-4.
        moe_every_n_layers: Replace every Nth feedforward layer with MoE. Default: 1.
        moe_layers: Explicit list of feedforward layer indices to replace.
            Overrides ``moe_every_n_layers`` if provided. Default: None.
        glu: Whether to use GLU variant in the feedforward. Default: True.
        mult: Expansion factor for the feedforward inner dimension. Default: 4.
        dropout: Dropout rate for the feedforward. Default: 0.0.
        no_bias: Whether to disable bias in the feedforward. Default: False.
        zero_init_output: Whether to zero-init the output projection of each expert. Default: True.
        batched_experts: Whether to use stacked/batched expert computation. Default: True.
        max_batch_size: Maximum batch size for capacity calculation. Default: 1.
        flash_attention: Whether to use flash attention in the decoder. Default: True.
        use_hca: Whether to prepend a Heavily Compressed Attention block. Default: False.
        use_csa: Whether to prepend a Compressed Sparse Attention block. Default: False.
        kv_dim: Dimension of the compressed key-value pairs for HCA/CSA. Shared by both. Default: 128.
        num_query_heads: Number of query heads for HCA/CSA multi-query attention. Shared by both. Default: 8.
        compression_rate: Number of tokens compressed into one KV pair. Shared by both. Default: 8.
        num_groups: Number of query head groups for grouped output in HCA/CSA. Shared by both. Default: 1.
        group_out_dim: Output dimension per group in HCA/CSA. Only used when ``num_groups > 1``. Default: None.
        window_size: Sliding window size for local attention in HCA/CSA. Set to 0 to disable. Shared by both. Default: 32.
        use_attention_sink: Whether to use a learnable attention sink token in HCA/CSA. Shared by both. Default: True.
        use_partial_rope: Whether to use partial rotary position embeddings in HCA/CSA. Shared by both. Default: True.
        rope_dim: Number of dimensions to apply rotary embeddings to in HCA/CSA. Shared by both. Default: 64.
        attn_dropout: Dropout rate for HCA/CSA attention. Shared by both. Default: 0.0.
        csa_top_k_blocks: Number of top-K blocks to select in CSA indexer. CSA-only. Default: 32.
        csa_indexer_dim: Dimension of the CSA block indexer projections. CSA-only. Default: None (dim // 4).
        sigmoid_routing: Deepkseekv3 - use sigmoid for routing instead of softmax.
        num_shared_experts: Deepkseekv3 - Add shared experts that always receive all tokens.
        granularity_factor: DeepSeekMoE - Splits each expert into m smaller sub-experts (inner_dim divided by granularity_factor), activating top_k * m sub-experts for more combinatorial flexibility.
        aux_loss_free: Deepseekv3 - Auxiliary-Loss-Free Load Balancing - only the seq balance loss is computed. IMPORTANT: Call model.update_routing_biases() after each optimizer step, so overloaded experts get bias decreased, underloaded get bias increased.
        bias_update_speed: Deepseekv3 - Speed of bias adjustments.
        seq_balance_loss_weight: Deepseekv3 - Weight of the sequence balance loss.
        sqrt_softplus_routing: DeepSeekV4 - use sqrt(softplus(·)) for routing instead of softmax or sigmoid. Mutually exclusive with sigmoid_routing.
        hash_routing: DeepSeekV4 - use hash-based deterministic routing by token ID. Replaces learned routing with a hash function for initial layers.
        num_hash_functions: DeepSeekV4 - number of independent hash seeds for hash routing. Default: 4.
        anticipatory_routing: DeepSeekV4 - use cached gate weights from the previous step for routing decisions during training. Call model.update_anticipatory_weights() after each optimizer step.
        swiglu_clamp_value: DeepSeekV4 - clamp SwiGLU linear component to [-c, c] and gate component upper bound to c. 0 means disabled. Default: 0.0.
    """

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
        batched_experts: bool = True,
        max_batch_size: int = 1,
        flash_attention: bool = True,
        use_hca: bool = False,
        use_csa: bool = False,
        kv_dim: int = 128,
        num_query_heads: int = 8,
        compression_rate: int = 8,
        num_groups: int = 1,
        group_out_dim: Optional[int] = None,
        window_size: int = 32,
        use_attention_sink: bool = True,
        use_partial_rope: bool = True,
        rope_dim: int = 64,
        attn_dropout: float = 0.0,
        csa_top_k_blocks: int = 32,
        csa_indexer_dim: Optional[int] = None,
        sigmoid_routing: bool = False,
        num_shared_experts: int = 0,
        granularity_factor: int = 1,
        aux_loss_free: bool = False,
        bias_update_speed: float = 0.01,
        seq_balance_loss_weight: float = 0.0,
        sqrt_softplus_routing: bool = False,
        hash_routing: bool = False,
        num_hash_functions: int = 4,
        anticipatory_routing: bool = False,
        swiglu_clamp_value: float = 0.0,
    ):
        super().__init__()

        assert routing_strategy in ("top_k", "expert_choice"), (
            f"routing_strategy must be 'top_k' or 'expert_choice', got '{routing_strategy}'"
        )
        assert not (sigmoid_routing and sqrt_softplus_routing), (
            "sigmoid_routing and sqrt_softplus_routing are mutually exclusive"
        )
        assert num_hash_functions >= 1, (
            f"num_hash_functions must be >= 1, got {num_hash_functions}"
        )
        assert swiglu_clamp_value >= 0, (
            f"swiglu_clamp_value must be >= 0, got {swiglu_clamp_value}"
        )
        assert num_experts > 0, f"num_experts must be positive, got {num_experts}"
        assert 0 < expert_top_k <= num_experts, (
            f"expert_top_k must be in (0, num_experts], got {expert_top_k} with num_experts={num_experts}"
        )
        assert capacity_factor > 0, (
            f"capacity_factor must be positive, got {capacity_factor}"
        )
        assert moe_every_n_layers >= 1, (
            f"moe_every_n_layers must be >= 1, got {moe_every_n_layers}"
        )
        if moe_layers is not None:
            assert all(idx >= 0 for idx in moe_layers), (
                f"moe_layers must contain non-negative indices, got {moe_layers}"
            )
        assert mult > 0, f"mult must be positive, got {mult}"
        assert granularity_factor >= 1, (
            f"granularity_factor must be >= 1, got {granularity_factor}"
        )
        assert num_shared_experts >= 0, (
            f"num_shared_experts must be >= 0, got {num_shared_experts}"
        )
        if granularity_factor > 1 and num_shared_experts > 0:
            effective_top_k = expert_top_k * granularity_factor
            assert effective_top_k > num_shared_experts, (
                f"expert_top_k*granularity_factor ({effective_top_k}) must be > "
                f"num_shared_experts ({num_shared_experts})"
            )
        if use_hca or use_csa:
            assert kv_dim > 0, f"kv_dim must be positive, got {kv_dim}"
            assert num_query_heads > 0, (
                f"num_query_heads must be positive, got {num_query_heads}"
            )
            assert compression_rate > 0, (
                f"compression_rate must be positive, got {compression_rate}"
            )
            assert num_groups >= 1, f"num_groups must be >= 1, got {num_groups}"
            if num_groups > 1:
                assert num_query_heads % num_groups == 0, (
                    f"num_query_heads ({num_query_heads}) must be divisible by num_groups ({num_groups})"
                )
            if group_out_dim is not None:
                assert group_out_dim > 0, (
                    f"group_out_dim must be positive, got {group_out_dim}"
                )
            assert window_size >= 0, (
                f"window_size must be non-negative, got {window_size}"
            )
            assert rope_dim > 0, f"rope_dim must be positive, got {rope_dim}"

        if use_csa:
            assert csa_top_k_blocks >= 0, (
                f"csa_top_k_blocks must be non-negative, got {csa_top_k_blocks}"
            )
            if csa_indexer_dim is not None:
                assert csa_indexer_dim > 0, (
                    f"csa_indexer_dim must be positive, got {csa_indexer_dim}"
                )

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
            sigmoid_routing=sigmoid_routing,
            num_shared_experts=num_shared_experts,
            granularity_factor=granularity_factor,
            aux_loss_free=aux_loss_free,
            bias_update_speed=bias_update_speed,
            seq_balance_loss_weight=seq_balance_loss_weight,
            sqrt_softplus_routing=sqrt_softplus_routing,
            hash_routing=hash_routing,
            num_hash_functions=num_hash_functions,
            anticipatory_routing=anticipatory_routing,
            swiglu_clamp_value=swiglu_clamp_value,
        )

        self.num_experts = num_experts
        self.expert_top_k = expert_top_k
        self.routing_strategy = routing_strategy
        self.use_hca = use_hca
        self.use_csa = use_csa
        self.sigmoid_routing = sigmoid_routing
        self.num_shared_experts = num_shared_experts
        self.granularity_factor = granularity_factor
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed
        self.seq_balance_loss_weight = seq_balance_loss_weight
        self.sqrt_softplus_routing = sqrt_softplus_routing
        self.hash_routing = hash_routing
        self.num_hash_functions = num_hash_functions
        self.anticipatory_routing = anticipatory_routing
        self.swiglu_clamp_value = swiglu_clamp_value
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

        dim = getattr(attn_layers, "dim", None)
        assert dim is not None, (
            "Could not determine dim from transformer attention layers"
        )

        self.model_config = {
            "dim": dim,
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
            "use_hca": use_hca,
            "use_csa": use_csa,
            "kv_dim": kv_dim,
            "num_query_heads": num_query_heads,
            "compression_rate": compression_rate,
            "num_groups": num_groups,
            "group_out_dim": group_out_dim,
            "window_size": window_size,
            "use_attention_sink": use_attention_sink,
            "use_partial_rope": use_partial_rope,
            "rope_dim": rope_dim,
            "attn_dropout": attn_dropout,
            "csa_top_k_blocks": csa_top_k_blocks,
            "csa_indexer_dim": csa_indexer_dim,
            "sigmoid_routing": sigmoid_routing,
            "num_shared_experts": num_shared_experts,
            "granularity_factor": granularity_factor,
            "aux_loss_free": aux_loss_free,
            "bias_update_speed": bias_update_speed,
            "seq_balance_loss_weight": seq_balance_loss_weight,
            "sqrt_softplus_routing": sqrt_softplus_routing,
            "hash_routing": hash_routing,
            "num_hash_functions": num_hash_functions,
            "anticipatory_routing": anticipatory_routing,
            "swiglu_clamp_value": swiglu_clamp_value,
        }

        self.ds4_attention = None
        if use_hca or use_csa:
            hca_cfg = None
            csa_cfg = None

            if use_hca:
                hca_cfg = {
                    "kv_dim": kv_dim,
                    "num_query_heads": num_query_heads,
                    "compression_rate": compression_rate,
                    "num_groups": num_groups,
                    "group_out_dim": group_out_dim,
                    "window_size": window_size,
                    "use_attention_sink": use_attention_sink,
                    "use_partial_rope": use_partial_rope,
                    "rope_dim": rope_dim,
                    "dropout": attn_dropout,
                }

            if use_csa:
                csa_cfg = {
                    "kv_dim": kv_dim,
                    "num_query_heads": num_query_heads,
                    "compression_rate": compression_rate,
                    "top_k_blocks": csa_top_k_blocks,
                    "num_groups": num_groups,
                    "group_out_dim": group_out_dim,
                    "window_size": window_size,
                    "use_attention_sink": use_attention_sink,
                    "use_partial_rope": use_partial_rope,
                    "rope_dim": rope_dim,
                    "indexer_dim": csa_indexer_dim,
                    "dropout": attn_dropout,
                }

            self.ds4_attention = HybridAttentionBlock(
                dim=dim,
                hca_config=hca_cfg,
                csa_config=csa_cfg,
            )
            self.ds4_norm = nn.LayerNorm(
                transformer.emb_dim if hasattr(transformer, "emb_dim") else dim
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

    def update_routing_biases(self):
        for module in self.modules():
            if isinstance(module, MoEFFN):
                module.update_routing_bias()

    def update_anticipatory_weights(self):
        for module in self.modules():
            if isinstance(module, MoEFFN):
                module.update_anticipatory_weights()

    def enable_gradient_checkpointing(self):
        self._gradient_checkpointing = True
        if hasattr(self.transformer, "attn_layers"):
            attn_layers = self.transformer.attn_layers
            if hasattr(attn_layers, "grad_checkpointing"):
                attn_layers.grad_checkpointing = True
        count = enable_gradient_checkpointing(self)
        return count

    def forward(self, x, **kwargs):
        # Note: ds4_attention (HCA/CSA) is an experimental block for compressed
        # cross-sequence attention. It is created when use_hca/use_csa is True
        # but is not yet integrated into the standard forward pass (requires
        # embedding before the attention block, which happens inside the
        # autoregressive wrapper).
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
