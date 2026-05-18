import argparse
from pathlib import Path

import torch

from x_transformers import TransformerWrapper, Decoder

from easy_moe.wrapper import MoETransformerWrapper


def build_model_from_config(model_config, vocab_size, max_seq_len):
    decoder = Decoder(
        dim=model_config.get("dim", 256),
        depth=model_config.get("depth", 6),
        heads=model_config.get("heads", 8),
        ff_glu=not model_config.get("no_ff_glu", False),
        ff_mult=model_config.get("ff_mult", 4),
        ff_dropout=0.0,
        attn_dropout=0.0,
        layer_dropout=0.0,
        rotary_pos_emb=not model_config.get("no_rotary_pos_emb", False),
        ff_no_bias=not model_config.get("ff_bias", False),
    )

    transformer = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        attn_layers=decoder,
        emb_dropout=0.0,
        tie_embedding=True,
        use_abs_pos_emb=model_config.get("no_rotary_pos_emb", False),
    )

    ds4_attention = None
    if model_config.get("use_hca", False) or model_config.get("use_csa", False):
        from easy_moe.attention import HybridAttentionBlock
        hca_cfg = None
        csa_cfg = None
        if model_config.get("use_hca", False):
            hca_cfg = {
                "kv_dim": model_config.get("hca_kv_dim", 128),
                "num_query_heads": model_config.get("hca_num_heads", 8),
                "compression_rate": model_config.get("hca_compression_rate", 8),
                "num_groups": model_config.get("hca_num_groups", 1),
                "window_size": model_config.get("hca_window_size", 32),
                "use_attention_sink": model_config.get("hca_use_sink", True),
                "use_partial_rope": model_config.get("hca_use_rope", True),
                "rope_dim": model_config.get("hca_rope_dim", 64),
            }
        if model_config.get("use_csa", False):
            csa_cfg = {
                "kv_dim": model_config.get("csa_kv_dim", 128),
                "num_query_heads": model_config.get("csa_num_heads", 8),
                "compression_rate": model_config.get("csa_compression_rate", 4),
                "top_k_blocks": model_config.get("csa_top_k_blocks", 32),
                "num_groups": model_config.get("csa_num_groups", 1),
                "window_size": model_config.get("csa_window_size", 32),
                "use_attention_sink": model_config.get("csa_use_sink", True),
                "use_partial_rope": model_config.get("csa_use_rope", True),
                "rope_dim": model_config.get("csa_rope_dim", 64),
            }
        ds4_attention = HybridAttentionBlock(
            dim=model_config.get("dim", 256), hca_config=hca_cfg, csa_config=csa_cfg
        )

    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=model_config.get("num_experts", 8),
        expert_top_k=model_config.get("expert_top_k", 2),
        capacity_factor=model_config.get("capacity_factor", 1.25),
        routing_strategy=model_config.get("routing_strategy", "top_k"),
        load_balance_loss_weight=model_config.get("load_balance_loss_weight", 0.01),
        z_loss_weight=model_config.get("z_loss_weight", 1e-4),
        moe_every_n_layers=model_config.get("moe_every_n_layers", 1),
        moe_layers=model_config.get("moe_layers", None),
        glu=not model_config.get("no_ff_glu", False),
        mult=model_config.get("ff_mult", 4),
        dropout=0.0,
        no_bias=not model_config.get("ff_bias", False),
        zero_init_output=True,
        ds4_attention=ds4_attention,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained MoE Transformer")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer JSON file")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt for generation")
    parser.add_argument("--num-stories", type=int, default=5, help="Number of stories to generate")
    parser.add_argument("--seq-len", type=int, default=256, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=str, default="top_k", help="Filter logits fn (top_k, top_p, min_p)")
    parser.add_argument("--top-k-val", type=int, default=50, help="Top-k value for filtering")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) value for filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_config = checkpoint["model_config"]

    print("Loading tokenizer...")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    vocab_size = checkpoint.get("vocab_size", tokenizer.get_vocab_size())
    max_seq_len = checkpoint.get("max_seq_len", model_config.get("max_seq_len", 256))

    print("Building model...")
    model = build_model_from_config(model_config, vocab_size, max_seq_len)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    num_params = model.num_params
    print(f"Model loaded. Parameters: {num_params:,}")
    print(f"Checkpoint val PPL: {checkpoint.get('val_ppl', 'N/A')}")
    print()

    filter_kwargs = {}
    if args.top_k == "top_k":
        filter_kwargs = {"k": args.top_k_val}
    elif args.top_k == "top_p":
        filter_kwargs = {}
    elif args.top_k == "min_p":
        filter_kwargs = {"min_p": 0.05}

    for i in range(args.num_stories):
        if args.prompt:
            prompt_text = args.prompt
        else:
            prompt_text = ""

        if prompt_text:
            prompt_tokens = tokenizer.encode(prompt_text).ids
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        else:
            prompt_tensor = torch.zeros(args.batch_size, 1, dtype=torch.long, device=device)
            prompt_tensor[:, 0] = tokenizer.token_to_id("<eos>") if eos_id is not None else 0

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            output = model.generate(
                prompt_tensor,
                seq_len=args.seq_len,
                temperature=args.temperature,
                filter_logits_fn=args.top_k,
                filter_kwargs=filter_kwargs,
                eos_token=eos_id,
                cache_kv=True,
            )

        for b in range(output.shape[0]):
            tokens = output[b].tolist()
            text = tokenizer.decode(tokens)
            text = text.replace("<pad>", "").replace("<eos>", "")
            text = text.strip()
            print(f"--- Story {i * args.batch_size + b + 1} ---")
            print(text)
            print()


if __name__ == "__main__":
    main()