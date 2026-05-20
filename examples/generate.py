import argparse

import torch

from easy_moe.trainer import build_model_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate text with a trained MoE Transformer"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--tokenizer", type=str, required=True, help="Path to tokenizer JSON file"
    )
    parser.add_argument(
        "--prompt", type=str, default="", help="Text prompt for generation"
    )
    parser.add_argument(
        "--num-stories", type=int, default=5, help="Number of stories to generate"
    )
    parser.add_argument(
        "--seq-len", type=int, default=256, help="Max generation length"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="top_k",
        help="Filter logits fn (top_k, top_p, min_p)",
    )
    parser.add_argument(
        "--top-k-val", type=int, default=50, help="Top-k value for filtering"
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p (nucleus) value for filtering"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for generation"
    )
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
            prompt_tensor = torch.tensor(
                [prompt_tokens], dtype=torch.long, device=device
            )
        else:
            prompt_tensor = torch.zeros(
                args.batch_size, 1, dtype=torch.long, device=device
            )
            prompt_tensor[:, 0] = (
                tokenizer.token_to_id("<eos>") if eos_id is not None else 0
            )

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
