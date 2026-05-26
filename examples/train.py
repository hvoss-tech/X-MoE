import os
import math
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

from accelerate import Accelerator

from x_transformers import TransformerWrapper, Decoder

from x_moe.wrapper import MoETransformerWrapper
from x_moe.optimizer import Muon, MuonWithAdamW, configure_muon_optimizer
from x_moe.perf import (
    DataPrefetcher,
    ThroughputLogger,
    CUDAGraphCapturer,
    get_linear_warmup_cosine_scheduler,
)


def train_tokenizer(texts, vocab_size=4096, save_path="tokenizer.json"):
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel()
    tokenizer.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<eos>", "<unk>"],
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


def encode(text, tokenizer, max_len):
    encoded = tokenizer.encode(text).ids
    if len(encoded) > max_len:
        encoded = encoded[:max_len]
    return encoded


class TinyStoriesDataset(torch.utils.data.Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.texts = texts
        self.eos_id = tokenizer.token_to_id("<eos>")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = encode(text, self.tokenizer, self.max_seq_len - 1)
        tokens = tokens + [self.eos_id]
        tokens = tokens[: self.max_seq_len]
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch, pad_id=0):
    max_len = max(b.shape[0] for b in batch)
    padded = []
    for b in batch:
        pad_len = max_len - b.shape[0]
        if pad_len > 0:
            padded.append(F.pad(b, (0, pad_len), value=pad_id))
        else:
            padded.append(b)
    return torch.stack(padded)


def get_collate_fn(pad_id=0):
    def _collate(batch):
        return collate_fn(batch, pad_id=pad_id)

    return _collate


def compute_perplexity(model, dataloader, pad_id=0):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            loss = model(batch)
            if isinstance(loss, tuple):
                loss = loss[0]
            num_tokens = (batch[:, 1:] != pad_id).sum().item()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    if total_tokens == 0:
        return float("inf")
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(min(avg_loss, 20))
    return perplexity


def build_model(args, vocab_size):
    decoder_kwargs = dict(
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        ff_glu=not args.no_ff_glu,
        ff_mult=args.ff_mult,
        ff_dropout=args.ff_dropout,
        attn_dropout=args.attn_dropout,
        layer_dropout=args.layer_dropout,
        rotary_pos_emb=not args.no_rotary_pos_emb,
        ff_no_bias=not args.ff_bias,
    )
    if args.flash_attention:
        decoder_kwargs["attn_flash"] = True

    decoder = Decoder(**decoder_kwargs)

    transformer = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=args.max_seq_len,
        attn_layers=decoder,
        emb_dropout=args.emb_dropout,
        tie_embedding=True,
        use_abs_pos_emb=args.no_rotary_pos_emb,
    )

    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=args.num_experts,
        expert_top_k=args.expert_top_k,
        capacity_factor=args.capacity_factor,
        routing_strategy=args.routing_strategy,
        load_balance_loss_weight=args.load_balance_loss_weight,
        z_loss_weight=args.z_loss_weight,
        moe_every_n_layers=args.moe_every_n_layers,
        moe_layers=args.moe_layers,
        glu=not args.no_ff_glu,
        mult=args.ff_mult,
        dropout=args.ff_dropout,
        no_bias=not args.ff_bias,
        zero_init_output=True,
        batched_experts=args.batched_experts,
        max_batch_size=args.batch_size,
        flash_attention=args.flash_attention,
        use_hca=args.use_hca,
        use_csa=args.use_csa,
        kv_dim=args.kv_dim,
        num_query_heads=args.num_query_heads,
        compression_rate=args.compression_rate,
        num_groups=args.num_groups,
        group_out_dim=args.group_out_dim if args.group_out_dim > 0 else None,
        window_size=args.window_size,
        use_attention_sink=args.use_attention_sink,
        use_partial_rope=args.use_partial_rope,
        rope_dim=args.rope_dim,
        attn_dropout=args.attn_dropout,
        csa_top_k_blocks=args.csa_top_k_blocks,
        csa_indexer_dim=args.csa_indexer_dim if args.csa_indexer_dim > 0 else None,
    )

    return model


def main():
    parser = argparse.ArgumentParser(description="Train MoE Transformer on TinyStories")

    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)

    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--expert-top-k", type=int, default=2)
    parser.add_argument(
        "--routing-strategy",
        type=str,
        default="top_k",
        choices=["top_k", "expert_choice"],
    )
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--load-balance-loss-weight", type=float, default=0.01)
    parser.add_argument("--z-loss-weight", type=float, default=1e-4)
    parser.add_argument("--moe-every-n-layers", type=int, default=1)
    parser.add_argument(
        "--moe-layers",
        type=str,
        default=None,
        help="Comma-separated FFN layer indices to make MoE (e.g. '0,2,4'). "
        "If set, overrides --moe-every-n-layers.",
    )
    parser.add_argument(
        "--batched-experts",
        action="store_true",
        default=False,
        help="Use batched expert computation with stacked parameters",
    )

    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--tokenizer-path", type=str, default=None)

    parser.add_argument("--ff-glu", default=True, help="Use GLU in FFN (default: True)")
    parser.add_argument(
        "--no-ff-glu", action="store_true", default=False, help="Disable GLU in FFN"
    )
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument(
        "--ff-bias",
        action="store_true",
        default=False,
        help="Use bias in FFN linear layers (default: no bias)",
    )
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--ff-dropout", type=float, default=0.1)
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--layer-dropout", type=float, default=0.0)
    parser.add_argument("--no-rotary-pos-emb", action="store_true", default=False)

    parser.add_argument(
        "--use-hca",
        action="store_true",
        default=True,
        help="Use Heavily Compressed Attention (HCA)",
    )
    parser.add_argument(
        "--use-csa",
        action="store_true",
        default=False,
        help="Use Compressed Sparse Attention (CSA)",
    )
    parser.add_argument(
        "--kv-dim", type=int, default=128, help="Compressed KV dimension for HCA/CSA"
    )
    parser.add_argument(
        "--num-query-heads",
        type=int,
        default=8,
        help="Number of query heads for HCA/CSA",
    )
    parser.add_argument(
        "--compression-rate", type=int, default=8, help="Compression rate for HCA/CSA"
    )
    parser.add_argument(
        "--num-groups",
        type=int,
        default=1,
        help="Number of query head groups for HCA/CSA",
    )
    parser.add_argument(
        "--group-out-dim", type=int, default=0, help="Output dim per group (0 = auto)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=32,
        help="Sliding window size for HCA/CSA (0 to disable)",
    )
    parser.add_argument(
        "--use-attention-sink",
        action="store_true",
        default=True,
        help="Use attention sink in HCA/CSA",
    )
    parser.add_argument(
        "--use-partial-rope",
        action="store_true",
        default=True,
        help="Use partial rotary pos emb in HCA/CSA",
    )
    parser.add_argument(
        "--rope-dim", type=int, default=64, help="RoPE dimension for HCA/CSA"
    )
    parser.add_argument(
        "--csa-top-k-blocks", type=int, default=32, help="Top-K blocks for CSA indexer"
    )
    parser.add_argument(
        "--csa-indexer-dim",
        type=int,
        default=0,
        help="CSA indexer dimension (0 = auto)",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="muon",
        choices=["adamw", "muon"],
        help="Optimizer to use (adamw or muon)",
    )
    parser.add_argument("--muon-lr", type=float, default=1e-3)
    parser.add_argument("--muon-momentum", type=float, default=0.9)
    parser.add_argument("--muon-rms-factor", type=float, default=1.0)
    parser.add_argument("--adamw-for-non-muon-lr", type=float, default=3e-4)

    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Mixed precision training mode via accelerate",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Compile the model with torch.compile for faster training",
    )
    parser.add_argument(
        "--flash-attention",
        action="store_true",
        default=True,
        help="Enable Flash Attention in the decoder",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--prefetch-data",
        action="store_true",
        default=True,
        help="Enable CUDA data prefetching for faster data loading",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        default=True,
        help="Capture CUDA graphs for training steps (fixed shape input)",
    )
    parser.add_argument(
        "--aux-loss-every",
        type=int,
        default=1,
        help="Compute MoE aux loss every N steps. >1 enables lazy computation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )

    args = parser.parse_args()

    if args.moe_layers is not None:
        args.moe_layers = [int(x.strip()) for x in args.moe_layers.split(",")]
    else:
        args.moe_layers = None

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        gradient_accumulation_steps=args.gradient_accumulate,
    )

    accelerator.print(
        f"Accelerator: device={accelerator.device}, "
        f"mixed_precision={accelerator.mixed_precision}, "
        f"num_processes={accelerator.num_processes}"
    )

    accelerator.wait_for_everyone()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if accelerator.num_processes > 1:
        from accelerate.utils import set_seed

        set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    accelerator.print("Loading TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    train_texts = ds["train"]["text"]
    val_texts = ds["validation"]["text"]

    if args.max_samples is not None:
        train_texts = train_texts[: args.max_samples]
        val_texts = val_texts[: min(args.max_samples // 10, len(val_texts))]

    accelerator.print(
        f"Train samples: {len(train_texts)}, Val samples: {len(val_texts)}"
    )

    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        accelerator.print(f"Loading tokenizer from {args.tokenizer_path}...")
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    else:
        accelerator.print("Training tokenizer...")
        tokenizer_save_path = str(save_dir / "tokenizer.json")
        tokenizer = train_tokenizer(
            train_texts[: min(len(train_texts), 100000)],
            vocab_size=args.vocab_size,
            save_path=tokenizer_save_path,
        )
        accelerator.print(f"Tokenizer saved to {tokenizer_save_path}")

    actual_vocab_size = tokenizer.get_vocab_size()
    accelerator.print(f"Vocabulary size: {actual_vocab_size}")

    pad_id = tokenizer.token_to_id("<pad>")
    eos_id = tokenizer.token_to_id("<eos>")

    accelerator.print("Creating datasets...")
    train_dataset = TinyStoriesDataset(train_texts, tokenizer, args.max_seq_len)
    val_dataset = TinyStoriesDataset(val_texts, tokenizer, args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(pad_id=pad_id),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(pad_id=pad_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    accelerator.print("Building MoE Transformer...")
    model = build_model(args, actual_vocab_size)

    if args.gradient_checkpointing:
        count = model.enable_gradient_checkpointing()
        accelerator.print(f"Enabled gradient checkpointing on {count} MoE layers")

    accelerator.print(f"Model parameters: {model.num_params:,}")

    if args.compile:
        accelerator.print("Compiling model with torch.compile...")
        model = torch.compile(model, dynamic=True)

    if args.optimizer == "muon":
        muon_opt, adamw_opt = configure_muon_optimizer(
            model,
            lr=args.muon_lr,
            momentum=args.muon_momentum,
            weight_decay=args.weight_decay,
            adamw_lr=args.adamw_for_non_muon_lr,
            adamw_weight_decay=args.weight_decay,
            rms_rescale_factor=args.muon_rms_factor,
        )
        optimizer = MuonWithAdamW(muon_opt, adamw_opt)
        accelerator.print(
            f"Using Muon optimizer (muon_lr={args.muon_lr}, adamw_lr={args.adamw_for_non_muon_lr})"
        )
    else:
        optimizer = AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        accelerator.print(f"Using AdamW optimizer (lr={args.lr})")

    warmup_steps = args.warmup_steps
    total_steps = args.epochs * len(train_loader) // args.gradient_accumulate

    if args.optimizer == "muon":
        muon_scheduler = get_linear_warmup_cosine_scheduler(
            muon_opt, warmup_steps, total_steps, eta_min=0.1
        )
        adamw_scheduler = get_linear_warmup_cosine_scheduler(
            adamw_opt, warmup_steps, total_steps, eta_min=0.1
        )
        accelerator.print(
            f"Using warmup+cosine scheduler: warmup={warmup_steps}, total={total_steps}"
        )
    else:
        scheduler = get_linear_warmup_cosine_scheduler(
            optimizer, warmup_steps, total_steps, eta_min=0.1
        )
        accelerator.print(
            f"Using warmup+cosine scheduler: warmup={warmup_steps}, total={total_steps}"
        )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    throughput_logger = ThroughputLogger(log_interval=args.log_interval)

    best_val_ppl = float("inf")
    global_step = 0

    accelerator.print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_start = time.time()
        throughput_logger.start_epoch()

        use_prefetcher = args.prefetch_data and accelerator.device.type == "cuda"

        if use_prefetcher:
            half = args.mixed_precision in ("fp16", "bf16")
            data_iter = DataPrefetcher(train_loader, accelerator.device, half=half)
        else:
            data_iter = train_loader

        for batch_idx, batch in enumerate(data_iter):
            with accelerator.autocast():
                loss = model(batch)
                if isinstance(loss, tuple):
                    loss = loss[0]

                should_compute_aux = (args.aux_loss_every <= 1) or (
                    (batch_idx + 1) % args.aux_loss_every == 0
                )
                if not should_compute_aux:
                    model.set_aux_loss_compute(False)

                moe_aux = model.moe_aux_loss
                model.reset_moe_aux_loss()
                model.set_aux_loss_compute(True)

                total_loss = loss + moe_aux

            accelerator.backward(total_loss)

            if (batch_idx + 1) % args.gradient_accumulate == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                if args.optimizer == "muon":
                    muon_scheduler.step()
                    adamw_scheduler.step()
                else:
                    scheduler.step()

                optimizer.zero_grad()
                global_step += 1

            num_tokens = (batch[:, 1:] != pad_id).sum().item()
            epoch_loss += loss.item() * num_tokens
            epoch_tokens += num_tokens

            throughput_logger.log_step(
                num_tokens=num_tokens,
                step=batch_idx,
                extra_info={
                    "loss": f"{loss.item():.4f}",
                    "moe_aux": f"{moe_aux.item():.4f}",
                },
            )

            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = epoch_loss / epoch_tokens
                ppl = math.exp(min(avg_loss, 20))
                lr = optimizer.param_groups[0]["lr"]
                accelerator.print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Step {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss {avg_loss:.4f} | "
                    f"PPL {ppl:.2f} | "
                    f"LR {lr:.6f} | "
                    f"MoE aux {moe_aux.item():.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        train_ppl = math.exp(min(avg_train_loss, 20))
        perf_summary = throughput_logger.epoch_summary()

        accelerator.print(f"\n--- Epoch {epoch} Summary ---")
        accelerator.print(
            f"Train Loss: {avg_train_loss:.4f} | Train PPL: {train_ppl:.2f} | Time: {epoch_time:.1f}s"
        )
        if perf_summary:
            accelerator.print(
                f"Throughput: {perf_summary['tokens_per_sec']:.0f} tokens/s | "
                f"{perf_summary['total_tokens']} tokens total"
            )

        val_ppl = None
        if epoch % args.val_interval == 0:
            accelerator.print("Computing validation perplexity...")
            model.eval()
            val_loss = 0.0
            val_tokens = 0
            with torch.no_grad():
                for batch in val_loader:
                    with accelerator.autocast():
                        v_loss = model(batch)
                        if isinstance(v_loss, tuple):
                            v_loss = v_loss[0]
                    num_v_tokens = (batch[:, 1:] != pad_id).sum().item()
                    val_loss += v_loss.item() * num_v_tokens
                    val_tokens += num_v_tokens
            if val_tokens > 0:
                val_ppl = math.exp(val_loss / val_tokens)
            else:
                val_ppl = float("inf")
            accelerator.print(f"Validation PPL: {val_ppl:.2f}")
            model.train()

            if val_ppl < best_val_ppl and accelerator.is_main_process:
                best_val_ppl = val_ppl
                best_path = save_dir / "best_model.pt"
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_ppl": val_ppl,
                        "train_ppl": train_ppl,
                        "model_config": unwrapped_model.model_config,
                        "vocab_size": actual_vocab_size,
                        "max_seq_len": args.max_seq_len,
                    },
                    best_path,
                )
                accelerator.print(
                    f"New best model saved to {best_path} (PPL: {val_ppl:.2f})"
                )

        if accelerator.is_main_process:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrapped_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_ppl": val_ppl,
                    "model_config": unwrapped_model.model_config,
                    "vocab_size": actual_vocab_size,
                    "max_seq_len": args.max_seq_len,
                },
                checkpoint_path,
            )

    accelerator.print(f"\nTraining complete! Best validation PPL: {best_val_ppl:.2f}")


if __name__ == "__main__":
    main()
