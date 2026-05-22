# X-MoE

A Mixture of Experts wrapper around [x-transformers](https://github.com/lucidrains/x-transformers) for training state-of-the-art MoE language models.
This project was primarily done because I really like the x-transformers library, but would like to easily add MoE training on top of it. This library enables exactly that.

X-MoE takes any x-transformers `Decoder` / `TransformerWrapper` model and replaces its feed-forward layers with MoE routing layers, giving you sparse, expert-driven architectures with minimal code changes. It ships with two routing strategies, custom DS4 attention mechanisms, the Muon optimizer, CUDA performance utilities, and a full-featured `Trainer`.

---

## Features

### Core MoE

- **MoEFFN layer** — drop-in replacement for FFN layers with configurable experts, top-k routing, and capacity factor
- **Two routing strategies** — `top_k` (tokens choose experts) and `expert_choice` (experts choose tokens)
- **Auxiliary losses** — load balance loss and z-loss to prevent routing collapse, both with configurable weights
- **Selective MoE placement** — replace every Nth FFN, or target specific layers by index
- **Batched experts** — stack expert parameters into single tensors for a vectorized einsum-based forward pass

### DS4 Attention (Dual-State Sparse Streaming)

- **HCA** (Heavily Compressed Attention) — KV compression with learned soft-merging, sliding window, and attention sinks
- **CSA** (Compressed Sparse Attention) — overlapped block compression + top-K block retrieval via a learned indexer
- **SharedKVMQA** — multi-query attention over a compressed KV cache with optional grouped output projections
- **HybridAttentionBlock** — composable block combining HCA and/or CSA layers

### Muon Optimizer

- **Muon** — momentum optimizer using Newton-Schulz orthogonalization for 2D+ weight matrices
- **MuonWithAdamW** — combined optimizer: Muon for 2D+ weights, AdamW for 1D params (biases, norms, embeddings, gates)
- **Auto parameter classification** — `configure_muon_optimizer()` splits model parameters into the right groups

### Performance Utilities

- **DataPrefetcher** — CUDA-stream-based async data prefetching
- **ThroughputLogger** — tokens/sec and step-time tracking with rolling window averaging
- **CUDAGraphCapturer** — capture and replay CUDA graphs for fixed-shape training steps
- **LR schedulers** — linear warmup + cosine decay, with a Muon-specific variant

### Trainer

- Full training loop powered by [HuggingFace Accelerate](https://github.com/huggingface/accelerate)
- Multi-GPU, mixed precision (`bf16`/`fp16`), gradient accumulation
- `torch.compile()` support and gradient checkpointing
- Validation perplexity tracking and best-model checkpointing
- `Trainer.load()` for resuming from checkpoint
- Built-in `chat()` and `generate()` for interactive text generation

---

## Installation

Requires Python >= 3.11 and a CUDA-capable PyTorch install.

```bash
# Clone
git clone git@github.com:hvoss-techfak/X-MoE.git
cd X-MoE

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e ".[dev]"
```

---

## Quick Start

### Train with the Trainer API

```python
from datasets import load_dataset
from x_transformers import TransformerWrapper, Decoder
from x_moe import MoETransformerWrapper, Trainer
from x_moe.data import TextDataset, train_tokenizer

# 1. Load data
ds = load_dataset("roneneldan/TinyStories")
train_texts = ds["train"]["text"]
val_texts = ds["validation"]["text"]

# 2. Train a tokenizer
tokenizer = train_tokenizer(train_texts, vocab_size=4096, save_path="tokenizer.json")

# 3. Create datasets
train_ds = TextDataset(train_texts, tokenizer, max_seq_len=256)
val_ds = TextDataset(val_texts, tokenizer, max_seq_len=256)

# 4. Build the model
decoder = Decoder(
    dim=256, depth=12, heads=8,
    ff_glu=True, ff_mult=4,
    rotary_pos_emb=True, ff_no_bias=True,
)
transformer = TransformerWrapper(
    num_tokens=tokenizer.get_vocab_size(),
    max_seq_len=256,
    attn_layers=decoder,
    tie_embedding=True, use_abs_pos_emb=False,
)
model = MoETransformerWrapper(
    transformer=transformer,
    num_experts=32,
    expert_top_k=2,
    routing_strategy="top_k",
    load_balance_loss_weight=0.01,
    z_loss_weight=1e-4,
)

# 5. Train
trainer = Trainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, val_dataset=val_ds,
    epochs=10, batch_size=32, lr=3e-4, optimizer="muon",
)
trainer.train()
trainer.save()

# 6. Load and generate
trainer = Trainer.load("checkpoints/best_model.pt", tokenizer=tokenizer)
print(trainer.chat("Once upon a time"))
```

### Train with the CLI

```bash
python examples/train.py \
  --dim 256 --depth 12 --heads 8 \
  --num-experts 32 --expert-top-k 2 \
  --routing-strategy top_k \
  --optimizer muon --muon-lr 1e-3 \
  --batch-size 32 --epochs 10 \
  --mixed-precision bf16 \
  --flash-attention --compile \
  --max-seq-len 256 --vocab-size 4096
```

### Generate from a checkpoint

```bash
python examples/generate.py \
  --checkpoint checkpoints/best_model.pt \
  --tokenizer tokenizer.json \
  --prompt "Once upon a time" \
  --seq-len 256 \
  --temperature 0.8
```

---

## Citations

```bibtex
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```