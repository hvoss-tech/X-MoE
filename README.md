<div align="center">

# X-MoE

A Mixture of Experts wrapper around [x-transformers](https://github.com/lucidrains/x-transformers) for training state-of-the-art MoE language models with minimal code changes.

<p>
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c" alt="PyTorch 2.0+" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License: MIT" />
  <img src="https://img.shields.io/badge/CUDA-Required-76b900" alt="CUDA Required" />
</p>

</div>

---

**You want to train a MoE model. You love x-transformers. You don't want to rewrite your entire codebase.**

X-MoE takes any x-transformers `Decoder` / `TransformerWrapper` and replaces its feed-forward layers with sparse, expert-driven routing. Two lines of code and you're running a Mixture of Experts model.

---

## ✨ Features

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🧠 MoE Routing</h3>
      <p>Drop-in <code>MoEFFN</code> layer, configurable experts, top-k routing, capacity factor. Two strategies: <b>top_k</b> (tokens choose) and <b>expert_choice</b> (experts choose).</p>
    </td>
    <td width="50%" valign="top">
      <h3>⚖️ Auxiliary Losses</h3>
      <p>Load balance loss + z-loss prevent routing collapse. Configurable weights, zero integration effort.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🎯 Selective Placement</h3>
      <p>Replace every Nth FFN, or target specific layers by index. Put MoE where it matters.</p>
    </td>
    <td width="50%" valign="top">
      <h3>⚡ Batched Experts</h3>
      <p>Stack expert parameters into single tensors for vectorized einsum-based forward passes. No loops, no overhead.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🔬 DS4 Attention</h3>
      <p>Heavily Compressed Attention (HCA) + Compressed Sparse Attention (CSA) + SharedKVMQA. composable via <code>HybridAttentionBlock</code>.</p>
    </td>
    <td width="50%" valign="top">
      <h3>🚀 Muon Optimizer</h3>
      <p>Newton-Schulz orthogonalization for 2D+ weights. <code>MuonWithAdamW</code> auto-splits params, Muon for matrices, AdamW for the rest.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>🔧 CUDA Utilities</h3>
      <p>Async <code>DataPrefetcher</code>, <code>ThroughputLogger</code>, <code>CUDAGraphCapturer</code>, squeeze every token/second out of your hardware.</p>
    </td>
    <td width="50%" valign="top">
      <h3>🏋️ Trainer</h3>
      <p>Full training loop powered by HuggingFace Accelerate. Multi-GPU, mixed precision, gradient checkpointing, <code>torch.compile()</code>, and built-in <code>chat()</code>.</p>
    </td>
  </tr>
</table>

---

## 🚀 Quick Start

### Installation

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

### Train a model

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

# 4. Build the model from x-transformer
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
# 5. Add the MoE Wrapper
model = MoETransformerWrapper(
    transformer=transformer,
    num_experts=32,
    expert_top_k=2,
    routing_strategy="top_k",
    load_balance_loss_weight=0.01,
    z_loss_weight=1e-4,
)

# 6. Train
trainer = Trainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, val_dataset=val_ds,
    epochs=10, batch_size=32, lr=3e-4, optimizer="muon",
)
trainer.train()
trainer.save()

# 7. Load and generate
trainer = Trainer.load("checkpoints/best_model.pt", tokenizer=tokenizer)
print(trainer.chat("Once upon a time"))
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

## 🔧 Under the Hood

### How MoE wraps x-transformers

X-MoE wraps x-transformers. `MoETransformerWrapper` takes your existing `TransformerWrapper`, walks its layers, and replaces FFN modules with `MoEFFN` instances based on your placement config (every Nth layer, or specific indices). All expert parameters are batched into single tensors for vectorized forward passes.

The routing strategies differ in who initiates selection:

- **top_k** - each token chooses its top-k experts. Classic GShard/Switch-style routing.
- **expert_choice** - each expert chooses its top-k tokens. Better load balancing, inspired by Expert Choice Routing.

Both strategies support auxiliary losses (load balance + z-loss) that feed into the training loss automatically via `collect_moe_aux_loss()`.

### DS4 Attention

HCA and CSA implement the dual-state attention from DeepSeek-V4. HCA compresses the KV cache with learned soft-merging and sliding windows. CSA adds overlapped block compression + top-K block retrieval via a learned indexer. They compose into a `HybridAttentionBlock` so you can mix and match within the same model.

> **Note:** This is an experimental addition and will likely be removed if x-transformers implements it natively.

---

## 📄 Citation

```bibtex
@misc{deepseekai2026deepseekv4,
      title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
      author={DeepSeek-AI},
      year={2026},
}
```

<p align="center">
  MIT License &copy; <a href="https://github.com/hvoss-techfak">hvoss-techfak</a>
</p>