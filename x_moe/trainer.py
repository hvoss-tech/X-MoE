import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW

from accelerate import Accelerator

from x_transformers import TransformerWrapper, Decoder

from x_moe.wrapper import MoETransformerWrapper
from x_moe.optimizer import MuonWithAdamW, configure_muon_optimizer
from x_moe.perf import (
    DataPrefetcher,
    ThroughputLogger,
    get_linear_warmup_cosine_scheduler,
)


def build_model_from_config(
    model_config: dict, vocab_size: int, max_seq_len: int
) -> MoETransformerWrapper:
    decoder_kwargs = dict(
        dim=model_config.get("dim", 256),
        depth=model_config.get("depth", 12),
        heads=model_config.get("heads", 8),
        ff_glu=not model_config.get("no_ff_glu", False),
        ff_mult=model_config.get("ff_mult", 4),
        ff_dropout=model_config.get("ff_dropout", 0.1),
        attn_dropout=model_config.get("attn_dropout", 0.1),
        layer_dropout=model_config.get("layer_dropout", 0.0),
        rotary_pos_emb=not model_config.get("no_rotary_pos_emb", False),
        ff_no_bias=not model_config.get("ff_bias", False),
    )
    if model_config.get("flash_attention", True):
        decoder_kwargs["attn_flash"] = True

    decoder = Decoder(**decoder_kwargs)

    transformer = TransformerWrapper(
        num_tokens=vocab_size,
        max_seq_len=max_seq_len,
        attn_layers=decoder,
        emb_dropout=model_config.get("emb_dropout", 0.1),
        tie_embedding=True,
        use_abs_pos_emb=model_config.get("no_rotary_pos_emb", False),
    )

    ds4_attention = None
    ds4_config = {}
    if model_config.get("use_hca", False) or model_config.get("use_csa", False):
        from x_moe.attention import HybridAttentionBlock

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
            ds4_config.update(
                {
                    "use_hca": True,
                    "hca_kv_dim": model_config.get("hca_kv_dim", 128),
                    "hca_num_heads": model_config.get("hca_num_heads", 8),
                    "hca_compression_rate": model_config.get("hca_compression_rate", 8),
                    "hca_num_groups": model_config.get("hca_num_groups", 1),
                    "hca_window_size": model_config.get("hca_window_size", 32),
                    "hca_use_sink": model_config.get("hca_use_sink", True),
                    "hca_use_rope": model_config.get("hca_use_rope", True),
                    "hca_rope_dim": model_config.get("hca_rope_dim", 64),
                }
            )
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
            ds4_config.update(
                {
                    "use_csa": True,
                    "csa_kv_dim": model_config.get("csa_kv_dim", 128),
                    "csa_num_heads": model_config.get("csa_num_heads", 8),
                    "csa_compression_rate": model_config.get("csa_compression_rate", 4),
                    "csa_top_k_blocks": model_config.get("csa_top_k_blocks", 32),
                    "csa_num_groups": model_config.get("csa_num_groups", 1),
                    "csa_window_size": model_config.get("csa_window_size", 32),
                    "csa_use_sink": model_config.get("csa_use_sink", True),
                    "csa_use_rope": model_config.get("csa_use_rope", True),
                    "csa_rope_dim": model_config.get("csa_rope_dim", 64),
                }
            )
        ds4_attention = HybridAttentionBlock(
            dim=model_config.get("dim", 256), hca_config=hca_cfg, csa_config=csa_cfg
        )

    model = MoETransformerWrapper(
        transformer=transformer,
        num_experts=model_config.get("num_experts", 32),
        expert_top_k=model_config.get("expert_top_k", 2),
        capacity_factor=model_config.get("capacity_factor", 1.25),
        routing_strategy=model_config.get("routing_strategy", "top_k"),
        load_balance_loss_weight=model_config.get("load_balance_loss_weight", 0.01),
        z_loss_weight=model_config.get("z_loss_weight", 1e-4),
        moe_every_n_layers=model_config.get("moe_every_n_layers", 1),
        moe_layers=model_config.get("moe_layers", None),
        glu=not model_config.get("no_ff_glu", False),
        mult=model_config.get("ff_mult", 4),
        dropout=model_config.get("ff_dropout", 0.1),
        no_bias=not model_config.get("ff_bias", False),
        zero_init_output=True,
        ds4_attention=ds4_attention,
        batched_experts=model_config.get("batched_experts", False),
        max_batch_size=model_config.get("max_batch_size", 1),
        flash_attention=model_config.get("flash_attention", True),
    )

    return model


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    gradient_accumulate: int = 1
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 500
    val_interval: int = 1
    log_interval: int = 50
    aux_loss_every: int = 4
    optimizer: str = "muon"
    muon_lr: float = 1e-3
    muon_momentum: float = 0.9
    muon_rms_factor: float = 1.0
    adamw_for_non_muon_lr: float = 3e-4
    mixed_precision: str = "bf16"
    compile: bool = True
    gradient_checkpointing: bool = False
    prefetch_data: bool = True
    num_workers: int = 8
    seed: int = 42
    save_dir: str = "checkpoints"
    max_seq_len: int = 256
    pad_to_max: bool = True


class Trainer:
    def __init__(
        self,
        model: MoETransformerWrapper,
        tokenizer,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        collate_fn: Optional[callable] = None,
        config: Optional[TrainConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = TrainConfig(**kwargs)
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        if tokenizer.decoder is None:
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder

            tokenizer.decoder = ByteLevelDecoder()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        pad_id = (
            tokenizer.token_to_id("<pad>") if hasattr(tokenizer, "token_to_id") else 0
        )
        self.pad_id = pad_id
        self.eos_id = (
            tokenizer.token_to_id("<eos>")
            if hasattr(tokenizer, "token_to_id")
            else None
        )

        if collate_fn is not None:
            self._collate_fn = collate_fn
        else:
            self._collate_fn = _make_collate_fn(
                pad_id,
                pad_to_max=self.config.pad_to_max,
                max_seq_len=self.config.max_seq_len,
            )

        self.accelerator = None
        self.optimizer = None
        self.scheduler = None
        self.muon_scheduler = None
        self.adamw_scheduler = None
        self.best_val_ppl = float("inf")
        self._trained_epochs = 0

    def _unwrapped_model(self):
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(self.model)
        return self.model

    def _setup(self):
        cfg = self.config

        self.accelerator = Accelerator(
            mixed_precision=(
                cfg.mixed_precision if cfg.mixed_precision != "no" else None
            ),
            gradient_accumulation_steps=cfg.gradient_accumulate,
        )

        self.accelerator.print(
            f"Accelerator: device={self.accelerator.device}, "
            f"mixed_precision={self.accelerator.mixed_precision}, "
            f"num_processes={self.accelerator.num_processes}"
        )

        self.accelerator.wait_for_everyone()

        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        if self.accelerator.num_processes > 1:
            from accelerate.utils import set_seed

            set_seed(cfg.seed)

        save_dir = Path(cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if cfg.gradient_checkpointing:
            count = self.model.enable_gradient_checkpointing()
            self.accelerator.print(
                f"Enabled gradient checkpointing on {count} MoE layers"
            )

        self.accelerator.print(f"Model parameters: {self.model.num_params:,}")

        if cfg.compile:
            compile_dynamic = not cfg.pad_to_max
            self.accelerator.print(
                f"Compiling model with torch.compile (dynamic={compile_dynamic})..."
            )
            self.model = torch.compile(self.model, dynamic=compile_dynamic)

        self._setup_optimizer()

        persistent = cfg.num_workers > 0
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=persistent,
        )

        self.val_loader = None
        if self.val_dataset is not None:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                collate_fn=self._collate_fn,
                num_workers=cfg.num_workers,
                pin_memory=True,
                persistent_workers=persistent,
            )

        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        )

        self._setup_scheduler()

    def _setup_optimizer(self):
        cfg = self.config

        if cfg.optimizer == "muon":
            muon_opt, adamw_opt = configure_muon_optimizer(
                self.model,
                lr=cfg.muon_lr,
                momentum=cfg.muon_momentum,
                weight_decay=cfg.weight_decay,
                adamw_lr=cfg.adamw_for_non_muon_lr,
                adamw_weight_decay=cfg.weight_decay,
                rms_rescale_factor=cfg.muon_rms_factor,
            )
            self.optimizer = MuonWithAdamW(muon_opt, adamw_opt)
            self._muon_opt = muon_opt
            self._adamw_opt = adamw_opt
            self.accelerator.print(
                f"Using Muon optimizer (muon_lr={cfg.muon_lr}, "
                f"adamw_lr={cfg.adamw_for_non_muon_lr})"
            )
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=(0.9, 0.95),
            )
            self._muon_opt = None
            self._adamw_opt = None
            self.accelerator.print(f"Using AdamW optimizer (lr={cfg.lr})")

    def _setup_scheduler(self):
        cfg = self.config
        total_steps = cfg.epochs * (len(self.train_loader) // cfg.gradient_accumulate)
        warmup_steps = cfg.warmup_steps

        if cfg.optimizer == "muon":
            self.muon_scheduler = get_linear_warmup_cosine_scheduler(
                self._muon_opt, warmup_steps, total_steps, eta_min=0.1
            )
            self.adamw_scheduler = get_linear_warmup_cosine_scheduler(
                self._adamw_opt, warmup_steps, total_steps, eta_min=0.1
            )
            self.scheduler = None
        else:
            self.scheduler = get_linear_warmup_cosine_scheduler(
                self.optimizer, warmup_steps, total_steps, eta_min=0.1
            )
            self.muon_scheduler = None
            self.adamw_scheduler = None

        self.accelerator.print(f"Scheduler: warmup={warmup_steps}, total={total_steps}")

    def train(self, validation_string=""):
        if self.train_dataset is None:
            raise ValueError(
                "train_dataset is required for training. Provide it in the Trainer constructor."
            )

        if self.accelerator is None:
            self._setup()

        cfg = self.config
        accelerator = self.accelerator
        pad_id = self.pad_id

        throughput_logger = ThroughputLogger(log_interval=cfg.log_interval)
        global_step = 0

        accelerator.print("Starting training...")
        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            epoch_tokens = 0
            epoch_start = time.time()
            throughput_logger.start_epoch()

            use_prefetcher = cfg.prefetch_data and accelerator.device.type == "cuda"

            if use_prefetcher:
                half = cfg.mixed_precision in ("fp16", "bf16")
                data_iter = DataPrefetcher(
                    self.train_loader, accelerator.device, half=half
                )
            else:
                data_iter = self.train_loader

            for batch_idx, batch in enumerate(data_iter):
                should_compute_aux = (cfg.aux_loss_every <= 1) or (
                    (batch_idx + 1) % cfg.aux_loss_every == 0
                )
                self.model.set_aux_loss_compute(should_compute_aux)

                with accelerator.autocast():
                    loss = self.model(batch)
                    if isinstance(loss, tuple):
                        loss = loss[0]

                    moe_aux = self.model.moe_aux_loss
                    self.model.reset_moe_aux_loss()
                    self.model.set_aux_loss_compute(True)

                    total_loss = loss + moe_aux

                accelerator.backward(total_loss)

                if (batch_idx + 1) % cfg.gradient_accumulate == 0:
                    accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    if cfg.optimizer == "muon":
                        self.muon_scheduler.step()
                        self.adamw_scheduler.step()
                    else:
                        self.scheduler.step()

                    self.optimizer.zero_grad()
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

                if (batch_idx + 1) % cfg.log_interval == 0:
                    avg_loss = epoch_loss / epoch_tokens
                    ppl = math.exp(min(avg_loss, 20))
                    lr = self.optimizer.param_groups[0]["lr"]
                    accelerator.print(
                        f"Epoch {epoch}/{cfg.epochs} | "
                        f"Step {batch_idx + 1}/{len(self.train_loader)} | "
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
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Train PPL: {train_ppl:.2f} | Time: {epoch_time:.1f}s"
            )
            if perf_summary:
                accelerator.print(
                    f"Throughput: {perf_summary['tokens_per_sec']:.0f} tokens/s | "
                    f"{perf_summary['total_tokens']} tokens total"
                )

            val_ppl = None
            if self.val_loader is not None and epoch % cfg.val_interval == 0:
                val_ppl = self._validate()
                if val_ppl < self.best_val_ppl and accelerator.is_main_process:
                    self.best_val_ppl = val_ppl
                    self._save_checkpoint("best_model.pt", epoch, val_ppl, train_ppl)
                    accelerator.print(f"New best model saved (PPL: {val_ppl:.2f})")

            if accelerator.is_main_process:
                self._save_checkpoint(
                    f"checkpoint_epoch_{epoch}.pt", epoch, val_ppl, train_ppl
                )

            self._trained_epochs = epoch

            if validation_string != "":
                print(validation_string + self.chat(validation_string))

        accelerator.print(
            f"\nTraining complete! Best validation PPL: {self.best_val_ppl:.2f}"
        )

    def _validate(self):
        accelerator = self.accelerator
        pad_id = self.pad_id

        self.model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in self.val_loader:
                with accelerator.autocast():
                    v_loss = self.model(batch)
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
        self.model.train()
        return val_ppl

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        val_ppl: Optional[float],
        train_ppl: Optional[float],
    ):
        unwrapped = self._unwrapped_model()
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = self._make_checkpoint(unwrapped, epoch, val_ppl, train_ppl)
        torch.save(checkpoint, save_dir / filename)

    def _make_checkpoint(
        self,
        model,
        epoch: int,
        val_ppl: Optional[float],
        train_ppl: Optional[float],
    ):
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                self.optimizer.state_dict() if self.optimizer else {}
            ),
            "val_ppl": val_ppl,
            "train_ppl": train_ppl,
            "model_config": model.model_config,
            "vocab_size": (
                self.tokenizer.get_vocab_size()
                if hasattr(self.tokenizer, "get_vocab_size")
                else 0
            ),
            "max_seq_len": self.config.max_seq_len,
            "train_config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "gradient_accumulate": self.config.gradient_accumulate,
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "warmup_steps": self.config.warmup_steps,
                "optimizer": self.config.optimizer,
                "mixed_precision": self.config.mixed_precision,
            },
        }

    def save(self, path: Optional[str] = None):
        unwrapped = self._unwrapped_model()

        checkpoint = self._make_checkpoint(
            unwrapped,
            self._trained_epochs,
            self.best_val_ppl if self.best_val_ppl != float("inf") else None,
            None,
        )

        if path:
            save_path = Path(path)
        else:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "best_model.pt"

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")

    @classmethod
    def load(
        cls,
        checkpoint_path: str,
        tokenizer,
        train_dataset: Optional[torch.utils.data.Dataset] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        config: Optional[TrainConfig] = None,
        device: Optional[str] = None,
        **kwargs,
    ) -> "Trainer":
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model_config = checkpoint["model_config"]
        vocab_size = checkpoint.get(
            "vocab_size",
            tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else 0,
        )
        max_seq_len = checkpoint.get(
            "max_seq_len", model_config.get("max_seq_len", 256)
        )

        print("Building model from config...")
        model = build_model_from_config(model_config, vocab_size, max_seq_len)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        print(f"Model loaded. Parameters: {model.num_params:,}")
        if "val_ppl" in checkpoint and checkpoint["val_ppl"] is not None:
            print(f"Checkpoint val PPL: {checkpoint['val_ppl']:.2f}")

        if config is None:
            config = TrainConfig(max_seq_len=max_seq_len, **kwargs)

        trainer = cls(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
        )
        trainer._trained_epochs = checkpoint.get("epoch", 0)
        trainer._device = device
        trainer._checkpoint = checkpoint

        return trainer

    @torch.no_grad()
    def chat(
        self,
        prompt: str,
        seq_len: int = 256,
        temperature: float = 0.8,
        filter_logits_fn: str = "top_k",
        filter_kwargs: Optional[dict] = None,
        eos_token: Optional[int] = None,
    ) -> str:
        model = self._unwrapped_model()
        device = next(model.parameters()).device
        model.eval()

        eos_id = eos_token if eos_token is not None else self.eos_id

        if prompt:
            prompt_tokens = self.tokenizer.encode(prompt).ids
            prompt_tensor = torch.tensor(
                [prompt_tokens], dtype=torch.long, device=device
            )
        else:
            prompt_tensor = torch.zeros(1, 1, dtype=torch.long, device=device)
            if eos_id is not None:
                prompt_tensor[:, 0] = eos_id

        if filter_kwargs is None:
            filter_kwargs = {"k": 50} if filter_logits_fn == "top_k" else {}

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            output = model.generate(
                prompt_tensor,
                seq_len=seq_len,
                temperature=temperature,
                filter_logits_fn=filter_logits_fn,
                filter_kwargs=filter_kwargs,
                eos_token=eos_id,
                cache_kv=True,
            )

        tokens = output[0].tolist()
        text = self.tokenizer.decode(tokens)
        text = text.replace("<pad>", "").replace("<eos>", "")
        return text.strip()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        seq_len: int = 256,
        temperature: float = 0.8,
        filter_logits_fn: str = "top_k",
        filter_kwargs: Optional[dict] = None,
        num_samples: int = 1,
    ) -> list:
        results = []
        for _ in range(num_samples):
            results.append(
                self.chat(
                    prompt=prompt,
                    seq_len=seq_len,
                    temperature=temperature,
                    filter_logits_fn=filter_logits_fn,
                    filter_kwargs=filter_kwargs,
                )
            )
        return results


def _make_collate_fn(
    pad_id: int = 0, pad_to_max: bool = False, max_seq_len: int = None
):
    def collate(batch):
        if pad_to_max and max_seq_len is not None:
            padded = []
            for b in batch:
                pad_len = max_seq_len - b.shape[0]
                if pad_len > 0:
                    padded.append(F.pad(b, (0, pad_len), value=pad_id))
                else:
                    padded.append(b[:max_seq_len] if b.shape[0] > max_seq_len else b)
            return torch.stack(padded)
        max_len = max(b.shape[0] for b in batch)
        padded = []
        for b in batch:
            pad_len = max_len - b.shape[0]
            if pad_len > 0:
                padded.append(F.pad(b, (0, pad_len), value=pad_id))
            else:
                padded.append(b)
        return torch.stack(padded)

    return collate
