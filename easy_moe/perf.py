import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


class DataPrefetcher:
    def __init__(self, loader: DataLoader, device: torch.device, half: bool = False):
        self.loader = iter(loader)
        self.device = device
        self.half = half
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self._preload()

    def _preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                if isinstance(self.next_batch, torch.Tensor):
                    self.next_batch = self.next_batch.to(self.device, non_blocking=True)
                    if self.half and self.next_batch.is_floating_point():
                        self.next_batch = self.next_batch.half()
                elif isinstance(self.next_batch, (list, tuple)):
                    self.next_batch = type(self.next_batch)(
                        self._move_tensor(t) for t in self.next_batch
                    )
        else:
            if isinstance(self.next_batch, torch.Tensor):
                self.next_batch = self.next_batch.to(self.device)
                if self.half and self.next_batch.is_floating_point():
                    self.next_batch = self.next_batch.half()
            elif isinstance(self.next_batch, (list, tuple)):
                self.next_batch = type(self.next_batch)(
                    self._move_tensor_cpu(t) for t in self.next_batch
                )

    def _move_tensor(self, t):
        if isinstance(t, torch.Tensor):
            t = t.to(self.device, non_blocking=True)
            if self.half and t.is_floating_point():
                t = t.half()
            return t
        return t

    def _move_tensor_cpu(self, t):
        if isinstance(t, torch.Tensor):
            t = t.to(self.device)
            if self.half and t.is_floating_point():
                t = t.half()
            return t
        return t

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        batch = self.next_batch
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        self._preload()
        return batch

    def next(self):
        try:
            return self.__next__()
        except StopIteration:
            return None


class ThroughputLogger:
    def __init__(self, log_interval: int = 50, window: int = 50):
        self.log_interval = log_interval
        self.window = window
        self._step_times = []
        self._tokens_per_step = []
        self._start_time = None
        self._total_tokens = 0
        self._total_steps = 0
        self._epoch_start = None

    def start_epoch(self):
        self._epoch_start = time.time()
        self._total_tokens = 0
        self._total_steps = 0
        self._step_times = []

    def log_step(self, num_tokens: int, step: int, extra_info: Optional[dict] = None):
        now = time.time()
        if self._start_time is not None:
            step_time = now - self._start_time
            self._step_times.append(step_time)
        self._start_time = now
        self._total_tokens += num_tokens
        self._total_steps += 1
        self._tokens_per_step.append(num_tokens)

        if (step + 1) % self.log_interval == 0 and len(self._step_times) > 0:
            recent_times = self._step_times[-self.window:]
            avg_step_time = sum(recent_times) / len(recent_times)
            recent_tokens = self._tokens_per_step[-self.window:]
            avg_tokens = sum(recent_tokens) / len(recent_tokens)
            tokens_per_sec = avg_tokens / avg_step_time if avg_step_time > 0 else 0

            info = {
                "tokens_per_sec": f"{tokens_per_sec:.0f}",
                "step_time_ms": f"{avg_step_time * 1000:.1f}",
                "avg_tokens_per_step": f"{avg_tokens:.0f}",
            }
            if extra_info:
                info.update(extra_info)

            parts = [f"{k}={v}" for k, v in info.items()]
            print(f"  [Perf] " + " | ".join(parts))

    def epoch_summary(self):
        if self._epoch_start is None:
            return {}
        epoch_time = time.time() - self._epoch_start
        tokens_per_sec = self._total_tokens / epoch_time if epoch_time > 0 else 0
        return {
            "epoch_time": epoch_time,
            "total_tokens": self._total_tokens,
            "tokens_per_sec": tokens_per_sec,
            "total_steps": self._total_steps,
        }


class CUDAGraphCapturer:
    def __init__(self, model, optimizer, gradient_accumulate=1):
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulate = gradient_accumulate
        self.graph = None
        self.static_input = None
        self.static_loss = None
        self.captured = False

    def capture(self, sample_input):
        if not torch.cuda.is_available():
            return

        device = next(self.model.parameters()).device
        self.graph = torch.cuda.CUDAGraph()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        for _ in range(3):
            output = self.model(sample_input)
            if isinstance(output, tuple):
                output = output[0]
            loss = output.sum() if output.dim() > 0 else output
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            del loss, output

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        with torch.cuda.graph(self.graph):
            self.static_input = sample_input
            output = self.model(self.static_input)
            if isinstance(output, tuple):
                output = output[0]
            self.static_loss = output.sum() if output.dim() > 0 else output
            self.static_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.captured = True

    def replay(self, sample_input):
        if not self.captured:
            return None
        self.static_input.copy_(sample_input)
        self.graph.replay()
        return self.static_loss


def get_linear_warmup_cosine_scheduler(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min: float = 0.0,
):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        if total_steps <= warmup_steps:
            return eta_min
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return eta_min + (1.0 - eta_min) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def get_warmup_cosine_scheduler_for_muon(
    muon_opt,
    adamw_opt,
    warmup_steps: int,
    total_steps: int,
    muon_lr: float,
    adamw_lr: float,
):
    muon_scheduler = get_linear_warmup_cosine_scheduler(
        muon_opt, warmup_steps, total_steps, eta_min=0.1
    )
    adamw_scheduler = get_linear_warmup_cosine_scheduler(
        adamw_opt, warmup_steps, total_steps, eta_min=0.1
    )
    return muon_scheduler, adamw_scheduler