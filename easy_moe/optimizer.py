import torch
from torch import Tensor
from torch.optim import Optimizer
import math

from easy_moe.moe import MoEFFN


class HybridNewtonSchulz:
    def __init__(self):
        self.stage1_coeffs = (3.4445, -4.7750, 2.0315)
        self.stage2_coeffs = (2.0, -1.5, 0.5)
        self.stage1_iters = 8
        self.stage2_iters = 2
        self.total_iters = self.stage1_iters + self.stage2_iters

    def orthogonalize(self, M: Tensor) -> Tensor:
        a1, b1, c1 = self.stage1_coeffs
        a2, b2, c2 = self.stage2_coeffs

        original_dtype = M.dtype
        m, n = M.shape

        M_float = M.float()

        frobenius_norm = M_float.norm().clamp(min=1e-7)
        X = M_float / frobenius_norm

        if m >= n:
            for i in range(self.total_iters):
                if i < self.stage1_iters:
                    a, b, c = a1, b1, c1
                else:
                    a, b, c = a2, b2, c2

                XtX = X.T @ X
                X = a * X + b * (X @ XtX) + c * (X @ (XtX @ XtX))
        else:
            for i in range(self.total_iters):
                if i < self.stage1_iters:
                    a, b, c = a1, b1, c1
                else:
                    a, b, c = a2, b2, c2

                XXt = X @ X.T
                X = a * X + b * (XXt @ X) + c * ((XXt @ XXt) @ X)

        return X.to(original_dtype)


class Muon(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        rms_rescale_factor: float = 1.0,
        newton_schulz: HybridNewtonSchulz | None = None,
        fused: bool = False,
    ):
        if newton_schulz is None:
            newton_schulz = HybridNewtonSchulz()

        self._newton_schulz = newton_schulz
        self._rms_rescale_factor = rms_rescale_factor
        self._nesterov = nesterov
        self._fused = fused

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            group["momentum_buffer"] = []

    @staticmethod
    def _is_muon_param(p: Tensor) -> bool:
        return p.dim() >= 2

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]

            if len(group["momentum_buffer"]) != len(group["params"]):
                group["momentum_buffer"] = [None] * len(group["params"])

            for idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                g = p.grad.data

                if wd > 0:
                    p.data.mul_(1 - lr * wd)

                buf = group["momentum_buffer"][idx]
                if buf is None:
                    buf = torch.zeros_like(p.data)
                    group["momentum_buffer"][idx] = buf

                buf.mul_(momentum).add_(g)

                if self._nesterov:
                    update = g + momentum * buf
                else:
                    update = buf.clone()

                if self._is_muon_param(p):
                    ns = self._newton_schulz
                    update_2d = update.view(update.shape[0], -1)
                    orth_update = ns.orthogonalize(update_2d)
                    orth_update = orth_update.view_as(p)

                    m_dim, n_dim = update_2d.shape
                    rescale = math.sqrt(max(m_dim, n_dim)) * self._rms_rescale_factor
                    orth_update = orth_update * rescale

                    p.data.add_(orth_update, alpha=-lr)
                else:
                    p.data.add_(update, alpha=-lr)

        return loss


def configure_muon_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    adamw_lr: float = 3e-4,
    adamw_weight_decay: float = 0.1,
    rms_rescale_factor: float = 1.0,
    fused: bool = False,
):
    muon_params = []
    adamw_params = []

    for module in model.modules():
        if isinstance(module, MoEFFN):
            for name, p in module.named_parameters():
                if "gate" in name:
                    adamw_params.append(p)
                else:
                    muon_params.append(p)
            continue

        module_type = type(module).__name__
        if module_type in ("RMSNorm", "LayerNorm"):
            adamw_params.extend(module.parameters())
            continue

        if isinstance(module, (torch.nn.Embedding,)):
            adamw_params.extend(module.parameters())
            continue

    seen = set(id(p) for p in adamw_params) | set(id(p) for p in muon_params)

    for name, p in model.named_parameters():
        if id(p) not in seen:
            if p.dim() >= 2:
                muon_params.append(p)
            else:
                adamw_params.append(p)
            seen.add(id(p))

    muon_opt = Muon(
        muon_params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        rms_rescale_factor=rms_rescale_factor,
        fused=fused,
    )

    adamw_kwargs = dict(
        lr=adamw_lr,
        weight_decay=adamw_weight_decay,
    )
    if fused and torch.cuda.is_available():
        try:
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                **adamw_kwargs,
                fused=True,
            )
        except (TypeError, RuntimeError):
            adamw_opt = torch.optim.AdamW(
                adamw_params,
                **adamw_kwargs,
            )
    else:
        adamw_opt = torch.optim.AdamW(
            adamw_params,
            **adamw_kwargs,
        )

    return muon_opt, adamw_opt


class MuonWithAdamW:
    def __init__(self, muon_opt: Muon, adamw_opt: torch.optim.AdamW):
        self.muon_opt = muon_opt
        self.adamw_opt = adamw_opt

    def step(self, closure=None):
        self.muon_opt.step(closure)
        self.adamw_opt.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon_opt.zero_grad(set_to_none=set_to_none)
        self.adamw_opt.zero_grad(set_to_none=set_to_none)

    @property
    def param_groups(self):
        return self.muon_opt.param_groups + self.adamw_opt.param_groups

    def state_dict(self):
        return {
            "muon": self.muon_opt.state_dict(),
            "adamw": self.adamw_opt.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.muon_opt.load_state_dict(state_dict["muon"])
        self.adamw_opt.load_state_dict(state_dict["adamw"])