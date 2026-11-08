"""Reference optimizer for Algorithm 1 of arXiv:2610.37712v1.

Reconstruction written for a new run. It is not the source of
``artifacts/results_csv``.
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer

GATE_MODES = ("shrink", "mean_preserving")


def reliability_gate(
    sign_hat: torch.Tensor,
    gamma: float,
    r_min: float,
    mode: str,
) -> torch.Tensor:
    """Coordinate gate from Equation (5), then (6) or (7).

    gamma = 0 returns ones, including where the sign EMA is zero, so the
    power is never evaluated at 0^0. The mean in Equation (7) is the mean
    over this tensor, which the caller passes as one parameter tensor.
    Mean 1 does not preserve the squared Euclidean norm of the update.
    """
    if mode not in GATE_MODES:
        raise ValueError(f"Unknown gate mode: {mode}")
    if gamma < 0.0:
        raise ValueError(f"Invalid gamma: {gamma}")
    if r_min < 0.0:
        raise ValueError(f"Invalid r_min: {r_min}")
    if gamma == 0.0:
        return torch.ones_like(sign_hat)
    reliability = sign_hat.abs().pow(gamma)
    if r_min > 0.0:
        reliability = torch.clamp(reliability, min=r_min)
    if mode == "shrink":
        return torch.clamp(reliability, max=1.0)
    mean = reliability.mean()
    # r_min > 0 makes the mean positive. The where-guard covers r_min = 0
    # and an all-zero tensor, which Equation (7) does not define.
    return torch.where(mean > 0, reliability / mean.clamp_min(torch.finfo(reliability.dtype).tiny), torch.ones_like(reliability))


class MGOSC(Optimizer):
    """AdamW-style moments plus a sign-EMA consistency gate.

    Decoupled weight decay matches torch.optim.AdamW: the decay multiplies
    the parameter and is not gated. gamma = 0 leaves that AdamW step unchanged.
    Coupled decay is available only as an explicit alternative.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        beta_s: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.5,
        r_min: float = 0.2,
        gate_mode: str = "mean_preserving",
        bias_correction_s: bool = False,
        decoupled_weight_decay: bool = True,
        collect_stats: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if not (0.0 <= beta_s < 1.0):
            raise ValueError(f"Invalid beta_s: {beta_s}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if gate_mode not in GATE_MODES:
            raise ValueError(f"Unknown gate mode: {gate_mode}")
        defaults = dict(
            lr=lr,
            betas=betas,
            beta_s=beta_s,
            eps=eps,
            weight_decay=weight_decay,
            gamma=float(gamma),
            r_min=float(r_min),
            gate_mode=gate_mode,
            bias_correction_s=bool(bias_correction_s),
            decoupled_weight_decay=bool(decoupled_weight_decay),
            collect_stats=bool(collect_stats),
        )
        super().__init__(params, defaults)
        self._reset_stat_buffers()

    def _reset_stat_buffers(self) -> None:
        self._gate_sum = None
        self._gate_count = None
        self._active_count = None
        self._update_sq = None
        self._grad_sq = None
        self._steps = None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta_s = float(group["beta_s"])
            lr = float(group["lr"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            gamma = float(group["gamma"])
            r_min = float(group["r_min"])
            mode = group["gate_mode"]
            correct_sign = bool(group["bias_correction_s"])
            decoupled = bool(group["decoupled_weight_decay"])
            collect = bool(group["collect_stats"])

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("MGOSC does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["exp_avg_sign"] = torch.zeros_like(param)

                if weight_decay != 0.0:
                    if decoupled:
                        param.mul_(1.0 - lr * weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg_sign = state["exp_avg_sign"]
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                exp_avg_sign.mul_(beta_s).add_(grad.sign(), alpha=1.0 - beta_s)

                state["step"] += 1
                step = int(state["step"])
                bias1 = 1.0 - beta1**step
                bias2 = 1.0 - beta2**step
                step_size = lr / bias1
                denom = exp_avg_sq.sqrt().div_(bias2**0.5).add_(eps)
                if correct_sign:
                    sign_hat = exp_avg_sign / (1.0 - beta_s**step)
                else:
                    sign_hat = exp_avg_sign

                gate = reliability_gate(sign_hat, gamma, r_min, mode)
                update = exp_avg / denom
                if gamma != 0.0:
                    update = update * gate
                param.add_(update, alpha=-step_size)

                if collect:
                    self._accumulate(gate, update, grad)

        return loss

    def _accumulate(self, gate: torch.Tensor, update: torch.Tensor, grad: torch.Tensor) -> None:
        device = gate.device
        if self._gate_sum is None or self._gate_sum.device != device:
            self._gate_sum = torch.zeros((), device=device)
            self._gate_count = torch.zeros((), device=device)
            self._active_count = torch.zeros((), device=device)
            self._update_sq = torch.zeros((), device=device)
            self._grad_sq = torch.zeros((), device=device)
            self._steps = torch.zeros((), device=device)
        self._gate_sum.add_(gate.sum())
        self._gate_count.add_(gate.numel())
        r_min = float(self.param_groups[0]["r_min"])
        self._active_count.add_((gate > r_min).sum())
        self._update_sq.add_(update.square().sum())
        self._grad_sq.add_(grad.detach().square().sum())
        self._steps.add_(1)

    def consume_stats(self) -> dict[str, float] | None:
        """One host sync for the stats gathered since the previous consume."""
        if self._gate_count is None or int(self._gate_count.item()) == 0:
            self._reset_stat_buffers()
            return None
        count = float(self._gate_count.item())
        steps = max(float(self._steps.item()), 1.0)
        stats = {
            "gate_mean": float(self._gate_sum.item()) / count,
            "gate_active_ratio": float(self._active_count.item()) / count,
            "update_rms": float(self._update_sq.item() / steps) ** 0.5,
            "grad_rms": float(self._grad_sq.item() / steps) ** 0.5,
            "steps": steps,
        }
        self._reset_stat_buffers()
        return stats
