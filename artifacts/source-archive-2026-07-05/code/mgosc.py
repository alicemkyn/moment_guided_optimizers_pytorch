import math
from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer


class MGOSC(Optimizer):
    """MGO-SC: Moment-Guided Optimizer with Sign-Consistency gating.

    Update:
        s_t = beta_s * s_{t-1} + (1-beta_s) * sign(g_t)
        r_t = clamp(|s_hat|^gamma, min=r_min)   (optionally with sign-EMA bias correction)
        theta <- theta - lr * r_t * m_hat / (sqrt(v_hat) + eps)

    Key switches:
      - gamma=0 recovers Adam-like behavior (r_t == 1)
      - bias_correction_s: debias the sign-EMA (s_hat = s_t / (1 - beta_s^t))
      - warmup_steps: disable gating for the first N steps (r_t==1)

    Stats (optional, controlled by collect_stats=True):
      - gate_mean: mean gate value r_t
      - active_ratio: fraction of elements with |s_hat| >= r_min
      - update_norm: L2 norm of the update (aggregated across parameters)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        beta_s: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.5,
        r_min: float = 0.2,
        bias_correction_s: bool = True,
        warmup_steps: int = 0,
        collect_stats: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if not (0.0 <= beta_s < 1.0):
            raise ValueError(f"Invalid beta_s: {beta_s}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if not (0.0 <= r_min <= 1.0):
            raise ValueError(f"Invalid r_min: {r_min}")
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps: {warmup_steps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta_s=beta_s,
            eps=eps,
            weight_decay=weight_decay,
            gamma=float(gamma),
            r_min=float(r_min),
            bias_correction_s=bool(bias_correction_s),
            warmup_steps=int(warmup_steps),
            collect_stats=bool(collect_stats),
        )
        super().__init__(params, defaults)
        self._last_step_stats: Dict[str, float] = {}

    def get_last_step_stats(self) -> Dict[str, float]:
        """Returns stats from the most recent optimizer.step() call (if enabled)."""
        return dict(self._last_step_stats)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Aggregated step stats (computed only if enabled)
        step_gate_sum = 0.0
        step_active_sum = 0.0
        step_elem_count = 0
        step_update_sq_sum = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            beta_s = group["beta_s"]
            eps = group["eps"]
            wd = group["weight_decay"]
            gamma = float(group["gamma"])
            r_min = float(group["r_min"])
            bcs = bool(group["bias_correction_s"])
            warmup_steps = int(group["warmup_steps"])
            collect_stats = bool(group["collect_stats"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("MGOSC does not support sparse gradients.")

                # Adam-style coupled weight decay (keep consistent with Adam baseline used in experiments)
                if wd != 0.0:
                    g = g.add(p, alpha=wd)

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)  # sign EMA

                state["t"] += 1
                t = int(state["t"])
                m = state["m"]
                v = state["v"]
                s = state["s"]

                # moments
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                # sign EMA
                s.mul_(beta_s).add_(torch.sign(g), alpha=1.0 - beta_s)
                if bcs:
                    s_hat = s / (1.0 - (beta_s ** t))
                else:
                    s_hat = s

                # bias correction (Adam)
                m_hat = m / (1.0 - (b1 ** t))
                v_hat = v / (1.0 - (b2 ** t))

                denom = v_hat.sqrt().add_(eps)

                # gate
                if warmup_steps > 0 and t <= warmup_steps:
                    r = None  # treat as 1
                    reliability = None
                elif gamma == 0.0:
                    r = None
                    reliability = None
                else:
                    reliability = s_hat.abs()
                    r = reliability.pow(gamma)
                    if r_min > 0.0:
                        r = r.clamp_(min=r_min)

                # apply update (and gather stats)
                if r is None:
                    upd = -lr * (m_hat / denom)
                    p.add_(upd)
                    if collect_stats:
                        step_update_sq_sum += float((upd * upd).sum().item())
                        # gate stats are undefined in this branch (treat as 1)
                        step_gate_sum += float(upd.numel())  # pretend gate mean == 1
                        step_active_sum += float(upd.numel())
                        step_elem_count += int(upd.numel())
                else:
                    upd = -lr * (r * (m_hat / denom))
                    p.add_(upd)
                    if collect_stats:
                        step_update_sq_sum += float((upd * upd).sum().item())
                        step_gate_sum += float(r.sum().item())
                        # active: reliability >= r_min (before pow), so it's interpretable
                        if reliability is not None:
                            step_active_sum += float((reliability >= r_min).float().sum().item())
                        else:
                            step_active_sum += float(upd.numel())
                        step_elem_count += int(upd.numel())

        # store last step stats
        if step_elem_count > 0:
            gate_mean = step_gate_sum / step_elem_count
            active_ratio = step_active_sum / step_elem_count
            update_norm = math.sqrt(max(step_update_sq_sum, 0.0))
            self._last_step_stats = {
                "gate_mean": float(gate_mean),
                "active_ratio": float(active_ratio),
                "update_norm": float(update_norm),
            }
        else:
            self._last_step_stats = {}

        return loss
