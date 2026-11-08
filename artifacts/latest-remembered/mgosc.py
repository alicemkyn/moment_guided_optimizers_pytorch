# mgosc.py
import math
import torch
from torch.optim import Optimizer


class MGOSC(Optimizer):
    """
    Moment-Guided Optimizer with Sign-Consistency (MGO-SC).

    Core idea:
      - Maintain Adam-style m, v
      - Maintain sign-EMA s
      - Build reliability gate r = |s_hat|^gamma (optionally bias-corrected)
      - Clamp with r_min (optional)
      - Update: theta <- theta - lr * r * m_hat / (sqrt(v_hat) + eps)

    When gamma = 0, r = 1 everywhere and this reduces to Adam update.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        beta_s=0.9,
        eps=1e-8,
        weight_decay=0.0,
        gamma=1.0,
        r_min=0.0,
        bias_correction_s=False,
    ):
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if not (0.0 <= betas[0] < 1.0 and 0.0 <= betas[1] < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if not (0.0 <= beta_s < 1.0):
            raise ValueError(f"Invalid beta_s: {beta_s}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if gamma < 0.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        if r_min < 0.0:
            raise ValueError(f"Invalid r_min: {r_min}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta_s=beta_s,
            eps=eps,
            weight_decay=weight_decay,
            gamma=float(gamma),
            r_min=float(r_min),
            bias_correction_s=bool(bias_correction_s),
        )
        super().__init__(params, defaults)

        # Step-level stats for mechanism diagnostics (set on each step call)
        self._last_step_stats = None

    def get_last_step_stats(self):
        """Return dict with step-level stats (or None)."""
        return self._last_step_stats

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Reset stats; will be overwritten at end of each param_group loop
        self._last_step_stats = None

        for group in self.param_groups:
            lr = float(group["lr"])
            b1, b2 = group["betas"]
            beta_s = float(group["beta_s"])
            eps = float(group["eps"])
            wd = float(group["weight_decay"])
            gamma = float(group["gamma"])
            r_min = float(group.get("r_min", 0.0))
            bias_correction_s = bool(group.get("bias_correction_s", False))

            # Accumulators across parameters for this optimizer.step() call
            gate_sum = 0.0
            gate_cnt = 0
            gate_active = 0
            update_norm2_sum = 0.0
            grad_norm2_sum = 0.0

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("MGOSC does not support sparse gradients.")

                # Decoupled weight decay (AdamW-style) – more standard in modern setups
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)  # sign-EMA

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]
                s = state["s"]

                # Adam moments
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                # Sign-consistency EMA
                s.mul_(beta_s).add_(torch.sign(g), alpha=1.0 - beta_s)

                # Bias correction (Adam)
                m_hat = m / (1.0 - (b1 ** t))
                v_hat = v / (1.0 - (b2 ** t))

                # Optional bias correction for sign EMA (important early epochs)
                if bias_correction_s:
                    s_hat = s / (1.0 - (beta_s ** t))
                else:
                    s_hat = s

                denom = v_hat.sqrt().add_(eps)

                if gamma == 0.0:
                    # Reduce to Adam update
                    update = lr * (m_hat / denom)
                    p.add_(update, alpha=-1.0)

                    n = int(g.numel())
                    gate_cnt += n
                    gate_sum += float(n) * 1.0
                    gate_active += n
                else:
                    # Gate: r in [0,1] (with clamp)
                    r = s_hat.abs().pow(gamma)
                    if r_min > 0.0:
                        r = torch.clamp(r, min=r_min, max=1.0)

                    update = lr * (r * (m_hat / denom))
                    p.add_(update, alpha=-1.0)

                    gate_cnt += int(r.numel())
                    gate_sum += float(r.sum().item())
                    # active gate count (strictly > r_min)
                    gate_active += int((r > r_min).sum().item())

                # Norm stats for mechanism/stability narrative
                grad_norm2_sum += float((g.detach() * g.detach()).sum().item())
                update_norm2_sum += float((update.detach() * update.detach()).sum().item())

            # Store stats for this step
            self._last_step_stats = dict(
                gate_sum=gate_sum,
                gate_cnt=gate_cnt,
                gate_active=gate_active,
                grad_norm2_sum=grad_norm2_sum,
                update_norm2_sum=update_norm2_sum,
            )

        return loss

