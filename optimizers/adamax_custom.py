import torch
from torch.optim.optimizer import Optimizer

class AdaMaxCustom(Optimizer):
    """
    Implementation of the AdaMax optimizer, 
    an L-infinity variant of Adam (Kingma & Ba, 2015).
    """
    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdaMaxCustom, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaMaxCustom does not support sparse gradients')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_inf'] = torch.zeros_like(p.data)

                exp_avg, exp_inf = state['exp_avg'], state['exp_inf']
                beta1, beta2 = group['betas']

                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_inf = torch.max(beta2 * exp_inf, grad.abs() + group['eps'])
                state['exp_inf'] = exp_inf

                bias_correction = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                p.data.addcdiv_(exp_avg, exp_inf, value=-step_size)

        return loss
