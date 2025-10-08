import torch
from torch.optim.optimizer import Optimizer

class AdaGradCustom(Optimizer):
    """
    Implementation of AdaGrad optimizer.
    """
    def __init__(self, params, lr=1e-2, eps=1e-10, weight_decay=0.0):
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super(AdaGradCustom, self).__init__(params, defaults)

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
                state = self.state[p]

                if len(state) == 0:
                    state['sum'] = torch.zeros_like(p.data)

                sum_sq = state['sum']
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                sum_sq.addcmul_(grad, grad, value=1.0)
                std = sum_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(grad, std, value=-group['lr'])

        return loss
