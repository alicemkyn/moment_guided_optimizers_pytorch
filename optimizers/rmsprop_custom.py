import torch
from torch.optim.optimizer import Optimizer

class RMSPropCustom(Optimizer):
    """
    RMSProp optimizer implementation with exponential moving average
    of squared gradients.
    """
    def __init__(self, params, lr=1e-3, alpha=0.99, eps=1e-8, weight_decay=0.0, momentum=0.0):
        defaults = dict(lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum)
        super(RMSPropCustom, self).__init__(params, defaults)

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
                    state['square_avg'] = torch.zeros_like(p.data)
                    state['momentum_buffer'] = torch.zeros_like(p.data)

                square_avg = state['square_avg']
                alpha = group['alpha']
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
                avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    p.data.add_(buf, alpha=-group['lr'])
                else:
                    p.data.addcdiv_(grad, avg, value=-group['lr'])

        return loss
