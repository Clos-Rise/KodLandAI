import torch
from torch.optim import Optimizer

class Kod(Optimizer):
    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if 'momentum' not in state:
                    state['momentum'] = torch.zeros_like(p.data)

                momentum = state['momentum']
                momentum.mul_(beta).add_(grad, alpha=1 - beta)
                state['momentum'] = momentum
                p.data.add_(momentum, alpha=-lr)

        return loss
