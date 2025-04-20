## higgs.py
import torch
from torch.optim import Optimizer
from typing import Iterable


class HiggsOptimizer(Optimizer):
    """
    Реализация HIGGS для трансформерных моделей любого типа. Подходит новичкам , легкий в использовании.
    """

    def __init__(self, params: Iterable, lr: float = 1e-4, beta: float = 0.9, eps: float = 1e-8):
        defaults = dict(lr=lr, beta=beta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Выполняет один шаг"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                #создаем состояние
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p.data)
                    state['grad_norm'] = torch.zeros_like(p.data)

                #вычисления от HIGGS
                state['step'] += 1
                beta = group['beta']
                state['momentum'].mul_(beta).add_(grad, alpha=1 - beta)
                grad_norm = torch.norm(grad)
                param_norm = torch.norm(p.data)
                scale_factor = (param_norm + group['eps']) / (grad_norm + group['eps'])

                #обновлем параметры
                p.data.add_(
                    state['momentum'] * -group['lr'] * scale_factor
                )

        return loss


if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch.nn as nn
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    optimizer = HiggsOptimizer(model.parameters(), lr=2e-5, beta=0.95)
    inputs = tokenizer("Hello, Man!", return_tensors="pt")
    labels = torch.tensor([1])
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
