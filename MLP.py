import torch
import torch.nn as nn
from keras.src.optimizers import AdamW


class MLP(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


def train_mlp(model, X, y, optim_name='adamw', epochs=100, lr=0.001):
    model = model.train()
    cn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), optim_name, lr=lr).optimizer

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = cn(outputs, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Эпоха {epoch + 1}/{epochs} | Потери: {loss.item():.4f}')


if __name__ == "__main__":
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)

    model = MLP()
    train_mlp(model, X, y, optim_name='lion', lr=0.0005)