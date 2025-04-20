import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1,
                 num_layers=3, dropout=0.2, rnn_type='lstm'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()

        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)
        else:  # дефолтная
            self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                            dropout=dropout if num_layers > 1 else 0,
                            batch_first=True)

        #доп. слои
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        if self.rnn_type == 'lstm':
            c0 = torch.zeros_like(h0)
            out, _ = self.rnn(x, (h0, c0))
        else:
            out, _ = self.rnn(x, h0)

        #attn
        attn_weights = torch.softmax(self.attention(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)

        return self.fc(context)