import torch
import torch.nn as nn

class UAF_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None, compute_gamma=False):
        emb = self.embedding(x)
        if hidden is None:
            h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(x.device)
            hidden = (h0, c0)
        out, hidden = self.lstm(emb, hidden)
        out = self.dropout(out)
        logits = self.fc(out)

        gamma = None
        if compute_gamma:
            # gamma = средняя косинусная близость между последовательными скрытыми состояниями (последние 10 шагов)
            h_states = out[:, -10:, :]
            h_norm = h_states / (h_states.norm(dim=2, keepdim=True) + 1e-8)
            cos_sim = (h_norm[:, 1:, :] * h_norm[:, :-1, :]).sum(dim=2)
            gamma = cos_sim.mean()
        return logits, hidden, gamma
