import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTower(nn.Module):
    def __init__(self, input_dim=300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=1)  # normalize for cosine similarity


class DocumentTower(nn.Module):
    def __init__(self, input_dim=300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=1)  # normalize for cosine similarity


class RNNTower(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        rnn_class = nn.GRU
        self.rnn = rnn_class(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Linear(128, 64)
        )

    def forward(self, x, lengths):
        # x: (batch, seq_len)
        x = self.embedding(x)

        # pad sequence so that all sequences in the batch have the same length
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )

        # pass through RNN
        _, h_n = self.rnn(packed)

        # h_n (final hidden state): (num_directions, batch, hidden_size)
        h_n = torch.cat(
            [h_n[-2], h_n[-1]], dim=1
        )  # need to combine h_n from both directions

        out = self.fc(h_n)
        return F.normalize(out, p=2, dim=1)  # normalize for cosine similarity
