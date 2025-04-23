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
