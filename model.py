import torch.nn as nn
import torch.nn.functional as F

class QueryTower(nn.Module):
    def __init__(self, input_dim=300, output_dim=300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=1) # normalize for cosine similarity


class DocumentTower(nn.Module):
    def __init__(self, input_dim=300, output_dim=300):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return F.normalize(x, p=2, dim=1) # normalize for cosine similarity
