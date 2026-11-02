"""
Gated Recurrent Unit (GRU) Architecture for Actigraphy Classification.

Processes chronological activity sequences with a comparable parameter budget:
- Input shape: (Batch, Timesteps, Channels=1)
- Multi-layer GRU with configurable hidden dimension and dropout
- Feature extraction hook for latent representations (Phase 5 fusion)
- Classification head returning unnormalized logits
"""

import torch
import torch.nn as nn
from deep_learning.lstm import count_trainable_parameters


class ActigraphyGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        pooling: str = "last",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation vector before the classification layer.
        Parameters:
            x: Tensor of shape (Batch, Timesteps, 1)
        Returns:
            latent_features: Tensor of shape (Batch, hidden_dim)
        """
        out, hn = self.gru(x)
        if self.pooling == "last":
            latent = hn[-1] # shape (Batch, hidden_dim)
        elif self.pooling == "mean":
            latent = torch.mean(out, dim=1)
        else:
            latent = hn[-1]
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning binary classification logits.
        Parameters:
            x: Tensor of shape (Batch, Timesteps, 1)
        Returns:
            logits: Tensor of shape (Batch,)
        """
        latent = self.extract_features(x)
        dropped = self.dropout(latent)
        logits = self.classifier(dropped).squeeze(-1)
        return logits
