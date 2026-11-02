"""
Long Short-Term Memory (LSTM) Architecture for Actigraphy Classification.

Processes chronological activity sequences:
- Input shape: (Batch, Timesteps, Channels=1)
- Multi-layer LSTM with configurable hidden dimension and dropout
- Temporal reduction via last hidden state / mean pooling
- Classification head returning unnormalized logits
- Latent feature extraction hook for feature-level fusion
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


class ActigraphyLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        pooling: str = "mean",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent representation vector before the final classification layer.
        Parameters:
            x: Tensor of shape (Batch, Timesteps, 1)
        Returns:
            latent_features: Tensor of shape (Batch, hidden_dim)
        """
        out, (hn, _) = self.lstm(x)
        if self.pooling == "last":
            # Last layer hidden state: hn[-1] has shape (Batch, hidden_dim)
            latent = hn[-1]
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


def count_trainable_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
