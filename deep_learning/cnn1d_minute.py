"""
One-Dimensional Convolutional Neural Network for High-Resolution (1-Minute) Actigraphy.

Architectural details:
- Input shape: (Batch, Timesteps=1440, Channels=1) or (Batch, 1, 1440)
- Multi-scale temporal feature extraction over 1,440 continuous minute intervals:
  * Stage 1: Conv1D (k=15, s=2) + BN + ReLU + MaxPool(4) -> receptive field ~30 mins, T: 1440 -> 180
  * Stage 2: Conv1D (k=7, s=1) + BN + ReLU + MaxPool(3)  -> receptive field ~2 hours, T: 180 -> 60
  * Stage 3: Conv1D (k=5, s=1) + BN + ReLU + AdaptiveAvgPool(1) -> 64-dim latent temporal vector
- Latent feature extraction hook (dim=64) for multimodal fusion compatibility
- Binary classification head with dropout regularization
"""

from typing import Tuple
import torch
import torch.nn as nn


class ActigraphyCNN1DMinute(nn.Module):
    """
    1D-CNN tailored for raw 1-minute actigraphy series (T=1440 steps per 24-hour frame).
    """
    def __init__(
        self,
        in_channels: int = 1,
        num_filters: Tuple[int, int, int] = (32, 64, 64),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.latent_dim = num_filters[2]  # 64

        # Stage 1: Local burst filtering (k=15 covers 15 mins)
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters[0], kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(dropout * 0.5),
        )

        # Stage 2: Multi-hour rhythm extraction (k=7 covers ~1 hour of aggregated state)
        self.block2 = nn.Sequential(
            nn.Conv1d(num_filters[0], num_filters[1], kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(num_filters[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(dropout * 0.5),
        )

        # Stage 3: Global aggregation
        self.block3 = nn.Sequential(
            nn.Conv1d(num_filters[1], num_filters[2], kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(num_filters[2]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 64-dimensional latent temporal representation.
        Parameters:
            x: Tensor of shape (Batch, 1440, 1) or (Batch, 1, 1440)
        Returns:
            latent: Tensor of shape (Batch, 64)
        """
        if x.ndim == 3 and x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        h1 = self.block1(x)
        h2 = self.block2(h1)
        h3 = self.block3(h2)
        return h3.squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized logits."""
        latent = self.extract_features(x)
        logits = self.classifier(latent).squeeze(-1)
        return logits
