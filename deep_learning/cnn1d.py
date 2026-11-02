"""
One-Dimensional Convolutional Neural Network (1D-CNN) for Temporal Actigraphy.

Extracts localized temporal motifs and behavioral patterns:
- Flexible input shape: accepts both (Batch, Timesteps, 1) and (Batch, 1, Timesteps)
- Multi-stage 1D convolution with Batch Normalization and ReLU
- Global Adaptive Average Pooling for temporal invariance
- Latent feature extraction hook (dim=64) for feature-level fusion
- Classification head returning unnormalized logits
"""

from typing import Tuple
import torch
import torch.nn as nn
from deep_learning.lstm import count_trainable_parameters


class ActigraphyCNN1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_filters: Tuple[int, int] = (32, 64),
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_filters = num_filters

        # Block 1: Conv -> BN -> ReLU -> MaxPool
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters[0], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout * 0.5),
        )

        # Block 2: Conv -> BN -> ReLU -> AdaptiveAvgPool
        self.block2 = nn.Sequential(
            nn.Conv1d(num_filters[0], num_filters[1], kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_filters[1]),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), # Flattens any sequence length T to (Batch, num_filters[1], 1)
        )

        self.latent_dim = num_filters[1] # 64
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.latent_dim, 1)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract latent 64-dimensional temporal representation vector.
        Parameters:
            x: Tensor of shape (Batch, Timesteps, 1) or (Batch, 1, Timesteps)
        Returns:
            latent: Tensor of shape (Batch, 64)
        """
        # Ensure channels-first shape: (Batch, Channels=1, Timesteps)
        if x.ndim == 3 and x.shape[-1] == 1 and x.shape[1] != 1:
            x = x.transpose(1, 2)
        elif x.ndim == 2:
            x = x.unsqueeze(1)

        h1 = self.block1(x)
        h2 = self.block2(h1) # shape: (Batch, 64, 1)
        latent = h2.squeeze(-1) # shape: (Batch, 64)
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning binary classification logits.
        """
        latent = self.extract_features(x)
        dropped = self.dropout(latent)
        logits = self.classifier(dropped).squeeze(-1)
        return logits
