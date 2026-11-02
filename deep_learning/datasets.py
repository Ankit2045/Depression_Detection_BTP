"""
PyTorch Dataset and Data Loading Utilities for Actigraphy Sequences.

Supports both:
- Hourly downsampled sequences (shape: T=24, C=1)
- Minute-level sequences (shape: T=1440, C=1)

Strict Leakage Prevention:
Standardization parameters (mean and std) are computed exclusively on training windows.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class ActigraphyDataset(Dataset):
    """
    PyTorch Dataset for actigraphy sequence windows.
    """
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        pids: np.ndarray,
        window_indices: np.ndarray,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        channels_first: bool = False,
    ):
        """
        Parameters:
            sequences: Array of shape (N, T, 1) or (N, T)
            labels: Array of shape (N,) with binary labels (0 or 1)
            pids: Array of shape (N,) with participant IDs
            window_indices: Array of shape (N,) with global window indices
            mean: Training-set mean for standardization
            std: Training-set std for standardization
            channels_first: If True, returns tensor of shape (1, T) (useful for 1D-CNN)
        """
        if sequences.ndim == 2:
            sequences = sequences[:, :, np.newaxis]
        self.sequences = sequences.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.pids = np.asarray(pids)
        self.window_indices = np.asarray(window_indices, dtype=np.int64)
        self.channels_first = channels_first

        # Apply standardization if parameters provided
        if mean is not None and std is not None:
            self.mean = float(mean)
            self.std = float(std) if float(std) > 1e-6 else 1.0
            self.sequences = (self.sequences - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        x = self.sequences[idx] # shape (T, 1)
        if self.channels_first:
            x = np.transpose(x, (1, 0)) # shape (1, T)
        x_tensor = torch.from_numpy(x).float()
        y_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        win_idx = int(self.window_indices[idx])
        pid = str(self.pids[idx])
        return x_tensor, y_tensor, win_idx, pid


def compute_training_stats(sequences: np.ndarray) -> Tuple[float, float]:
    """Compute mean and std exclusively across training windows."""
    mean_val = float(np.mean(sequences))
    std_val = float(np.std(sequences))
    if std_val < 1e-6:
        std_val = 1.0
    return mean_val, std_val
