"""
Temporal Self-Attention Transformer Architecture for Actigraphy Analysis.

Module: ai_system.transformer_attention
Integrates a modern Multi-Head Self-Attention Transformer mechanism to discover
dynamic temporal attention patterns across the 24-hour circadian actigraphy cycle.

Key Features:
1. Circadian Positional Embeddings: Maps continuous hourly time steps (t=0..23) to latent space.
2. Multi-Head Self-Attention (MHSA): Learns cross-temporal dependencies across the 24 hours.
3. Attention Extraction Hook: Captures hour-by-hour attention weights (alpha_t in [0, 1])
   to identify which hours of the day (e.g. afternoon sedentary bouts vs nocturnal restlessness)
   the AI prioritizes when diagnosing depression.
4. Standalone & Modular: Completely isolated from the core research folders.
"""

from typing import Tuple, Dict, Any, Optional
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


class TemporalCircadianEmbedding(nn.Module):
    """Encodes continuous 24-hour periodic circadian time steps."""
    def __init__(self, embed_dim: int = 64, max_len: int = 24):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Timesteps, Embed_Dim)
        return x + self.pos_embedding[:, :x.size(1), :]


class ActigraphyTemporalTransformer(nn.Module):
    """
    Multi-Head Temporal Self-Attention Transformer for 24-hour Actigraphy Sequences.
    """
    def __init__(
        self,
        input_dim: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 1. Linear Input Projection
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.pos_encoder = TemporalCircadianEmbedding(embed_dim=embed_dim, max_len=24)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 2. Multi-Head Self-Attention Block
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # 3. Feedforward Subnetwork
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )

        # 4. Attentive Temporal Pooling Head
        self.pool_attn = nn.Linear(embed_dim, 1)

        # 5. Classification Output
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        self._last_attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning classification logits and optional 24-hour attention map.
        Parameters:
            x: Tensor of shape (Batch, 24, 1) or (Batch, 24)
            return_attention: Whether to return the attention matrix (Batch, 24, 24)
        Returns:
            logits: Tensor of shape (Batch,)
            attn_weights: Tensor of shape (Batch, 24, 24) or None
        """
        if x.ndim == 2:
            x = x.unsqueeze(-1)

        # 1. Project to embedding space & inject circadian positional embeddings
        h = self.input_proj(x)
        h = self.pos_encoder(h)
        h = self.norm1(h)

        # 2. Multi-Head Self-Attention
        attn_out, attn_weights = self.self_attn(
            query=h,
            key=h,
            value=h,
            need_weights=True,
            average_attn_weights=True,
        )
        h = h + attn_out
        h = self.norm2(h)

        # 3. Feedforward Block
        h = h + self.ffn(h)

        # 4. Temporal Attention Pooling
        pool_weights = torch.softmax(self.pool_attn(h), dim=1)  # (Batch, 24, 1)
        pooled = torch.sum(h * pool_weights, dim=1)            # (Batch, embed_dim)

        # 5. Classification Logits
        logits = self.classifier(pooled).squeeze(-1)

        self._last_attention_weights = attn_weights
        if return_attention:
            return logits, attn_weights
        return logits, None

    def get_circadian_attention_profile(self, x: torch.Tensor) -> np.ndarray:
        """
        Extract the 1D hour-by-hour (0..23) attention profile for input actigraphy frames.
        Returns:
            hourly_importance: Array of shape (24,) summing attention allocated per hour.
        """
        self.eval()
        with torch.no_grad():
            if x.ndim == 2:
                x = x.unsqueeze(-1)
            _, attn_matrix = self.forward(x, return_attention=True)
            # attn_matrix: (Batch, 24, 24) -> average across all queries to get key hourly importance
            hourly_weights = attn_matrix.mean(dim=(0, 1)).cpu().numpy()
            hourly_weights = hourly_weights / (hourly_weights.sum() + 1e-8)
            return hourly_weights


def train_transformer_and_extract_attention(
    sequences_path: str = "data/processed/windows_hourly.npy",
    metadata_path: str = "data/processed/window_metadata.csv",
    epochs: int = 40,
    lr: float = 0.001,
    seed: int = 42,
) -> Tuple[ActigraphyTemporalTransformer, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Train the Temporal Self-Attention Transformer and extract circadian attention profiles
    differentiating Depressed (Condition) vs Healthy Control cohorts.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    sequences = np.load(sequences_path)  # (1331, 24, 1)
    meta_df = pd.read_csv(metadata_path)
    labels = meta_df["label"].values.astype(int)

    # Standardize
    m, s = sequences.mean(), sequences.std() + 1e-8
    seq_norm = (sequences - m) / s

    X_tensor = torch.tensor(seq_norm, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    model = ActigraphyTemporalTransformer(embed_dim=64, num_heads=4, dim_feedforward=128, dropout=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    # Extract 24-hour attention profiles per cohort
    idx_dep = np.where(labels == 1)[0]
    idx_ctrl = np.where(labels == 0)[0]

    attn_dep = model.get_circadian_attention_profile(X_tensor[idx_dep])
    attn_ctrl = model.get_circadian_attention_profile(X_tensor[idx_ctrl])

    # Per-participant attention profiles
    part_records = []
    for pid in meta_df["participant_id"].unique():
        p_mask = (meta_df["participant_id"] == pid).values
        p_tensor = X_tensor[p_mask]
        p_attn = model.get_circadian_attention_profile(p_tensor)
        p_label = int(meta_df[p_mask]["label"].iloc[0])
        rec = {"participant_id": pid, "label": p_label}
        for h in range(24):
            rec[f"hour_{h:02d}"] = float(np.round(p_attn[h], 5))
        part_records.append(rec)

    attention_df = pd.DataFrame(part_records)
    return model, attn_dep, attn_ctrl, attention_df


def plot_figure10_circadian_attention(
    attn_dep: np.ndarray,
    attn_ctrl: np.ndarray,
    output_path: str = "ai_system/results/figures/figure10_circadian_attention_heatmap.png",
):
    """
    Generate publication-ready 300 DPI visualization of Transformer Self-Attention weights
    across the 24-hour circadian cycle comparing Depressed vs Control cohorts.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hours = np.arange(24)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), dpi=300, sharex=True)

    # 1. Hour-by-Hour Attention Trajectory
    ax1.plot(hours, attn_dep * 100, color="#d95f02", lw=2.5, marker="o", markersize=5, label="Depressed Cohort (Condition)")
    ax1.plot(hours, attn_ctrl * 100, color="#1b9e77", lw=2.5, marker="s", markersize=5, linestyle="--", label="Healthy Controls")
    
    # Highlight salient temporal zones
    ax1.axvspan(1, 5, color="#7570b3", alpha=0.15, label="Nocturnal Vulnerability Window (01:00-05:00)")
    ax1.axvspan(13, 17, color="#e7298a", alpha=0.15, label="Diurnal Psychomotor Retardation (13:00-17:00)")
    
    ax1.set_ylabel("Attention Weight (%)", fontweight="bold", fontsize=10)
    ax1.set_title("Temporal Transformer AI: 24-Hour Circadian Self-Attention Distribution", fontweight="bold", fontsize=12)
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Attention Disparity Heatmap (Depressed Attention - Control Attention)
    diff = (attn_dep - attn_ctrl) * 100
    colors = ["#d95f02" if d > 0 else "#1b9e77" for d in diff]
    bars = ax2.bar(hours, diff, color=colors, width=0.6, edgecolor="black", alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    
    ax2.set_xlabel("Hour of the Day (00:00 to 23:00)", fontweight="bold", fontsize=10)
    ax2.set_ylabel(r"Attention Shift $\Delta$ (%)", fontweight="bold", fontsize=10)
    ax2.set_title(r"Differential Attention Focus: $\Delta = \text{Attention}_{\text{Depressed}} - \text{Attention}_{\text{Control}}$", fontweight="bold", fontsize=11)
    ax2.set_xticks(hours)
    ax2.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45, ha="right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Transformer Circadian Attention Figure to {output_path}")
