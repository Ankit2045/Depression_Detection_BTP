"""
Unit tests for PyTorch Deep Learning Architectures and Datasets.
"""

import torch
import numpy as np
import pytest

from deep_learning.datasets import ActigraphyDataset, compute_training_stats
from deep_learning.lstm import ActigraphyLSTM, count_trainable_parameters
from deep_learning.gru import ActigraphyGRU
from deep_learning.cnn1d import ActigraphyCNN1D


def test_lstm_forward_and_feature_extraction():
    model = ActigraphyLSTM(input_dim=1, hidden_dim=64, num_layers=2)
    x_hourly = torch.randn(8, 24, 1) # batch=8, timesteps=24, dim=1
    
    # Forward pass
    logits = model(x_hourly)
    assert logits.shape == (8,)

    # Feature extraction hook
    latent = model.extract_features(x_hourly)
    assert latent.shape == (8, 64)

    # Minute-level sequence test
    x_minute = torch.randn(4, 1440, 1)
    logits_min = model(x_minute)
    assert logits_min.shape == (4,)


def test_gru_forward_and_feature_extraction():
    model = ActigraphyGRU(input_dim=1, hidden_dim=64, num_layers=2)
    x = torch.randn(8, 24, 1)

    logits = model(x)
    assert logits.shape == (8,)

    latent = model.extract_features(x)
    assert latent.shape == (8, 64)


def test_cnn1d_forward_and_feature_extraction():
    model = ActigraphyCNN1D(in_channels=1, num_filters=(32, 64))
    
    # Test (Batch, T, 1)
    x1 = torch.randn(8, 24, 1)
    logits1 = model(x1)
    assert logits1.shape == (8,)
    latent1 = model.extract_features(x1)
    assert latent1.shape == (8, 64)

    # Test (Batch, 1, T)
    x2 = torch.randn(8, 1, 24)
    logits2 = model(x2)
    assert logits2.shape == (8,)

    # Test Minute resolution (Batch, 1440, 1)
    x3 = torch.randn(4, 1440, 1)
    logits3 = model(x3)
    assert logits3.shape == (4,)


def test_parameter_counts_reasonable():
    lstm = ActigraphyLSTM(input_dim=1, hidden_dim=64, num_layers=2)
    gru = ActigraphyGRU(input_dim=1, hidden_dim=64, num_layers=2)
    cnn = ActigraphyCNN1D(in_channels=1, num_filters=(32, 64))

    p_lstm = count_trainable_parameters(lstm)
    p_gru = count_trainable_parameters(gru)
    p_cnn = count_trainable_parameters(cnn)

    # Ensure parameter counts are compact (< 100k parameters) to prevent overfitting on 55 subjects
    assert 5000 < p_lstm < 100000
    assert 5000 < p_gru < 100000
    assert 1000 < p_cnn < 100000


def test_dataset_and_standardization():
    seqs = np.array([[[10.0], [20.0]], [[30.0], [40.0]]], dtype=np.float32) # (2, 2, 1)
    labels = np.array([1, 0])
    pids = np.array(["condition_1", "control_1"])
    win_idx = np.array([0, 1])

    mean, std = compute_training_stats(seqs)
    assert np.isclose(mean, 25.0)

    dataset = ActigraphyDataset(seqs, labels, pids, win_idx, mean=mean, std=std)
    assert len(dataset) == 2
    x_t, y_t, w_idx, pid = dataset[0]
    assert x_t.shape == (2, 1)
    assert y_t.item() == 1.0
    assert w_idx == 0
    assert pid == "condition_1"
