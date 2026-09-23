"""Tests for the PyTorch Matrix-Free Neural Network Hessian trace estimation benchmark."""

import os
import pytest
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.run_pytorch_hessian_benchmark import (
    SmallConvNet,
    generate_synthetic_vision_data,
    train_model,
    run_pytorch_hessian_benchmark,
)
from src.pytorch_matvec_oracle import PyTorchHessianOracle


def test_small_convnet_instantiation_and_forward():
    """Verify SmallConvNet architecture and output dimensions."""
    model = SmallConvNet(in_channels=1, num_classes=10)
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4254


def test_pytorch_hessian_benchmark_smoke_run(tmp_path):
    """Run a fast smoke test of the benchmark with minimal budget and trials."""
    df_raw, df_summary = run_pytorch_hessian_benchmark(
        n_trials=2,
        budgets=(30,),
        seed=123,
        output_dir=tmp_path,
    )
    assert isinstance(df_raw, pd.DataFrame)
    assert (tmp_path / "pytorch_hessian_benchmark_trials.csv").is_file()
    assert (df_raw["query_count"] == df_raw["budget_m"]).all()
    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_raw) == 2 * 4  # 2 trials * 4 estimators
    assert len(df_summary) == 4
    for col in ["budget_m", "algorithm", "median_rel_err", "mse", "gate_trigger_rate"]:
        assert col in df_summary.columns
    # Check error non-negativity
    assert (df_raw["rel_err"] >= 0).all()
    assert (df_raw["sq_err"] >= 0).all()
