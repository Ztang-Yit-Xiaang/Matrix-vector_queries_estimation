"""Real Matrix-Free Machine Learning Benchmark: Neural Network Hessian Trace Estimation.

Evaluates Adaptive TwoStageGated Hutch++, Standard Hutch++, and Classical Hutchinson
on neural network loss landscapes using implicit Hessian-Vector Products (HVP).

Demonstrates:
1. Matrix-free execution without ever allocating the O(d^2) dense Hessian matrix.
2. Exact query budget accounting (q + r_actual + ell == m) via PyTorchHessianOracle.
3. Exploitation of the spiked/step eigenspectrum naturally present in trained neural networks.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections.abc import Sequence
from torch.utils.data import DataLoader, TensorDataset

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

from src.pytorch_matvec_oracle import PyTorchHessianOracle
from src.trace_baseline import (
    Hutchinson,
    Hutch_pplus,
    Gaussian_Hutch_pplus,
    Adaptive_Hutch_pplus_TwoStageGated,
)

RESULTS_DIR = os.path.join(project_dir, "results")


class SmallConvNet(nn.Module):
    """Compact Convolutional Neural Network for exact ground-truth trace verification.

    Total parameters d ~ 1,800 - 2,500, allowing exact column-wise trace computation.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, bias=False)  # 4 * 1 * 9 = 36
        self.pool = nn.MaxPool2d(2, 2)  # 14 x 14
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=False)  # 8 * 4 * 9 = 288
        # pool -> 7 x 7
        self.fc = nn.Linear(8 * 7 * 7, num_classes, bias=True)  # 392 * 10 + 10 = 3,930
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def generate_synthetic_vision_data(
    num_samples: int = 500, in_dim: tuple = (1, 28, 28), num_classes: int = 10, seed: int = 42
):
    """Generate structured synthetic image classification data."""
    torch.manual_seed(seed)
    X = torch.randn(num_samples, *in_dim)
    # Target depends on linear combination of patch sums
    patch_sums = X.view(num_samples, -1)[:, :num_classes].sum(dim=1)
    y = (patch_sums > 0).long() % num_classes
    return X, y


def train_model(model: nn.Module, loader: DataLoader, epochs: int = 5, lr: float = 0.01):
    """Train model to reach realistic loss curvature with dominant spectral spikes."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    model.eval()


def run_pytorch_hessian_benchmark(
    n_trials: int = 30,
    budgets: Sequence[int] = (30, 60, 90, 120),
    seed: int = 2026,
    output_dir: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute the synthetic-data Hessian benchmark; writing requires output_dir."""
    print("=" * 70)
    print("STARTING PYTORCH MATRIX-FREE ML HESSIAN TRACE BENCHMARK")
    print("=" * 70)

    # 1. Setup Data & Model
    X, y = generate_synthetic_vision_data(num_samples=400, in_dim=(1, 28, 28), num_classes=10)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SmallConvNet(in_channels=1, num_classes=10)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Instantiated Model: SmallConvNet with d = {num_params} trainable parameters.")
    print("Training model for 5 epochs to induce realistic spectral knee curvature...")
    train_model(model, loader, epochs=5, lr=0.02)

    # Evaluation Batch for Hessian Computation
    eval_inputs = X[:128]
    eval_targets = y[:128]
    criterion = nn.CrossEntropyLoss()

    oracle = PyTorchHessianOracle(model, criterion, eval_inputs, eval_targets)
    d = oracle.num_params
    print(f"PyTorchHessianOracle ready: d = {d} parameters.")

    # 2. Compute Exact Ground Truth Hessian Trace
    print(f"Computing exact ground truth Hessian trace via {d} column-wise HVPs...")
    t0 = time.time()
    exact_trace = oracle.compute_exact_trace_columnwise()
    t_exact = time.time() - t0
    print(f"Exact Hessian Trace = {exact_trace:.6f} (computed in {t_exact:.2f}s).")

    # 3. Run Comparative Estimator Sweep
    rng = np.random.default_rng(seed)
    records = []

    for m in budgets:
        print(f"\n--- Testing Query Budget m = {m} ({n_trials} trials per estimator) ---")
        q_0 = m // 3

        for trial in range(n_trials):
            # A. Classical Hutchinson
            oracle.query_count = 0
            t_start = time.time()
            est_hutch = Hutchinson(oracle, m=m, d=d, rng=rng)
            time_hutch = time.time() - t_start
            assert oracle.query_count == m
            err_hutch = abs(est_hutch - exact_trace) / exact_trace

            records.append({
                "budget_m": m,
                "trial": trial,
                "algorithm": "Classical Hutchinson",
                "trace_estimate": est_hutch,
                "rel_err": err_hutch,
                "sq_err": (est_hutch - exact_trace) ** 2,
                "gate_triggered": False,
                "q_target": 0,
                "time_sec": time_hutch,
            })

            # B. Standard Hutch++ (Rademacher)
            oracle.query_count = 0
            t_start = time.time()
            est_hpp = Hutch_pplus(oracle, m=m, d=d, rng=rng)
            time_hpp = time.time() - t_start
            assert oracle.query_count == m
            err_hpp = abs(est_hpp - exact_trace) / exact_trace

            records.append({
                "budget_m": m,
                "trial": trial,
                "algorithm": "Standard Hutch++",
                "trace_estimate": est_hpp,
                "rel_err": err_hpp,
                "sq_err": (est_hpp - exact_trace) ** 2,
                "gate_triggered": False,
                "q_target": q_0,
                "time_sec": time_hpp,
            })

            # C. Standard Gaussian Hutch++
            oracle.query_count = 0
            t_start = time.time()
            est_ghpp = Gaussian_Hutch_pplus(oracle, m=m, d=d, rng=rng)
            time_ghpp = time.time() - t_start
            assert oracle.query_count == m
            err_ghpp = abs(est_ghpp - exact_trace) / exact_trace

            records.append({
                "budget_m": m,
                "trial": trial,
                "algorithm": "Gaussian Hutch++",
                "trace_estimate": est_ghpp,
                "rel_err": err_ghpp,
                "sq_err": (est_ghpp - exact_trace) ** 2,
                "gate_triggered": False,
                "q_target": q_0,
                "time_sec": time_ghpp,
            })

            # D. TwoStageGated (Ours)
            oracle.query_count = 0
            t_start = time.time()
            est_gated, diag = Adaptive_Hutch_pplus_TwoStageGated(
                oracle,
                m=m,
                d=d,
                b_0=8,
                tau_gap=1.2,
                p_oversample=2,
                rng=rng,
                return_diagnostics=True,
            )
            time_gated = time.time() - t_start
            assert oracle.query_count == m
            err_gated = abs(est_gated - exact_trace) / exact_trace

            records.append({
                "budget_m": m,
                "trial": trial,
                "algorithm": "TwoStageGated (Ours)",
                "trace_estimate": est_gated,
                "rel_err": err_gated,
                "sq_err": (est_gated - exact_trace) ** 2,
                "gate_triggered": bool(diag["is_gated_trigger"]),
                "q_target": int(diag["q_target"]),
                "time_sec": time_gated,
            })

    df = pd.DataFrame(records)
    df["seed"] = seed
    df["data_seed"] = 42
    df["dimension"] = d
    df["exact_trace"] = exact_trace
    df["query_count"] = df["budget_m"]
    df["ground_truth_hvp_count"] = d

    # Summary table
    summary_rows = []
    for (budget_m, algo), grp in df.groupby(["budget_m", "algorithm"]):
        summary_rows.append({
            "budget_m": budget_m,
            "algorithm": algo,
            "median_rel_err": grp["rel_err"].median(),
            "mean_rel_err": grp["rel_err"].mean(),
            "mse": grp["sq_err"].mean(),
            "mean_time_sec": grp["time_sec"].mean(),
            "gate_trigger_rate": grp["gate_triggered"].mean(),
            "mean_q_target": grp["q_target"].mean(),
        })

    df_summary = pd.DataFrame(summary_rows)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(os.path.join(output_dir, "pytorch_hessian_benchmark_trials.csv"), index=False)
        df_summary.to_csv(os.path.join(output_dir, "pytorch_hessian_benchmark_summary.csv"), index=False)

    return df, df_summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    run_pytorch_hessian_benchmark(output_dir=parser.parse_args().output_dir)
