"""Unit tests for PyTorch Hessian-Vector Product Matrix-Free Oracle."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.pytorch_matvec_oracle import PyTorchHessianOracle
from src.trace_baseline import (
    Hutchinson,
    Hutch_pplus,
    Adaptive_Hutch_pplus_TwoStageGated,
)


class SimpleQuadraticModel(nn.Module):
    """Model whose Hessian is analytically known for exact mathematical testing."""

    def __init__(self, Q: torch.Tensor):
        super().__init__()
        self.d = Q.shape[0]
        self.w = nn.Parameter(torch.randn(self.d))
        self.register_buffer("Q", Q)

    def forward(self, x):
        # f(w) = 0.5 * w^T Q w
        return 0.5 * torch.dot(self.w, torch.mv(self.Q, self.w))


def test_hvp_linearity_and_symmetry():
    """Verify that PyTorchHessianOracle satisfies mathematical linearity and symmetry."""
    torch.manual_seed(42)
    np.random.seed(42)
    d = 20

    # Create random symmetric positive definite matrix Q
    A = torch.randn(d, d)
    Q = A.T @ A + 0.1 * torch.eye(d)
    expected_trace = float(torch.trace(Q).item())

    model = SimpleQuadraticModel(Q)
    dummy_input = torch.tensor([1.0])
    dummy_target = torch.tensor([0.0])

    def loss_fn(pred, target):
        return pred

    oracle = PyTorchHessianOracle(model, loss_fn, dummy_input, dummy_target)
    assert oracle.num_params == d

    # 1. Test Linearity: H(a*u + b*v) == a*Hu + b*Hv
    u = np.random.randn(d)
    v = np.random.randn(d)
    a, b = 2.5, -1.3

    Hu = oracle(u)
    Hv = oracle(v)
    H_linear = oracle(a * u + b * v)
    assert np.allclose(H_linear, a * Hu + b * Hv, atol=1e-5)

    # 2. Test Symmetry: u^T (H v) == v^T (H u)
    u_Hv = float(np.dot(u, Hv))
    v_Hu = float(np.dot(v, Hu))
    assert np.isclose(u_Hv, v_Hu, atol=1e-5)

    # 3. Test Exact Trace against analytic Q
    computed_trace = oracle.compute_exact_trace_columnwise()
    assert np.isclose(computed_trace, expected_trace, rtol=1e-4)


def test_estimator_integration_with_pytorch_oracle():
    """Verify that Hutchinson, Hutch++, and TwoStageGated execute cleanly on PyTorch oracle."""
    torch.manual_seed(123)
    np.random.seed(123)
    d = 30
    m = 30

    # MLP with 1 hidden layer
    model = nn.Sequential(
        nn.Linear(5, 5, bias=False),  # 25 params
        nn.ReLU(),
        nn.Linear(5, 1, bias=False),  # 5 params -> total 30 params
    )
    inputs = torch.randn(10, 5)
    targets = torch.randn(10, 1)

    loss_fn = nn.MSELoss()
    oracle = PyTorchHessianOracle(model, loss_fn, inputs, targets)
    assert oracle.num_params == d

    exact_tr = oracle.compute_exact_trace_columnwise()
    assert np.isfinite(exact_tr)

    # Reset oracle query count
    oracle.query_count = 0

    # 1. Hutchinson
    est_hutch = Hutchinson(oracle, m=m, d=d)
    assert oracle.query_count == m
    assert np.isfinite(est_hutch)

    # 2. Standard Hutch++
    oracle.query_count = 0
    est_hpp = Hutch_pplus(oracle, m=m, d=d)
    assert oracle.query_count == m
    assert np.isfinite(est_hpp)

    # 3. TwoStageGated
    oracle.query_count = 0
    est_gated, diag = Adaptive_Hutch_pplus_TwoStageGated(
        oracle, m=m, d=d, b_0=8, return_diagnostics=True
    )
    assert oracle.query_count == m
    assert np.isfinite(est_gated)
    assert "is_gated_trigger" in diag
