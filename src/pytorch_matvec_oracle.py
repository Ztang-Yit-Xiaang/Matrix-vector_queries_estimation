"""PyTorch Hessian-Vector Product (HVP) Matrix-Free Oracle.

Wraps PyTorch models to provide matrix-free Hessian-vector products:
    v -> H v = nabla_theta ( <nabla_theta L(theta), v> )
without ever constructing the dense d x d Hessian.

Integrates with MatVecOracle query budget accounting and supports
both single-vector and batch-vector matrix-free multiplications.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Sequence


class PyTorchHessianOracle:
    """Matrix-free Hessian-Vector Product Oracle with exact query accounting."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        inputs: torch.Tensor,
        targets: torch.Tensor,
        device: torch.device | str = "cpu",
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.inputs = inputs.to(device)
        self.targets = targets.to(device)
        self.device = device
        self.query_count = 0

        # Extract trainable parameters
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.num_params = sum(p.numel() for p in self.params)

        # Compute initial gradient at current weights
        self.model.eval()
        outputs = self.model(self.inputs)
        loss = self.loss_fn(outputs, self.targets)
        self.grads = torch.autograd.grad(
            loss, self.params, create_graph=True, retain_graph=True
        )

    def _unflatten_vector(self, v_flat: torch.Tensor) -> list[torch.Tensor]:
        """Unflatten a 1D tensor of size d into a list of tensors matching model parameters."""
        v_list = []
        offset = 0
        for p in self.params:
            numel = p.numel()
            v_list.append(v_flat[offset : offset + numel].view_as(p))
            offset += numel
        return v_list

    def _hvp_single(self, v_np: np.ndarray) -> np.ndarray:
        """Compute single Hessian-vector product H v via autograd."""
        v_tensor = torch.from_numpy(v_np).float().to(self.device)
        v_list = self._unflatten_vector(v_tensor)

        # Compute directional derivative: sum <grad_i, v_i>
        grad_v_prod = sum(
            torch.sum(g * v) for g, v in zip(self.grads, v_list)
        )

        # Second backward pass to get H v
        hvp_tensors = torch.autograd.grad(
            grad_v_prod, self.params, retain_graph=True
        )

        # Flatten result back to 1D numpy array
        hvp_flat = torch.cat([h.contiguous().view(-1) for h in hvp_tensors])
        self.query_count += 1
        return hvp_flat.detach().cpu().numpy().astype(np.float64)

    def __call__(self, V: np.ndarray) -> np.ndarray:
        """Matrix-vector product evaluation for V in R^{d} or R^{d x k}."""
        V = np.asarray(V, dtype=np.float64)
        if V.ndim == 1:
            if V.shape[0] != self.num_params:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.num_params}, got {V.shape[0]}"
                )
            return self._hvp_single(V)
        elif V.ndim == 2:
            if V.shape[0] != self.num_params:
                raise ValueError(
                    f"Matrix dimension mismatch: expected {self.num_params}, got {V.shape[0]}"
                )
            k = V.shape[1]
            out = np.zeros((self.num_params, k), dtype=np.float64)
            for j in range(k):
                out[:, j] = self._hvp_single(V[:, j])
            return out
        else:
            raise ValueError(f"Expected 1D or 2D array, got {V.ndim}D")

    def compute_exact_trace_columnwise(self) -> float:
        """Compute exact Hessian trace by column-wise HVP: tr(H) = sum_i e_i^T (H e_i).

        Feasible when d <= 5,000 for verification benchmarks.
        """
        exact_trace = 0.0
        d = self.num_params
        saved_count = self.query_count

        for i in range(d):
            e_i = np.zeros(d, dtype=np.float64)
            e_i[i] = 1.0
            h_i = self._hvp_single(e_i)
            exact_trace += h_i[i]

        # Reset query count so benchmark starts clean
        self.query_count = saved_count
        return float(exact_trace)
