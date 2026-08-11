import os
import zipfile
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path

class RealWorldMatrix:
    """
    Container for a real-world matrix dataset.
    Exposes a matrix-vector oracle fn, exact trace, dimension d, 
    Frobenius norm, effective rank, and name.
    """
    def __init__(self, name, matvec_fn, d, exact_trace, frobenius_norm=None, effective_rank=None, metadata=None):
        self.name = name
        self.matvec_fn = matvec_fn
        self.d = d
        self.exact_trace = float(exact_trace)
        self.frobenius_norm = float(frobenius_norm) if frobenius_norm is not None else None
        self.effective_rank = float(effective_rank) if effective_rank is not None else None
        self.metadata = metadata or {}

    def summary(self):
        frob_str = f"{self.frobenius_norm:.4f}" if self.frobenius_norm is not None else "N/A"
        reff_str = f"{self.effective_rank:.4f}" if self.effective_rank is not None else "N/A"
        return (
            f"Dataset: {self.name}\n"
            f"  Dimension d: {self.d}\n"
            f"  Exact Trace tr(A): {self.exact_trace:.4f}\n"
            f"  Frobenius Norm ||A||_F: {frob_str}\n"
            f"  Effective Rank r_eff: {reff_str}\n"
        )

def load_wiki_vote():
    """
    Loads Wiki-Vote network graph as sparse adjacency matrix B.
    Defines A = B^T @ B (PSD matrix of dimension d=7115).
    tr(A) = ||B||_F^2 = number of edges.
    """
    wiki_path = Path(__file__).resolve().parent.parent / "wiki-Vote.txt"
    if not wiki_path.exists():
        raise FileNotFoundError(f"Wiki-Vote dataset not found at {wiki_path}")

    edges = []
    max_node = 0
    with open(wiki_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                u, v = int(parts[0]), int(parts[1])
                edges.append((u, v))
                max_node = max(max_node, u, v)

    # Remap node IDs to 0..d-1
    unique_nodes = sorted(list(set([u for u, _ in edges] + [v for _, v in edges])))
    node_map = {node: i for i, node in enumerate(unique_nodes)}
    d = len(unique_nodes)

    rows = [node_map[u] for u, _ in edges]
    cols = [node_map[v] for _, v in edges]
    data = np.ones(len(edges), dtype=np.float64)

    B = sp.csr_matrix((data, (rows, cols)), shape=(d, d))
    
    # A = B.T @ B
    exact_trace = float(B.nnz)  # tr(B^T B) = sum_ij B_ij^2 = nnz
    
    # Calculate A = B.T @ B explicitly (d=7115 sparse) to get exact ||A||_F and r_eff
    A_sparse = B.T @ B
    frobenius_norm = float(sp.linalg.norm(A_sparse, ord='fro'))
    effective_rank = (exact_trace ** 2) / (frobenius_norm ** 2)

    def matvec_fn(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return B.T @ (B @ v)
        else:
            return B.T @ (B @ v)

    return RealWorldMatrix(
        name="Wiki-Vote Graph (A=B^T B)",
        matvec_fn=matvec_fn,
        d=d,
        exact_trace=exact_trace,
        frobenius_norm=frobenius_norm,
        effective_rank=effective_rank,
        metadata={"nnz": B.nnz, "nodes": d}
    )

def load_year_prediction(nrows=50000):
    """
    Loads YearPredictionMSD features X (nrows x d=90).
    Defines covariance matrix A = X^T @ X (PSD matrix of dimension d=90).
    tr(A) = ||X||_F^2.
    """
    zip_path = Path(__file__).resolve().parent.parent.parent / "leverage_score" / "data" / "leverage_score" / "yearpredictionmsd.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"YearPredictionMSD zip not found at {zip_path}")

    with zipfile.ZipFile(zip_path) as archive:
        member = next(name for name in archive.namelist() if "yearpredictionmsd" in name.lower())
        with archive.open(member) as stream:
            df = pd.read_csv(stream, header=None, nrows=nrows)

    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    # Standardize features
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-8)
    
    d = X.shape[1] # 90 features
    exact_trace = float(np.sum(X ** 2))
    
    # Compute A = X.T @ X (90x90 matrix) explicitly
    A_dense = X.T @ X
    frobenius_norm = float(np.linalg.norm(A_dense, ord='fro'))
    effective_rank = (exact_trace ** 2) / (frobenius_norm ** 2)

    def matvec_fn(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return X.T @ (X @ v)
        else:
            return X.T @ (X @ v)

    return RealWorldMatrix(
        name=f"YearPredictionMSD (A=X^T X, n={nrows})",
        matvec_fn=matvec_fn,
        d=d,
        exact_trace=exact_trace,
        frobenius_norm=frobenius_norm,
        effective_rank=effective_rank,
        metadata={"nrows": nrows, "features": d}
    )

def load_synthetic_decay(d=500, c=1.0, seed=42):
    """
    Generates synthetic PSD matrix A = U diag(i^-c) U^T.
    Computes exact trace, Frobenius norm, and effective rank.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(X)
    i_vals = np.arange(1, d + 1, dtype=np.float64)
    lambdas = i_vals ** (-c)
    
    exact_trace = float(np.sum(lambdas))
    frobenius_norm = float(np.sqrt(np.sum(lambdas ** 2)))
    effective_rank = (exact_trace ** 2) / (frobenius_norm ** 2)
    
    A = Q @ (lambdas[:, None] * Q.T)

    def matvec_fn(v):
        return A @ v

    return RealWorldMatrix(
        name=f"Synthetic Decay (c={c:.1f}, d={d})",
        matvec_fn=matvec_fn,
        d=d,
        exact_trace=exact_trace,
        frobenius_norm=frobenius_norm,
        effective_rank=effective_rank,
        metadata={"c": c, "d": d}
    )

if __name__ == "__main__":
    print("Testing Dataset Loaders...")
    ds_synthetic = load_synthetic_decay(d=500, c=1.0)
    print(ds_synthetic.summary())
    
    ds_year = load_year_prediction(nrows=50000)
    print(ds_year.summary())
    
    ds_wiki = load_wiki_vote()
    print(ds_wiki.summary())
