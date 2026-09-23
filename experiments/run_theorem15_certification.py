"""Build a small Theorem 15 certification map and empirical sanity check."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from theorem15_certification import (  # noqa: E402
    certify_gaussian_knee,
    gaussian_hmt_product_bound,
    guaranteed_ritz_ratio,
    minimum_certifiable_oversampling,
    ratio_threshold_from_alignment,
    ratio_threshold_from_log,
)


DEFAULT_GAMMA_GAP = 1.5
DEFAULT_TAU_RATIO = ratio_threshold_from_log(DEFAULT_GAMMA_GAP)


def build_certification_map():
    rows = []
    d = 500
    delta = 0.05
    for r_star in (10, 20):
        for eta in (0.01, 0.05, 0.10, 0.20):
            threshold_cases = (
                ("fixed_code_gamma", DEFAULT_TAU_RATIO),
                ("fixed_alignment_rho_0.25", ratio_threshold_from_alignment(eta, 0.25)),
            )
            for threshold_mode, tau_ratio in threshold_cases:
                minimum_p = minimum_certifiable_oversampling(
                    d, r_star, eta, delta, tau_ratio
                )
                for p in (4, 8, 10):
                    row = certify_gaussian_knee(
                        d=d,
                        r_star=r_star,
                        p=p,
                        eta=eta,
                        delta=delta,
                        tau_ratio=tau_ratio,
                    ).to_dict()
                    row["threshold_mode"] = threshold_mode
                    row["minimum_certifiable_p"] = minimum_p
                    rows.append(row)
    return pd.DataFrame(rows)


def _step_trial(d, r_star, p, eta, tau_ratio, distribution, rng, U=None):
    b = r_star + p
    if U is None:
        U = np.eye(d, r_star)
    if U.shape != (d, r_star) or not np.allclose(U.T @ U, np.eye(r_star)):
        raise ValueError("U must have shape (d, r_star) and orthonormal columns.")

    if distribution == "gaussian":
        S = rng.normal(size=(d, b))
    elif distribution == "rademacher":
        S = rng.choice((-1.0, 1.0), size=(d, b))
    else:
        raise ValueError(f"Unknown sketch distribution: {distribution}")

    S_1 = U.T @ S
    S_residual = S - U @ S_1
    full_row_rank = np.linalg.matrix_rank(S_1) == r_star
    product_norm = np.nan
    if full_row_rank:
        # U_perp is unnecessary: left multiplication by its orthonormal basis
        # preserves the nonzero singular values of this residual product.
        product_norm = float(np.linalg.norm(S_residual @ np.linalg.pinv(S_1), ord=2))

    Y = eta * S + (1.0 - eta) * U @ S_1
    U_y, singular_values, _ = np.linalg.svd(Y, full_matrices=False)
    tolerance = max(Y.shape) * np.finfo(Y.dtype).eps * singular_values[0]
    r_y = int(np.sum(singular_values > tolerance))
    Q = U_y[:, :r_y]
    ratio_defined = r_y >= r_star + 1
    knee_ratio = np.nan
    if ratio_defined:
        Q_signal = U.T @ Q
        ritz_matrix = eta * np.eye(r_y) + (1.0 - eta) * (Q_signal.T @ Q_signal)
        ritz_values = np.linalg.eigvalsh(ritz_matrix)[::-1]
        knee_ratio = float(ritz_values[r_star - 1] / ritz_values[r_star])

    lower_bound = np.nan
    implication_holds = np.nan
    if full_row_rank and ratio_defined:
        lower_bound = guaranteed_ritz_ratio(eta, product_norm)
        implication_holds = knee_ratio + 1e-10 >= lower_bound

    return {
        "full_row_rank": full_row_rank,
        "sample_rank": r_y,
        "ratio_defined": ratio_defined,
        "product_norm": product_norm,
        "knee_ratio": knee_ratio,
        "detected": bool(ratio_defined and knee_ratio > tau_ratio),
        "deterministic_ratio_lower_bound": lower_bound,
        "deterministic_implication_holds": implication_holds,
    }


def run_empirical_sanity_check(n_trials=200):
    d = 500
    r_star = 20
    eta = 0.01
    delta = 0.05
    tau_ratio = DEFAULT_TAU_RATIO
    summary_rows = []
    orientation_rng = np.random.default_rng(14_999)
    U, _ = np.linalg.qr(orientation_rng.normal(size=(d, r_star)), mode="reduced")

    for p in (4, 8, 10):
        k_gaussian = gaussian_hmt_product_bound(d, r_star, p, delta)
        certificate = certify_gaussian_knee(
            d, r_star, p, eta, delta, tau_ratio
        )
        for distribution_index, distribution in enumerate(("gaussian", "rademacher")):
            trial_rows = []
            for trial in range(n_trials):
                seed = 15_000 + 10_000 * distribution_index + 100 * p + trial
                trial_rows.append(
                    _step_trial(
                        d,
                        r_star,
                        p,
                        eta,
                        tau_ratio,
                        distribution,
                        np.random.default_rng(seed),
                        U=U,
                    )
                )

            trial_df = pd.DataFrame(trial_rows)
            full_rank_df = trial_df[trial_df["full_row_rank"]]
            summary_rows.append(
                {
                    "distribution": distribution,
                    "d": d,
                    "r_star": r_star,
                    "p": p,
                    "b": r_star + p,
                    "eta": eta,
                    "delta": delta,
                    "n_trials": n_trials,
                    "gamma_gap": DEFAULT_GAMMA_GAP,
                    "tau_ratio": tau_ratio,
                    "gaussian_k_bound": k_gaussian,
                    "gaussian_certificate_applies": distribution == "gaussian",
                    "gaussian_parameter_certificate": (
                        certificate.certified if distribution == "gaussian" else False
                    ),
                    "full_row_rank_rate": float(trial_df["full_row_rank"].mean()),
                    "ratio_defined_rate": float(trial_df["ratio_defined"].mean()),
                    "observed_product_bound_rate": (
                        float((full_rank_df["product_norm"] <= k_gaussian).mean())
                        if distribution == "gaussian" and not full_rank_df.empty
                        else np.nan
                    ),
                    "observed_detection_rate": float(trial_df["detected"].mean()),
                    "mean_knee_ratio": float(trial_df["knee_ratio"].mean()),
                    "mean_product_norm": (
                        float(full_rank_df["product_norm"].mean())
                        if not full_rank_df.empty
                        else np.nan
                    ),
                    "deterministic_implication_violations": int(
                        (trial_df["deterministic_implication_holds"] == False).sum()  # noqa: E712
                    ),
                }
            )
    return pd.DataFrame(summary_rows)


def main():
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    certification_df = build_certification_map()
    certification_path = results_dir / "theorem15_gaussian_certification_map.csv"
    certification_df.to_csv(certification_path, index=False)

    empirical_df = run_empirical_sanity_check()
    empirical_path = results_dir / "theorem15_sketch_distribution_check.csv"
    empirical_df.to_csv(empirical_path, index=False)

    print("Theorem 15 Gaussian certification map")
    print(
        certification_df.groupby(["threshold_mode", "r_star", "eta"])
        .agg(
            tested_p_certified_fraction=("certified", "mean"),
            minimum_certifiable_p=("minimum_certifiable_p", "first"),
        )
        .to_string()
    )
    print("\nSketch-distribution sanity check")
    print(empirical_df.to_string(index=False))
    print(f"\nSaved {certification_path}")
    print(f"Saved {empirical_path}")


if __name__ == "__main__":
    main()
