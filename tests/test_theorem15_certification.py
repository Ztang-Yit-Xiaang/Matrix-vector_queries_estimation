import math
import sys
from pathlib import Path

import numpy as np
import pytest


SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from theorem15_certification import (  # noqa: E402
    admissible_product_radius,
    certify_gaussian_knee,
    gaussian_hmt_product_bound,
    guaranteed_ritz_ratio,
    log_threshold_from_ratio,
    minimum_certifiable_oversampling,
    ratio_threshold_from_alignment,
    ratio_threshold_from_log,
    required_squared_alignment,
)
from experiments.run_theorem15_certification import _step_trial  # noqa: E402


def test_ratio_and_log_thresholds_round_trip():
    gamma_gap = 1.5
    tau_ratio = ratio_threshold_from_log(gamma_gap)
    assert tau_ratio == pytest.approx(math.exp(gamma_gap))
    assert log_threshold_from_ratio(tau_ratio) == pytest.approx(gamma_gap)


def test_alignment_parameterization_round_trip():
    eta = 0.05
    rho = 0.25
    tau_ratio = ratio_threshold_from_alignment(eta, rho)
    assert 1.0 < tau_ratio < 1.0 / eta
    assert required_squared_alignment(eta, tau_ratio) == pytest.approx(rho)


def test_current_step_regime_matches_audited_values():
    certificate = certify_gaussian_knee(
        d=500,
        r_star=20,
        p=4,
        eta=0.01,
        delta=0.05,
        tau_ratio=math.exp(1.5),
    )
    assert certificate.k_gaussian == pytest.approx(164.4441926585429)
    assert certificate.kappa == pytest.approx(523.7792928524184)
    assert certificate.certification_margin == pytest.approx(3.1851492253059384)
    assert certificate.certified
    assert certificate.guaranteed_ratio > certificate.tau_ratio
    assert minimum_certifiable_oversampling(
        500, 20, 0.01, 0.05, math.exp(1.5)
    ) == 4


def test_certificate_inequality_matches_ritz_lower_bound():
    eta = 0.10
    tau_ratio = 3.0
    kappa = admissible_product_radius(eta, tau_ratio)
    assert guaranteed_ritz_ratio(eta, 0.99 * kappa) > tau_ratio
    assert guaranteed_ritz_ratio(eta, kappa) == pytest.approx(tau_ratio)
    assert guaranteed_ritz_ratio(eta, 1.01 * kappa) < tau_ratio


@pytest.mark.parametrize("distribution", ["gaussian", "rademacher"])
def test_empirical_trial_respects_deterministic_implication(distribution):
    orientation_rng = np.random.default_rng(7)
    U, _ = np.linalg.qr(orientation_rng.normal(size=(80, 8)), mode="reduced")
    result = _step_trial(
        d=80,
        r_star=8,
        p=4,
        eta=0.05,
        tau_ratio=3.0,
        distribution=distribution,
        rng=np.random.default_rng(42),
        U=U,
    )
    assert result["full_row_rank"]
    assert result["ratio_defined"]
    assert result["deterministic_implication_holds"]


@pytest.mark.parametrize(
    "call",
    [
        lambda: gaussian_hmt_product_bound(20, 5, 3, 0.05),
        lambda: gaussian_hmt_product_bound(20, 18, 4, 0.05),
        lambda: gaussian_hmt_product_bound(20, 5, 4, 0.0),
        lambda: admissible_product_radius(0.1, 1.0),
        lambda: admissible_product_radius(0.1, 10.0),
        lambda: ratio_threshold_from_alignment(0.1, 1.0),
        lambda: ratio_threshold_from_log(1e308),
        lambda: minimum_certifiable_oversampling(7, 4, 0.1, 0.05, 3.0),
    ],
)
def test_invalid_theorem_domains_are_rejected(call):
    with pytest.raises((TypeError, ValueError)):
        call()
