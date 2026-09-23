"""Numerical certificates for the Gaussian step-spectrum knee theorem.

The functions in this module implement the sufficient condition proved in
``docs/proof_step_ritz_gap.md``.  They do not estimate an empirical detection
probability and they do not provide a certificate for Rademacher sketches.
"""

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
import sys


def _positive_integer(name, value, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


def _finite_real(name, value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def ratio_threshold_from_log(gamma_gap):
    """Convert the code-level log-gap threshold to a Ritz-ratio threshold."""
    gamma_gap = _finite_real("gamma_gap", gamma_gap)
    if gamma_gap <= 0.0:
        raise ValueError("gamma_gap must be positive.")
    try:
        return math.exp(gamma_gap)
    except OverflowError as exc:
        raise ValueError("gamma_gap is too large for a finite ratio threshold.") from exc


def log_threshold_from_ratio(tau_ratio):
    """Convert a Ritz-ratio threshold to the code-level log-gap threshold."""
    tau_ratio = _finite_real("tau_ratio", tau_ratio)
    if tau_ratio <= 1.0:
        raise ValueError("tau_ratio must be greater than 1.")
    return math.log(tau_ratio)


def ratio_threshold_from_alignment(eta, rho):
    """Return the ratio threshold requiring squared alignment greater than rho."""
    eta = _finite_real("eta", eta)
    rho = _finite_real("rho", rho)
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between 0 and 1.")
    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie strictly between 0 and 1.")
    return 1.0 + ((1.0 - eta) / eta) * rho


def required_squared_alignment(eta, tau_ratio):
    """Return rho_tau = eta(tau_ratio - 1)/(1 - eta)."""
    eta = _finite_real("eta", eta)
    tau_ratio = _finite_real("tau_ratio", tau_ratio)
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between 0 and 1.")
    if not 1.0 < tau_ratio < 1.0 / eta:
        raise ValueError("tau_ratio must satisfy 1 < tau_ratio < 1/eta.")
    return eta * (tau_ratio - 1.0) / (1.0 - eta)


def gaussian_hmt_product_bound(d, r_star, p, delta):
    """Return the conservative HMT-derived bound K^G.

    This is Theorem 15A's bound for an i.i.d. standard Gaussian sketch with
    ``b = r_star + p`` columns.  Its assumptions require ``p >= 4`` and
    ``b <= d``.
    """
    d = _positive_integer("d", d)
    r_star = _positive_integer("r_star", r_star)
    p = _positive_integer("p", p, minimum=4)
    delta = _finite_real("delta", delta)
    if r_star + p > d:
        raise ValueError("The Gaussian theorem requires r_star + p <= d.")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie strictly between 0 and 1.")

    b = r_star + p
    log_two_over_delta = math.log(2.0) - math.log(delta)
    u_delta = math.sqrt(2.0 * log_two_over_delta)
    try:
        z_p_delta = math.exp(log_two_over_delta / (p + 1.0))
    except OverflowError as exc:
        raise ValueError("delta is too small for a finite Gaussian bound.") from exc
    s2_bound = math.sqrt(d - r_star) + math.sqrt(b) + u_delta
    s1_pinv_bound = math.e * math.sqrt(b) * z_p_delta / (p + 1.0)
    product_bound = s2_bound * s1_pinv_bound
    if not math.isfinite(product_bound):
        raise ValueError("The Gaussian bound is not finite for these parameters.")
    return product_bound


def admissible_product_radius(eta, tau_ratio):
    """Return the deterministic radius kappa(eta, tau_ratio)."""
    eta = _finite_real("eta", eta)
    tau_ratio = _finite_real("tau_ratio", tau_ratio)
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between 0 and 1.")
    if not 1.0 < tau_ratio < 1.0 / eta:
        raise ValueError("tau_ratio must satisfy 1 < tau_ratio < 1/eta.")
    numerator = 1.0 - eta * tau_ratio
    denominator = eta**3 * (tau_ratio - 1.0)
    return math.sqrt(numerator / denominator)


def guaranteed_ritz_ratio(eta, product_bound):
    """Return the deterministic Ritz-ratio lower bound for a product bound K."""
    eta = _finite_real("eta", eta)
    product_bound = _finite_real("product_bound", product_bound)
    if not 0.0 < eta < 1.0:
        raise ValueError("eta must lie strictly between 0 and 1.")
    if product_bound < 0.0:
        raise ValueError("product_bound must be nonnegative.")
    scaled_product = eta * product_bound
    if scaled_product > math.sqrt(sys.float_info.max):
        return 1.0
    return 1.0 + (1.0 - eta) / (eta * (1.0 + scaled_product**2))


@dataclass(frozen=True)
class GaussianKneeCertificate:
    """One sufficient Gaussian certification result."""

    d: int
    r_star: int
    p: int
    b: int
    eta: float
    delta: float
    tau_ratio: float
    gamma_gap: float
    rho_required: float
    k_gaussian: float
    kappa: float
    certification_margin: float
    guaranteed_ratio: float
    certified: bool

    def to_dict(self):
        return asdict(self)


def certify_gaussian_knee(d, r_star, p, eta, delta, tau_ratio):
    """Evaluate the sufficient Gaussian knee-detection condition.

    ``certified`` means Theorem 15A guarantees detection with probability at
    least ``1 - delta``.  A false value does not imply detection failure.
    """
    k_gaussian = gaussian_hmt_product_bound(d, r_star, p, delta)
    kappa = admissible_product_radius(eta, tau_ratio)
    tau_ratio = float(tau_ratio)
    eta = float(eta)
    return GaussianKneeCertificate(
        d=int(d),
        r_star=int(r_star),
        p=int(p),
        b=int(r_star + p),
        eta=eta,
        delta=float(delta),
        tau_ratio=tau_ratio,
        gamma_gap=log_threshold_from_ratio(tau_ratio),
        rho_required=required_squared_alignment(eta, tau_ratio),
        k_gaussian=k_gaussian,
        kappa=kappa,
        certification_margin=kappa / k_gaussian,
        guaranteed_ratio=guaranteed_ritz_ratio(eta, k_gaussian),
        certified=k_gaussian < kappa,
    )


def minimum_certifiable_oversampling(
    d, r_star, eta, delta, tau_ratio, *, max_p=None
):
    """Return the smallest integer p certified by Theorem 15A, if one exists.

    The search is finite because a Gaussian sketch must satisfy
    ``r_star + p <= d``.  Returning ``None`` means that this sufficient bound
    certifies no admissible ``p``; it does not prove detection impossible.
    """
    d = _positive_integer("d", d)
    r_star = _positive_integer("r_star", r_star)
    if r_star + 4 > d:
        raise ValueError("The Gaussian theorem requires room for p >= 4.")
    upper_p = d - r_star
    if max_p is not None:
        upper_p = min(upper_p, _positive_integer("max_p", max_p, minimum=4))

    for p in range(4, upper_p + 1):
        if certify_gaussian_knee(
            d, r_star, p, eta, delta, tau_ratio
        ).certified:
            return p
    return None
