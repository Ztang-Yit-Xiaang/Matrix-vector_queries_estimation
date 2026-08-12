import numpy as np
import scipy.linalg as la

class MatVecOracle:
    """
    Wrapper around a matrix or matrix-vector product function.
    Tracks every single matrix-vector query.
    """
    def __init__(self, A_or_fn, d=None):
        if callable(A_or_fn):
            self.fn = A_or_fn
            self.d = d
        else:
            self.A = A_or_fn
            self.d = A_or_fn.shape[0]
            self.fn = lambda V: self.A @ V
        self.query_count = 0

    def reset_query_count(self):
        self.query_count = 0

    def __call__(self, V):
        """
        Computes A @ V for a vector (d,) or matrix (d, k).
        Increments query_count by k (number of columns).
        """
        V = np.asarray(V)
        if V.ndim == 1:
            self.query_count += 1
            return self.fn(V)
        elif V.ndim == 2:
            self.query_count += V.shape[1]
            try:
                return self.fn(V)
            except Exception:
                return np.column_stack([self.fn(V[:, j]) for j in range(V.shape[1])])
        else:
            raise ValueError(f"Expected 1D or 2D array, got {V.ndim}D array")

def _rank_aware_qr(W, reference_scale=None, rtol=1e-12, atol=0.0):
    """
    Return an orthonormal basis for the numerically significant columns of W.
    
    Parameters:
    -----------
    W : np.ndarray
        Matrix of size (d, k).
    reference_scale : float or None
        Scale relative to original unprojected matrix (e.g. Frobenius norm).
        Prevents numerical dust from being misidentified as true basis directions.
    rtol : float
        Relative tolerance multiplier.
    atol : float
        Absolute tolerance floor.
    """
    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array.")
    if W.shape[1] == 0:
        return np.empty((W.shape[0], 0), dtype=W.dtype), 0
    if not np.all(np.isfinite(W)):
        raise ValueError("W contains NaN or infinite values.")

    Q_full, R_full, _ = la.qr(W, mode='economic', pivoting=True)
    diag_R = np.abs(np.diag(R_full))

    if reference_scale is None:
        reference_scale = float(la.norm(W, ord='fro'))

    cutoff = max(float(atol), float(rtol) * float(reference_scale))
    rank = int(np.count_nonzero(diag_R > cutoff))

    return Q_full[:, :rank], rank

def Hutchinson(oracle, m, d, rng=None):
    """
    Classical Hutchinson trace estimator.
    Queries: m
    """
    if rng is None:
        rng = np.random.default_rng()
    
    G = rng.choice([-1.0, 1.0], size=(d, m))
    AG = oracle(G)
    return float(np.sum(G * AG)) / m

def Hutch_pplus(oracle, m, d, rng=None):
    """
    Corrected Hutch++ estimator with rank-aware QR and double residual projection.
    Queries: m
    """
    if rng is None:
        rng = np.random.default_rng()
        
    m1 = m // 3
    m2 = m // 3
    m3 = m - m1 - m2  # Ensures m1 + m2 + m3 == m
    
    S = rng.choice([-1.0, 1.0], size=(d, m1))
    AS = oracle(S)
    
    scale_AS = float(la.norm(AS, ord='fro'))
    Q, k = _rank_aware_qr(AS, reference_scale=scale_AS)
    
    AQ = oracle(Q) if k > 0 else np.empty((d, 0), dtype=AS.dtype)
    m3_eff = m3 + (m1 - k)
    
    G = rng.choice([-1.0, 1.0], size=(d, m3_eff))
    B_G = G - Q @ (Q.T @ G) if k > 0 else G
    ABG = oracle(B_G)
    
    trace_low_rank = float(np.sum(Q * AQ)) if k > 0 else 0.0
    trace_residual = float(np.sum(B_G * ABG)) / m3_eff
    
    return trace_low_rank + trace_residual

def Gaussian_Hutch_pplus(oracle, m, d, rng=None):
    """
    Corrected Gaussian-Hutch++ estimator with rank-aware QR and double residual projection.
    Queries: m
    """
    if rng is None:
        rng = np.random.default_rng()
        
    m1 = (m + 2) // 4
    m2 = (m + 2) // 4
    m3 = m - m1 - m2  # m3 == (m - 2) // 2 when m is even
    
    S = rng.normal(loc=0.0, scale=1.0, size=(d, m1))
    AS = oracle(S)
    
    scale_AS = float(la.norm(AS, ord='fro'))
    Q, k = _rank_aware_qr(AS, reference_scale=scale_AS)
    
    AQ = oracle(Q) if k > 0 else np.empty((d, 0), dtype=AS.dtype)
    m3_eff = m3 + (m1 - k)
    
    G = rng.choice([-1.0, 1.0], size=(d, m3_eff))
    B_G = G - Q @ (Q.T @ G) if k > 0 else G
    ABG = oracle(B_G)
    
    trace_low_rank = float(np.sum(Q * AQ)) if k > 0 else 0.0
    trace_residual = float(np.sum(B_G * ABG)) / m3_eff
    
    return trace_low_rank + trace_residual

def NA_Hutch_pplus(oracle, m, d, c1=0.25, c2=0.5, c3=0.25, rng=None):
    """
    Non-Adaptive Hutch++ (NA-Hutch++) estimator.
    Queries: m
    """
    if rng is None:
        rng = np.random.default_rng()
        
    m1 = int(m * c1)
    m2 = int(m * c2)
    m3 = m - m1 - m2  # m1 + m2 + m3 == m
    
    S = rng.choice([-1.0, 1.0], size=(d, m1))
    R = rng.choice([-1.0, 1.0], size=(d, m2))
    G = rng.choice([-1.0, 1.0], size=(d, m3))
    
    W = oracle(S)
    Z = oracle(R)
    AG = oracle(G)
    
    STZ = S.T @ Z
    STZ_pinv = la.pinv(STZ)
    
    term1 = float(np.trace(STZ_pinv @ (W.T @ Z)))
    term2 = float(np.sum(G * AG))
    term3 = float(np.trace(G.T @ (Z @ (STZ_pinv @ (W.T @ G)))))
    
    return term1 + (1.0 / m3) * (term2 - term3)

def Adaptive_Hutch_pplus(
    oracle,
    m,
    d,
    b=10,
    min_fit_points=4,
    r2_min=0.8,
    probe_mode='rademacher',
    rng=None,
    return_diagnostics=False
):
    """
    Corrected Pilot-Driven Adaptive Hutch++ Trace Estimator (Hard Allocation).
    """
    if rng is None:
        rng = np.random.default_rng()

    probe_mode = probe_mode.lower()
    if probe_mode not in {'gaussian', 'rademacher'}:
        raise ValueError("probe_mode must be 'gaussian' or 'rademacher'.")

    queries_before = oracle.query_count

    q_max = min(d, (m - 2) // 2)
    min_budget = 2 * b + 2
    if m < min_budget or b > q_max:
        q_target = min(q_max, max(b, m // 3)) if q_max >= b else b
        fallback_used = True
        fallback_reason = "budget_too_small_for_pilot"
        c_hat = None
        r_squared = 0.0
        fit_is_reliable = False
    else:
        fallback_used = False
        fallback_reason = None
        c_hat = None
        r_squared = 0.0
        fit_is_reliable = False

    # PHASE 1: Pilot Stage
    if not fallback_used:
        S_0 = rng.choice([-1.0, 1.0], size=(d, b))
        W_0 = oracle(S_0)
        scale_0 = float(la.norm(W_0, ord='fro'))
        
        Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
        Z_0 = oracle(Q_0) if r_0 > 0 else np.empty((d, 0), dtype=W_0.dtype)
        
        pos_ritz = np.empty(0)
        if r_0 >= min_fit_points:
            M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
            ritz_vals = la.eigvalsh(M_0)
            ritz_vals = np.sort(ritz_vals)[::-1]
            
            theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0
            if theta_max > 0.0:
                ritz_cutoff = 1e-12 * theta_max
                pos_ritz = ritz_vals[ritz_vals > ritz_cutoff]
                
                if len(pos_ritz) >= min_fit_points:
                    j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
                    log_j = np.log(j_indices)
                    log_theta = np.log(pos_ritz)
                    
                    slope, intercept = np.polyfit(log_j, log_theta, 1)
                    c_hat_candidate = float(max(0.0, -slope))
                    
                    fit_vals = intercept + slope * log_j
                    ss_res = np.sum((log_theta - fit_vals) ** 2)
                    ss_tot = np.sum((log_theta - np.mean(log_theta)) ** 2)
                    r2_candidate = float(1.0 - (ss_res / (ss_tot + 1e-12)))
                    
                    if r2_candidate >= r2_min:
                        c_hat = c_hat_candidate
                        r_squared = r2_candidate
                        fit_is_reliable = True

    # PHASE 2: Finite-Sum Oracle Risk Allocation
    if not fallback_used:
        if fit_is_reliable:
            i_vals = np.arange(1, d + 1, dtype=np.float64)
            w_i = i_vals ** (-2.0 * c_hat)
            T_cum = np.zeros(d + 1, dtype=np.float64)
            T_cum[:-1] = np.cumsum(w_i[::-1])[::-1]
            
            best_risk = float("inf")
            q_target = b
            
            for q_cand in range(b, q_max + 1):
                l_cand = m - 2 * q_cand
                if l_cand <= 0:
                    continue
                risk = 2.0 * T_cum[q_cand] / l_cand
                if risk < best_risk:
                    best_risk = risk
                    q_target = q_cand
        else:
            q_target = min(q_max, max(b, m // 3))
            fallback_used = True
            fallback_reason = "unreliable_power_law_fit"

    # PHASE 3: Pilot Basis Extension
    k_extra = q_target - b
    if k_extra > 0:
        S_1 = rng.choice([-1.0, 1.0], size=(d, k_extra))
        W_1 = oracle(S_1)
        scale_1 = float(la.norm(W_1, ord='fro'))
        
        if r_0 > 0:
            W1_tilde = W_1 - Q_0 @ (Q_0.T @ W_1)
            W1_tilde = W1_tilde - Q_0 @ (Q_0.T @ W1_tilde)
        else:
            W1_tilde = W_1

        Q_1, r_1 = _rank_aware_qr(W1_tilde, reference_scale=scale_1)
        
        if r_1 > 0 and r_0 > 0:
            Q_1 = Q_1 - Q_0 @ (Q_0.T @ Q_1)
            Q_1, r_1 = _rank_aware_qr(Q_1, reference_scale=1.0)
            
        Z_1 = oracle(Q_1) if r_1 > 0 else np.empty((d, 0), dtype=W_1.dtype)
        
        if r_0 > 0 and r_1 > 0:
            Q = np.column_stack([Q_0, Q_1])
            Z = np.column_stack([Z_0, Z_1])
        elif r_0 > 0:
            Q, Z = Q_0, Z_0
        else:
            Q, Z = Q_1, Z_1
    else:
        Q, Z = Q_0, Z_0
        r_1 = 0

    r_actual = Q.shape[1]

    # PHASE 4: Exact Budget Accounting
    ell_eff = m - q_target - r_actual
    if ell_eff < 2:
        raise RuntimeError(
            f"Invalid query allocation: fewer than 2 residual probes remain. "
            f"m={m}, q_target={q_target}, r_actual={r_actual}, ell_eff={ell_eff}"
        )

    # PHASE 5: Double Residual Projection
    if probe_mode == 'gaussian':
        G = rng.normal(loc=0.0, scale=1.0, size=(d, ell_eff))
    else:
        G = rng.choice([-1.0, 1.0], size=(d, ell_eff))
        
    if r_actual > 0:
        RG = G - Q @ (Q.T @ G)
    else:
        RG = G
        
    ARG = oracle(RG)
    
    queries_used = oracle.query_count - queries_before
    if queries_used != m:
        raise RuntimeError(
            f"Query budget mismatch: expected {m} queries, actually used {queries_used}."
        )

    trace_low = float(np.sum(Q * Z)) if r_actual > 0 else 0.0
    trace_res = float(np.sum(RG * ARG)) / ell_eff
    
    trace_est = trace_low + trace_res
    
    if return_diagnostics:
        diag = {
            "c_hat": c_hat,
            "r_squared": r_squared,
            "q_target": q_target,
            "r_actual": r_actual,
            "ell_eff": ell_eff,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "probe_mode": probe_mode
        }
        return trace_est, diag
        
    return trace_est

def Adaptive_Hutch_pplus_Soft(
    oracle,
    m,
    d,
    b=10,
    min_fit_points=4,
    r2_min=0.8,
    probe_mode='rademacher',
    rng=None,
    return_diagnostics=False
):
    """
    Soft Safe Adaptive Hutch++ Trace Estimator.
    Computes a convex combination target rank:
        q_soft = round((1 - gamma) * q_0 + gamma * q_adapt)
    where q_0 = floor(m / 3) is the safe standard Hutch++ split,
    q_adapt = argmin_q F_hat(q) is the pilot adaptive choice,
    and gamma in [0, 1] is confidence derived from R^2 and Ritz spectral concentration C_h.
    """
    if rng is None:
        rng = np.random.default_rng()

    probe_mode = probe_mode.lower()
    if probe_mode not in {'gaussian', 'rademacher'}:
        raise ValueError("probe_mode must be 'gaussian' or 'rademacher'.")

    queries_before = oracle.query_count
    q_max = min(d, (m - 2) // 2)
    q_0 = min(q_max, max(b, m // 3)) if q_max >= b else b
    min_budget = 2 * b + 2

    if m < min_budget or b > q_max:
        q_target = q_0
        gamma = 0.0
        fallback_used = True
        fallback_reason = "budget_too_small_for_pilot"
        c_hat = None
        r_squared = 0.0
        fit_is_reliable = False
        q_adapt = q_0
    else:
        fallback_used = False
        fallback_reason = None
        c_hat = None
        r_squared = 0.0
        fit_is_reliable = False
        gamma = 0.0
        q_adapt = q_0

    # PHASE 1: Pilot Stage
    if not fallback_used:
        S_0 = rng.choice([-1.0, 1.0], size=(d, b))
        W_0 = oracle(S_0)
        scale_0 = float(la.norm(W_0, ord='fro'))
        Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
        Z_0 = oracle(Q_0) if r_0 > 0 else np.empty((d, 0), dtype=W_0.dtype)

        if r_0 >= min_fit_points:
            M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
            ritz_vals = la.eigvalsh(M_0)[::-1]
            theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0

            if theta_max > 0.0:
                ritz_cutoff = 1e-12 * theta_max
                pos_ritz = ritz_vals[ritz_vals > ritz_cutoff]

                if len(pos_ritz) >= min_fit_points:
                    j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
                    log_j = np.log(j_indices)
                    log_theta = np.log(pos_ritz)

                    slope, intercept = np.polyfit(log_j, log_theta, 1)
                    c_hat_candidate = float(max(0.0, -slope))

                    fit_vals = intercept + slope * log_j
                    ss_res = np.sum((log_theta - fit_vals) ** 2)
                    ss_tot = np.sum((log_theta - np.mean(log_theta)) ** 2)
                    r2_candidate = float(1.0 - (ss_res / (ss_tot + 1e-12)))

                    if r2_candidate >= r2_min:
                        c_hat = c_hat_candidate
                        r_squared = r2_candidate
                        fit_is_reliable = True

                        # Compute Spectral Concentration C_h
                        h = min(4, len(pos_ritz))
                        c_h = float(np.sum(pos_ritz[:h]**2) / (np.sum(pos_ritz**2) + 1e-12))
                        # Confidence weight gamma
                        gamma = float(np.clip(r_squared * c_h, 0.0, 1.0))

    # PHASE 2: Convex Combination Allocation
    if not fallback_used:
        if fit_is_reliable:
            i_vals = np.arange(1, d + 1, dtype=np.float64)
            w_i = i_vals ** (-2.0 * c_hat)
            T_cum = np.zeros(d + 1, dtype=np.float64)
            T_cum[:-1] = np.cumsum(w_i[::-1])[::-1]

            best_risk = float("inf")
            q_adapt = b
            for q_cand in range(b, q_max + 1):
                l_cand = m - 2 * q_cand
                if l_cand <= 0:
                    continue
                risk = 2.0 * T_cum[q_cand] / l_cand
                if risk < best_risk:
                    best_risk = risk
                    q_adapt = q_cand

            # Soft allocation combination
            q_soft_float = (1.0 - gamma) * float(q_0) + gamma * float(q_adapt)
            q_target = int(np.clip(round(q_soft_float), b, q_max))
        else:
            q_target = q_0
            gamma = 0.0
            fallback_used = True
            fallback_reason = "unreliable_power_law_fit"

    # PHASE 3: Pilot Extension
    k_extra = q_target - b
    if k_extra > 0:
        S_1 = rng.choice([-1.0, 1.0], size=(d, k_extra))
        W_1 = oracle(S_1)
        scale_1 = float(la.norm(W_1, ord='fro'))

        if r_0 > 0:
            W1_tilde = W_1 - Q_0 @ (Q_0.T @ W_1)
            W1_tilde = W1_tilde - Q_0 @ (Q_0.T @ W1_tilde)
        else:
            W1_tilde = W_1

        Q_1, r_1 = _rank_aware_qr(W1_tilde, reference_scale=scale_1)

        if r_1 > 0 and r_0 > 0:
            Q_1 = Q_1 - Q_0 @ (Q_0.T @ Q_1)
            Q_1, r_1 = _rank_aware_qr(Q_1, reference_scale=1.0)

        Z_1 = oracle(Q_1) if r_1 > 0 else np.empty((d, 0), dtype=W_1.dtype)

        if r_0 > 0 and r_1 > 0:
            Q = np.column_stack([Q_0, Q_1])
            Z = np.column_stack([Z_0, Z_1])
        elif r_0 > 0:
            Q, Z = Q_0, Z_0
        else:
            Q, Z = Q_1, Z_1
    else:
        Q, Z = Q_0, Z_0
        r_1 = 0

    r_actual = Q.shape[1]

    # PHASE 4: Exact Budget Accounting
    ell_eff = m - q_target - r_actual
    if ell_eff < 2:
        raise RuntimeError(
            f"Invalid query allocation: fewer than 2 residual probes remain. "
            f"m={m}, q_target={q_target}, r_actual={r_actual}, ell_eff={ell_eff}"
        )

    # PHASE 5: Double Residual Projection
    if probe_mode == 'gaussian':
        G = rng.normal(loc=0.0, scale=1.0, size=(d, ell_eff))
    else:
        G = rng.choice([-1.0, 1.0], size=(d, ell_eff))

    if r_actual > 0:
        RG = G - Q @ (Q.T @ G)
    else:
        RG = G

    ARG = oracle(RG)

    queries_used = oracle.query_count - queries_before
    if queries_used != m:
        raise RuntimeError(
            f"Query budget mismatch: expected {m} queries, actually used {queries_used}."
        )

    trace_low = float(np.sum(Q * Z)) if r_actual > 0 else 0.0
    trace_res = float(np.sum(RG * ARG)) / ell_eff
    trace_est = trace_low + trace_res

    if return_diagnostics:
        diag = {
            "c_hat": c_hat,
            "r_squared": r_squared,
            "gamma": gamma,
            "q_0": q_0,
            "q_adapt": q_adapt,
            "q_target": q_target,
            "r_actual": r_actual,
            "ell_eff": ell_eff,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "probe_mode": probe_mode
        }
        return trace_est, diag

    return trace_est

def Adaptive_Hutch_pplus_GaussianResidual(oracle, m, d, b=10, rng=None, return_diagnostics=False):
    """Adaptive Hutch++ with Gaussian residual probes (theory-aligned mode)."""
    return Adaptive_Hutch_pplus(
        oracle, m, d, b=b, probe_mode='gaussian', rng=rng, return_diagnostics=return_diagnostics
    )

def Adaptive_Hutch_pplus_RademacherResidual(oracle, m, d, b=10, rng=None, return_diagnostics=False):
    """Adaptive Hutch++ with Rademacher residual probes (practical mode)."""
    return Adaptive_Hutch_pplus(
        oracle, m, d, b=b, probe_mode='rademacher', rng=rng, return_diagnostics=return_diagnostics
    )

def Adaptive_Hutch_pplus_Soft_RademacherResidual(oracle, m, d, b=10, rng=None, return_diagnostics=False):
    """Soft Safe Adaptive Hutch++ with Rademacher residual probes."""
    return Adaptive_Hutch_pplus_Soft(
        oracle, m, d, b=b, probe_mode='rademacher', rng=rng, return_diagnostics=return_diagnostics
    )

def Adaptive_Hutch_pplus_ModelAveraged(
    oracle,
    m,
    d,
    b=10,
    min_fit_points=4,
    temperature=0.1,
    use_safety_shrinkage=True,
    probe_mode='rademacher',
    rng=None,
    return_diagnostics=False
):
    """
    Uncertainty-Aware Model-Averaged Adaptive Hutch++ Trace Estimator.
    
    1. Fits candidate tail models T_P(q) (Power-Law), T_E(q) (Exponential), T_S(q) (Step/Gap) to pilot Ritz values.
    2. Computes predictive loss L_j = (1/r_0) * sum_k (ln theta_k - ln theta_hat_k^{(j)})^2.
    3. Computes model probability weights w_j = softmax(-L_j / temperature).
    4. Combines tail energy: T_mix(q) = w_P T_P(q) + w_E T_E(q) + w_S T_S(q).
    5. Computes predicted risk curve R_mix(q) = 2 * T_mix(q) / (m - 2*q) and selects q_adapt = argmin R_mix(q).
    6. (If use_safety_shrinkage=True) Computes model agreement dispersion D_q and sets two-layer confidence gamma.
       Sets q_final = round((1 - gamma) * q_0 + gamma * q_adapt).
    """
    if rng is None:
        rng = np.random.default_rng()

    probe_mode = probe_mode.lower()
    if probe_mode not in {'gaussian', 'rademacher'}:
        raise ValueError("probe_mode must be 'gaussian' or 'rademacher'.")

    queries_before = oracle.query_count
    q_max = min(d, (m - 2) // 2)
    q_0 = min(q_max, max(b, m // 3)) if q_max >= b else b
    min_budget = 2 * b + 2

    if m < min_budget or b > q_max:
        q_target = q_0
        gamma = 0.0
        weights = {"power": 0.333, "exp": 0.333, "step": 0.334}
        fallback_used = True
        fallback_reason = "budget_too_small_for_pilot"
        q_adapt = q_0
    else:
        fallback_used = False
        fallback_reason = None
        gamma = 0.0
        q_adapt = q_0
        weights = {"power": 0.333, "exp": 0.333, "step": 0.334}

    # PHASE 1: Pilot Stage & Model Averaging
    if not fallback_used:
        S_0 = rng.choice([-1.0, 1.0], size=(d, b))
        W_0 = oracle(S_0)
        scale_0 = float(la.norm(W_0, ord='fro'))
        
        Q_0, r_0 = _rank_aware_qr(W_0, reference_scale=scale_0)
        Z_0 = oracle(Q_0) if r_0 > 0 else np.empty((d, 0), dtype=W_0.dtype)

        if r_0 >= min_fit_points:
            M_0 = 0.5 * (Q_0.T @ Z_0 + Z_0.T @ Q_0)
            ritz_vals = la.eigvalsh(M_0)[::-1]
            theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0

            if theta_max > 0.0:
                ritz_cutoff = 1e-12 * theta_max
                pos_ritz = ritz_vals[ritz_vals > ritz_cutoff]

                if len(pos_ritz) >= min_fit_points:
                    j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
                    log_j = np.log(j_indices)
                    log_theta = np.log(pos_ritz)
                    i_vals = np.arange(1, d + 1, dtype=np.float64)

                    # Model 1: Power-Law
                    slope_p, intercept_p = np.polyfit(log_j, log_theta, 1)
                    c_hat = float(max(0.0, -slope_p))
                    fit_p = intercept_p + slope_p * log_j
                    loss_p = float(np.mean((log_theta - fit_p) ** 2))

                    w_pow = i_vals ** (-2.0 * c_hat)
                    T_cum_p = np.cumsum(w_pow[::-1])[::-1]
                    q_p = b
                    best_risk_p = float("inf")
                    for q_cand in range(b, q_max + 1):
                        l_cand = m - 2 * q_cand
                        if l_cand > 0:
                            risk = 2.0 * T_cum_p[q_cand - 1] / l_cand
                            if risk < best_risk_p:
                                best_risk_p = risk
                                q_p = q_cand

                    # Model 2: Exponential
                    slope_e, intercept_e = np.polyfit(j_indices, log_theta, 1)
                    alpha_hat = float(max(1e-5, -slope_e))
                    fit_e = intercept_e + slope_e * j_indices
                    loss_e = float(np.mean((log_theta - fit_e) ** 2))

                    w_exp = np.exp(-2.0 * alpha_hat * i_vals)
                    T_cum_e = np.cumsum(w_exp[::-1])[::-1]
                    q_e = b
                    best_risk_e = float("inf")
                    for q_cand in range(b, q_max + 1):
                        l_cand = m - 2 * q_cand
                        if l_cand > 0:
                            risk = 2.0 * T_cum_e[q_cand - 1] / l_cand
                            if risk < best_risk_e:
                                best_risk_e = risk
                                q_e = q_cand

                    # Model 3: Step/Gap Model
                    ratios = pos_ritz[:-1] / (pos_ritz[1:] + 1e-12)
                    max_ratio = float(np.max(ratios)) if len(ratios) > 0 else 1.0
                    r_elbow = int(np.argmax(ratios) + 1) if len(ratios) > 0 else 1
                    loss_s = 0.05 if max_ratio > 3.0 else 2.0
                    q_s = min(q_max, max(b, r_elbow + 2))

                    tail_s_val = float((pos_ritz[r_elbow] ** 2) if r_elbow < len(pos_ritz) else pos_ritz[-1]**2)
                    T_cum_s = np.zeros(d, dtype=np.float64)
                    for idx_i in range(d):
                        curr_q = idx_i + 1
                        if curr_q < r_elbow:
                            T_cum_s[idx_i] = (r_elbow - curr_q) * 1.0 + (d - r_elbow) * tail_s_val
                        else:
                            T_cum_s[idx_i] = (d - curr_q) * tail_s_val

                    # Compute Softmax Weights with Calibrated Temperature
                    losses = np.array([loss_p, loss_e, loss_s], dtype=np.float64)
                    logits = -losses / max(1e-4, float(temperature))
                    exp_logits = np.exp(logits - np.max(logits))
                    w_arr = exp_logits / np.sum(exp_logits)

                    w_P, w_E, w_S = float(w_arr[0]), float(w_arr[1]), float(w_arr[2])
                    weights = {"power": w_P, "exp": w_E, "step": w_S}

                    # Combine Tail Energy T_mix(q)
                    T_mix = w_P * T_cum_p + w_E * T_cum_e + w_S * T_cum_s

                    # Risk Minimization over R_mix(q)
                    best_risk_mix = float("inf")
                    q_adapt = b
                    for q_cand in range(b, q_max + 1):
                        l_cand = m - 2 * q_cand
                        if l_cand > 0:
                            risk_mix = 2.0 * T_mix[q_cand - 1] / l_cand
                            if risk_mix < best_risk_mix:
                                best_risk_mix = risk_mix
                                q_adapt = q_cand

                    # Compute Model Disagreement & Two-Layer Confidence gamma
                    D_q = w_P * abs(q_p - q_adapt) + w_E * abs(q_e - q_adapt) + w_S * abs(q_s - q_adapt)
                    avg_loss = w_P * loss_p + w_E * loss_e + w_S * loss_s
                    
                    if use_safety_shrinkage:
                        gamma = float(np.clip(np.exp(-avg_loss) * np.exp(-D_q / 25.0), 0.0, 1.0))
                        q_soft_float = (1.0 - gamma) * float(q_0) + gamma * float(q_adapt)
                        q_target = int(np.clip(round(q_soft_float), b, q_max))
                    else:
                        gamma = 1.0
                        q_target = q_adapt
                else:
                    q_target = q_0
                    fallback_used = True
                    fallback_reason = "insufficient_positive_ritz_values"
        else:
            q_target = q_0
            fallback_used = True
            fallback_reason = "insufficient_pilot_rank"


    # PHASE 3: Pilot Basis Extension
    k_extra = q_target - b
    if k_extra > 0:
        S_1 = rng.choice([-1.0, 1.0], size=(d, k_extra))
        W_1 = oracle(S_1)
        scale_1 = float(la.norm(W_1, ord='fro'))

        if r_0 > 0:
            W1_tilde = W_1 - Q_0 @ (Q_0.T @ W_1)
            W1_tilde = W1_tilde - Q_0 @ (Q_0.T @ W1_tilde)
        else:
            W1_tilde = W_1

        Q_1, r_1 = _rank_aware_qr(W1_tilde, reference_scale=scale_1)

        if r_1 > 0 and r_0 > 0:
            Q_1 = Q_1 - Q_0 @ (Q_0.T @ Q_1)
            Q_1, r_1 = _rank_aware_qr(Q_1, reference_scale=1.0)

        Z_1 = oracle(Q_1) if r_1 > 0 else np.empty((d, 0), dtype=W_1.dtype)

        if r_0 > 0 and r_1 > 0:
            Q = np.column_stack([Q_0, Q_1])
            Z = np.column_stack([Z_0, Z_1])
        elif r_0 > 0:
            Q, Z = Q_0, Z_0
        else:
            Q, Z = Q_1, Z_1
    else:
        Q, Z = Q_0, Z_0
        r_1 = 0

    r_actual = Q.shape[1]

    # PHASE 4: Exact Budget Accounting
    ell_eff = m - q_target - r_actual
    if ell_eff < 2:
        raise RuntimeError(
            f"Invalid query allocation: fewer than 2 residual probes remain. "
            f"m={m}, q_target={q_target}, r_actual={r_actual}, ell_eff={ell_eff}"
        )

    # PHASE 5: Double Residual Projection
    if probe_mode == 'gaussian':
        G = rng.normal(loc=0.0, scale=1.0, size=(d, ell_eff))
    else:
        G = rng.choice([-1.0, 1.0], size=(d, ell_eff))

    if r_actual > 0:
        RG = G - Q @ (Q.T @ G)
    else:
        RG = G

    ARG = oracle(RG)

    queries_used = oracle.query_count - queries_before
    if queries_used != m:
        raise RuntimeError(
            f"Query budget mismatch: expected {m} queries, actually used {queries_used}."
        )

    trace_low = float(np.sum(Q * Z)) if r_actual > 0 else 0.0
    trace_res = float(np.sum(RG * ARG)) / ell_eff
    trace_est = trace_low + trace_res

    if return_diagnostics:
        diag = {
            "weights": weights,
            "gamma": gamma,
            "q_0": q_0,
            "q_adapt": q_adapt,
            "q_target": q_target,
            "r_actual": r_actual,
            "ell_eff": ell_eff,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "probe_mode": probe_mode
        }
        return trace_est, diag

    return trace_est


    if return_diagnostics:
        diag = {
            "weights": weights,
            "gamma": gamma,
            "q_0": q_0,
            "q_adapt": q_adapt,
            "q_target": q_target,
            "r_actual": r_actual,
            "ell_eff": ell_eff,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "probe_mode": probe_mode
        }
        return trace_est, diag

    return trace_est


def Adaptive_Hutch_pplus_SequentialPilot(
    oracle,
    m,
    d,
    b_0=8,
    delta_b=4,
    b_max=None,
    tau_gap=1.5,
    p_min=1,
    max_extrapolation_dist=40,
    probe_mode='rademacher',
    rng=None,
    return_diagnostics=False
):
    """
    Direction 1: Sequential / Adaptive Pilot Stopping Hutch++ Estimator (Upgraded).
    
    1. Starts with initial pilot size b_0.
    2. Sequentially acquires delta_b pilot queries.
    3. Horizon-Resolution & Multi-Condition Safe Stopping Rule:
       - Condition A (Allocation Stable): |q_curr - q_prev| <= 1.
       - Condition B (Horizon Resolution / Knee Detectability):
         Computes log gaps g_j = log(theta_j / theta_{j+1}).
         If peak gap g_{r_gap} >= tau_gap and post-gap count b - r_gap >= p_min,
         the knee is already resolved with p >= p_min directions.
       - Condition C (Extrapolation Distance Safe): q_curr - b_curr <= max_extrapolation_dist.
    4. Exact budget accounting identity: q_target + r_actual + ell_eff == m.
    """
    if rng is None:
        rng = np.random.default_rng()

    queries_before = oracle.query_count
    q_max = min(d, (m - 2) // 2)
    q_0 = min(q_max, max(b_0, m // 3)) if q_max >= b_0 else b_0

    if b_max is None:
        b_max = min(q_max // 2, max(b_0, (m - 10) // 4))

    b_curr = b_0
    Q_pilot = None
    Z_pilot = None
    q_prev = None
    stopped_early = False
    stop_reason = "max_pilot_reached"
    max_adjacent_log_gap = 0.0
    gap_location = 0
    post_gap_obs = 0

    # Sequential Pilot Acquisition Loop
    while b_curr <= b_max:
        if Q_pilot is None:
            S_chunk = rng.choice([-1.0, 1.0], size=(d, b_0))
            W_chunk = oracle(S_chunk)
            scale_chunk = float(la.norm(W_chunk, ord='fro'))
            Q_pilot, r_pilot = _rank_aware_qr(W_chunk, reference_scale=scale_chunk)
            Z_pilot = oracle(Q_pilot) if r_pilot > 0 else np.empty((d, 0), dtype=W_chunk.dtype)
        else:
            S_chunk = rng.choice([-1.0, 1.0], size=(d, delta_b))
            W_chunk = oracle(S_chunk)
            scale_chunk = float(la.norm(W_chunk, ord='fro'))
            
            W_tilde = W_chunk - Q_pilot @ (Q_pilot.T @ W_chunk)
            W_tilde = W_tilde - Q_pilot @ (Q_pilot.T @ W_tilde)
            
            Q_delta, r_delta = _rank_aware_qr(W_tilde, reference_scale=scale_chunk)
            if r_delta > 0:
                Q_delta = Q_delta - Q_pilot @ (Q_pilot.T @ Q_delta)
                Q_delta, r_delta = _rank_aware_qr(Q_delta, reference_scale=1.0)
                Z_delta = oracle(Q_delta)

                Q_pilot = np.column_stack([Q_pilot, Q_delta])
                Z_pilot = np.column_stack([Z_pilot, Z_delta])

        r_curr = Q_pilot.shape[1]
        if r_curr >= 4:
            M_curr = 0.5 * (Q_pilot.T @ Z_pilot + Z_pilot.T @ Q_pilot)
            ritz_vals = la.eigvalsh(M_curr)[::-1]
            theta_max = float(ritz_vals[0]) if len(ritz_vals) > 0 else 0.0

            if theta_max > 0.0:
                pos_ritz = ritz_vals[ritz_vals > 1e-12 * theta_max]
                if len(pos_ritz) >= 4:
                    j_indices = np.arange(1, len(pos_ritz) + 1, dtype=np.float64)
                    log_j = np.log(j_indices)
                    log_theta = np.log(pos_ritz)
                    i_vals = np.arange(1, d + 1, dtype=np.float64)

                    # Horizon Resolution Log-Gap Analysis
                    log_gaps = log_theta[:-1] - log_theta[1:]
                    max_adjacent_log_gap = float(np.max(log_gaps)) if len(log_gaps) > 0 else 0.0
                    gap_location = int(np.argmax(log_gaps) + 1) if len(log_gaps) > 0 else 0
                    post_gap_obs = len(pos_ritz) - gap_location

                    # Model 1: Power-Law
                    slope_p, _ = np.polyfit(log_j, log_theta, 1)
                    c_hat = float(max(0.0, -slope_p))
                    w_pow = i_vals ** (-2.0 * c_hat)
                    T_cum_p = np.cumsum(w_pow[::-1])[::-1]

                    # Model 2: Exponential
                    slope_e, _ = np.polyfit(j_indices, log_theta, 1)
                    alpha_hat = float(max(1e-5, -slope_e))
                    w_exp = np.exp(-2.0 * alpha_hat * i_vals)
                    T_cum_e = np.cumsum(w_exp[::-1])[::-1]

                    T_mix = 0.5 * T_cum_p + 0.5 * T_cum_e
                    best_risk = float("inf")
                    q_curr = b_curr
                    for q_cand in range(b_curr, q_max + 1):
                        l_cand = m - 2 * q_cand
                        if l_cand > 0:
                            risk = 2.0 * T_mix[q_cand - 1] / l_cand
                            if risk < best_risk:
                                best_risk = risk
                                q_curr = q_cand

                    # Horizon Resolution Rule
                    has_resolved_knee = (max_adjacent_log_gap >= tau_gap and post_gap_obs >= p_min)
                    extrapolation_safe = (q_curr - b_curr <= max_extrapolation_dist)
                    allocation_stable = (q_prev is not None and abs(q_curr - q_prev) <= 1)

                    if allocation_stable and extrapolation_safe:
                        if has_resolved_knee or (max_adjacent_log_gap < tau_gap):
                            stopped_early = True
                            stop_reason = f"horizon_resolved_stop_at_b={b_curr}"
                            q_adapt_final = q_curr
                            break

                    q_prev = q_curr

        b_curr += delta_b

    b_final = Q_pilot.shape[1] if Q_pilot is not None else b_0
    q_target = q_prev if q_prev is not None else q_0
    q_target = int(np.clip(q_target, b_final, q_max))

    # Phase 3: Basis Extension to q_target
    k_extra = q_target - b_final
    if k_extra > 0:
        S_ext = rng.choice([-1.0, 1.0], size=(d, k_extra))
        W_ext = oracle(S_ext)
        scale_ext = float(la.norm(W_ext, ord='fro'))

        W_ext_tilde = W_ext - Q_pilot @ (Q_pilot.T @ W_ext)
        W_ext_tilde = W_ext_tilde - Q_pilot @ (Q_pilot.T @ W_ext_tilde)

        Q_ext, r_ext = _rank_aware_qr(W_ext_tilde, reference_scale=scale_ext)
        if r_ext > 0:
            Q_ext = Q_ext - Q_pilot @ (Q_pilot.T @ Q_ext)
            Q_ext, r_ext = _rank_aware_qr(Q_ext, reference_scale=1.0)
            Z_ext = oracle(Q_ext)
            Q = np.column_stack([Q_pilot, Q_ext])
            Z = np.column_stack([Z_pilot, Z_ext])
        else:
            Q, Z = Q_pilot, Z_pilot
    else:
        Q, Z = Q_pilot, Z_pilot

    r_actual = Q.shape[1]

    # Phase 4 & 5: Residual Estimation & Exact Budget Accounting Identity: q_target + r_actual + ell_eff == m
    ell_eff = m - q_target - r_actual
    if ell_eff < 2:
        ell_eff = 2
        q_target = m - r_actual - ell_eff

    if probe_mode == 'gaussian':
        G = rng.normal(loc=0.0, scale=1.0, size=(d, ell_eff))
    else:
        G = rng.choice([-1.0, 1.0], size=(d, ell_eff))

    RG = G - Q @ (Q.T @ G) if r_actual > 0 else G
    ARG = oracle(RG)

    queries_used = oracle.query_count - queries_before
    if queries_used != m:
        raise RuntimeError(f"Sequential pilot query budget mismatch: expected {m}, got {queries_used}")

    tr_low = float(np.sum(Q * Z)) if r_actual > 0 else 0.0
    tr_res = float(np.sum(RG * ARG)) / ell_eff
    tr_est = tr_low + tr_res

    if return_diagnostics:
        diag = {
            "b_final": b_final,
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "q_target": q_target,
            "r_actual": r_actual,
            "ell_eff": ell_eff,
            "max_adjacent_log_gap": max_adjacent_log_gap,
            "gap_location": gap_location,
            "post_gap_observations": post_gap_obs,
            "extrapolation_distance": q_target - b_final
        }
        return tr_est, diag

    return tr_est




