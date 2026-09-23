# Phase 1B-A: Budget-Aware Direct Rademacher-Risk Certification

## Executive verdict

**NET BENEFIT + TAIL-INSURANCE TRADEOFF**

This is a preregistered empirical verdict conditional on the frozen Phase 1A orientations and paths. It is not a theorem-level confidence certificate.

## Accounting result

**PROVED.** Nested construction and cached higher-prefix products make the committed construction cost

$$
c_{\rm pre}=\max\{q_0+r_0,q_a+r_a\}.
$$

After $s$ certification queries, both final bases share

$$
\ell_{\rm paid}=m-c_{\rm pre}-s.
$$

If the original baseline numerator is positive, returning to that basis after paying cannot restore its original risk because $\ell_{\rm paid}<\ell_0$.

## Primary result

The primary selected/original mean-risk ratio is **0.036329**, with conditional 95% percentile interval **[0.014628, 0.639830]**. The paid-oracle ratio is **0.035411**, while always falling back after paying has ratio **1.172197**.

| $r_\star$ | numerator threshold | paid fallback | paid oracle | empirical | median path | paths harmed |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 0.880 | 1.13636 | 0.00129155 | 0.00131495 | 1.13636 | 96.00% |
| 15 | 0.862 | 1.16071 | 2.1671e-05 | 2.20885e-05 | 1.16071 | 96.00% |
| 30 | 0.820 | 1.21951 | 0.104919 | 0.10765 | 1.21951 | 95.50% |

## Certification-cost sensitivity

| $s$ | paid fallback | paid oracle | empirical | accept | abstain |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.0513 | 0.0309 | 0.0356 | 7.10% | 84.20% |
| 8 | 1.0886 | 0.0323 | 0.0339 | 4.41% | 91.47% |
| 16 | 1.1722 | 0.0354 | 0.0363 | 4.19% | 94.11% |
| 32 | 1.3875 | 0.0439 | 0.0447 | 4.15% | 95.40% |

## Interpretation

**EMPIRICALLY ESTABLISHED.** The paid oracle separates architecture economics from decision quality. The empirical selector is evaluated only after candidate construction and certification are charged. Paid-order truth compares $\sigma_a^2$ and $\sigma_0^2$; net-benefit truth compares the complete paid candidate with never starting the procedure.

**EMPIRICAL DECISION RULE.** The multiplicative numerator guard is not a finite-sample certificate.

**OPEN.** A simultaneous confidence theorem and a genuinely online schedule remain open. Whether theorem work is warranted is determined by the paid-oracle and empirical results above.

## Limitations

The analysis reuses frozen Phase 1A probes and paths and performs no new matrix-vector experiment. Bootstrap intervals are conditional on that finite frozen population. The original comparator is the baseline action in each frozen adjacent pair, not a claim about every Standard Hutch++ allocation.

The complete preregistered secondary grid is retained. It contains **256** summary rows whose paid residual capacity is nonpositive; their paid risks are undefined and they are not silently dropped or used to renormalize equal-rank conclusions. The primary configuration is feasible in every rank.
