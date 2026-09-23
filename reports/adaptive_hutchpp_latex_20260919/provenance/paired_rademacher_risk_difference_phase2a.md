# Phase 2A: Direct Paired Rademacher Risk-Difference Audit

**Verdict:** `PAIRING SIGNAL GO`

## Executive result

This artifact-only continuation study issued no new matrix--vector experiment.
It tested whether the same certification probes reduce uncertainty in

$$
\widehat\Delta_R=S_a^2/\ell_a-S_0^2/\ell_0
$$

relative to the exact-form independence variance benchmark
$V_{\rm ind}=\operatorname{Var}(S_a^2/\ell_a)+\operatorname{Var}(S_0^2/\ell_0)$.
The reported $V_{\rm ind}$ values are empirical plug-in estimates of that
mathematically exact independence formula.

At the frozen primary configuration, the equal-rank pairing variance ratio is
**0.151032** with conditional path-bootstrap interval
**[0.137069, 0.166735]**. The corresponding
covariance contribution is **0.848968** with interval
**[0.833785, 0.862865]**.

The three ranks contain 37 net-beneficial
and 563 net-harmful paths in total. Top-5%
true-better eligibility by rank is
{5: 10, 15: 10, 30: 10}.

## What the GO verdict does and does not mean

The preregistered aggregate gate passes decisively: the common-probe statistic
retains only about 15.1% of the plug-in
independence-benchmark variance. The within-batch shifted/decorrelated
comparator has equal-rank mean ratio
**0.999563**,
close to one, which supports the interpretation that the original same-probe
alignment is responsible for the aggregate cancellation.

The gain is not uniform. Net-beneficial paths have mean pairing ratio
**0.804673**, whereas net-harmful paths
have mean ratio **0.108075**. The top-5%
catastrophic paths have mean ratio
**0.899565**, compared with
**0.111635** on ordinary paths. Thus the
global GO establishes a strong paired signal, but the cancellation is weakest
precisely on the rare beneficial tail paths that motivate certification. This
tail limitation must be carried into Phase 2B rather than hidden by the
aggregate verdict.

## Primary results by rank

| $r_\star$ | paths | truly better | pairing ratio | covariance contribution |
|---:|---:|---:|---:|---:|
| 5 | 200 | 11 | 0.205836 | 0.794164 |
| 15 | 200 | 14 | 0.132231 | 0.867769 |
| 30 | 200 | 12 | 0.115029 | 0.884971 |

The pathwise ratios are computed first within each frozen path, then averaged
within rank, then averaged equally across ranks. Individual certification
repetitions are not treated as independent research paths.

## Evidence classification

- `PROVED`: conditional unbiasedness, paired Hoeffding decomposition, and exact
  variance/covariance identities in the frozen plan and tested module.
- `EMPIRICALLY ESTABLISHED`: the reported covariance and variance-ratio results
  conditional on the frozen orientations and path population.
- `DESCRIPTIVE COMPARATOR`: the within-batch cyclic shift; it is not called an
  independent sequence. $V_{\rm ind}$ remains the mathematical independence
  benchmark.
- `OPEN`: a finite-sample one-sided radius for $\Delta_R$ and every online
  allocator consequence.

## Accounting scope

The formula $c_{\rm pre}=\max\{q_a+r_a,q_0+r_0\}$ is used only because the
primary frozen actions are nested shared prefixes. Other architectures must use
their actual committed construction-query count from a query ledger.

Every one of the 43,200 Phase 2A accounting rows was cross-checked against the
frozen Phase 1B table. Infeasible rows remain in the artifact with an explicit
flag; no denominator was silently changed or repaired.

## Recommendation

The preregistered gate authorizes a separate Phase 2B one-sided theorem
feasibility study for the direct paired difference. It does not establish a
finite-sample radius, and this report does not implement or authorize an
allocator.
