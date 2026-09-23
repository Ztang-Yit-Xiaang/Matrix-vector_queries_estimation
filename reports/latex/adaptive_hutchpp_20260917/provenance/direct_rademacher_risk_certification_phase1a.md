# Phase 1A: Direct Realized Rademacher-Risk Certification

## Executive verdict

**QUALIFIED GO**

This is a preregistered empirical verdict for the frozen orientations and basis-path
population. It is not a theorem-level safety certificate.

## Exact estimand

**PROVED.** For a basis fixed before a fresh Rademacher probe,

$$
\sigma_Q^2=\operatorname{Var}(g^TR_QAR_Qg\mid Q)
=2\sum_{i\ne j}(R_QAR_Q)_{ij}^2,
$$

and the frozen conditional estimator risk is

$$
\mathcal R_Q(m)=\frac{\sigma_Q^2}{m-q-r}.
$$

Phase 1A estimates only the numerator. The external certification probes are not
deducted from the frozen denominator.

## Reconstruction and accounting audit

- Dimensions: 500.
- Step ranks: (5, 15, 30).
- Tail levels: (1e-10, 1e-06).
- Frozen paths per rank: 200.
- Certification repetitions per path and tail: 200.
- Every repetition uses exactly 32 certification matrix-vector queries.
- Cached $AQ$ products are paid for only by the reconstruction counter.
- Historical bridge artifacts are unchanged.

## Estimators and decision rule

The experiment compares ordinary sample variance, paired mean, mom_w1, and mom_w2.
The verdict uses sample variance, mom_w1, and mom_w2. At the primary
$\varepsilon=1/3$, the empirical rule accepts only when the estimated candidate
risk is at most one half of the estimated baseline risk.

**EMPIRICAL DECISION RULE.** This multiplicative rule is selective evidence, not
a finite-sample confidence bound.

## Primary gate

| s | estimator | evaluable | passed | passing batches |
|---:|:---|:---:|:---:|---:|
| 16 | sample_variance | True | True | 4 |
| 16 | mom_w1 | True | False | 0 |
| 16 | mom_w2 | True | False | 0 |
| 32 | sample_variance | True | True | 4 |
| 32 | mom_w1 | True | False | 0 |
| 32 | mom_w2 | True | False | 0 |

## Primary bootstrap metrics at s=16

| estimator | metric | point | conditional 95% percentile interval |
|:---|:---|---:|:---|
| sample_variance | false_safe | 0.0134% | [0.0028%, 0.0305%] |
| sample_variance | false_rejection | 0.5157% | [0.2991%, 0.7722%] |
| sample_variance | catastrophic_detection | 79.6667% | [69.0662%, 89.2833%] |
| sample_variance | better_acceptance | 52.0991% | [40.3055%, 63.5069%] |
| sample_variance | control_false_safe | 0.0008% | [0.0000%, 0.0025%] |
| mom_w1 | false_safe | 4.6971% | [4.5486%, 4.8518%] |
| mom_w1 | false_rejection | 6.8796% | [5.2527%, 8.6241%] |
| mom_w1 | catastrophic_detection | 72.0000% | [63.0833%, 80.8333%] |
| mom_w1 | better_acceptance | 51.8454% | [42.8916%, 60.8373%] |
| mom_w1 | control_false_safe | 4.2100% | [4.1000%, 4.3220%] |
| mom_w2 | false_safe | 1.5478% | [1.4513%, 1.6504%] |
| mom_w2 | false_rejection | 3.7787% | [2.7139%, 4.8907%] |
| mom_w2 | catastrophic_detection | 75.6000% | [66.2662%, 84.4671%] |
| mom_w2 | better_acceptance | 52.4556% | [42.6741%, 62.3038%] |
| mom_w2 | control_false_safe | 1.3044% | [1.2351%, 1.3743%] |

## Sample-variance sensitivity

| s | false safe | false rejection | true-better acceptance | top-5% detection | abstention |
|---:|---:|---:|---:|---:|---:|
| 4 | 3.2141% | 7.4648% | 49.8398% | 69.1833% | 84.0517% |
| 8 | 0.2960% | 2.4796% | 51.1278% | 74.6833% | 91.3633% |
| 16 | 0.0134% | 0.5157% | 52.0991% | 79.6667% | 94.0183% |
| 32 | 0.0000% | 0.0537% | 51.6880% | 81.1500% | 95.3692% |

The sample-variance rule becomes substantially safer as $s$ grows. At $s=16$,
the point false-safe rate is 0.0134%, the conditional bootstrap upper endpoint
is 0.0305%, and top-5% catastrophic detection is 79.67%. The rule passes the
complete point gate in all 4 batches. Its high
abstention rate is intentional: useful acceptance is measured separately and is
52.10% among truly better paths.

## Numerator-estimation accuracy at s=16

These error summaries use the primary family, $m=160$, and the candidate
$q=r_\star+1$.

| estimator | median relative error | p90 | p95 | p99 |
|:---|---:|---:|---:|---:|
| mom_w1 | 58.3752% | 86.6235% | 91.1858% | 97.8256% |
| mom_w2 | 42.0410% | 79.0370% | 88.2535% | 145.6340% |
| paired_mean | 35.6545% | 79.6284% | 105.0441% | 178.4284% |
| sample_variance | 27.7332% | 64.4602% | 81.5610% | 137.0069% |

Accurate point estimation is not itself the gate. The primary scientific outcome
is the candidate-versus-baseline selective decision.

## MoM outcome

The small-block MoM candidates do not pass Phase 1A. At $s=16$, mom_w1 passes
0 batches and mom_w2 passes
0 batches. Their false-safe rates are 4.6971%
and 1.5478%, respectively, and their benign-control false-safe rates are 4.2100%
and 1.3044%. These are well above the preregistered 0.5% point threshold.

This negative result applies to the declared mom_w1 and mom_w2 constructions at
$s\le32$. It does not refute the proved fourth-moment bound or every possible
median-of-means confidence construction. It shows that these very small-block
empirical plug-in rules are not safe enough for the planned allocator.

## Interpretation and recommendation

**EMPIRICALLY ESTABLISHED.** The tables and figures report whether small fresh
probe batches carry useful information about the realized conditional risk under
the frozen population.

The preregistered verdict is **QUALIFIED GO** because ordinary sample variance
passes at $s=16$, while neither MoM candidate passes. Therefore the direct
realized-risk signal is empirically usable, but estimator choice matters.

The next mathematical study should focus on whether the paired sample-variance
difference can receive a valid simultaneous finite-sample bound, or whether a
different robust construction can retain the sample-variance power while
recovering theorem-level safety. Phase 1B should also charge certification queries
before any online estimator modification.

**OPEN.** A simultaneous finite-sample confidence theorem, zero-risk boundary,
budget-charged online timing, and a physically feasible fallback architecture
remain future Phase 1B/Phase 2 work.

## Limitations

The bootstrap resamples eligible frozen paths within rank and is conditional on
the three fixed signal-subspace orientations. Cross-tail, cross-budget,
cross-action, cross-estimator, and cross-s comparisons use common probes and are
paired. The result does not establish orientation-universal safety.
