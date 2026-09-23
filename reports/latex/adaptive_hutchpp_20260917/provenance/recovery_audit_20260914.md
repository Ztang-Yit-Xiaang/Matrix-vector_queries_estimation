# September implementation recovery — 2026-09-14

**Status: complete.** Validated reruns were published on September 14 at 18:25:59 UTC. Exactly six named CSVs changed; the other 118 pre-existing result artifacts retain their SHA-256 checksums. The full suite passed 217 tests, followed by nine targeted mathematical checks and a no-query serialization check for the two-stage benchmark.

## Cause and scope

The September 14 inspection ran the existing Haar and PyTorch smoke tests. Both tests accepted `tmp_path` but ignored it; both experiment runners wrote directly to production filenames. This replaced the untracked Haar CSV pair with a 36-row smoke run and the Hessian CSV pair with an 8-row smoke run. The inspection caused the overwrite and should have checked output isolation before execution.

The original raw-file hashes were unavailable. Recovery therefore reruns the original recorded protocols and validates their counts, keys, budgets, and reported statistics. It does not claim byte-identical restoration. Wall-clock measurements are environment-dependent.

The smoke-sized files, original Phase 2C grids/proof/report, and pre-recovery tracker, memory, and current-state documents are retained under `results/recovery_20260914/before/`. All pre-recovery result checksums are retained in `before_checksums.json`.

## Prevention

- Haar, Hessian, gating diagnostics, the two-stage benchmark, and Phase 2C grid callable runners write only when an explicit `output_dir` is supplied. Their command-line entry points require `--output-dir`.
- Both affected smoke tests write to their temporary directories and verify the expected output files.
- A session-level test fixture hashes production results before and after the suite and fails if any file is added, removed, or changed. The separate recovery staging directory is excluded so a production recovery can run concurrently.
- The two recovery runs stage results before validating and publishing. Publication checks that every canonical result still matches the pre-recovery snapshot, verifies staged manifests, and permits replacement of exactly six named CSVs.
- All four benchmark methods have explicit query-count assertions. New metadata records seeds, dimensions, budgets, and actual accounting; the Hessian table separately records the 4,254 reference HVPs.

## Reproduction protocol

| Run | Frozen settings | Required output |
|---|---|---|
| Haar | d=100, m=60, seed=2026, six spectra, 30 Haar orientations plus coordinate orientation, 10 trials/orientation | 1,860 rows, each containing four method estimates; 12 summary rows |
| Hessian | original synthetic data seed 42, SmallConvNet with 4,254 parameters, NumPy seed 2026, m=30/60/90/120, 30 trials/method | 480 rows; 16 summary rows |
| Phase 2C | original radius and structural-ratio grids | 252 radius rows; 36 illustrative kurtosis rows |

The Haar gate retains gap threshold 1.5 and oversampling 2. The Hessian gate retains threshold 1.2 and oversampling 2. Both runs leave `tau_contrast=None`, as in the original runners. They therefore do not validate the later proposed contrast threshold 5. No allocator logic or random sampling parameters were changed in recovery.

The Haar rerun reproduces the reported rotated rank-5 step MSE ratio 0.169740 (5.89 times lower), and the coordinate rank-5 ratio 4582.446 (much worse). The latter is an observed heuristic failure. Standard Hutch++ MSE there is approximately 7.21e-7, not zero. Keeping both outcomes prevents an orientation-selective research claim.

The Hessian rerun reproduces the reference trace 31.581270 at the precision printed in the original report. The data generator produces labels 0 and 1 with a ten-output model. This is a synthetic-data neural-network benchmark, not a ten-observed-class real dataset. Its Hessian is not assumed PSD.

| Hessian budget | Standard Hutch++ MSE | Gated MSE |
|---:|---:|---:|
| 30 | 5.858498 | 5.334005 |
| 60 | 1.417330 | 0.841176 |
| 90 | 1.006494 | 0.570674 |
| 120 | 0.457086 | 0.370210 |

These empirical MSE values reproduce the earlier table to its printed precision. The recovery ran alongside validation, so its timing columns are not a controlled performance comparison. The recovery consumed 446,400 Haar matrix-vector queries and 36,000 Hessian estimator HVPs plus 4,254 reference HVPs; smoke-test work is separate. Protocol details and software versions are saved in the per-run manifests.

## Mathematical corrections

1. **Theorem 16:** nested projectors give H_a=R_a H_0 R_a and Frobenius contraction. The finite Cantelli radius is conditional on a valid supplied norm bound and nesting. Neither assumption can be certified from the supplied observation arrays. Finite radius alone does not establish useful acceptance.
2. **Theorem 17:** the earlier fourth-moment expansion had incorrect coefficients. The corrected identity is

   $$\mathbb E Z^4=3\sigma^4+48\operatorname{tr}(C^4)-96\sum_i(C^2)_{ii}^2+32\sum_{ij}C_{ij}^4.$$

   Direct even-degree graph counting establishes this identity; exhaustive small-matrix enumeration checks the implementation. It retains the bound (3+12 kappa^2) sigma^4. This degree-two result does not bound the kurtosis of the degree-four signed-pair variable.
3. **Theorem 18:** strict feasibility is exactly

   $$\delta_{\rm scale}n(n-1)>(K-1)(n-1)+2.$$

   At K=3 and delta_scale=0.05, n=41 is equality; the minimum is n=42, or s=84 probes. At K=3.12 the corrected minimum is n=44. Exact decimal arithmetic avoids rounding the equality into acceptance.
4. **Scope:** these are limits of the audited confidence construction. They do not prove that empirical certification is universally impossible or that a norm prior is necessary.
5. **Haar diagnostic:** `diag_concentration` measures an ideal top-q eigenspace residual. It cannot be used as the realized randomized Hutch++ residual's concentration. Observed trigger stability across 30 orientations is not a rotation-invariance theorem for Rademacher sketches.
6. **Gating interpretation:** the contrast threshold is a development-grid heuristic. Empirical fixed-q grid minima are noisy references, not exact risk oracles. The report's 23 stored configurations and ideal-capture assumptions are now explicit.

## Verification and remaining boundary

The complete maintained suite passed **217 tests**, including result immutability. The corrected threshold and fourth-moment tests also received a focused rerun after the final test-example cleanup. Compilation and whitespace checks are included in the recovery audit.

The recovery manifest records validated publication, final checksums, software versions, and query totals. September additions remain local and uncommitted; unrelated source/PDF changes are preserved. No online certification claim follows from these repairs. The next research action is to consolidate the report around the verified failure mechanisms; any further gate evaluation needs a separately frozen held-out design.
