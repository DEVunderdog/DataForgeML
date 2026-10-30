# Why do 12 cores run no faster than 2?

Measured 2026-10-30 for the wayfinder ticket "Why do 12 cores run no faster than 2?" (#663), on the map "a CI policy that keeps PRs fast and main strict" (#662).

**Question.** A CI job took about 18 min with 2 xdist workers. Locally the suite took about 16 min, which looked like 12 cores. Why do the extra cores barely help? Is it inner parallelism, one long test, subprocess startup, or something else? What would the CI runner (4 logical, 2 physical) see under other setups?

**This note decides nothing.** It records what each setup is worth. The threshold and topology tickets make the calls.

## 1. Summary

- **The premise was off in two ways.** Locally `-n auto` starts **6** workers, not 12: xdist counts physical cores. And "18 min" was one sample: the last five green CI runs had pytest jobs of 15 to 32 min on the same suite, median 24, with no Python version consistently slowest.
- **Workers do scale, once two brakes are off.** On the same 4 pinned CPUs, the CI shape took 31.3 min. Four workers with work-stealing and inner threads capped took 16.9 min. All 2,225 tests passed.
- **Brake 1: inner OpenMP threads, when CPUs run out.** The C2ST classifier (`HistGradientBoostingClassifier`) starts one OpenMP thread per CPU in every worker. With 4 workers on 4 CPUs, three tests went from about 30 s to 1,100–1,500 s each, and the run was stopped unfinished after 63 min. With `OMP_NUM_THREADS=1` the same test file took 13 s. With spare CPUs (6 workers on 12) the effect is small. That thread contention is the mechanism is inferred from these timings, not observed directly.
- **Brake 2: `--dist load`.** It hands out tests up front and never rebalances. At 6 workers one worker ran 1,162 s while four others stopped at 614–668 s. `--dist worksteal` took 19.5 min to 13.1 min, though noise is part of that gap (section 3).
- **`-n logical` alone is a trap on CI.** Without the thread cap, that setup is the one that stalled.
- **One test sets a floor of about 5 min.** `test_no_operation_deletes_or_changes_anything_in_the_workspace` took 254–357 s in every run. No worker count beats it unless it gets cheaper or is split.
- **Memory caps this box, not CI.** Each worker peaks at about 1 GB, and this box has about 2 GB free. Six workers swap; `-n 12` would need about 12 GB and was not run. GitHub's standard runners have 16 GB, so 4 workers should fit.
- **The test work itself is still large.** It is 57–66 worker-minutes, and the contract folder is about 83–94% of it. Faster scheduling only divides that total, so the sharing ticket still matters.

## 2. Method

- Machine: AMD Ryzen 5 5600H, 6 cores / 12 threads, 7 GB RAM with about 5 GB already used by other software. Python 3.12, pytest 9.1.1, pytest-xdist 3.8.0.
- Each run is the full suite of 2,225 tests. A throwaway plugin logged each test's worker, start, end and peak RSS. `vmstat` sampled CPU and swap every 5 s.
- The CI shape was copied with `taskset -c 0-3`: two physical cores and their SMT siblings.
- Each setup ran once, on a box in normal use. Treat differences under about 10% as noise.
- "Sum" is the total of all test durations, in worker-minutes. "LPT ideal" is the wall time if the same tests were packed perfectly, longest first. It is a lower bound.
- CI job times come from `gh run view` and include setup steps. Per-step times were not available.

## 3. Results

| Setup | Workers | Wall | Sum (worker-min) | LPT ideal |
|---|---|---|---|---|
| 12 CPUs: `-n auto`, `load` (the default) | 6 | 19.5 min | 77.4 | 12.9 min |
| 12 CPUs: `worksteal` | 6 | 13.1 min | 64.2 | 10.7 min |
| 12 CPUs: `worksteal`, threads capped | 6 | 11.4 min | 62.5 | 10.4 min |
| 4 CPUs: `-n 2`, `load` (the CI shape) | 2 | 31.3 min | 56.7 | 28.4 min |
| 4 CPUs: `-n 4`, `worksteal` | 4 | stalled: stopped at 63 min, 2,163 of 2,225 done | n/a | n/a |
| 4 CPUs: `-n 4`, `worksteal`, threads capped | 4 | 16.9 min | 66.3 | 16.6 min |

"Threads capped" means `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` all set to 1. The last row changes three things at once compared with the CI shape: workers, scheduling and the cap. They are not separated.

Worker-seconds in the two C2ST test files (`test_imputation_score_c2st.py`, `test_c2st_seam.py`):

| Setup | Worker-seconds |
|---|---|
| 12 CPUs, 6 workers, uncapped | 86 |
| 12 CPUs, 6 workers, capped | 37 |
| 4 CPUs, 2 workers, uncapped (CI shape) | 432 |
| 4 CPUs, 4 workers, uncapped | over 5,000, with 35 of 51 tests in one file unfinished |

Notes:
- The default run's sum (77) is higher than the worksteal run's (64) with the same workers and similar swap. That gap is unexplained; other load on the box is one candidate. So the 6.4 min saving is scheduling plus noise.
- At 2 workers, `load` cost about 10% against the ideal (1,871 s vs 1,701 s).
- About a third of the CPU stays idle at 6 workers on 12. Most test work is single-threaded, so 6 workers fill about 6 logical CPUs.

Asked for by the ticket, and not fully measured:
- `-n 0`: not run. The sum is its estimate, about 57 min or more.
- `-n logical` on this box: not run, for memory. The 4-CPU runs stand in for it.
- Library core budget forced to 1: not run. In the capped 4-worker run the test harness already gave each worker a budget of 1 (4 cores // 4 workers).
- Without the heaviest tests: arithmetic only, from the capped 4-worker run. Dropping the top 10 removes 23% of the sum; the top 100 removes 58%.
- Subprocess startup: not isolated. One `dataforge load` of Ames took 2.9 s wall and 257 MB on an idle box. The number of subprocess calls in the suite was not counted, so its share of the total is unknown.

## 4. Where the time goes

Slowest tests in the CI-shape run:

| Test | Time |
|---|---|
| `test_workspace_operations.py::test_no_operation_deletes_or_changes_anything_in_the_workspace` | 264 s |
| `test_operation_contract.py::test_result_json_is_byte_stable[evaluate_imputation]` | 177 s |
| `test_operation_contract.py::test_result_json_is_byte_stable[c2st]` | 122 s |
| `test_operation_contract.py::test_every_omitted_operation_is_a_real_command[evaluate_imputation]` | 109 s |
| `test_operation_contract.py::test_worst_case_result_is_under_the_result_budget[evaluate_imputation]` | 104 s |
| `test_c2st_seam.py::test_null_calibration_is_standard_normal_at_the_nominal_level` | 52 s |

The top 10 tests are 23–29% of the sum in every run, and the top 100 are 58–63%.

## 5. What the CI runner would likely see

The 4-CPU runs copy CI's shape, not its hardware. These are inferences until a real CI run confirms them.

- **Today:** 2 workers, uncapped. CI jobs run 15–32 min, and the slowest job in a run is what a PR waits for: 24–32 min in the five runs read.
- **`-n 4 --dist worksteal` plus the thread cap:** 46% faster here (31.3 to 16.9 min). If that ratio holds, a 30 min job becomes about 16 min. Runner memory should allow it.
- **`-n 4` without the cap:** do not do this. It stalled here.
- **The cap on its own, at 2 workers:** not run. The C2ST files alone would save about 3 min of wall time here.
- **Why CI jobs vary 2-fold:** not measured. Thread contention at 2 workers is one candidate, and runner hardware is another.

The cap must reach subprocesses too. An environment variable set on the CI step does that.

## 6. Reproduce

The plugin and analysis script were throwaway and are not committed.

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 0-3 uv run pytest -n 4 --dist worksteal --durations=30
```
