# Why do about 180 contract tests take 3–5 s each?

Measured 2026-10-30 for the wayfinder ticket "Why do about 180 contract tests take 3–5 s each?" (#667), on the map "a CI policy that keeps PRs fast and main strict" (#662).

**Question.** `dataforge --help` takes 0.14 s, so basic CLI startup doesn't seem to explain 3–5 s tests. Where does the time go, and how much comes from each cause?

**This note decides nothing.** It records where the time goes. The Ames ticket and any later plan make the calls.

## 1. Summary

- **The cause is CLI startup.** `--help` is the wrong yardstick, because it deliberately skips the heavy imports. Every real Operation call pays about 2 s before it does anything, even when it only rejects a bad flag.
- **The Operation's own work is tiny.** A median CLI call took 2.13 s, and 0.009 s of that was the Operation.
- **Startup is 59.5% of all contract test time:** 26.6 of 44.7 worker-minutes, from 747 CLI calls. The real work inside those calls is 8.1%. Code Mode is 30.6%.
- **Mid-sized tests are nearly all startup.** Startup is 92% of the 2–3 s tests and 81% of the 3–5 s tests. A test costs about 2 s per CLI call.
- **"About 180 at 3–5 s" did not reproduce.** This run had 52 tests at 3–5 s and 257 at 2–3 s. The source of 180 was not recovered. One call slows from 1.9 s to about 3 s when CPUs are busy, which would lift many of the 257 into the 3–5 s band. That link is inferred.
- **Sharing worst-case calls barely touches this.** It removes about 40 of the 747 CLI calls. Marking tests slow doesn't either: 581 calls sit in tests under 10 s, and startup would be about 80% of a fast tier's time.
- **For the Ames ticket: the costliest worst case is not Ames.** `evaluate_imputation` on a synthetic 101-column frame costs 272 s across its three tests. The Ames worst cases together cost about 126 s.

## 2. Method

- Machine as in "Why do 12 cores run no faster than 2?": Ryzen 5 5600H, 6 cores / 12 threads, 7 GB RAM, most of it used by other software. Python 3.12.3, pytest 9.1.1, pytest-xdist 3.8.0.
- One full run of `tests/contract`: 3 workers, `worksteal`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1. All 922 tests passed in 17.0 min, 44.7 worker-minutes.
- A throwaway pytest plugin timed every Code Mode call, every subprocess and every tokeniser call, per test. It swapped the `dataforge` script for a stand-in that also timed the import stages and the call to `invoke_json`. Neither was committed.
- "Startup" means everything in a CLI subprocess outside `invoke_json`: process start, imports, argument parsing, printing and exit. "Work" means the time inside `invoke_json`.
- The stand-in adds a little: a median call took about 2.1 s under it and 2.0 s without it, in a matching 3-process run. Startup is overstated by up to about 5%.
- The box was in normal use and swapped in bursts: 26 of 75 `vmstat` samples passed 5 MB/s combined. Child wall time stayed below child CPU time (ratio 0.83), so CLI calls did not visibly wait on disk. Treat differences under about 10% as noise. The full run happened once.

## 3. Where the time goes

Share of each band's time, by test duration:

| Test duration | Tests | Worker-min | CLI startup | CLI work | Code Mode | Other |
|---|---:|---:|---:|---:|---:|---:|
| under 2 s | 467 | 0.4 | 45% | 0% | 17% | 37% |
| 2–3 s | 257 | 10.2 | 92% | 1% | 7% | 0% |
| 3–5 s | 52 | 3.4 | 81% | 3% | 14% | 2% |
| 5–10 s | 104 | 11.7 | 72% | 3% | 26% | 0% |
| 10 s and over | 42 | 19.0 | 31% | 17% | 50% | 3% |
| **All** | **922** | **44.7** | **59.5%** | **8.1%** | **30.6%** | **1.9%** |

- Of the tests under 2 s, 429 took under 0.1 s. Most of those make no CLI call.
- The 3–5 s tests are 29 with one CLI call, 22 with two, and one with none.
- The 2–3 s tests all make exactly one CLI call.
- Tokenising for the budget check totals 2.7 s, so it does not matter. Loading the encoding totals 0.9 s.

## 4. What one CLI call spends

Median stages of a call, in the full run:

| Stage | Time |
|---|---:|
| Process start and the light CLI imports (what `--help` pays) | 0.13 s |
| Importing the Operation machinery | 1.38 s |
| Importing the other Operations and parsing arguments | 0.32 s |
| The Operation itself | 0.009 s |
| Printing and process exit | 0.29 s |

On an idle box, each of these took about 1.9 s: `dataforge list-handles`, `load` on a 6-row CSV, a bad path, and an unknown flag.

Why the imports are so large:

- `_boundary/_workspace.py:29` imports one constant, `FORMAT_SCHEMA_VERSION`, from `_serialization`. That module imports `imputation`, which reaches scikit-learn, and `profiling`. So the base of every Operation loads the whole stack.
- `operations()` imports all 20 Operations to dispatch one.
- By import-time self cost: scipy 493 ms, `dataforge_ml` itself 368 ms, scikit-learn 150 ms, pandas 108 ms, polars 94 ms, numpy 85 ms.

How low startup could go was not measured. Removing that one import may not be enough, since the Operation modules may pull the same stack in themselves.

Agents pay this too. It is not only a test cost: each `dataforge` call in real use starts with the same 1.9 s.

## 5. Why a 2 s call can look like 3–5 s

The same `dataforge list-handles`, run in parallel loops:

| CPUs | Processes | Threads capped | Median | Max |
|---|---:|---|---:|---:|
| 12 | 1 | no | 1.88 s | 1.90 s |
| 12 | 3 | no | 2.24 s | 2.36 s |
| 12 | 3 | yes | 2.02 s | 2.10 s |
| 12 | 6 | no | 2.81 s | 3.28 s |
| 4 | 2 | no | 2.32 s | 2.72 s |
| 4 | 2 | yes | 2.09 s | 2.26 s |
| 4 | 4 | no | 3.08 s | 3.22 s |
| 4 | 4 | yes | 3.00 s | 3.19 s |

- Each process made 6 calls, so a row holds 6 to 36 samples. This is a small sample.
- A test at the 4-worker setup that "How does the policy map onto the workflow files?" chose would pay about 3 s per call on these 4 CPUs. That is about 1.6 times the idle cost. The thread cap does not help there.
- Real CI hardware was not measured. Expect the same shape, not these numbers.

## 6. The heavy end, for the Ames ticket

`profile` is the costly Code Mode call. It totals 586 s of Code Mode's 823 s. Measured alone, with threads capped:

| Frame | `load` | `profile` |
|---|---:|---:|
| `train.csv`, 6 rows | 0.12 s | 0.01 s |
| Synthetic, 150 columns × 40 rows | 0.01 s | 3.1 s |
| Ames, 2,930 rows × 82 columns | 1.36 s | 13 s |

Of the 502 Code Mode `profile` calls, 19 took over 8 s (316 s in total, Ames-sized) and 48 took 2–8 s (167 s, wide-fixture-sized). Another 372 took 0.2–0.5 s (97 s). Which fixture those use was not traced.

The three worst-case tests of each Operation, in seconds:

| Operation | Each of the three tests | Fixture |
|---|---:|---|
| `evaluate_imputation` | 97, 91, 85 | synthetic, 101 columns × 90 rows |
| `c2st` | 22, 21, 21 | synthetic, 80 rows, 100 repeats |
| `split` | 21, 20, 20 | Ames |
| `describe_target` | 18, 18, 17 | Ames |
| `profile` | 11, 11, 9 | synthetic, 150 columns |
| `describe_c2st_columns` | 10, 10, 10 | synthetic, 21 columns |
| `query_frame` | 4, 4, 4 | Ames |
| The other 13 Operations | 2–8 each | synthetic or small |

- These 60 tests total 12.1 min, 27% of the contract sum.
- The Ames rows are about 126 s of that. `evaluate_imputation` alone is 272 s, and it uses no Ames data.
- Ames is declared in 25 tests, which total 8.6 min. 213 s of that is the no-delete test, which took 4–6 min in earlier runs.

The slowest tests in this run:

| Test | Time |
|---|---:|
| `test_workspace_operations.py::test_no_operation_deletes_or_changes_anything_in_the_workspace` | 213 s |
| `test_operation_contract.py::test_worst_case_result_is_under_the_result_budget[evaluate_imputation]` | 97 s |
| `test_operation_contract.py::test_result_json_is_byte_stable[evaluate_imputation]` | 91 s |
| `test_operation_contract.py::test_every_omitted_operation_is_a_real_command[evaluate_imputation]` | 85 s |
| `test_profile.py::test_ames_digest_fits_the_result_budget_whole` | 32 s |

## 7. What the decided work would and wouldn't reach

This is arithmetic on the run above, not a measurement of the changes.

- **Sharing worst-case calls ("How do contract tests share one worst-case call?").** Merging three tests into one per Operation drops about two thirds of the 12.1 min, so about 8 min. It removes about 40 CLI calls, about 90 s of startup. Almost all of the saving is Code Mode and real work in the heaviest tests.
- **The slow marker ("What counts as slow, and how is the rule enforced?").** Taking out tests of 10 s and over removes 19.0 min and leaves 25.7 min. Of that, about 20.7 min is startup, so about 80%. This run's 10 s line is a local one; the rule is set on CI seconds.
- **Startup itself.** With 747 calls, each 0.1 s off one call saves about 75 worker-seconds. Halving startup would save about 13 worker-minutes, or 30% of this contract run. No decided work touches this.
- **In-process CLI.** `harness.call_in_process` skips the subprocess, but only 2 lines outside `conftest.py` use it, and it cost nothing measurable here. It also gives up the check that a real process behaves the same.
