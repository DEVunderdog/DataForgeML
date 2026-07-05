# ADR 0049: Datetime columns are excluded from stratification; temporal splitting stays user-invoked

`profile_stratified_split` builds its label matrix from every profile signal that protects a downstream phase, yet it ignores `Datetime` columns entirely — no datetime stratification signal is ever emitted, and the presence of a datetime column never causes the profile-stratified path to auto-route to a chronological split. A future reader who sees the full `DatetimeProfiler` (range, granularity, gap regularity, MNAR flags) will reasonably wonder why none of it reaches the splitter. This records that the omission is deliberate.

## Context

Stratification and temporal splitting are opposites, not variants of each other. Stratification assumes rows are **exchangeable**: it shuffles, then rebalances so both partitions carry proportional shares of every signal. A temporal split assumes rows are **not** exchangeable: train = past, test = future, precisely so no future information leaks. "Stratifying on a datetime column" — quantile-binning the timestamp and balancing the buckets — would shuffle future rows into train and past rows into test, which is exactly the leakage `time_split` exists to prevent. A datetime signal is therefore not a smaller version of temporal handling; it is the thing temporal handling forbids.

Two distinct rules fall out of this:

- **Data rule (automatic):** a `Datetime` column never contributes a stratification signal. Under the current phase set it also fails Gate 1 of the viability contract (ADR-0047) — there is no datetime imputer in Phase 2 (`Datetime` routes to Passthrough), and the Phase 5 temporal feature-extraction signals were removed (commit `4c8f75c`) — so no downstream phase depends on datetime-distribution balance.
- **Intent rule (the user's call):** whether a dataset should be split chronologically is **modeling intent**, not a data property. The library cannot distinguish a forecast (order matters) from a cross-sectional task that merely carries a date feature (a `signup_date` column on an i.i.d. churn model). The same data — "a `Datetime` column exists" — appears in both cases.

## Decision

The profile-stratified path ignores datetime columns and never infers temporality. Temporal splitting is served exclusively by the explicit, user-invoked `DataSplitter.time_split(time_column, ...)`. The user selects the split method that matches their intent; the library makes no guess.

## Considered options

- **Auto-route to `time_split` when any opted-in `Datetime` column exists.** Rejected: silently wrong for every cross-sectional dataset that happens to contain a date feature — it would chronologically sort and cut a churn model, discarding the target-balance and rare-label guarantees `profile_stratified_split` provides, with no way for the user to opt back into stratification.
- **Quantile-bin the timestamp into a stratification signal.** Rejected: this shuffles across time and induces the exact temporal leakage `time_split` is built to prevent; it also fails Gate 1 (no downstream phase consumes datetime balance).
- **Warn, then fall back to random, when the label matrix collapses and a datetime column is present.** Rejected for now: consistent with the library's no-post-hoc-imbalance-check stance, the docs carry the guidance instead. The user owns the split-method choice for every non-stratified method.

## Consequences

- A user with a genuinely temporal dataset who calls `profile_stratified_split` gets a random-fallback split (leaky) with no runtime warning. This is the accepted cost of not guessing intent; the docs direct temporal datasets to `time_split`.
- Reversing this later (adding datetime-aware auto-routing or a warning) is a behavior change, which is why the deliberate no- is recorded here rather than left implicit.
