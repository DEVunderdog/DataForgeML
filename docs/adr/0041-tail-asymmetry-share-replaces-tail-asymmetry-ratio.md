# ADR 0041: Tail asymmetry share replaces tail asymmetry ratio; zero-spread bands get a defined outcome

`NumericStats.tail_asymmetry_ratio` is redefined from `(p99 - p95) / (p5 - p1)` to a bounded share, `(p99 - p95) / ((p99 - p95) + (p5 - p1))`, and renamed `tail_asymmetry_share`. The reformulation removes a structural division-by-zero blind spot in ADR-0033's original formula, at the cost of touching a formula and config fields ADR-0033 had already locked in.

## Context

ADR-0033 defined `tail_asymmetry_ratio = (p99 - p95) / (p5 - p1)` and specified that a zero denominator (`p5 == p1`, a flat left tail) stores `tail_asymmetry_ratio = None` and leaves `TailAsymmetryTag` unset — treated as "can't determine."

That rule conflates two different situations:

- **True 0/0** — both extreme bands are flat (`p5 == p1` and `p99 == p95`). Genuinely no signal; neither side showed spread to compare.
- **One-sided zero** — the left band is flat (`p5 == p1`) but the right band has real width (`p99 - p95 > 0`). The ratio diverges to `+∞` as the denominator shrinks to zero — a well-defined maximal `RightHeavy` signal, not an unknown. The old formula discarded this case as `None`, silently dropping a real, unambiguous signal.

This asymmetry is an artifact of the formula's shape, not of the underlying concept: the right band sits in the numerator (a zero there resolves safely to `ratio = 0`, which flows into `LeftHeavy` via the normal threshold comparison), while the left band sits in the denominator (a zero there is undefined). Which side is "safe at zero" depends only on which band was arbitrarily assigned to which position.

## Decision

Replace the ratio with a **share**: `tail_asymmetry_share = (p99 - p95) / ((p99 - p95) + (p5 - p1))`, bounded to `[0, 1]`. Both band widths now appear only in sums, never as a lone denominator, so a zero band no longer produces an asymmetric failure mode — the one-sided-zero case above now resolves to `share = 1.0` (maximal right-heavy) through the ordinary threshold comparison, with no special-casing required.

The only value that remains undefined is true 0/0 (both bands flat): `tail_asymmetry_share = None`, `TailAsymmetryTag` unset. This is not classified as `Symmetric` — `Symmetric` asserts that both tails were measured and found balanced; 0/0 means neither tail showed any spread to measure. Overstating a null result as a balanced one would misrepresent what was actually observed.

Renames, since "share" is a different kind of quantity than "ratio" (bounded `[0, 1]` vs. unbounded) and keeping the old name would mislead readers:

- `NumericStats.tail_asymmetry_ratio` → `NumericStats.tail_asymmetry_share`
- `NumericProfileConfig.tail_asymmetry_right_threshold` → `tail_asymmetry_right_share_threshold`, default `2.0` → `2/3`
- `NumericProfileConfig.tail_asymmetry_left_threshold` → `tail_asymmetry_left_share_threshold`, default `0.5` → `1/3`

Threshold defaults are converted exactly, not re-guessed: `share = ratio / (1 + ratio)`, so the prior calibration (`ratio > 2.0` for `RightHeavy`, `ratio < 0.5` for `LeftHeavy`) is preserved losslessly as `share > 2/3` and `share < 1/3`.

No config migration path is provided. The library has no external users yet — breaking the field names and defaults for anyone who saved a `PipelineConfig` under the old scheme is accepted as a non-issue at this stage, not a general policy for future breaking changes.

## Considered options

**Patch the ratio formula instead of replacing it** — keep `tail_asymmetry_ratio` as-is, and for the one-sided-zero case, set `TailAsymmetryTag.RightHeavy` while leaving `ratio = None` (direction known, magnitude not expressible as a finite ratio). Rejected once the share formula was on the table: the share formula produces a real finite value for that exact case, making the patch unnecessary, and it leaves the true root cause (numerator/denominator asymmetry) in place for any future band comparison built the same way.

**Keep the `tail_asymmetry_ratio` name for the share value.** Rejected — a share and a ratio are different kinds of numbers (bounded vs. unbounded); keeping the old name would silently mislead anyone expecting a value that can exceed `1`. There is no compatibility cost to renaming since nothing depends on the old field name yet.

**Round the converted thresholds to `0.67` / `0.33` for readability.** Rejected in favor of the exact fractions `2/3` / `1/3` — these are internal escalation thresholds, not user-facing display numbers, so faithfully preserving the prior calibration outweighs cosmetic tidiness.

## Consequences

- Supersedes the "Tail asymmetry ratio" section of ADR-0033 (formula, field names, and division-by-zero handling). ADR-0033's escalation rule (`RightHeavy`/`LeftHeavy` upgrades effective `SkewSeverity` by one level) is unchanged — it keys off the `TailAsymmetryTag` enum value only, never the numeric field, so it is unaffected by this rename.
- `_compute_tail_asymmetry` in `_numeric_profiler.py` needs its zero-handling branch rewritten: only `numerator == 0 and denominator == 0` remains a `None`/`None` case.
- `to_dict`/`from_dict` in `_numeric_config.py` and the `NumericProfileConfig`/`NumericStats` docstrings need the field renames applied throughout (per ADR-0034's documentation rule, since these are in-scope Config dataclass fields).
- No consumer currently reads `tail_asymmetry_ratio`/`tail_asymmetry_tag` — the `SkewSeverity` upgrade described in ADR-0033 is not yet wired into `_strategy_router.py`. This change has no blast radius on existing routing behavior; it only shapes the contract that future wiring will consume.
