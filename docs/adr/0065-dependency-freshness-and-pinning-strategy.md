# Dependency freshness rests on retrain-only recovery, ranged pins guarded by tightened numeric caps, and an automated fresh-and-safe update pipeline

ADR-0063 named the **DataForgeML library version the single master key** for reconstructing a persisted `FittedUnit`, on the premise that "same library version" transitively guarantees a compatible sklearn/numpy/joblib environment — a premise that only holds if each release pins the dependency *ranges* it was tested against. It deferred the machinery that makes that true to a sibling ticket. This ADR is that ticket (#339): it fixes what happens on a version mismatch, how the dependency *ranges* are shaped, how the tested environment is frozen for CI, and how the project stays current and free of known vulnerabilities over time. It is the last of the persistence-boundary contracts charted in the wayfinder map (#333).

## The problem, stated plainly

Two questions drive the whole ticket:

- **Mismatch on load.** A saved model records the versions it was produced with. What happens when those don't match the versions running now?
- **Upstream drift.** scikit-learn/numpy/etc. release new versions and occasional security fixes. How does DataForgeML take them in without silently invalidating its own correctness guarantees?

Both are answered by one linchpin: **every DataForgeML version corresponds to one known-good, tested dependency environment**, and the *runtime version stamp* — not the install-time pins — is what enforces artifact correctness.

## Locked decisions

- **Recovery on mismatch is retrain-only.** ADR-0063 already refuses a load whose stamped versions don't match the running environment (silent numerical drift being the unacceptable failure). This ADR fixes the *recovery*: the **only** sanctioned path is **retrain under the new version**. There is no "rebuild the old environment to load the old artifact" path. A fitted artifact with no training data behind it is simply not portable across versions — that is the deal. This **supersedes ADR-0063's consequence** that "a pure-inference consumer must reconstruct the exact pinned environment": reconstruction is no longer a supported workflow, and consequently the lockfile below is *not* an artifact-recovery mechanism.

- **`pyproject.toml` publishes ranges, with tightened upper caps on the numeric/pickle-critical deps.** DataForgeML is a co-installable library, so its declared dependencies stay **ranges** — exact `==` pins would make it un-installable next to any project needing a different sklearn, and would still leave the transitive graph floating. But the numerically-critical, pickle-affecting deps — `scikit-learn`, `numpy`, `joblib`, `scipy` — get **upper caps to the next untested minor** (tested on `1.8.x` → `>=1.8,<1.9`) so a fresh install can never silently pull an *untested* upstream that breaks DataForgeML's code or numerics. The remaining deps (`polars`, `pandas`, `chardet`, `iterative-stratification`, `diptest`) keep open-ended lower bounds. The current open `>=` upper bounds (`scikit-learn>=1.0.0`, `numpy>=2.0.0`, `scipy>=1.10.0`) must be tightened accordingly. The caps advertise the tested band; the **artifact-level enforcement is the runtime `produced_with` stamp from ADR-0063**, not the pins — and the per-dependency stamp refuse is retained as belt-and-suspenders against range slack (the "Alice and Bob install the same DataForgeML version but resolve different sklearn" hole).

- **A committed lockfile freezes CI, and nothing else.** The project commits a **`uv.lock`** and CI runs from it. With reconstruction off the table, its sole job is **deterministic CI**: an upstream release can only reach the project inside a deliberate dependency-bump PR — it can never ambush an unrelated commit (e.g. a docstring fix going red because scipy shipped a floating-point patch overnight). A red build therefore always means *our code* broke. The lock is **not shipped in the wheel** and is not a consumer artifact. `uv` is adopted for environment/lock management; the `setuptools` build backend is unchanged, so published wheels and the `pip install dataforge-ml` experience (ranges) are unaffected.

- **Freshness and safety are an automated three-move pipeline.** (1) **Dependabot** opens PRs for new dependency versions (weekly) and for security advisories (immediately) — it covers both jobs. (2) **`pip-audit`** runs in CI and **fails the build** if any resolved dependency carries a known CVE. (3) The **full test suite is the gate**; green → merge → cut a new DataForgeML release. A flagged CVE is not special-cased — it rides the same update→test→release path, prioritized.

- **Any accepted change to a pickle-critical dependency forces a new DataForgeML version.** Two environments must never wear the same DataForgeML version label while carrying different sklearn/numpy/joblib — that is exactly the range-slack hole the master-key model cannot tolerate. So a bump to a pickle-critical dep is only ever released *as* a new library version, and that new version refuses (→ retrain) artifacts made by the old one. This is the accepted churn cost of the master-key model, and it is consonant with the pre-release, no-backward-compatibility stance.

## Status

accepted

## Considered Options

- **Library-version-only load gate (drop the per-dependency checks).** Appealing simplicity — "the library version already implies the deps." Rejected: because published ranges let two installs of the *same* library version resolve *different* sklearn, a library-only gate would load a cross-version pickle and drift silently. The per-dependency stamp refuse closes exactly that hole and is nearly free (a string compare against versions the manifest already records).
- **Exact `==` pins in `pyproject.toml`.** Rejected: hostile to co-installation (conflict-resolves against every other package in a consumer's environment), and it *still* doesn't lock the transitive graph — so it fails to deliver the robustness it promises. Exactness belongs in the lockfile, not the published metadata.
- **Reconstruct-the-old-environment recovery.** Rejected in favour of retrain-only. Supporting artifact portability across versions would require preserving and rebuilding old dependency environments for load — complexity the project explicitly refuses under a no-backward-compat, pre-release stance.
- **No lockfile (CI installs newest-in-range each run).** Rejected: upstream releases would ambush unrelated PRs with numeric-drift test failures, fighting the bit-identical / accuracy-over-speed posture. The lock costs effectively nothing (auto-generated by uv, auto-bumped by Dependabot).
- **`safety` for vulnerability scanning.** Rejected: its full advisory database is now gated behind a paid account. `pip-audit` is free and draws from the official PyPA Advisory Database.
- **Renovate for update automation.** Deferred: more configurable (grouping, custom schedules, auto-merge) than currently needed. Dependabot is lower-friction, built into GitHub, and already covers both routine and security update PRs; revisit if finer control is wanted.

## Consequences

- **ADR-0063's "pure-inference consumer reconstructs the pinned environment" consequence is superseded.** The recovery contract is now retrain-only; the lockfile is a CI concern, not an artifact-recovery mechanism.
- **`pyproject.toml`'s open upper bounds must be tightened** for the numeric/pickle-critical deps as specified above.
- **New project artifacts:** a committed `uv.lock`, a `.github/dependabot.yml`, and a `pip-audit` step in CI; `uv` becomes the env/lock tool (build backend unchanged). The `pip freeze`-style `requirements.txt` is retired in favour of the lock.
- **More frequent minor releases** as caps are bumped and deps refreshed — accepted churn, and precisely the freshness the ticket set out to achieve. Each such release invalidates prior artifacts (→ retrain), by design.
- The **per-dependency stamp refuse in ADR-0063 is retained**, not redundant: it is the safety net for range slack under a matching library version.
