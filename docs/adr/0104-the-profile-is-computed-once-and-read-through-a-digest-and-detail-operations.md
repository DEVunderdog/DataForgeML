# The profile is computed once and read through a digest and detail Operations

The full profile can't go in front of a model. On Ames (2,930 rows, 82 columns) its compact JSON is about 304k tokens with correlation on and 47k with it off (`o200k_base`, measured). Of the 304k, 129k is `target_correlations`, a byte-identical copy of the feature matrices (G1). Correlation profiling is off by default, yet it silently decides whether routing can reach KNN and MICE (D4). The profile also doesn't record the config that produced it (G13). ADR-0102 gives every Operation one bounded shape and a Result Budget (8k then, 10k since ADR-0109), and says detail comes from narrower Operations over a Handle. This ADR decides how "describe the dataset" splits into those calls.

## Status

accepted. Resolves [How is describing the dataset split into small calls?](https://github.com/DEVunderdog/DataForgeML/issues/576) on [Map: an agent-native DataForgeML](https://github.com/DEVunderdog/DataForgeML/issues/568). It amends ADR-0103 (Lineage records arguments) and the profiling defaults of ADR-0030. Planning only: no code changes yet.

**Amended by [ADR-0112](0112-agents-reach-the-library-through-a-cli-and-a-skill.md) (2026-10-11).** The Dataset Digest's column rows and every detail list go on the wire in Table Layout. Default page sizes (`describe_columns` 15, and the others) may shrink to meet the 25,000-character Result Budget.

## Decisions

- **The profile is computed once, and every detail Operation reads the stored profile.** `profile` computes everything, writes a profile Handle, and returns the **Dataset Digest**. Detail Operations never recompute and never touch the frame. Gained: write-once Handles stay true (ADR-0103), and detail calls are cheap, stateless and safe to run in parallel. Given up: work the Agent never asks for is still paid for up front.
- **Correlation and nonlinearity profiling are always on. `compute_correlation` and `compute_nonlinearity` are deleted.** On Ames, turning correlation on changes 7 routes. Three scalar fills (Lot Frontage, Mas Vnr Area and Garage Area) and two GMM fills move to KNN or MICE. Gained: the better routing is the only routing, and the Agent can't silently cut its own imputation quality without knowing. D4 dissolves, because there is no switch to explain. Given up: speed. Profiling Ames takes 8.2 s instead of 1.1 s, and the cost grows with the square of the column count. Nothing quadratic goes on the wire, so the cost is time and Workspace space, not tokens. A bound for very wide tables belongs to the map's resource-bounds item.
- **"Probe" is not used for these calls.** The glossary already uses "probe" for the nonlinearity model probe behind Explainable Variance. The calls are detail Operations over the profile Handle. Gained: one meaning per word. Given up: nothing.
- **The Dataset Digest is the dataset card plus at most 100 column rows.** The card holds the row and column counts, duplicates, sparsity, targets with problem types, and what the Agent set: type overrides, sentinels and exclusions. Each column row holds the semantic type, numeric kind, null ratio and severity, cardinality, skew, outlier density and flags. The digest has no literal values. Flagged columns come first, then ties in dataset order. Measured on Ames at full precision: 4,377 tokens, about 53 per column, so 82 columns fit whole. Gained: one call shows the whole dataset at low resolution, within budget up to 100 columns. Given up: a wider table needs a second call, `describe_columns`, to see the rest. The cap counts columns, not tokens, so very long column names can still push it past budget. Only the fixture test catches that.
- **Five detail Operations, each with a fixed cap.**
  - `describe_columns(names=null, flagged_only=null, offset, limit)`: one column's full profile per item, without literal values, at most **15** per call. The worst Ames column is 443 tokens.
  - `top_associations(column=null, metric=null, k, min_abs=null)`: Pearson, Cramér's V and η² in one list sorted by strength, at most **100** pairs. That is about 28 tokens per pair, and the top 10 cost about 310 tokens.
  - `missingness(column=null, k)`: the row-missingness distribution plus the strongest co-missing pairs, at most **100** pairs. The top 10 cost about 330 tokens.
  - `describe_target(name)`: one target's problem type and stats.
  - `value_summary(column, k)`: top values, rare labels, bounds and percentiles, at most **20** of each. It is the only Operation that returns literal values, and whether it is offered waits on the literal-value policy.

  Gained: everything in the profile is reachable, and each call fits well inside the Result Budget. Given up: more Operations to document and test.
- **Histograms leave the wire.** No Operation returns one. Gained: a bounded but real literal-value leak is gone, and no list on the wire grows with the row count (`bins="auto"`). Given up: the shape of a distribution is reduced to skew, kurtosis, tail tags and percentiles. An Agent that needs the full shape computes it in Code Mode.
- **`target_correlations` is deleted (G1).** Feature-to-target strength is `top_associations(column=<target>)`, and the target's own stats are `describe_target`. Gained: 129k tokens on Ames and one source of truth. Given up: nothing measured. The block was a copy.
- **The config that produced a Handle is kept in its Lineage, not echoed on results (G13).** Lineage now records the Operation's arguments as passed, as well as the Operation and the Handles it read. Arguments are already JSON-able (ADR-0099), and `null` means default (ADR-0101), so the arguments plus the library version pin the resolved config. Gained: one answer to "how was this made" for every Handle kind, not only the profile, and no second copy on the digest to drift. Given up: Lineage records what was passed, not the resolved thresholds. Reading the resolved value of a default means knowing that library version's default.

## Considered Options

- **Each detail Operation computes lazily on demand.** Rejected: a result would depend on when it was called, a lazy call needs the frame as well as the profile, and the Handle would no longer name a fixed object.
- **Keep the correlation switch and show its effect on routing to the Agent.** Rejected: showing what it would change means computing correlation anyway, and a choice the Agent gets wrong without knowing is the worst kind of default.
- **Echo the full ProfileConfig on the digest.** Rejected: about 60 thresholds on every call, plus a second copy of what Lineage holds.
- **A `describe_profile_config` detail Operation.** Rejected: it fixes the profile only, and every other Handle kind would need its own.
- **Paging the digest by tokens or bytes.** Rejected by ADR-0102: caps are counts, so a cut result still matches its schema.

## Consequences

- ADR-0103 is amended: a Handle's Lineage records its Operation's arguments as passed.
- The routing branches that read `feature_correlation is None` no longer depend on a config switch. `feature_correlation` is empty only when the data has nothing to correlate.
- The Operation inventory inherits the five detail Operations and the digest's fields. The fixture test for the Result Budget runs them on Ames and on a synthetic wide table.
- `value_summary` is the literal-value policy's first consumer.
