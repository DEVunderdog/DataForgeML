# DataForgeML Domain Glossary

## Pipeline & Phases

- **Pipeline** — An ordered sequence of **Phases** that transform a dataset from raw input to ML-ready features.
- **Phase** — A discrete stage of the pipeline with a specific responsibility. Currently defined phases are:
    1. **Profiling** — Structural analysis and type detection.
    2. **Imputation** — Handling missing values.
    3. **Outlier Detection** — Identifying and handling extreme values.
    4. **Normalization** — Transforming distributions (e.g. Log, Box-Cox).
    5. **Encoding** — Converting categorical data to numeric.
    6. **Scaling** — Rescaling numeric features.

## Column Exclusions

- **Hard Exclusion** — The immediate removal of a column from the dataset. Dropped columns are invisible to all downstream phases. Managed via `exclude_columns`. `PipelineConfig.add_exclusions(cols)` is the programmatic entry point for promoting columns into the Hard Exclusion set — used by `FittedImputer.apply_exclusions` to propagate Phase 2 `DropCandidate` columns into all downstream phases. See ADR 0023.
- **Soft Exclusion** — Bypassing the logic of one or more phases for a specific column while retaining that column in the dataset. Declared explicitly via `phase_exclusions` in `PipelineConfig`. Distinct from assigning `SemanticType.Identifier`, which is a type-routing decision, not an exclusion declaration.

## Phase Orchestration

- **Phase Orchestrator** — The single entry point for a Phase. Owns all column routing, exclusion, and sequencing decisions for that phase. Calls `PipelineConfig.resolve_active_columns` to obtain the active column set and hands sub-processors only pre-decided column lists. Sub-processors hold no configuration reference and trust the column list they receive completely. For Phase 1, the Phase Orchestrator is `StructuralProfiler`. For Phase 2, the Phase Orchestrator is `ImputationOrchestrator`.
- **Sub-processor** — A focused computation unit within a phase (e.g. `NumericProfiler`, `MissingnessProfiler`). Receives `(DataFrame, list[str])` and profiles exactly those columns — no routing, no eligibility checks, no access to `ProfileConfig` or `PipelineConfig`. If a sub-processor needs a computation parameter (e.g. a threshold), it receives a **Phase Sub-Config** in its constructor — a purpose-built parameter bundle containing only that sub-processor's threshold constants. The Phase Sub-Config carries no routing state; all routing and eligibility decisions remain with the Phase Orchestrator. See ADR-0002, ADR-0030.
- **Imputation Sub-processor** — A SemanticType-scoped computation unit within Phase 2 (e.g. `NumericImputer`, `CategoricalImputer`). Receives `(DataFrame, list[str], StructuralProfileResult)` and fills missing values for exactly those columns using signals from the profile. Registered in `_IMPUTATION_REGISTRY` keyed by `SemanticType`. `SemanticType.Text` and `SemanticType.Identifier` have no registered imputer and are skipped silently.

## Configuration

- **PipelineConfig** — The master configuration object for the entire pipeline. Cross-phase: owns global exclusions, phase exclusions, column type overrides, references to each phase's Sub-Configuration, and `random_seed: Optional[int]` (seed for all stochastic pipeline operations, including GMM Sampling during bimodal imputation; `None` produces non-deterministic output). Lives at the package root, not inside any phase module.
  _Avoid_: pipeline settings, global config
- **Phase Config** — A specialized configuration object (e.g. `ProfileConfig`, `ImputationConfig`) that holds parameters relevant to exactly one phase. Nested inside `PipelineConfig` as a named field. Phase configs are phase-owned; they live in their respective phase module. Threshold configurability is scoped to the phase that owns the threshold: definitional thresholds (e.g. what counts as `MissingSeverity.High`) live in the phase that *produces* the label; operational thresholds (e.g. `knn_max_rows`) live in the phase that *consumes* it. Phase Configs may nest **Phase Sub-Configs** — one per sub-processor — composed as named fields (e.g. `ProfileConfig.missingness`, `ProfileConfig.numeric`) carrying only the threshold constants owned by that sub-processor. Orchestrator-level thresholds that no sub-processor owns (e.g. `row_drop_threshold`, `memory_threshold_mb`) remain as top-level fields on the Phase Config directly. See ADR-0030.
  _Avoid_: sub-config, modular config
- **Phase Sub-Config** — A purpose-built parameter bundle for a specific sub-processor within a phase. Contains only the threshold constants that sub-processor uses to produce its labels. Passed to the sub-processor's constructor by the Phase Orchestrator, which extracts it from the corresponding field on the Phase Config. Carries no routing state, exclusions, or cross-phase signals — those remain in `PipelineConfig` and `PhaseConfig`. Phase Sub-Configs for Phase 1: `MissingnessProfileConfig` (`ProfileConfig.missingness`), `NumericProfileConfig` (`ProfileConfig.numeric`), `TypeDetectionConfig` (`ProfileConfig.type_detection`), `CategoricalProfileConfig` (`ProfileConfig.categorical`), `CorrelationProfileConfig` (`ProfileConfig.correlation`), `DatetimeProfileConfig` (`ProfileConfig.datetime_`). The term "shared config object" in ADR-0002 refers to `ProfileConfig` and `PipelineConfig` — sub-processor constructors may accept a Phase Sub-Config. See ADR-0030.
  _Avoid_: profiler config, sub-processor config
- **SplitConfig** — The configuration object for `DataSplitter`. Carries `max_stratification_signals` (cap on the stratification label matrix width) and `boolean_minority_threshold` (minimum true/false ratio for a boolean column to generate a stratification signal). Nested on `PipelineConfig` as `split: SplitConfig`. `DataSplitter.__init__` accepts an optional `SplitConfig`, defaulting to `SplitConfig()`. See ADR-0030.
- **Column Override** — An explicit `SemanticType` assignment for a named column stored in `PipelineConfig.column_overrides`, bypassing auto-detection for that column in all downstream phases. Set via `set_column_type(column, type)` for a single column or `set_columns_type(columns, type)` for a batch of columns sharing the same type. Storage is always `dict[str, SemanticType]` (column → type), regardless of how the override was declared.
  _Avoid_: type override, manual type
- **resolve_active_columns** — A centralized method in `PipelineConfig` that calculates the final set of columns a phase should process by reconciling hard exclusions, phase-specific soft exclusions, and the available data.
- **Public API** — The curated set of names importable directly from `dataforge_ml` (the package root). Includes: entry points, config objects, shared enums, and result types users type-hint against. Diagnostic and internal types (e.g. `TypeFlag`, `NumericKind`, per-modality stats dataclasses) are accessible via submodule imports but are not part of the Public API.

## NonlinearityTag

A per-column signal computed by Phase 1 (`NumericProfiler`) and stored in `NumericStats`. Captures the degree to which the relationship between a numeric column and its predictors departs from linearity. Used by Phase 2 to auto-select the estimator inside the Regression `IterativeImputer`.

- **Linear** — Pearson ≈ Spearman across predictors; mutual information consistent with linear dependence; R² gap between linear and tree model is small.
- **MonotonicNonlinear** — Spearman rank correlation significantly exceeds Pearson; relationship is non-linear but monotonic.
- **ComplexNonlinear** — high mutual information not explained by Pearson or Spearman alone; R² gap confirms tree model meaningfully outperforms linear model; relationship has non-monotonic or interaction-driven structure.
- **Unpredictable** — both linear and tree model R² are near zero (< 0.1); no model family provides meaningful uplift over a scalar fill. Acts as an **unconditional pre-routing guard** in Phase 2: regardless of missingness mechanism (MCAR, MAR) or severity, the column routes to Median and no model-based strategy is attempted. The MAR mechanism may still be real, but if no numeric predictor achieves meaningful R², model-based imputation produces no uplift over a scalar fill. Recorded in `ColumnImputationRecord.signals`. See ADR-0016. **Exception (bimodal):** when `NumericFlag.Bimodal` is also set and no correlated features are available, the column routes to GMM Sampling instead of Median — GMM Sampling fills from the distribution shape and is not subject to the R² guard. See ADR-0032.

Computed from four signals, all always applied: (1) Spearman/Pearson discrepancy, (2) `mutual_info_regression`, (3) R² gap test (LinearRegression vs shallow RandomForestRegressor on complete rows), (4) Breusch-Pagan heteroscedasticity test on linear model residuals. Phase 2 uses this tag first as a pre-routing guard (`Unpredictable` → Median, unconditionally), then as an estimator selector for non-Unpredictable columns: `Linear` → `Pipeline([StandardScaler, BayesianRidge])`; `MonotonicNonlinear` → `RandomForestRegressor`; `ComplexNonlinear` → `GradientBoostingRegressor` (large datasets) or `RandomForestRegressor` (small).
  _Avoid_: non-linearity score, linearity flag

## NumericKind

The discrete/continuous classification of a numeric column, computed by `NumericProfiler` and stored in `NumericStats.numeric_kind`. Lives in `models/_data_types.py` because both Phase 1 (profiling) and Phase 2 (imputation strategy routing) consume it.

- **BoundedDiscrete** — a numeric column whose values form a closed, finite set. Classified by a compound four-signal test applied in `_classify_numeric_kind` (Phase 1); all four signals must pass: (1) tight sequence — observed values fill every integer slot between min and max (`range_span == n_unique`); (2) small range — `max - min ≤ 20`; (3) low cardinality — `n_unique / n_rows < 0.05` OR `n_unique ≤ 10` (the floor protects small datasets); (4) standard origin — `min == 0 or min == 1`. For float columns, an additional pre-check is required: all non-null values must be whole numbers (`value % 1 == 0` for all values); float columns with any fractional values are always `Continuous`. `n_rows` is passed into `_classify_numeric_kind` to support signal 3. In Phase 2, BoundedDiscrete columns are captured at **Priority 3** (the BoundedDiscrete gate) and run their own sub-chain: MAR and MCAR routing apply normally, but every model-based prediction is **domain-snapped** (`clip(round(prediction), min, max)`) to enforce the finite domain. Mode is the terminal fallback when all model-based size guards fail, and also the unconditional fill for `Unpredictable` and `NearConstant` sub-cases where no model-based uplift is available. Snap bounds (`NumericStats.min`, `NumericStats.max`) are stored in `ColumnImputationRecord.domain_snap_bounds` at fit time so `transform()` can apply the snap without the profile. NearConstant minority defined by exact equality to mode value. **Exception (bimodal):** when `NumericFlag.Bimodal` is also set, the Bimodal Imputation Framework applies (domain-constrained GMM Sampling and Cluster-Conditional fills snap to the nearest valid discrete value) — deferred to a future scope. See ADR-0035 (amends ADR-0018), ADR-0032.
- **Continuous** — all other numeric columns. Includes integer-typed and integer-valued float columns that fail any of the four signals above. NearConstant minority defined by values outside `mode ± 0.5 * IQR`.
  _Avoid_: integer column, categorical numeric, discrete column

## TailAsymmetryTag

A per-column signal computed by Phase 1 (`NumericProfiler`) and stored in `NumericStats.tail_asymmetry_tag`. Captures the asymmetry between the extreme right and extreme left tails, derived from the existing `PercentileSnapshot` as `(p99 - p95) / (p5 - p1)`. The raw ratio is stored as `NumericStats.tail_asymmetry_ratio`; `None` when `p5 == p1` (flat left tail — division undefined).

- **RightHeavy** — right extreme tail is significantly heavier than the left (ratio > `tail_asymmetry_right_threshold`, configurable in `NumericProfileConfig`)
- **LeftHeavy** — left extreme tail is significantly heavier than the right (ratio < `tail_asymmetry_left_threshold`, configurable in `NumericProfileConfig`)
- **Symmetric** — both extreme tails are balanced

For routing: `RightHeavy` or `LeftHeavy` upgrades the effective `SkewSeverity` by one level at routing time (stored severity unchanged). Provides finer signal than `SkewSeverity` alone — a `SkewSeverity.Moderate` column with a disproportionately heavy extreme tail is escalated to KNN the same as `SkewSeverity.Severe`. See ADR-0033.
  _Avoid_: tail ratio, asymmetry ratio

## NumericFlag

A set of per-column diagnostic annotations computed by Phase 1 (`NumericProfiler`) and stored as a list on `NumericStats.flags`. Used in Phase 2 routing via `NumericStats.has_flag(flag)`.

- **ScaleAnomaly** — column values span an anomalous number of orders of magnitude, suggesting unit mixing or data quality issues.
- **NearConstant** — mode frequency exceeds `near_constant_threshold` (default 90%): 90%+ of values share the same value. Caps distribution shape escalation in Phase 2 — model-based imputation learns near-constant predictions regardless of tail shape, so escalation is wasted. De-escalation takes priority over all distribution shape signals.
- **Bimodal** — the column's distribution has two distinct peaks, detected via Hartigan's Dip Test (`p < bimodal_dip_p_value_threshold`, configurable in `NumericProfileConfig`). When set, `NumericStats.bimodal_stats` is populated with `BimodalStats`. Routes to the Bimodal Imputation Framework in Phase 2. Mutually exclusive with `NearConstant` (a 90%-mode column cannot be bimodal). See ADR-0031, ADR-0032.
- **HighOutlierDensity** — the fraction of values beyond `outlier_sigma_threshold` standard deviations from the mean exceeds `high_outlier_density_threshold` (default 0.05). Stored as `NumericStats.outlier_density`. An independent escalation trigger in Phase 2's Priority 7 distribution shape condition, alongside `KurtosisTag.Leptokurtic`. See ADR-0033.

## BimodalStats

A dataclass stored as `NumericStats.bimodal_stats`. Present (non-`None`) when and only when `NumericFlag.Bimodal` is set; `None` for all other columns. Contains:

- **`dip_statistic`** — the raw dip statistic from Hartigan's Dip Test
- **`dip_p_value`** — the p-value from Hartigan's Dip Test; the detection threshold is `NumericProfileConfig.bimodal_dip_p_value_threshold`
- **`center1`**, **`center2`** — the two cluster means from the 2-component GMM fitted after the dip test fires. Used by the Bimodal Imputation Framework for cluster-conditional fill assignment and domain-constrained GMM Sampling.

The invariant is bidirectional: `NumericFlag.Bimodal` present ↔ `bimodal_stats` is not `None`. Part of the Phase 1 output; consumed by Phase 2 without re-computation.
  _Avoid_: GMM stats, bimodal metadata

## SemanticType

The ML-level interpretation assigned to a column by the type detector. One of:

- **Numeric** — a continuous or discrete quantitative variable. Profiled with descriptive statistics, histogram, percentiles.
- **Categorical** — a fixed-vocabulary string or integer label. Low cardinality, suitable for one-hot or ordinal encoding. Profiled with frequency distribution.
- **Text** — a natural-language string column. Values are multi-token, near-unique, and not fixed-vocabulary (names, descriptions, addresses). Excluded from correlation matrices; routed to the text profiler.
- **Boolean** — a binary column (native `pl.Boolean`, integer `{0,1}`, or string `{"true","false","yes","no"}`).
- **Datetime** — a temporal column (native date/datetime dtype or successfully coerced from string).
- **Identifier** — a unique key column with no statistical signal (UUIDs, codes, sequential indices). Stats are skipped entirely.

### Text vs Categorical

A string column is **Text** when its values are natural-language content — names, descriptions, addresses, free-form input — where each value is unique or near-unique and contains multiple tokens. **Categorical** strings are low-cardinality, fixed-vocabulary values where the set of labels is small and meaningful (e.g. `"S"`, `"C"`, `"Q"` for embarkation port).

The distinction matters for ML: Categorical columns go through frequency profiling and enter the correlation matrix (Cramér's V); Text columns do not — encoding a `Name` column as one-hot produces noise, not signal.

## Effective Null

A value that counts as missing beyond a standard Polars `null`. Detection combines dtype-driven automatic rules and user-declared numeric sentinels:

- **String / Utf8 columns** — standard null + empty string + sentinel strings (`"NA"`, `"NAN"`, `"NULL"`, `"NONE"`, `"?"`)
- **Float32 / Float64 columns** — standard null + `NaN` + `Inf`
- **Numeric sentinel** — a user-declared value (e.g. `-999`, `9999`) for a specific column, treated as effective null for that column only. Declared per-column in `ProfileConfig`; surfaced in `StructuralProfileResult` so Phase 2 can apply the same sentinel rules without a separate config reference. Not auto-detected from data. Applies to any numeric dtype.
- **All other dtypes** — standard null only (unless a numeric sentinel is declared)

## TypeFlag

A diagnostic annotation attached to a `ColumnProfile` to explain *why* the type detector reached its `SemanticType` verdict. Not used in any decision logic — read-only metadata in the output JSON.

Examples: `numeric_coerced` (string column successfully cast to float), `free_text_candidate` (heuristics detected multi-token natural-language content), `user_override` (the semantic type was explicitly set by the caller via `set_column_type` or `set_columns_type` — the detector's verdict was ignored for this column), `numeric_kind_override` (the `NumericKind` was explicitly set by the caller via `PipelineConfig.numeric_kind_overrides` — distinct from `user_override`, which is SemanticType-only; both flags can coexist on the same column when the user overrides both).

## Phase 2 — Imputation

### Missingness Mechanism

The causal reason why values are missing in a column. Determines which imputation strategy is appropriate.

- **MCAR** (Missing Completely At Random) — missingness is unrelated to any observed or unobserved value. No `MARSuspect` flag from Phase 1 and not declared MNAR. Simple imputation (Mean, Median, Mode) is valid.
- **MAR** (Missing At Random) — missingness is correlated with other observed columns. Detected in Phase 1 via `MissingnessFlag.MARSuspect` (missingness indicator Pearson correlation exceeds `ProfileConfig.missingness.mar_correlation_threshold` — configurable, default 0.6 — with any other active column, regardless of that column's semantic type). Missingness correlations are computed across all column types; a numeric column's missingness may correlate with a categorical column's missingness and still qualify as `MARSuspect`. Model-based imputation (MICE, Regression, KNN) is appropriate when numeric predictors are available; if `NonlinearityTag.Unpredictable` fires, the column falls back to Median regardless.
- **MNAR** (Missing Not At Random) — missingness is caused by the unobserved value itself. Cannot be detected from data alone. Declared explicitly by the user via `ImputationConfig.mnar_columns`. Handled with a data-derived fill (observed mean when `SkewSeverity.Normal`, observed median otherwise — computed from non-missing rows only) and a binary missingness indicator. The fill centres the missing rows within the observed distribution rather than placing them at an arbitrary sentinel, minimising directional bias for linear and distance-based downstream models while preserving the indicator as the primary MNAR signal. Phase 2 output remains null-free.
  _Avoid_: non-random missing, structurally missing

### Imputation Strategy

The method applied to fill missing values in a column. Selected by the `ImputationOrchestrator` based on SemanticType, missingness mechanism, severity, distribution shape, and dataset size. One of:

- **Mean** — fill with column mean. Applied to `SemanticType.Numeric` MCAR columns with `SkewSeverity.Normal`, `MissingSeverity.Minor`, and `KurtosisTag` not `Leptokurtic`. A `Leptokurtic` column escalates to KNN even at Minor severity because heavy tails make the mean an unreliable central-tendency estimate for missing rows.
- **Median** — fill with column median. The scalar fallback after distribution shape escalation is not triggered or model-based size guards are not met. Applied to: MCAR `Minor` with skew present but `KurtosisTag` not `Leptokurtic`; MCAR `Moderate` without Leptokurtic or `SkewSeverity.Severe`; MAR `Minor`/`Moderate` without distribution shape escalation; any severity or mechanism when `NonlinearityTag.Unpredictable` fires; any severity when `NumericFlag.NearConstant` is present (de-escalation cap — model-based is wasted when 90%+ of values share the mode). Also the terminal fallback when all model-based size guards fail.
- **Mode** — fill with most frequent value. Applied to `NumericKind.BoundedDiscrete` columns and `SemanticType.Categorical` / `SemanticType.Boolean` columns.
- **KNN** — fill using k-nearest neighbour rows. Applied in three routing contexts: (1) `MARSuspect` columns where Regression is disqualified by size; (2) `MissingSeverity.High` MCAR columns, subject to size guards; (3) Distribution shape escalation — any MCAR or MAR column at `Minor` or `Moderate` severity where `KurtosisTag.Leptokurtic` OR `SkewSeverity.Severe` is present, subject to size guards. Not applied when `NonlinearityTag.Unpredictable` is present (pre-routing guard) or when `NumericFlag.NearConstant` is present (de-escalation cap). Falls back to Regression if size guards are exceeded; falls back to Median if all size guards fail. `n_neighbors` is computed dynamically at Phase 2 fit time from three signals via a multiplicative formula: `n_features` (dimensionality raises k), feature matrix missingness fraction (high missingness raises k), and complete row fraction (low completeness raises k); bounded by `knn_min_neighbors` and `knn_max_neighbors` in `NumericImputationConfig`. `weights` is `"distance"` when distances are reliable (`miss_frac < knn_distance_weight_max_null_ratio` AND `n_features ≤ knn_distance_weight_max_features`); otherwise `"uniform"`. Feature columns are NaN-safe scaled (nanmean/nanstd from `train_df`) before fit; imputed values are inverse-scaled. Stored as `_FittedKNN(model, col_means, col_stds)` under `"knn"` in `FittedImputer.models`. The chosen `n_neighbors`, `weights`, and scaling status are recorded in `ColumnImputationRecord.signals`.
- **Regression** — fill using a single-column `IterativeImputer` that handles missing feature values iteratively, eliminating the need to drop incomplete rows during fit. The internal estimator is auto-selected from Phase 1's `NonlinearityTag`: `Linear` → `Pipeline([StandardScaler, BayesianRidge])`; `MonotonicNonlinear` → `RandomForestRegressor`; `ComplexNonlinear` → `GradientBoostingRegressor` (large datasets) or `RandomForestRegressor` (small); `Unpredictable` → falls back to Median. `max_iter` and `tol` are set dynamically in Phase 2 from seven data signals: NonlinearityTag, feature missingness count, R² strength, inter-feature correlation among missing features, complete row fraction, and scale-relative tol (derived from column IQR). Post-fit convergence is monitored via `n_iter_`; non-convergence is recorded in `ColumnImputationRecord.signals`. Applied when `MARSuspect` + `MissingSeverity.High` and KNN is disqualified by size. Requires `n_rows >= regression_min_rows`; falls back to Median otherwise.
- **MICE** — iterative multi-model imputation across all missing columns jointly. Applied when `MARSuspect` + `MissingSeverity.Severe`, or multiple columns are simultaneously `MARSuspect`, or `MissingSeverity.Severe` on any MCAR column (20–50% missing). Subject to dataset-size guards. The internal estimator is auto-selected by taking the most complex `NonlinearityTag` across all MICE columns and passing it to `RegressionEstimatorFactory` (the same factory used by the Regression strategy). If all MICE columns are `Unpredictable`, the block is skipped and each column falls back to Median individually. `max_iter` and `tol` are computed dynamically using the same seven-signal framework as Regression, with conservative aggregation across MICE columns: most-complex `NonlinearityTag`, feature matrix missingness fraction, minimum R²_linear across MICE columns, maximum pairwise Pearson `|r|` among MICE columns (from `CorrelationProfiler`), complete row fraction, minimum IQR across MICE columns for scale-relative `tol`, and `ComplexNonlinear` tol tightening. `initial_strategy` is `"median"` if any MICE column has `SkewSeverity >= Moderate`; otherwise `"mean"`. `imputation_order` stays at the sklearn default `"ascending"` (fewest-missing first — this is already the better-grounded order). `n_nearest_features` is set only when the MICE block exceeds `mice_n_nearest_features_min_cols` columns; value is the median count of MICE columns with `|Pearson r| > mice_correlation_threshold` against each target, capped at `mice_max_nearest_features`. Predictor relevance is derived from `CorrelationProfiler` value-level correlations, not from missingness `correlated_with`. `sample_posterior` stays `False`; posterior sampling is deferred to a future multiple-imputation scope. Convergence warning (`n_iter_ == max_iter`) and the chosen estimator are appended to every MICE column's `ColumnImputationRecord.signals`. Depends on Scope 0 (`NonlinearityProfiler`, `RegressionEstimatorFactory`, `NonlinearityTag` in `NumericStats`) being shipped first.
- **MNAR** (strategy) — fill with the observed mean (`SkewSeverity.Normal`) or observed median (any other severity) computed from the non-missing rows of the MNAR column. Always adds a binary missingness indicator (`{col}_missing`). Applied exclusively to declared MNAR columns. The computed fill value and the skew severity that drove the choice are recorded in `ColumnImputationRecord.signals`. Serialised as `"mnar"`; `FittedImputer.from_dict()` migrates legacy `"constant"` values automatically. No user-configurable fill override — the fill is always data-derived.
  _Avoid_: imputation method, fill strategy, constant fill
- **GMM Sampling** — fill by drawing values from the fitted 2-component Gaussian Mixture Model stored in `BimodalStats`. Does not predict from features — it samples from the column's known distribution shape. Applied exclusively to `NumericFlag.Bimodal` columns with `NonlinearityTag.Unpredictable` and no correlated features available (branch 4 of the Bimodal Imputation Framework). Preserves the bimodal marginal distribution in the imputed output. Requires `PipelineConfig.random_seed` for determinism. For `NumericKind.BoundedDiscrete` columns, samples are snapped to the nearest valid discrete value (domain-constrained GMM Sampling). See ADR-0032.
  _Avoid_: GMM imputation, mixture sampling
- **Cluster-Conditional Imputation** — fill by assigning each missing row to one of the two distribution clusters identified by the Bimodal Imputation Framework, then filling with the cluster-appropriate statistic. Cluster assignment uses (in order of availability) the declared grouping variable, feature-centroid nearest-neighbour assignment, or GMM posterior. For continuous bimodal columns, the fill is the cluster mean or median (skew-driven). For `NumericKind.BoundedDiscrete` bimodal columns, the fill is always the cluster mode — preserving the finite domain. See ADR-0032.
  _Avoid_: cluster fill, bimodal fill
- **Indicator** — not a fill strategy. Assigned to `{col}_missing` columns that `transform()` produces when `indicator_added=True` for the source column. Written into `self.records` at fit time so the full output schema is inspectable before `transform()` runs. Carries `SemanticType.Boolean`. See ADR 0025.

### Missingness Indicator

A binary feature column (`column_missing = 1` if value was missing, else `0`) appended to the DataFrame alongside imputation. Always added for MNAR columns. Optionally added for any other column declared in `ImputationConfig.add_indicator_columns`. Each indicator column gets a `ColumnImputationRecord` with `ImputationStrategy.Indicator` and `SemanticType.Boolean` written into `self.records` at fit time (see ADR 0025). `apply_exclusions(config)` registers indicator columns as Soft Exclusions for Phases 3–6 so downstream phases skip them automatically.
  _Avoid_: missing flag column, null indicator

### ImputationResult

The output of Phase 2. Contains:
1. The imputed `pl.DataFrame` with all effective nulls filled and any missingness indicator columns appended.
2. A per-column audit log `dict[str, ColumnImputationRecord]` recording the strategy used, fill value (for scalar strategies), whether an indicator was added, the signals that drove the decision (severity, mechanism, size guards triggered, etc.), and for model-based strategies, convergence status (whether `IterativeImputer.n_iter_` reached `max_iter` without converging).
3. `dropped_columns` — column names removed because their `DropCandidate` flag fired (>50% effective missing). These columns are absent from the returned `dataframe`.
4. `exclusions_applied: bool` — `True` when `FittedImputer.apply_exclusions(config)` was called before `transform()`, meaning dropped columns have been registered as Hard Exclusions in `PipelineConfig`. Phase 3's orchestrator checks this flag and raises if `False`. Defaults to `False`; not serialised in `FittedImputer.to_dict()` because it is operational state, not fit state.
  _Avoid_: imputation output, filled dataset

### Numeric Imputation Decision Priority

For `SemanticType.Numeric` columns, the `ImputationOrchestrator` applies the following priority order (first match wins):

1. `DropCandidate` flag (`>50%` missing) → drop column entirely
1.5. **`per_column_strategy` override** — if declared for this column in `NumericImputationConfig.per_column_strategy`, use the declared strategy. All routing priorities below (2–7) are bypassed. Records `"per_column_strategy_override: user forced strategy=X"` in signals. If the declared strategy is model-based (KNN, Regression, MICE) and the dataset does not meet the size guard for that strategy, raises `ValueError` at fit time. See ADR 0029.
2. Declared MNAR → `ImputationStrategy.MNAR` (observed mean/median fill, skew-driven) + missingness indicator
3. **`NumericKind.BoundedDiscrete` gate** — fires before the Unpredictable guard and before MARSuspect routing. Inside the gate the column runs its own full sub-chain and exits; priorities 4–7 are never reached for BoundedDiscrete columns. Sub-chain (first match wins): (a) `NonlinearityTag.Unpredictable` → Mode — no model-based uplift available; Mode is always a valid domain member. (b) `NumericFlag.NearConstant` → Mode — 90%+ values share the mode; model-based is wasteful. (c) `NumericFlag.Bimodal` → deferred (Bimodal Imputation Framework for BoundedDiscrete is a future scope; currently falls through to the MCAR/MAR sub-chain below). (d) `MARSuspect` → domain-snapped MAR sub-chain (same MICE/Regression/KNN fallback order as Priority 5) → Mode terminal. (e) MCAR by severity → domain-snapped MCAR sub-chain (same severity routing as Priority 7) → Mode terminal. **Domain-snap**: every model-based prediction is post-processed as `clip(round(prediction), min, max)` using `NumericStats.min` / `NumericStats.max`, enforcing the finite domain. Scalar fills (Mean, Median) produced inside the gate are also snapped at fit time. Snap bounds are stored in `ColumnImputationRecord.domain_snap_bounds: Optional[tuple[float, float]]` so `transform()` can apply the snap without the profile. **Mode is the terminal fallback** for all sub-chain branches (replaces Median, which is not guaranteed to be a valid domain member). See ADR-0035 (amends ADR-0018).
4. **`NonlinearityTag.Unpredictable` (pre-routing guard)** — applies to non-BoundedDiscrete columns only (BoundedDiscrete handles Unpredictable inside its gate at Priority 3). → Median, unconditionally. Overrides MAR mechanism and all severity levels below. Recorded in signals. See ADR-0016.
5. `MARSuspect` → MICE (if Severe or multi-column MAR) → Regression (if High + `correlated_with` non-empty + size guards) → KNN (if High + no correlations, or Minor/Moderate + `KurtosisTag.Leptokurtic`/`SkewSeverity.Severe`, size-guarded) → Median
6. **`NumericFlag.NearConstant` (de-escalation cap)** → Median. Prevents distribution shape escalation; 90%+ values share the mode so model-based is wasteful.
7. MCAR by severity + distribution shape: `TailAsymmetryTag.RightHeavy` or `LeftHeavy` upgrades the effective `SkewSeverity` by one level before evaluation (stored severity unchanged). `NumericFlag.Bimodal` at any MCAR severity → Bimodal Imputation Framework. `Minor` + `SkewSeverity.Normal` + not `Leptokurtic` + not `HighOutlierDensity` → Mean; `Minor`/`Moderate` + (`KurtosisTag.Leptokurtic` OR `SkewSeverity.Severe` OR `NumericFlag.HighOutlierDensity`) → KNN (size-guarded) → Median; `Moderate` otherwise → Median; `High` → KNN → Regression → Median; `Severe` (20–50%) → MICE → KNN → Median. For MCAR routing, value-level Pearson correlations from `CorrelationProfiler` serve as a feature-predictability check: when max `|r| < 0.2` against all available numeric features, model-based strategies are skipped and Median is applied directly (this check does not apply when `NumericFlag.Bimodal` is set — bimodal columns follow the Bimodal Imputation Framework's own feature-count branches). See ADR-0017, ADR-0032, ADR-0033.

### Bimodal Imputation Framework

A four-branch routing sub-tree applied to any column with `NumericFlag.Bimodal` during Phase 2. Branches are evaluated in order; the first with sufficient evidence wins:

1. **Grouping variable available** — user-declared in `NumericImputationConfig.bimodal_grouping_variables` or auto-detected from Phase 1 profile. Fill with cluster-conditional statistic (mean or median within each group, skew-driven). Cleanest outcome.
2. **Many correlated features** — `≥ bimodal_min_correlated_features` numeric features with `|r| > 0.2`. Use MICE or KNN. Regression and MICE with tree-based estimators are preferred; KNN risks valley fills when neighbors span both clusters.
3. **Few correlated features** — fewer than `bimodal_min_correlated_features` features with `|r| > 0.2`. Use Cluster-Conditional Imputation: assign missing rows to the nearest cluster centroid (from GMM centers and cluster-conditional feature means), fill with cluster statistic.
4. **No correlated features** — GMM Sampling from the fitted 2-component GMM.

For continuous bimodal columns where `NonlinearityTag.Linear` is detected, the Regression estimator is overridden to `RandomForestRegressor` regardless of the tag — linear models produce valley predictions for bimodal targets. `bimodal_min_correlated_features: int = 3` in `NumericImputationConfig`.
  _Avoid_: bimodal strategy, bimodal routing

### ImputationFitDiagnostic

A per-column quality assessment computed during `ImputationOrchestrator.fit()` and attached to `ColumnImputationRecord.diagnostic`. Present for model-based strategies (KNN, Regression, MICE); `None` for scalar strategies (Mean, Median, Mode) and Passthrough, Dropped, MNAR.

- **`r2_train`** — R² on held-out complete rows; `None` when fewer than `refit_r2_min_complete_rows` complete rows are available or the strategy is scalar.
- **`rmse`** — root-mean-square error of imputed vs. true values on the same held-out complete rows used for `r2_train`; `None` under the same guard. Diagnostic-only — not consumed by `suggest_refit_config` because RMSE is in column-specific units and has no universal threshold.
- **`mae`** — mean absolute error on the same held-out rows; `None` under the same guard. Diagnostic-only for the same reason as `rmse`.
- **`converged`** — whether `IterativeImputer` stabilised before reaching `max_iter`; `None` for KNN and scalar strategies.
- **`n_iter`** — actual iteration count of `IterativeImputer`; `None` for KNN and scalar strategies.
- **`imputed_mean`**, **`imputed_std`** — distribution of values imputed for the null rows in `train_df`.
- **`observed_mean`**, **`observed_std`** — distribution of non-null values in `train_df`.
- **`variance_ratio`** — `imputed_std / observed_std`; values below `refit_variance_ratio_threshold` indicate distribution collapse (the model is predicting near-constant values for all null rows).

For MICE, all three of `r2_train`, `rmse`, and `mae` are computed per column from a single shared holdback: rows that are complete across **all** MICE columns are identified, 20% are held back, and each MICE column's metrics are evaluated against that same holdback set. If the intersection of complete rows is smaller than `refit_r2_min_complete_rows`, all three fields are `None` for every column in the block. Part of the Public API.
  _Avoid_: fit diagnostic, quality score

### suggest_refit_config

A standalone function importable from `dataforge_ml`. Accepts `dict[str, ColumnImputationRecord]` (from `FittedImputer.records`) and the current `NumericImputationConfig`; returns a new `NumericImputationConfig` with per-column overrides pre-populated from three diagnostic rules:

1. `converged=False` → sets `per_column_max_iter[col] = computed_max_iter × refit_max_iter_multiplier`
2. `r2_train < refit_r2_threshold` → sets `per_column_strategy[col] = Median`
3. `variance_ratio < refit_variance_ratio_threshold` → records a flag in signals; no automatic strategy change (requires user review)

When `r2_train` is `None` (insufficient complete rows), rule 2 is skipped for that column. Rule 2 also skips columns already in `mnar_columns` — MNAR columns do not go through the routing engine and their R² is not a meaningful signal; adding them to `per_column_strategy` would fail the construction-time conflict validation. The caller passes the returned config to a new `ImputationOrchestrator` and calls `fit()` again — the re-fit entry point is unchanged. Part of the Public API.

### Per-Column Imputation Override

An explicit assignment for a named column in `NumericImputationConfig` that bypasses the computed or dynamically-derived value during `NumericImputer.fit()`. Four override types:

- **`per_column_strategy`** — forces a specific `ImputationStrategy`, bypassing all routing priorities 2–7 (MNAR through MCAR). Fires at Priority 1.5 — after `DropCandidate` (which remains a hard gate) but before MNAR routing. Valid strategies: `Mean`, `Median`, `Mode`, `KNN`, `Regression`, `MICE`, `Constant`. `Passthrough`, `Indicator`, `MNAR`, and `Dropped` raise `ValueError` at construction. A column in both `mnar_columns` and `per_column_strategy` raises `ValueError` at construction. For model-based strategies, size guards are checked at fit time and raise `ValueError` if not met (no silent fallback). See ADR 0029.
- **`per_column_constant_fill`** — the user-specified fill value (float) for a column declared `Constant` in `per_column_strategy`. Required when any column is declared `Constant`; raises `ValueError` at construction if absent.
- **`per_column_max_iter`** — overrides the dynamically computed `max_iter` for Regression and MICE columns.
- **`per_column_n_neighbors`** — overrides the dynamically computed `n_neighbors` for KNN columns.

`per_column_strategy`, `per_column_max_iter`, and `per_column_n_neighbors` are populated automatically by `suggest_refit_config` or set manually by the user. `per_column_constant_fill` is set manually only — `suggest_refit_config` never declares `Constant` strategy and therefore never requires a companion fill value.
  _Avoid_: column parameter override, per-column config

## Strategy Routing vs Parameter Estimation

Two distinct responsibilities inside `ImputationOrchestrator.fit()`:

- **Strategy Routing** — selecting the imputation method for each column. Uses signals from the `StructuralProfileResult` (computed on the full dataset): `MissingSeverity`, `MissingnessFlag.MARSuspect`, `NumericKind`, `SkewSeverity`, `KurtosisTag`, `TailAsymmetryTag` (severity upgrade), `NonlinearityTag` (pre-routing guard), `NumericFlag.NearConstant` (de-escalation cap), `NumericFlag.Bimodal` (Bimodal Imputation Framework), `NumericFlag.HighOutlierDensity` (independent escalation trigger), and value-level Pearson correlations from `CorrelationProfiler` (feature-predictability check for MCAR routing and bimodal feature-count branching). Routing decisions do not embed any values from the test set into the training process — they are method-selection decisions only.
- **Parameter Estimation** — learning the actual fill values and models. Uses only `train_df`: computes medians, modes, fits KNN / Regression / MICE models. The test set is completely invisible during this step.

This distinction is why `ImputationOrchestrator.fit()` accepts both `(train_df, profile)`: the profile informs routing (best full-dataset description of each column's distribution); `train_df` informs estimation (the only data the model is allowed to see). Using full-dataset profile statistics for routing is not data leakage — no test-set values appear in the fitted parameters.
  _Avoid_: strategy selection, fill learning

## Fit/Transform Discipline

The principle that all transforming phases (Phases 2–6) must learn their parameters exclusively from training data and apply those learned parameters to any split — including the test set. Phase 1 (Profiling) is exempt: it is non-transforming and may run on the full dataset without leakage risk.

- **Fit** — the step where a transforming phase reads the training DataFrame and the `StructuralProfileResult`, selects strategies per column, and learns all fill parameters (scalar statistics, model weights). Returns a `FittedImputer` (or equivalent fitted object) that is independent of the orchestrator.
- **Transform** — the step where a `FittedImputer` applies its already-learned parameters to any DataFrame (train or test) and returns an `ImputationResult`. Does not re-learn any parameters. Raises `UnfittedColumnError` if the DataFrame contains a column with missing values for which no parameters were learned during fit.
- **FittedImputer** — the stateless, serializable output of `ImputationOrchestrator.fit()`. Stores the learned strategy and fill parameters per column as a plain data structure. Supports `to_dict()` / `from_dict()` round-trip for persistence. `self.records` is a complete manifest of the full train schema: every column present in `train_df` at fit time receives a `ColumnImputationRecord`, regardless of semantic type. Columns with no registered imputer (Text, Identifier, Categorical, Boolean, Datetime) receive `strategy=Passthrough`. Indicator columns (`{col}_missing`) receive `strategy=Indicator`, written at fit time. The invariant is: no record = column was never seen during fit. See ADR 0024 and ADR 0025. `apply_exclusions(config)` propagates all `ImputationStrategy.Dropped` columns into `PipelineConfig` as Hard Exclusions via `add_exclusions`, and registers all `ImputationStrategy.Indicator` columns as Soft Exclusions for Phases 3–6; sets an internal `_exclusions_applied` flag (not serialised) so that `transform()` stamps `ImputationResult.exclusions_applied = True`. Must be called again after `from_dict()` if phases are chained. See ADR 0023.
  _Avoid_: fitted pipeline, trained imputer
- **UnfittedColumnError** — raised by `FittedImputer.transform()` when the input DataFrame contains missing values in a column that had no missing values during `fit()` (and therefore no strategy was ever selected for it — recorded as `Passthrough`). Signals a split imbalance that must be corrected upstream.
- **UnseenColumnError** — raised by `FittedImputer.transform()` at the start of transform, before any fill logic, when the input DataFrame contains a column with no entry in `self.records`. Because `self.records` is a complete train schema manifest (ADR 0024), a missing record unambiguously means the column was never seen during fit. Fires for all unknown columns in a single raise, regardless of whether they have missing values. Distinct from `UnfittedColumnError`: "unfitted" means seen during fit with no missingness; "unseen" means never seen during fit at all. See ADR 0026.
- **FittedColumnAbsentError** — raised by `FittedImputer.transform()` when a column recorded in `self.records` with an active imputation strategy (any strategy other than `Dropped` or `Indicator`) is absent from the input DataFrame. Signals that the test DataFrame has fewer columns than the train DataFrame — always a pipeline bug, not an intentional removal. Distinct from `DroppedColumnAbsentWarning`, which covers columns the library itself decided to drop and where absence is a plausible intentional pre-processing step.
- **DroppedColumnAbsentWarning** — raised by `FittedImputer.transform()` when a column recorded as `ImputationStrategy.Dropped` is already absent from the input DataFrame. Indicates the input was pre-processed outside the library — either intentionally (user removed the column manually) or accidentally (wrong DataFrame passed). Fires once per absent column. Transform proceeds regardless. No library-level suppression mechanism; use Python's standard `warnings.filterwarnings` to suppress if the pre-removal is intentional.
- **TrainSplitImbalanceWarning** — raised by `ImputationOrchestrator.fit()` when the training split has a missing-value ratio below `split_imbalance_ratio_threshold × profile_effective_null_ratio` for any column that the full-dataset profile reports as having missingness. The check is proportional, not binary: a column that is 20% missing in the full dataset but only 2% missing in train triggers the warning even though train is not completely clean. Recommends using `DataSplitter.profile_stratified_split()`. String and float sentinels are normalised before the check (by `_resolve_effective_nulls`); numeric sentinel columns are exempt until Scope 5 ships.
  _Avoid_: SplitImbalanceWarning
- **TestSplitImbalanceWarning** — raised by `FittedImputer.transform()` when the test split has a missing-value ratio below `split_imbalance_ratio_threshold × profile_effective_null_ratio` for any column with a fitted imputation strategy. Fires for any split method, not only `profile_stratified_split`, because the `FittedImputer` receives the test DataFrame and can evaluate it independently. Signals that imputation quality cannot be reliably evaluated on test — the imputed model will run but the test set does not exercise the missingness distribution it was trained on.
- **fit_transform** — convenience method on `ImputationOrchestrator` that calls `fit(train_df, profile)` followed by `.transform(train_df)` and returns `tuple[FittedImputer, ImputationResult]`. The `FittedImputer` in the tuple is the same object that would be returned by a standalone `fit()` call; callers must use it to transform the test set via `fitted_imputer.transform(test_df)`. Returning both objects in the tuple prevents callers from discarding the `FittedImputer` and mistakenly re-fitting on the test set.

## RowMissingnessDistribution

A dataclass nested inside `MissingnessProfileResult` as the `row_distribution` field. Captures aggregate row-wise missingness statistics — how many columns are simultaneously missing per row across the dataset. Computed by `MissingnessProfiler` in Phase 1.

- **`row_missingness_p90`** — the 90th-percentile count of missing columns per row. A row whose missing-column count exceeds this threshold is considered "globally sparse." Used by `build_label_matrix` to generate the Compound missingness row signal for profile-stratified splitting.

## Profile-Stratified Split

A split mode on `DataSplitter` that consumes a `StructuralProfileResult` and produces a train/test partition whose distributional properties are representative across all signals that downstream phases depend on — not just the target column.

Derived stratification signals (auto-computed from the profile, no user configuration required):

- **Missingness density** — row-level count of columns with missing values; ensures columns with any missingness are proportionally represented in both splits (Phase 2)
- **Extreme value rows** — rows containing values in the tails of numeric distributions; ensures Phase 3 Outlier Detection sees the same extreme-value density in both splits
- **Rare label rows** — rows containing low-frequency categorical labels or rare non-mode values in near-constant columns; ensures Phase 5 Encoding has all labels in both splits

`DataSplitter.profile_stratified_split(profile, test_size)` is the canonical entry point for train/test splits. `DataSplitter.profile_stratified_kfold(profile, k)` is the equivalent for cross-validation. Both use `MultilabelStratifiedShuffleSplit` / `MultilabelStratifiedKFold` from the `iterative-stratification` package. When a target column is declared on `DataSplitter`, it is included in the label matrix automatically so target class proportions and distributional quality are preserved in a single pass. Users who need custom split logic (e.g. time-based splits) may use `random_split`, `time_split`, or `kfold` freely, but will receive a `TrainSplitImbalanceWarning` from `fit()` and a `TestSplitImbalanceWarning` from `transform()` if the split is distributional-quality unsafe.
  _Avoid_: smart split, missingness-aware split, profile-aware split

### Stratification Signal Taxonomy

The signals used to build the binary label matrix for profile-stratified splitting. Each signal produces one binary column in the matrix (1 = row exhibits this signal, 0 = does not). All derived per-row from the profile and raw DataFrame:

- **Missingness signal** — one label per column with `effective_null_ratio > 0` and without `MissingnessFlag.DropCandidate`. Derived from the DataFrame's null mask. `DropCandidate` columns (>50% missing) are excluded: they are dropped by Phase 2 and consume a label slot while protecting nothing downstream.
- **Joint MAR missingness signal** — one label per pair of `MARSuspect`-correlated columns. A row gets `1` if it is missing in BOTH correlated columns simultaneously, using the effective null mask (string sentinels included) for both columns. Preserves the correlation structure that MICE relies on.
- **Numeric extreme value signal** — one label per numeric column. A row gets `1` if its value is below `p5` or above `p95`. The upper threshold is extended to `p99` when `SkewSeverity.Severe` OR `KurtosisTag.Leptokurtic` — either condition alone is sufficient, because both indicate heavier-than-normal tails where p95 under-captures. Derived from `PercentileSnapshot` and `NumericStats.kurtosis_tag`.
- **Zero/negative value signal** — one label per numeric column where `SkewSeverity >= High` and `min <= 0`. A row gets `1` if its value is `<= 0`. Derived directly from the DataFrame (not in Phase 1 profile). Protects Phase 4 log-transform from unseen zero/negative values.
- **Rare categorical label signal** — one label per categorical column. A row gets `1` if it contains a value in `RareCategoryStats.rare_label_values` (values with row frequency below 5%). Derived from profile; no raw DataFrame re-query.
- **Boolean minority signal** — one label per boolean column where `min(true_ratio, false_ratio) < 0.05`. A row gets `1` if it holds the minority value. Derived from `BooleanStats`.
- **NearConstant numeric minority signal** — one label per numeric column with `NumericFlag.NearConstant` (mode frequency > 90%). A row gets `1` if its value differs from the column mode. For `NumericKind.Discrete` columns, exact equality is used; for `NumericKind.Continuous`, a band of `mode ± 0.5 * IQR` defines "at mode." Protects the structurally-rare non-constant rows that the extreme value signal misses in symmetric near-constant distributions.
- **Datetime future-date signal** — one label per datetime column with `DatetimeFlag.FutureDates`. A row gets `1` if its value is after the current timestamp. Future-dated rows produce out-of-distribution temporal features in Phase 5 Encoding if concentrated in a single split.
- **Compound missingness row signal** — one label for the dataset as a whole when `row_missingness_p90 > 0`. A row gets `1` if its count of missing columns exceeds `RowMissingnessDistribution.row_missingness_p90`. Protects "globally sparse" rows — those missing in many columns simultaneously — which the per-column missingness signals do not individually protect. These rows are the hardest inputs for MICE because they must be imputed jointly across all missing columns.
- **Target signal** — one label for the target column (if declared). Binary for classification; quantile-binned into 5 buckets for regression. Ensures class proportions are preserved alongside all distributional signals.

When the total number of signals exceeds the cap (`_MAX_STRATIFICATION_LABELS = 50`), signals are ranked by ascending proportion of 1s — rarest signals first — and the rarest 50 are retained. Rarest signals are most at risk of being zeroed out in a naive random split.
