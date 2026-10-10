# Schema generation probe: strict-portable JSON Schemas from DataForgeML types

Findings for wayfinder ticket #570 (map #568), run 2026-10-02.

## Resolution

**Short answer:** yes, for plain records and configs. Pydantic derives a schema from 30 of the 36 public dataclasses with no changes to the library. No schema produced so far is strict-portable as generated. Some of the fixes are mechanical and a transformer already does them. Five problems are structural, and no transformer fixes them: open maps, `Any`, live objects, forward refs hidden behind `TYPE_CHECKING`, and enums encoded two different ways. Recursion is **not** a problem: no type has a `$ref` cycle.

Run in a scratch venv against `main` @ `9b34b7d`, with the library installed unmodified. Versions: Python 3.12, pydantic 2.13.5, msgspec 0.22.0, anthropic 1.11.0, openai 3.23.0. Probe scripts were throwaway and are not stored.

### Method

- `TypeAdapter(T).json_schema()` in both `validation` and `serialization` modes, plus `msgspec.json.schema(T)`, over every dataclass and enum in `dataforge_ml.__all__`. Also run on the output-only internals `TypeFlag` and `EpochUnit`.
- Entry points: an args model built from each signature with `pydantic.create_model`, plus the return annotation. Covers `route`, `author`, `resolve_recipe`, `derive_units`, `fit_unit`, `c2st`, `imputation_score_c2st`, `serialize`/`deserialize`/`inspect`, and the public methods of `StructuralProfiler`, `DataSplitter`, `DataLoader`, `FittedImputer`, `FittedUnit`, `ImputationRecipe` and `ImputationRouting`.
- Each schema checked against the portable subset from the research doc §1.2. Then each schema was passed through **Anthropic's `transform_schema`** and **OpenAI's `_ensure_strict_json_schema`** to see what is left after the real transformers run.
- Second pass: forward refs injected into module namespaces (scratch only) to see past `NameError`s.
- Live check: built a real profile, routing, recipe, unit and `FitSignals` from a 300-row frame. Validated `TypeAdapter.dump_python(obj, mode="json")` and `obj.to_dict()` against the derived schema.

Validation and serialization modes gave identical schemas for every type. msgspec never did better than Pydantic, and it failed on three more types (see below).

### Mechanical gaps (every generated schema has them; a transformer fixes them)

- `additionalProperties` is never `false`.
- Fields with defaults are not `required`.
- `default` keys appear in the schema.
- A few constraint keywords: `format` (from `datetime`/`bytes`), and `minItems`/`maxItems` (from fixed tuples).

Anthropic's `transform_schema` fixes all of these when it doesn't crash. OpenAI's strict helper adds `additionalProperties:false` and `required` too.

### Per-type facts: structural blockers

| Type | Generates? | Structural blockers (after the mechanical fix) |
|---|---|---|
| `AuthoredColumn`, `ColumnRouting`, `ImputationUnit`, `FitSignals`, `PipelineEvent`, `SplitConfig`, all 7 profile Sub-Configs, `C2STResult`, `C2STScore`, `ColumnC2STResult` | yes | **none.** Pass after the mechanical fix |
| `ColumnEstimates` | yes | `domain_snap_bounds: tuple[float, float]` becomes `prefixItems` with `minItems`/`maxItems` (fixed tuple; not in the portable subset) |
| `ImputationConfig`, `NumericImputationConfig` | yes | 3 open maps (`dict[str, …]`). The private `_per_column_strategy`, `_per_column_constant_fill`, `_bimodal_grouping_variables` and `_mnar_columns` appear as **schema properties** with leading underscores |
| `ProfileConfig` | yes (Pydantic); **msgspec: no** (`InitVar` unsupported) | 8 open maps. Private `_numeric_sentinels`, `_string_sentinels`, `_datetime_epoch_units`, `_datetime_formats` exposed. Pydantic warns that the `InitVar` defaults are `property` objects |
| `PipelineConfig` | **no**: `NameError: NumericKind`. The name is imported under `TYPE_CHECKING` only. Generates once injected | 14 open maps. 12 underscore properties. `MappingProxyType` lives only on read-only properties, so it never reaches the schema, but the private backing fields do |
| `ImputationRouting` | yes | `mice_estimator: Any` gives a typeless `{}`. `column_routings` is an open map. A live estimator makes `dump_python` raise `PydanticSerializationError: Unable to serialize unknown type: BayesianRidge` |
| `ImputationRecipe` | yes | Everything in `ImputationRouting`, plus `dict[str, tuple[tuple[str, Any], ...]]` hyperparameter maps: open map, fixed-tuple `prefixItems`, and `Any`. 6 open maps total |
| `ColumnImputationRecord` | yes | `fill_value: Any \| None` gives a typeless branch |
| `FittedImputer` | yes | `units: list[Any]` plus `fill_value: Any`. 3 open maps |
| `StructuralProfileResult` | yes (Pydantic); **msgspec: no** (a union of 5 stats dataclasses; msgspec allows only one dataclass per union) | 12 open maps (`columns`, `targets`, the missingness matrix, sentinel and correlation maps). Typeless `Any` in `RareCategoryStats.rare_label_values` and `TopValueEntry.value`. Size: object depth 6, 317 properties inlined (under OpenAI's 10 / 5,000 limits) |
| `ColumnProfile` | yes (Pydantic); **msgspec: no** (same union) | The same two `Any` fields. `TypeFlag` inlines cleanly as an `enum`, so being internal is not a generation problem |
| `DatasetStats` | yes | 8 open maps |
| `C2STConfig`, `C2STProvenance`, `C2STReport` | yes | `classifier: Any \| None` is typeless. `C2STReport.columns` and `C2STProvenance.classifier_params` are open maps |
| `UnitFitResult` | **no**: `NameError: FittedUnit` (`TYPE_CHECKING`). Once injected: `FittedUnit` is an arbitrary class | live object |
| `ImputationResult`, `SplitResult`, `FoldResult`, `HoldoutCVResult` | **no**: `pl.DataFrame`. Even with `arbitrary_types_allowed=True`, the result is `PydanticInvalidForJsonSchema` | Payload Frames have no JSON Schema, full stop |
| All 12 public enums + `TypeFlag`, `EpochUnit` | yes: clean `{"enum": [values], "type": "string"}` | none, when used as a property (an enum alone is not an object root, as expected) |

### Entry points

| Entry point | Args | Return |
|---|---|---|
| `route`, `author`, `resolve_recipe`, `fit_unit`, `DataSplitter.profile_stratified_*`, `StructuralProfiler.__init__` | **fail**: `TYPE_CHECKING`-only annotations (`StructuralProfileResult`, `ImputationRecipe`, `NumericKind`) give a `NameError`. Same blockers as their types once resolved | as their types |
| `derive_units` | `routing`: 1 open map | `tuple[ImputationUnit, ...]`: array root (not an object) |
| `fit_unit`, `c2st`, `imputation_score_c2st`, `DataSplitter.__init__`, `FittedImputer.transform`, `FittedUnit.transform` | **fail**: a `pl.DataFrame` param | `C2STResult` clean; `C2STReport` as above; others are frames or live objects |
| `DataSplitter.random_split`/`kfold`/`holdout_cv`/`group_kfold`/`repeated_kfold` | `stratify` is **unannotated** and its default is an `object()` `_UNSET` sentinel, so no schema can express "not given" | `pl.DataFrame` results fail |
| `DataSplitter.time_split` | `cutoff: Any` is typeless | frames |
| `DataLoader.load` | **fail**: `PathOrBuffer` includes `io.IOBase` | `pl.DataFrame` |
| `StructuralProfiler.profile(data: Any)`, `serialize(obj: Any)`, `FittedImputer.compose(recipe: Any, results: Any)` | "generate", but each `Any` is typeless, so the schema accepts anything | `deserialize -> Any` is a typeless root. `inspect -> dict` is an open map |
| `ImputationRecipe.with_hyperparameters` / `.hyperparameters` | `dict[str, Any]`: open map plus `Any` | |
| `ImputationRecipe.with_estimates(column, **fields)` | `**fields` is invisible to any generator | |
| `ImputationRouting.with_model_choice(choice: ModelChoice \| Any)` | typeless branch | |

### Cross-cutting facts the schema decision waits on

1. **Transformers disagree, and one corrupts silently.** Anthropic's `transform_schema` turns an open map such as `dict[str, int]` into `{"type":"object","properties":{},"additionalProperties":false}`, which accepts **only `{}`**. Every per-column map (`column_routings`, `columns`, sentinels) would pass strict mode and accept no real value. It raises `ValueError: Schema must have a 'type'…` on any typeless `Any`. OpenAI's helper leaves open maps in place, and the API's strict mode then rejects them. So keyed-by-column maps need a schema shape, not a transformer.
2. **Two enum encodings for the same type.** `to_dict()` writes the imputation enums by **name** (`ImputationStrategy` → `"Mean"`, `"Passthrough"`; `SemanticType` → `"Numeric"`, at `imputation/_config.py:938-939, 1133, 1287`). The derived schema uses **values** (`"mean"`, `"passthrough"`). Live check: `routing.to_dict()`, `recipe.to_dict()` and `unit.to_dict()` all **fail** validation against their own derived schema, and `TypeAdapter.validate_python(obj.to_dict())` raises. `PipelineConfig` and `StructuralProfileResult` `to_dict()` do validate.
3. **Two key sets for configs.** `TypeAdapter.dump_python(cfg)` emits `_column_overrides`, `_exclude_columns`, … while `cfg.to_dict()` emits `column_overrides`, `exclude_columns`, …. The generated schema follows the private field names, so ADR-0043/0044's private-backed setter lockdown leaks into any derived schema.
4. **`TYPE_CHECKING`-only imports block derivation** on the main door (`route`, `author`, `resolve_recipe`, `fit_unit`) and on `PipelineConfig`/`UnitFitResult`. This is fixable by a runtime import or `TypeAdapter(…, _parent_namespace)`, but today it fails out of the box.
5. **Recursion: none.** There are no `$ref` cycles in any type, so Anthropic's no-recursion rule costs nothing.
6. **Size is fine.** The largest output (`StructuralProfileResult`) is depth 6, with 317 properties inlined.
7. **What does pass:** 7 Sub-Configs, `SplitConfig`, `ColumnRouting`, `ImputationUnit`, `AuthoredColumn`, `FitSignals`, `PipelineEvent`, and the three inner C2ST records. These are small, flat records of scalars, enums and `tuple[str, ...]`. They are strict-portable after the mechanical fix.

### Blocker → count of affected public types (Pydantic)

| Blocker | Types |
|---|---|
| Open map `dict[str, X]` | 11 |
| `Any` (typeless) | 9 |
| `pl.DataFrame` / live object | 5 (+ every frame-taking entry point) |
| `TYPE_CHECKING` forward ref | 2 types + 7 entry points |
| Fixed tuple (`prefixItems`) | 2 |
| Private `_field` exposed | 4 |
| `_UNSET` sentinel / unannotated param | 5 `DataSplitter` methods |
| `**kwargs` | 1 |
| Recursion | 0 |
