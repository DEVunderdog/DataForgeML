# Model units transform off a shared pre-model snapshot, not a chain

ADR-0067 pinned the order in which `FittedImputer.transform` applies model-based units, making the chain deterministic without making it right. It left the deeper question open on purpose: each unit consumes its predecessors' output, so a model trained on the raw train split may serve against features a *sibling* unit imputed — a train/serve skew — and the plan's unit order silently becomes load-bearing on imputation output. This ADR closes that question. **Every model-based unit now reads the same pre-model frame and contributes only the columns it owns.** The chain is gone.

The question was deferred as a genuine tension: the chain gives each model *better* features (MICE's fills are more informed than `IterativeImputer`'s internal round-robin) but ones it was not trained on; the snapshot gives each model *worse* features that match training exactly. That framing assumed the chain's accuracy advantage was real. It is not, and measuring it is what settled this.

**The chain's accuracy advantage is indistinguishable from noise.** Built to the exact shape the question describes — a MICE block over `a, b`, `regression:c` and `regression:d` reading it as features, a median-filled `e` — over synthetic data with known ground truth and 15% MCAR holes, scored as RMSE against truth on imputed cells only, across 30 seeds:

| column | chain wins | snapshot wins | mean relative difference | worst case |
|---|---|---|---|---|
| `c` | 15 | 15 | −0.18% | 2.7% |
| `d` | 14 | 16 | −0.21% | 2.5% |

A coin flip, with the mean leaning very slightly *toward* the snapshot. The "better estimate" argument — the whole reason this was a hard call rather than a bug — buys ~0.2% of noise and does not hold a consistent direction. Once accuracy cannot distinguish the two shapes, nothing weighs against the architectural case, and the tension collapses.

Two further facts narrow the change and correct the question as originally posed:

**Only regression units were ever chained.** `fit_mice_unit` and `fit_knn_unit` fit over `unit.columns` alone, so a MICE or KNN block reads exactly the columns it fills and never consumes a sibling's output. Their transform output is unchanged by this ADR, verified byte-identical under both shapes. `FittedRegression` is the sole unit type whose inputs are a strict superset of its targets, and therefore the only one this was ever about.

**The snapshot removes half the train/serve skew, not all of it.** The deferred question claimed it "removes the skew". It does not. Scalar fills are applied *before* the model loop, so a median-filled feature reaches a regression model that trained on that column's raw nulls under **both** shapes — `e` is a feature of `regression:c`, and fit saw 89 nulls there where transform sees zero. This ADR closes the sibling-model half of the skew and leaves the scalar half open and documented. Claiming a clean fix would be the more satisfying story and the false one.

Locked decisions:

- **Every model-based unit transforms off the same pre-model frame.** `transform` snapshots the frame after drops, indicators, and scalar fills, passes that one frame to every unit, and merges each unit's `target_columns` back. No unit observes another's output.
- **A unit's targets are part of the `FittedUnit` contract.** `target_columns` is a protocol member implemented by every unit type, not a type switch at the call site. The merge needs to know what a unit owns; the unit is the only thing that honestly knows. It also names the distinction the chain obscured — a regression unit's *inputs* are every feature column, its *target* is the one column it predicts.
- **Targets are disjoint, so the merge is order-independent by construction.** Each column routes to exactly one strategy and therefore exactly one unit. Order-independence is a property of the routing, not a convention the merge maintains — no unit can overwrite another's column, so there is nothing for an order to arbitrate.
- **The plan's unit order survives, demoted.** ADR-0067's pinning stays: `models` is still composed in plan order. It is no longer load-bearing on the result, and is kept because it is the order a human reads the plan in and the order progress events report.
- **ADR-0056's independence rule is restored to its original, unqualified form.** ADR-0067 narrowed it to a fit-time claim because transform genuinely violated it. Transform no longer does. Units are independent at both ends, and "concurrency never changes a result" is true as written again.

## Status

accepted

Amended by [Whether the KNN block reads the full active-numeric predictors](https://github.com/DEVunderdog/DataForgeML/issues/541) (map #528, ADR-0093): the KNN block now reads every active numeric column and writes back only its own, as the MICE block has since ADR-0079. Both joint blocks have inputs ⊃ targets; "a MICE or KNN block reads exactly the columns it fills" above is history. The snapshot, disjoint targets and order-independent merge all stand, and the KNN block fits on the scalar-filled frame it serves on, as MICE does.

## Considered Options

- **Keep the chain (ADR-0067's status quo).** Rejected. It costs nothing today, which was its whole appeal, but it preserves a "better features" benefit that 30 seeds cannot detect, in exchange for: an independence rule that must stay permanently caveated, a plan order silently coupled to imputation output, and a sibling train/serve skew. Every one of those is a real, ongoing cost paid for a measured non-benefit.
- **Chained fit — keep the chain but fit each unit against the same chained frame it will serve on.** The strongest alternative, and the one that closes the skew from the training side while preserving the chain's feature argument. Rejected on price: it reintroduces a *fit-time* ordering dependency between units, which lands directly on ADR-0061's resumable, independently-retrainable unit model — retraining one unit would require re-running its predecessors — and on ADR-0056's fit-time parallelism. That is a large architectural cost to buy the ~0.2% the measurement says is not there. Had the chain's advantage been real, this would have been the answer.
- **Merge only the columns a unit changed, rather than the columns it owns.** Rejected as a distinction without a difference that invents a failure mode: a unit's untouched target column is unchanged in its output, so merging it back is a no-op. Diffing to discover that would make the merge depend on value comparison where it can depend on a declared contract.

## Consequences

- **Imputation output changes for anyone with a regression unit reading another model unit's target.** This is the cost, and it is user-visible. A user with one model-based unit, or whose model units are all MICE/KNN, sees byte-identical output. There is no legacy engine left to diverge from — #368 deleted it, and #372's gate was re-expressed against the plan rather than a second engine — so the parity concern that motivated the deferral no longer has a counterparty.
- **Each regression unit falls back to its own internal round-robin for feature nulls.** Per-unit, this is the "worse features" the deferred question warned about, and it is real: the model no longer benefits from MICE's more informed fills. Measurement says the exchange is a wash, and what is bought is that the features now match the ones the model was fitted against.
- **The scalar half of the train/serve skew remains open.** Regression units still train on raw feature nulls and serve on scalar-filled features. Closing it means either fitting model units against the scalar-filled frame — which reintroduces an ordering dependency between the scalar and model layers — or accepting it. Not resolved here; recorded so the next reader does not mistake this ADR for a complete fix.
- **A future reorder of `decision.units` can no longer silently change imputation output.** ADR-0067 flagged the two-engine ordering coupling as load-bearing and dangerous. It is now inert.
- **`FittedUnit` gains a member, so any third-party implementation must supply it.** The protocol is `runtime_checkable`, and a unit without `target_columns` will now fail to satisfy it.
