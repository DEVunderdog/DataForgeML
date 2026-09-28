# Persistence saves profile, routing, recipe and fitted unit, and every load is strict

ADR-0072 made persistence three free functions over three kinds: profile, decision and fitted unit. The decision is gone. It split into routing (ADR-0088), derived units (ADR-0084) and the recipe (ADR-0089). This ADR says which of those layers `serialize` / `deserialize` / `inspect` handle, what a saved fitted unit holds, and how loading behaves now that old files don't need to load.

## Status

accepted — resolves [Persistence of the layered artifacts](https://github.com/DEVunderdog/DataForgeML/issues/535) on map #528. Amends ADR-0072 (the kinds change and the schema check covers every kind) and ADR-0083 (`from_dict` no longer gap-fills; a strict load is what licenses the fitters' hard subscripts).

## Decisions

- **Four kinds: profile, routing, recipe, fitted unit.** `ImputationDecision` leaves with its type. Only what a user hands to the next layer is saved. Not saved: `ImputationUnit` (`derive_units(routing)` rebuilds it for free), `FitSignals` and `UnitFitResult` (a fit's record, ADR-0074), `FittedImputer` (a recipe plus its units, rebuilt by `compose` — ADR-0072 unchanged) and `C2STReport` (a report you read, never fed back in). Given up: an old report or old fit signals can't be reloaded. The user keeps those, for example with `to_markdown`.
- **A saved recipe contains a full copy of its routing.** Routing also saves on its own. Both drop the live estimator (ADR-0090), so a reload is `Custom` with an empty slot and the fit raises. Gained: one blob is enough to fit, and a recipe can never load against the wrong routing. Given up: a user who saves both has two copies that nothing keeps in step. The recipe's copy is the one that fits.
- **A saved fitted unit carries no settings.** The header stays `unit_type`, `columns`, `produced_with` and the payload checksum. No dials, because the in-memory unit has none (#532) and `serialize(unit)` takes one argument. No `strategy` either: `FittedScalar` serves Mean, Median, Mode and Constant and doesn't know which, and putting the field back would undo ADR-0074. To learn a unit's dials or strategy, read the saved recipe by its columns. Given up: after later `with_hyperparameters` edits, a saved unit can't say which dials trained it, and its header alone can't tell a median fill from a mean fill.
- **Loading a recipe is strict. Gap-fill is deleted.** A missing or unknown dial row or key raises at `deserialize`, names what is wrong, and says to resolve again. The load also checks that the decided base's unit ids match `derive_units(routing)`. A loaded recipe is therefore complete or it raised, which is what licenses the fitters' hard subscripts now. Gained: a loaded recipe is exactly what was saved, and a bad file fails at load, not halfway through a fit. Given up: a recipe saved before a dial existed won't load. You resolve it again and apply the override delta again.
- **`FORMAT_SCHEMA_VERSION` goes to 2 and is checked on every kind.** Profile, routing and recipe used to skip the check that the fitted unit ran. A version-1 file, or a file of kind `decision`, raises `IncompatibleArtifactError` saying to profile, resolve or fit again. The check is less about old files than about a newer file meeting an older library. Given up: every profile saved before this change must be profiled again, even though its layout didn't change.
- **`inspect` stays header-only for every kind.** It answers "what is this, and can I load it safely?" and nothing else. It gives no summary of routing or recipe contents. Loading those is plain JSON with no unpickle, so `deserialize` is the way to look inside. Given up: to see a routing's strategies without loading it, you read the JSON yourself.

## Considered Options

- **Write the fit's dials into the unit header, by letting `serialize` accept a `UnitFitResult`.** Rejected: it makes one field of `FitSignals` saved state. ADR-0074 kept signals short-lived so they can't become a stale claim about a unit refitted elsewhere.
- **Hold the dials on `FittedUnit` again.** Rejected: it copies a recipe fact onto the artifact, which #532 and ADR-0074 removed.
- **A saved recipe points to a separately saved routing.** Rejected: nothing could prove at load time that the two belong together, which is the mix-up ADR-0089's by-value rule exists to prevent.
- **Keep ADR-0083's gap-fill.** Rejected: it exists only so older files keep working, and this map rules backward compatibility out. It also quietly gives a loaded recipe a dial it was never saved with.
- **No version bump, and let old files fail wherever they break.** Rejected: the failure would be a bare `KeyError` inside `from_dict`, and a newer file meeting an older library would fail the same unclear way.
