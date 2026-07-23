# The persistence boundary serializes each object by kind, with strict version-gated reconstruction

ADR-0060/0061 defined the three objects that cross the persistence boundary — the pure profile, the value-free `ImputationDecision`, and the set of `done` `FittedUnit`s that a checkpoint is made of (the executor is never pickled whole; `train_df` is never serialized at all). This ADR specifies *how* each is written and read back: the wire form per object kind, the version stamp that governs whether a saved artifact may be reconstructed, the mismatch policy on load, the security posture around unpickling fitted state, and the round-trip guarantee a caller may rely on. It is the reference instance of the "storage-agnostic serialization contract" charted in the wayfinder map (#333); the store *port*, cache *identity/keying*, and validation-on-load-relevance are the sibling concern (#338), and the dependency-freshness machinery this contract leans on is split out to its own ticket.

## The #337 / #338 seam

A serialized fitted unit carries **two independent version stories**, and this ADR owns exactly one of them:

- **Reconstruction / correctness (this ticket).** "Can these bytes be *safely and faithfully* rebuilt under the sklearn/numpy/joblib/format installed right now?" — independent of whether the inputs still match.
- **Cache identity / relevance (#338).** "Do these bytes still correspond to *my* `(data fingerprint, config, library version)`?" — the hash-keying and recompute-vs-error-on-input-change.

The DataForgeML library version appears in **both**, playing different roles (pickle-compat gate here; cache-invalidation key there); that is expected, not duplication. This ticket never decides "should I recompute" — only "are these bytes structurally loadable and faithful: yes, no, or warn."

## Locked decisions

- **Wire form is per object kind.** The pure profile and the `ImputationDecision` serialize as **single JSON documents** — they hold only scalars, enums, and tuples (including `config_snapshot`, reusing the existing `PipelineConfig.to_dict`/`from_dict`, and `profile_provenance`), no learned state, so there is nothing whose reconstruction needs a binary (de)serializer. A **`FittedUnit` serializes as two parts**: a JSON **manifest** (unit id, columns, strategy, the version stamp, the payload checksum, and a reference to the payload) plus a **separate opaque binary payload** (the joblib bytes of the sklearn model + numpy arrays, *not* base64-inlined). This replaces the current `_FittedKNN` pattern of base64-encoding the model *inside* the JSON dict. The manifest is fully readable without touching the payload, base64's ~33% bloat is gone, and the split maps cleanly onto the deferred hybrid store (metadata in a queryable row, artifact in object storage) without shoving binary into a JSON field.

- **The reconstruction stamp is asymmetric by object kind.** Pure objects (profile, decision) stamp only **format-schema version + DataForgeML library version** — the only two facts their reconstruction depends on. A `FittedUnit` manifest stamps a **`produced_with` block of all six**: format-schema, DataForgeML library, scikit-learn, numpy, joblib, and Python — the exact facts needed to judge whether the joblib payload will unpickle faithfully. Versions are captured at save time; a fitted unit is thereby self-describing for diagnosis.

- **Load is strict: mismatch of any reconstruction-critical version refuses; only Python and the joblib mechanism itself are lenient.** The dangerous failure is not `joblib.load` *crashing* (loud and safe) but a *successful* load that then transforms to subtly wrong numbers — silent numerical drift, unobservable and unacceptable under the library's accuracy-over-speed stance. So:
  - **Format-schema newer than this code supports → refuse** (`IncompatibleArtifactError`). Same-major is readable; higher-major is not. This is the one gate on *our own* layout.
  - **DataForgeML library-version mismatch → refuse, discard, retrain.** The library is pre-release with deliberate breaking changes and *no backward-compatibility promise*; a differently-versioned artifact is simply not trusted.
  - **scikit-learn / numpy / joblib mismatch → refuse.** Strict, to foreclose silent drift.
  - **Python mismatch → warn only.** CPython's pickle protocol is stable across minor versions, so refusing here is noise for no real safety gain.
  - A genuine `joblib.load` failure always propagates; it is never swallowed.

- **The DataForgeML library version is the single master key.** Because each library release pins the dependency *ranges* it is tested against, "same DataForgeML version" transitively guarantees a compatible sklearn/numpy/joblib environment. The per-dependency refuses are therefore belt-and-suspenders behind the library-version gate (they catch a force-installed rogue dependency under a matching library version). This makes the master-key idea depend on **tightly-bounded pins** — the current open `>=` upper bounds (`scikit-learn>=1.0.0`, `numpy>=2.0.0`) are in tension with it and must be tightened; that work, together with automated update PRs (Renovate/Dependabot + full-suite CI on every bump), `pip-audit` security scanning, and a lockfile for exact environment reproducibility, is deferred to its **own sibling ticket** rather than folded into this format ADR.

- **Security is integrity-now, authenticity-later.** Unpickling fitted state is arbitrary code execution the moment a payload is loaded. The library's default store is **local and single-tenant** (you load what you saved), so the contract is: store a **SHA-256 checksum of the payload in the manifest and verify it before load**, and **document the trust boundary** ("only load artifacts you produced or trust"). The checksum catches corruption and tampering-in-transit, *not* a malicious author — that is the authenticity tier. **Cryptographic signing** (HMAC/asymmetric, with a user- or service-supplied key) is the real answer to untrusted third-party artifacts, needs key management, and belongs to the deferred *service*; it is out of scope here. `skops`-style safe deserialization was weighed and rejected for now (heavy dependency, poor custom-estimator coverage, and it lags new sklearn versions — fighting the strict-version policy).

- **Round-trip guarantees are exact, and strictness makes them unconditional.** For a **decision (and profile)**: `load(save(x)) == x` — full structural equality across every field, with the `units` tuple re-deriving identically; enums serialize **by name** (`"BayesianRidge"`), never by ordinal, so reordering an enum later cannot silently corrupt an old plan. For a **`FittedUnit`**: `loaded.transform(X)` is **bit-identical** to `original.transform(X)` on equal input — not "within tolerance." This is honestly promisable *because* a version mismatch now refuses to load: the only case in which bit-exactness could fail never reaches `transform`, so the guarantee is unconditional in practice, and it dovetails with the serial/parallel determinism already committed in ADR-0056.

## Status

accepted; narrowed by ADR-0072

The version-gated **reconstruction** contract (the `produced_with` six-version stamp + SHA-256 checked before unpickle) and the ACE trust boundary survive unchanged, on the `FittedUnit` path only. Narrowed: the wire form moves to a JSON envelope + single opaque joblib tail reached through the free functions `serialize`/`deserialize`/`inspect`; the aggregate `FittedImputer` format and its base64 nesting are removed (an imputer rehydrates via `compose`); decision/profile carry only a light schema+`library_version` stamp with no reconstruction gate; and the ADR-0064 relevance/identity checks that ran alongside are gone (map [#377](https://github.com/DEVunderdog/DataForgeML/issues/377)).

## Considered Options

- **Inline base64 model inside one JSON artifact** (today's `_FittedKNN` pattern). Rejected for fitted units: ~33% binary bloat, metadata unreadable without parsing the blob, and a poor fit for the deferred hybrid store. Kept implicitly for the pure objects, which have no binary half.
- **Warn-and-load on dependency mismatch** (sklearn's own posture). Rejected: a cross-version pickle that loads but drifts numerically is exactly the unobservable failure the library cannot afford; the pre-release, no-backward-compat stance removes the operational cost that would otherwise justify leniency.
- **Refuse on Python-minor mismatch too** (zero exceptions). Rejected: pickle protocol is stable across CPython minors, so it is noise without safety gain.
- **`skops` safe serialization now.** Deferred: heavy dependency, incomplete custom-estimator coverage, and version lag that fights the strict-load gate; revisit if/when untrusted artifact exchange becomes real (i.e. the service tier).
- **A float-tolerance round-trip promise for fitted units.** Rejected: strictness already eliminated the drift case, so the stronger bit-identical promise is both achievable and easier to test.

## Consequences

- Refuse-and-retrain requires `train_df`, which ADR-0061 never serializes. A **pure-inference consumer** (holding only the fitted artifact, no training data) who hits a version mismatch cannot retrain. **Superseded by ADR-0065:** the reconstruct-the-old-environment path is *not* supported — the only sanctioned recovery on mismatch is **retrain under the new version**. The dependency lockfile therefore exists for deterministic CI, not for artifact reconstruction.
- The current `_FittedKNN.to_dict`/`from_dict` base64-in-JSON shape is superseded by the manifest + payload split when `_FittedKNN` is repackaged as a `FittedUnit` (ADR-0061); pre-release churn, no compat constraint.
- `pyproject.toml`'s open dependency upper bounds must be tightened for the master-key guarantee to hold; tracked by the new dependency-freshness sibling ticket, not here.
- The map (#333) gains an out-of-scope line for **artifact signing** (checksum → HMAC/asymmetric, user- or service-supplied key), which the service effort inherits.
