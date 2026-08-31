# The Rendering Contract — one `to_markdown()`, `__str__` delegates to it

The repo had two renderers, two formats, and disjoint sets of types using them. `StructuralProfileResult` had `to_markdown()` *and* `to_full_markdown()` but no `__str__`, so `print()` on the one type with a real renderer gave a raw dataclass repr. Six profiling result types hand-rolled a `__str__` emitting `=== ASCII Banners ===` and had no `to_markdown()` at all. Every imputation result type had `to_dict()` and no renderer of any kind. Which of the three shapes a type got was an accident of when it was written.

We adopt a single **Rendering Contract** binding every type the library *returns*: one renderer named `to_markdown() -> str`, with `__str__` delegating to it, `to_dict()` untouched, `__repr__` left alone, and nested types rendering as heading-bounded fragments. The `=== Banner ===` format is retired in favour of Markdown, which renders in a notebook, a GitHub issue, and a terminal alike. No backward compatibility is owed: callers depending on the old `str()` output break by design.

## Status

accepted — supersedes ADR-0040 outright.

## The five rules

1. **One renderer.** `to_markdown() -> str` is it. No second rendering method, no format flag, no verbosity argument.
2. **`__str__` returns `self.to_markdown()`.** `print(x)` and `x.to_markdown()` produce the same bytes, always.
3. **`to_dict()` is untouched.** It is the machine-readable serialiser and answers a different question. Types that have it keep it exactly as-is.
4. **`__repr__` is left alone.** Dataclass repr is for debugging and must stay unambiguous — a bare `x` in a REPL must not print a page of Markdown.
5. **Documents own `#` and `##`; fragments start at `###`.** Only a type the library returns emits a document. A nested type's `to_markdown()` returns rows or a subsection and never a line matching `^#{1,2} `, so a parent composes children without heading collisions.

## Locked decisions

- **The compact profile view is deleted, and the lossless one becomes `to_markdown()`.** ADR-0040 split `StructuralProfileResult`'s rendering in two — a depth-limited human view under `to_markdown()` and a lossless dump under `to_full_markdown()`. That split is the single largest violation of rule 1, and it sat on the only type in the repo that had a working renderer. Keeping it as a named exemption was considered and rejected: an exemption list is a thing that grows, and a contract with exemptions is not checkable by anyone who has not memorised them. The lossless renderer survives because it is the one that cannot silently mislead — a compact view that drops `missingness_matrix` and caps `top_values` at 3 is a summary presented as a report, and the reader cannot tell from the output what was withheld.

- **Consequence, accepted rather than mitigated: `print(profile)` is now roughly a megabyte on an 82-column dataset.** This is precisely what ADR-0040 was built to avoid, so it is recorded as a cost and not as an oversight. It is tolerable because rule 4 keeps `__repr__` short, which is what a REPL browser actually hits, and a user who explicitly calls `print()` on an entire structural profile has asked for the entire structural profile. The honest summary of a profile is `to_dict()` plus the caller's own selection, not a renderer guessing which fields matter.

- **Scope is a positive predicate, not a survey.** A type is in scope if the library **returns** it — directly from a public call, or nested inside something it returns and delegated to by that parent's renderer. Config objects are out: they are inputs the user *supplies*, so they keep `to_dict()` alone. The originating ticket scoped by "has `to_dict()` but no renderer", which silently missed six exported types that have neither — `SplitResult`, `FoldResult`, `HoldoutCVResult`, `ImputationResult`, `UnitFitResult`, `FitSignals` — i.e. exactly what `random_split`, `kfold`, `transform` and `fit_unit` hand back. Keying scope on an accident of which types happened to acquire a `to_dict()` is not a boundary anyone chose. Given up: scope roughly doubles, from ~11 types to ~17.

- **The fragment obligation follows the document, not the object graph.** A nested type owes a `to_markdown()` only when an in-scope parent's renderer *delegates to it*; anything a parent formats inline owes nothing. Unbounded reachability was rejected — the closure from `StructuralProfileResult` pulls in `ColumnProfile`, `DatasetStats`, `MemoryBreakdown`, `BimodalStats`, `ColumnTypeInfo` and more, and ADR-0050 already ruled that reachability is not the test for the public surface. This rule names a pattern the code already runs: `MissingnessProfileResult.__str__` calls `str(profile)` on each `ColumnMissingnessProfile` today. Given up: the fragment set is no longer knowable from the type declarations alone — you must read the parent's renderer to know whether a child is in it.

- **Rendering a result never dumps the payload frame.** Four in-scope types wrap a `pl.DataFrame`. Their renderers report the frame's shape and dtypes and never its rows, so `print(split_result)` stays bounded in the number of rows. The losslessness rule applies to *profile fields*, which are summaries and bounded by the column count; it does not extend to the data itself.

- **A `level=` argument on fragment renderers was rejected.** It reads as the natural way to let a parent place a child at the right depth, and it is a format flag wearing a different hat — banned by rule 1 on the same page. Fixed depths cost nesting headroom instead: a document owns two heading levels and a fragment gets four below it, so a chain three fragments deep would feel it. Today's deepest chain is document → column, so there is room.

## Consequences

- **Breaking, deliberately.** `to_full_markdown()` is removed, `to_markdown()` changes meaning on `StructuralProfileResult`, and `str()` changes output on six profiling types. The compact-view code and its tests are deleted; the losslessness test — every `to_dict()` leaf appears in the output — moves onto `to_markdown()` and becomes the contract's own regression guard.
- **Rule 5 is mechanically testable.** A fragment's output must contain no line matching `^#{1,2} `. One assertion, reused across every fragment type.
- **The `# pragma: no cover` markers on the hand-rolled `__str__` methods are dropped.** These become tested surface.
- **The C2ST types are born compliant.** `C2STReport`, `ColumnC2STResult` and their siblings do not exist yet; they are written to this contract rather than retrofitted to it, as are the later evaluation metrics' records.
- **ADR-0034 applies to every `to_markdown()` added to an in-scope symbol.** Result and report types are public API; their renderers need numpy-style docstrings.
