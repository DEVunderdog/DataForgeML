# DataForgeML

DataForgeML is an automated feature engineering and ML pipeline library that provides a structured, phase-based approach to data preparation.

```{toctree}
:maxdepth: 2
:caption: API Reference

api/pipeline
api/profiling
api/imputation
api/splitting
api/utilities
```

```{toctree}
:maxdepth: 1
:caption: Architecture Decision Records

adr/0001-type-detection-no-external-library
adr/0002-sub-processors-hold-no-config
adr/0003-pipeline-config-at-package-root
adr/0004-impute-before-normalize
adr/0005-imputation-registry-per-semantic-type
adr/0006-fitted-imputer-stateless-fit-returns-object
adr/0007-profile-stratified-split-full-profile-signals
adr/0008-stratification-signal-taxonomy-extension
adr/0009-phase2-effective-null-boundary-normalization
adr/0010-knn-nan-safe-manual-scaling
adr/0011-mice-estimator-most-complex-tag-wins
adr/0012-mice-sample-posterior-deferred
adr/0013-fit-quality-held-back-evaluation
adr/0014-mcar-rmse-mae-shares-r2-holdback
adr/0015-rmse-mae-diagnostic-only-excluded-from-suggest-refit
adr/0016-unpredictable-is-unconditional-pre-routing-guard
adr/0017-distribution-shape-is-universal-routing-signal
adr/0018-bounded-discrete-compound-classification
adr/0019-mnar-data-derived-fill-replaces-sentinel
adr/0020-test-split-imbalance-check-in-transform
adr/0021-fit-transform-returns-tuple
adr/0022-proportional-split-imbalance-check
adr/0023-dropcandiate-exclusion-propagation-caller-initiated-phase3-enforced
adr/0024-records-is-full-train-schema-manifest
adr/0025-indicator-records-written-at-fit-time
adr/0026-unseen-column-error-fires-regardless-of-missingness
adr/0027-regression-fitted-model-stores-target-idx
adr/0028-df-to-numpy-sentinel-ignorant-precondition
adr/0029-per-column-strategy-fires-after-dropcanidate-errors-on-size-guard
adr/0030-definitional-thresholds-promoted-to-phase-sub-configs
adr/0031-bimodality-dual-detection-dip-test-plus-gmm
adr/0032-bimodal-imputation-framework
adr/0033-tail-asymmetry-outlier-density-routing-signals
adr/0034-documentation-strategy-docstrings-mkdocs
adr/0035-bounded-discrete-model-based-routing-with-domain-snap
adr/0036-remove-proportional-split-imbalance-checks
adr/0037-knn-mice-block-override-scalars
adr/0038-sphinx-pydata-theme-replaces-mkdocs
```
