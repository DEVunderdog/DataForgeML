"""``split`` end to end, in both deliveries: one shape for every scheme, inherited declarations."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from dataforge_ml._boundary._split import SPLIT_DIALS, dial_defaults
from dataforge_ml._boundary._workspace import payload_path, read_record
from dataforge_ml._seed import DEFAULT_SEED
from dataforge_ml.operations.split import SchemeName, StratificationDials
from dataforge_ml.records import Conflict, Infeasible, InvalidInput

from .conftest import Harness, records, run_dataforge

#: Rows in the split fixture.
N_ROWS = 60
#: The one row whose ``label`` class appears once.
SINGLETON_ROW = 7


def _split_csv(harness: Harness) -> Path:
    # ``label`` has two balanced classes plus one singleton class, ``price`` is
    # a continuous target, ``grp`` has 12 groups, ``when`` is a date with one
    # gap, and ``amount`` has gaps and a ``-999`` sentinel.
    csv = harness.inputs / "split.csv"
    start = date(2020, 1, 1)
    lines = ["when,grp,amount,label,price"]
    for i in range(N_ROWS):
        when = "" if i == 30 else (start + timedelta(days=i)).isoformat()
        amount = "" if i % 6 == 2 else "-999" if i % 13 == 4 else f"{10.5 + (i * 7) % 23}"
        label = "zed" if i == SINGLETON_ROW else ("a" if i % 2 else "b")
        lines.append(f"{when},g{i % 12},{amount},{label},{1000.25 + i * 13.5}")
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    return csv


def _profile(harness: Harness, **arguments: Any) -> str:
    frame = harness.code_mode("load", path=str(_split_csv(harness))).json["frame"]
    return harness.code_mode("profile", frame=frame, **arguments).json["profile"]


def _frame(handle: str) -> pl.DataFrame:
    return pl.read_parquet(payload_path(handle, read_record(handle, "frame", param="frame")))


def _frame_handles(harness: Harness) -> list[str]:
    return [h for h in harness.handles() if h.startswith("frame_")]


#: One call per scheme, and whether it holds out a ``test`` frame and how many folds it makes.
SCHEMES: dict[str, tuple[dict[str, Any], bool, int, bool]] = {
    # kind: (scheme, has test, folds, single split)
    "random": ({"kind": "random", "test_size": 0.25}, True, 1, True),
    "time": ({"kind": "time", "time_column": "when", "test_size": 0.25}, True, 1, True),
    "group": ({"kind": "group", "group_column": "grp"}, True, 1, True),
    "kfold": ({"kind": "kfold", "k": 3}, False, 3, False),
    "group_kfold": ({"kind": "group_kfold", "group_column": "grp", "k": 3}, False, 3, False),
    "repeated_kfold": ({"kind": "repeated_kfold", "k": 3, "repeats": 2}, False, 6, False),
    "holdout_cv": ({"kind": "holdout_cv", "test_size": 0.25, "k": 3}, True, 3, False),
    "profile_stratified": ({"kind": "profile_stratified", "test_size": 0.25}, True, 1, True),
    "profile_stratified_kfold": ({"kind": "profile_stratified_kfold", "k": 3}, False, 3, False),
}


def test_the_scheme_list_covers_every_scheme() -> None:
    assert set(SCHEMES) == set(SchemeName.__args__)


# --- One shape for every scheme ----------------------------------------------------


@pytest.mark.parametrize("kind", sorted(SCHEMES))
def test_every_scheme_returns_test_and_folds_of_frame_handles(harness: Harness, kind: str) -> None:
    scheme, has_test, n_folds, single = SCHEMES[kind]
    profile = _profile(harness)

    out = harness.call("split", profile=profile, scheme=scheme)

    assert not out.is_error, out.json
    assert out.json["scheme"] == kind and out.json["profile"] == profile
    made = set(_frame_handles(harness))
    folds = records(out.json["folds"])
    assert len(folds) == n_folds
    assert (out.json["test"] is not None) == has_test
    partitions = [out.json["test"]] if has_test else []
    for fold in folds:
        assert (fold["validation"] is None) == single
        partitions += [fold["train"], *([fold["validation"]] if fold["validation"] else [])]
        assert _frame(fold["train"]).height == fold["train_rows"]
    assert set(partitions) <= made
    if has_test:
        assert _frame(out.json["test"]).height == out.json["test_rows"]
    if single:
        assert folds[0]["train_rows"] + out.json["test_rows"] == N_ROWS
    elif kind != "holdout_cv":
        repeats = {f["repeat"] for f in folds}
        for repeat in repeats:
            sizes = [f["validation_rows"] for f in folds if f["repeat"] == repeat]
            routed = (out.json["stratification"] or {}).get("rows_to_train", 0)
            assert sum(sizes) == N_ROWS - routed, "each row validates at most once per repeat"


def test_the_frames_are_the_source_rows_and_columns(harness: Harness) -> None:
    profile = _profile(harness)

    out = harness.call("split", profile=profile, scheme={"kind": "kfold", "k": 3})

    source = _frame(out.json["frame"])
    fold = records(out.json["folds"])[0]
    train, validation = _frame(fold["train"]), _frame(fold["validation"])
    assert train.columns == source.columns and train.schema == source.schema
    assert pl.concat([train, validation]).sort("price").equals(source.sort("price"))


def test_a_null_scheme_is_the_profile_stratified_split(harness: Harness) -> None:
    out = harness.call("split", profile=_profile(harness))

    assert out.json["scheme"] == "profile_stratified"
    assert out.json["test_rows"] == round(N_ROWS * 0.2)
    assert out.json["stratification"]["kept"] > 0


# --- Inherited Declarations --------------------------------------------------------


def test_folds_inherit_the_profiles_declarations(harness: Harness) -> None:
    profile = _profile(
        harness,
        targets=["label"],
        semantic_types=[{"column": "grp", "semantic_type": "categorical"}],
        sentinels=[{"column": "amount", "values": [-999.0]}],
        exclude_columns=["price"],
    )
    declared = read_record(profile, "profile", param="profile")["declarations"]

    out = harness.call("split", profile=profile, scheme={"kind": "kfold", "k": 3})

    fold = records(out.json["folds"])[0]
    for handle in (fold["train"], fold["validation"]):
        assert read_record(handle, "frame", param="frame")["declarations"] == declared
    reprofiled = harness.call("profile", frame=fold["train"]).json
    assert reprofiled["declarations"]["targets"] == ["label"]
    assert reprofiled["declarations"]["exclude_columns"] == ["price"]
    assert records(reprofiled["declarations"]["sentinels"]) == [
        {"column": "amount", "values": [-999.0]}
    ]


def test_a_fold_may_add_declarations_but_not_contradict_them(harness: Harness) -> None:
    profile = _profile(harness, semantic_types=[{"column": "grp", "semantic_type": "categorical"}])
    train = records(
        harness.call("split", profile=profile, scheme={"kind": "kfold", "k": 3}).json["folds"]
    )[0]["train"]

    added = harness.call("profile", frame=train, targets=["label"])
    contradiction = harness.call(
        "profile", frame=train, semantic_types=[{"column": "grp", "semantic_type": "text"}]
    )

    assert not added.is_error
    assert added.json["declarations"]["targets"] == ["label"]
    assert contradiction.code == "declaration_conflict"
    assert isinstance(contradiction.error, Conflict)
    assert contradiction.json["columns"] == ["grp"]


# --- Stratification ----------------------------------------------------------------


def test_stratify_null_balances_the_declared_target(harness: Harness) -> None:
    profile = _profile(harness, targets=["grp"])

    out = harness.call("split", profile=profile, scheme={"kind": "kfold", "k": 3})
    plain = harness.call(
        "split", profile=profile, scheme={"kind": "kfold", "k": 3, "stratify": False}
    )

    assert out.json["stratified_on"] == "grp"
    for fold in records(out.json["folds"]):
        counts = _frame(fold["validation"])["grp"].value_counts()["count"].to_list()
        assert max(counts) - min(counts) <= 1, "every group's share is kept"
    assert plain.json["stratified_on"] is None


def test_a_class_too_small_to_stratify_raises_the_same_code(harness: Harness) -> None:
    profile = _profile(harness, targets=["label"])

    out = harness.call("split", profile=profile, scheme={"kind": "random"})

    assert out.code == "stratify_on_continuous_target"
    assert "zed" not in out.json["message"], "counts, never the label"


def test_stratify_on_a_continuous_target_raises_its_registered_code(harness: Harness) -> None:
    profile = _profile(harness, targets=["price"])
    before = harness.handles()

    out = harness.call("split", profile=profile, scheme={"kind": "kfold", "k": 3})

    assert out.code == "stratify_on_continuous_target"
    assert isinstance(out.error, Infeasible)
    assert out.json["columns"] == ["price"]
    assert harness.handles() == before, "no Handle is written on failure"
    # The profile-stratified scheme bins a continuous target instead.
    binned = harness.call("split", profile=profile, scheme={"kind": "profile_stratified_kfold", "k": 3})
    assert binned.json["stratified_on"] == "price"


def test_stratify_without_a_target_is_refused(harness: Harness) -> None:
    out = harness.call(
        "split", profile=_profile(harness), scheme={"kind": "kfold", "k": 3, "stratify": True}
    )

    assert out.code == "stratify_without_target"
    assert isinstance(out.error, InvalidInput)


def test_several_targets_need_one_named(harness: Harness) -> None:
    profile = _profile(harness, targets=["label", "price"])

    unnamed = harness.call("split", profile=profile, scheme={"kind": "profile_stratified"})
    named = harness.call(
        "split", profile=profile, scheme={"kind": "profile_stratified", "target": "price"}
    )
    undeclared = harness.call(
        "split", profile=profile, scheme={"kind": "profile_stratified", "target": "amount"}
    )

    assert unnamed.code == "invalid_argument" and unnamed.json["param"] == "scheme.target"
    assert named.json["stratified_on"] == "price"
    assert undeclared.code == "invalid_argument"


# --- Fallbacks ---------------------------------------------------------------------


def _fallbacks(out: Any) -> dict[str, dict[str, Any]]:
    rows = records(out.json["fallbacks"])
    return {row["fallback"]: row for row in rows}


def test_rows_below_the_floor_go_to_train_and_report_a_count(harness: Harness) -> None:
    profile = _profile(harness, targets=["label"])

    out = harness.call("split", profile=profile, scheme={"kind": "profile_stratified_kfold", "k": 3})

    fallback = _fallbacks(out)["rows_to_train"]
    assert fallback["count"] == 1
    assert fallback["dials"] == [
        {"operation": "split", "dial": "scheme.k", "value": 3, "source": "argument"}
    ]
    # ``zed`` is both a target class and a rare value of ``label``.
    assert out.json["stratification"]["labels_to_train"] == 2
    for fold in records(out.json["folds"]):
        assert SINGLETON_ROW not in _row_ids(fold["validation"]), "the singleton never validates"
    warning = next(w for w in records(out.json["warnings"]) if w["code"] == "rare_rows_to_train")
    assert warning["param"] == "scheme.k"
    assert "zed" not in str(out.json), "counts, never the label"


def _row_ids(handle: str) -> set[int]:
    # ``price`` is 1000.25 + 13.5 * row, so it names the source row.
    return {round((p - 1000.25) / 13.5) for p in _frame(handle)["price"].to_list()}


def test_an_unaffordable_target_falls_back_to_random_with_the_cap_dials(harness: Harness) -> None:
    profile = _profile(harness, targets=["label"])

    out = harness.call(
        "split",
        profile=profile,
        scheme={"kind": "profile_stratified"},
        thresholds={"stratification": {"rows_per_signal": N_ROWS + 1}},
    )

    fallback = _fallbacks(out)["unstratified"]
    assert fallback["count"] == N_ROWS
    assert [d["dial"] for d in fallback["dials"]] == [
        "thresholds.stratification.max_stratification_signals",
        "thresholds.stratification.rows_per_signal",
    ]
    assert [d["source"] for d in fallback["dials"]] == ["default", "argument"]
    assert out.json["stratification"]["kept"] == 0
    assert out.json["stratified_on"] is None
    assert "split_unstratified" in [w["code"] for w in records(out.json["warnings"])]


def test_a_cap_below_the_signals_reports_how_many_were_left_out(harness: Harness) -> None:
    profile = _profile(harness, targets=["label"])

    out = harness.call(
        "split",
        profile=profile,
        scheme={"kind": "profile_stratified", "test_size": 0.5},
        thresholds={"stratification": {"max_stratification_signals": 1}},
    )

    summary = out.json["stratification"]
    fallback = _fallbacks(out)["signals_capped"]
    assert summary["kept"] == 1 and summary["non_redundant"] > 1
    assert fallback["count"] == summary["non_redundant"] - 1
    assert fallback["dials"][0] == {
        "operation": "split",
        "dial": "thresholds.stratification.max_stratification_signals",
        "value": 1,
        "source": "argument",
    }


def test_the_plain_schemes_report_no_stratification(harness: Harness) -> None:
    out = harness.call("split", profile=_profile(harness), scheme={"kind": "kfold", "k": 3})

    assert out.json["stratification"] is None
    assert records(out.json["fallbacks"]) == []


def test_dials_are_every_split_config_field_with_its_default() -> None:
    from dataforge_ml.splitting._config import SplitConfig

    assert list(StratificationDials.model_fields) == list(SPLIT_DIALS)
    assert dial_defaults() == {
        f"stratification.{k}": v for k, v in SplitConfig().to_dict().items()
    }
    for name, info in StratificationDials.model_fields.items():
        assert f"Null means {getattr(SplitConfig(), name)}." in info.description


# --- Seeds -------------------------------------------------------------------------


def test_a_null_seed_is_the_fixed_default_and_gives_the_same_split(harness: Harness) -> None:
    profile = _profile(harness)
    scheme = {"kind": "kfold", "k": 3}

    first = harness.call("split", profile=profile, scheme=scheme)
    again = harness.call("split", profile=profile, scheme=scheme)
    explicit = harness.call("split", profile=profile, scheme=scheme, seed=DEFAULT_SEED)
    other = harness.call("split", profile=profile, scheme=scheme, seed=DEFAULT_SEED + 1)

    def validation_rows(out: Any) -> list[set[int]]:
        return [_row_ids(f["validation"]) for f in records(out.json["folds"])]

    assert first.json["seed"] == DEFAULT_SEED
    assert validation_rows(first) == validation_rows(again) == validation_rows(explicit)
    assert validation_rows(other) != validation_rows(first)


# --- Time and group ----------------------------------------------------------------


def test_time_holds_out_the_latest_rows_and_sends_timeless_rows_to_train(harness: Harness) -> None:
    out = harness.call(
        "split",
        profile=_profile(harness),
        scheme={"kind": "time", "time_column": "when", "test_size": 0.25},
    )

    train = _frame(records(out.json["folds"])[0]["train"])
    test = _frame(out.json["test"])
    assert train["when"].null_count() == 1 and test["when"].null_count() == 0
    assert train["when"].max() < test["when"].min()
    warning = records(out.json["warnings"])[0]
    assert (warning["code"], warning["columns"]) == ("null_time_rows_to_train", ["when"])


def test_a_time_cutoff_is_an_iso_datetime_on_the_cli(harness: Harness) -> None:
    profile = _profile(harness)
    scheme = '{"kind":"time","time_column":"when","cutoff":"2020-02-15T00:00:00"}'

    out = run_dataforge(["split", "--profile", profile, "--scheme", scheme])

    assert out.exit_code == 0, out.stdout
    assert _frame(out.json["test"])["when"].min() == date(2020, 2, 15)
    assert out.json["test_rows"] == N_ROWS - 45


def test_group_splits_keep_every_group_whole(harness: Harness) -> None:
    profile = _profile(harness)

    one = harness.call("split", profile=profile, scheme={"kind": "group", "group_column": "grp"})
    folds = harness.call(
        "split", profile=profile, scheme={"kind": "group_kfold", "group_column": "grp", "k": 4}
    )

    train = set(_frame(records(one.json["folds"])[0]["train"])["grp"].to_list())
    assert train.isdisjoint(_frame(one.json["test"])["grp"].to_list())
    for fold in records(folds.json["folds"]):
        assert set(_frame(fold["train"])["grp"]).isdisjoint(_frame(fold["validation"])["grp"])


# --- Argument checks ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("scheme", "code", "param"),
    [
        ({"kind": "group", "group_column": "gorup"}, "unknown_column", "scheme.group_column"),
        ({"kind": "time", "time_column": "label"}, "invalid_argument", "scheme.time_column"),
        (
            {"kind": "time", "time_column": "when", "test_size": 0.2, "cutoff": 3},
            "invalid_argument",
            "scheme.cutoff",
        ),
        ({"kind": "kfold", "k": N_ROWS + 1}, "invalid_argument", "scheme.k"),
        ({"kind": "repeated_kfold", "k": 10, "repeats": 11}, "invalid_argument", "scheme.repeats"),
        ({"kind": "group_kfold", "group_column": "grp", "k": 13}, "invalid_argument", "scheme.k"),
        ({"kind": "kfold", "k": 1}, "invalid_argument", "scheme.kfold.k"),
        ({"kind": "kfold", "k": 3, "stratify": False, "target": "label"}, "invalid_argument", "scheme.target"),
    ],
)
def test_bad_schemes_fail_before_any_work(
    harness: Harness, scheme: dict[str, Any], code: str, param: str
) -> None:
    profile = _profile(harness, targets=["label"])
    before = harness.handles()

    out = harness.call("split", profile=profile, scheme=scheme)

    assert out.code == code
    assert param in out.json["params"]
    assert harness.handles() == before


def test_an_excluded_column_cannot_order_or_group_the_split(harness: Harness) -> None:
    profile = _profile(harness, exclude_columns=["grp", "when"])

    group = harness.call("split", profile=profile, scheme={"kind": "group", "group_column": "grp"})
    time = harness.call("split", profile=profile, scheme={"kind": "time", "time_column": "when"})

    assert group.code == time.code == "declaration_conflict"


def test_a_frame_handle_is_the_wrong_kind(harness: Harness) -> None:
    frame = harness.code_mode("load", path=str(_split_csv(harness))).json["frame"]

    out = harness.call("split", profile=frame)

    assert out.code == "wrong_handle_kind"
