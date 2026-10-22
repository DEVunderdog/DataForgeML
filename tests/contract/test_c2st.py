"""``c2st`` end to end, in both deliveries: two frame Handles in, a C2ST result by value."""

from __future__ import annotations

from pathlib import Path

from dataforge_ml._seed import DEFAULT_SEED

from .conftest import Harness, records

#: Rows in each drift fixture.
N_ROWS = 120


def _csv(harness: Harness, name: str, shift: float = 0.0, extra: bool = False) -> Path:
    # ``x`` and ``y`` are numeric, ``y`` with gaps; ``city`` is a string column.
    # ``shift`` moves ``x``, so two files with different shifts differ.
    csv = harness.inputs / name
    header = "x,y,city" + (",z" if extra else "")
    lines = [header]
    for i in range(N_ROWS):
        y = "" if i % 9 == 4 else f"{(i * 7) % 31 + 0.5}"
        row = f"{(i * 13) % 47 + shift},{y},{'nse'[i % 3]}"
        lines.append(row + (f",{i}" if extra else ""))
    csv.write_text("\n".join(lines) + "\n")
    harness.declare(csv)
    return csv


def _frame(harness: Harness, csv: Path) -> str:
    return harness.code_mode("load", path=str(csv)).json["frame"]


def _excluded(harness: Harness, csv: Path) -> str:
    # A frame that declares ``city`` excluded: a ``split`` fold's training frame.
    frame = _frame(harness, csv)
    profile = harness.code_mode("profile", frame=frame, exclude_columns=["city"]).json["profile"]
    return profile


def _train_and_test(harness: Harness) -> tuple[str, str]:
    profile = _excluded(harness, _csv(harness, "drift.csv"))
    out = harness.code_mode(
        "split", profile=profile, scheme={"kind": "random", "test_size": 0.3}
    ).json
    return records(out["folds"])[0]["train"], out["test"]


# --- The result ---------------------------------------------------------------------


def test_it_compares_a_split_s_train_and_test_frames(harness: Harness) -> None:
    train, test = _train_and_test(harness)

    out = harness.call("c2st", reference=train, candidate=test)

    assert not out.is_error, out.json
    result = out.json
    assert result["reference"] == train
    assert result["candidate"] == test
    assert result["reference_rows"] + result["candidate_rows"] == N_ROWS
    assert (result["columns"], result["skipped"]) == (2, 1)
    assert (result["scheme"], result["repeats"], result["seed"]) == ("split", 5, DEFAULT_SEED)
    assert len(result["repeat_z"]) == 5
    balanced = min(result["reference_rows"], result["candidate_rows"])
    assert result["n_test"] == 2 * round(balanced * 0.3)
    assert abs(result["mean_repeat_z"]) < 3
    assert result["degenerate"] is False


def test_it_writes_no_handle(harness: Harness) -> None:
    train, test = _train_and_test(harness)
    before = harness.handles()

    harness.call("c2st", reference=train, candidate=test)

    assert harness.handles() == before


def test_shifted_frames_score_far_above_chance(harness: Harness) -> None:
    reference = _excluded(harness, _csv(harness, "a.csv"))
    candidate = _excluded(harness, _csv(harness, "b.csv", shift=40.0))
    frames = [
        records(harness.code_mode("split", profile=p, scheme={"kind": "kfold", "k": 2}).json["folds"])[0]["train"]
        for p in (reference, candidate)
    ]

    out = harness.call("c2st", reference=frames[0], candidate=frames[1])

    assert not out.is_error, out.json
    assert out.json["mean_repeat_z"] > 3
    assert out.json["test_accuracy"] > 0.9


def test_the_scheme_repeats_and_seed_are_arguments(harness: Harness) -> None:
    train, test = _train_and_test(harness)
    arguments = {
        "reference": train,
        "candidate": test,
        "repeats": 3,
        "scheme": "cross_validation",
        "seed": 7,
    }

    first = harness.call("c2st", **arguments)
    second = harness.code_mode("c2st", **arguments)

    assert not first.is_error, first.json
    assert (first.json["scheme"], first.json["repeats"], first.json["seed"]) == (
        "cross_validation",
        3,
        7,
    )
    assert len(first.json["repeat_z"]) == 3
    balanced = min(first.json["reference_rows"], first.json["candidate_rows"])
    assert first.json["n_test"] == 2 * balanced
    assert second.json == first.json


# --- Errors -------------------------------------------------------------------------


def test_a_string_column_is_a_code_naming_it(harness: Harness) -> None:
    frame = _frame(harness, _csv(harness, "drift.csv"))

    out = harness.call("c2st", reference=frame, candidate=frame)

    assert out.code == "c2st_string_column"
    assert out.json["columns"] == ["city"]


def test_frames_below_the_floor_are_a_code_naming_the_dial(harness: Harness) -> None:
    train, test = _train_and_test(harness)
    # The test frame has 36 rows; a floor of 40 refuses it.
    out = harness.call("c2st", reference=train, candidate=test, thresholds={"min_rows": 40})

    assert out.code == "c2st_sample_floor"
    assert out.json["param"] == "thresholds.min_rows"

    lowered = harness.call("c2st", reference=train, candidate=test, thresholds={"min_rows": 10})
    assert not lowered.is_error, lowered.json


def test_columns_in_only_one_frame_are_not_aligned(harness: Harness) -> None:
    reference = _excluded(harness, _csv(harness, "a.csv"))
    candidate = _excluded(harness, _csv(harness, "b.csv", extra=True))
    frames = [
        records(harness.code_mode("split", profile=p, scheme={"kind": "kfold", "k": 2}).json["folds"])[0]["train"]
        for p in (reference, candidate)
    ]

    out = harness.call("c2st", reference=frames[0], candidate=frames[1])

    assert out.code == "frames_not_aligned"
    assert out.json["columns"] == ["z"]


def test_frames_with_every_column_excluded_have_nothing_to_test(harness: Harness) -> None:
    frame = _frame(harness, _csv(harness, "drift.csv"))
    profile = harness.code_mode(
        "profile", frame=frame, exclude_columns=["x", "y", "city"]
    ).json["profile"]
    split = harness.code_mode(
        "split", profile=profile, scheme={"kind": "random", "test_size": 0.3}
    ).json

    out = harness.call("c2st", reference=split["test"], candidate=split["test"])

    assert out.code == "invalid_argument"
    assert out.json["param"] == "reference"


def test_a_handle_of_another_kind_is_refused(harness: Harness) -> None:
    train, _ = _train_and_test(harness)
    profile = _excluded(harness, _csv(harness, "a.csv"))

    out = harness.call("c2st", reference=train, candidate=profile)

    assert out.code == "wrong_handle_kind"
