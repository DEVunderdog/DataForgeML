
import polars as pl
import pytest


@pytest.fixture
def round_trip():
    """Round-trip a FittedImputer through the bare-bytes boundary (ADR-0072).

    A whole imputer has no aggregate serialize format: it persists as its
    decision plus its units and rehydrates through ``compose``. This helper puts
    each model-based unit through the real ``serialize`` / ``deserialize`` door
    and reassembles the aggregate, carrying the structural records and sentinel
    maps across in-process the way the decision would.
    """

    def _round_trip(imputer, key: str = "imputer"):
        from dataforge_ml import deserialize, serialize
        from dataforge_ml import FittedImputer

        restored_units = [deserialize(serialize(unit)) for unit in imputer.units]
        return FittedImputer(
            records=dict(imputer.records),
            units=restored_units,
            numeric_sentinels={k: list(v) for k, v in imputer.numeric_sentinels.items()},
            string_sentinels={k: list(v) for k, v in imputer.string_sentinels.items()},
            random_seed=imputer.random_seed,
        )

    return _round_trip


@pytest.fixture(scope="session")
def override_df():
    n = 60
    return pl.DataFrame(
        {
            "score": pl.Series([float(i) for i in range(n)], dtype=pl.Float64),
            "category": pl.Series(["A", "B", "C"] * (n // 3), dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="session")
def target_df(rng):
    rng = rng(seed=42)
    n = 100
    features = rng.normal(0, 1, size=n).tolist()
    labels = ["pos", "neg"] * (n // 2)
    return pl.DataFrame(
        {
            "feature": pl.Series(features, dtype=pl.Float64),
            "label": pl.Series(labels, dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="session")
def empty_df():
    return pl.DataFrame(
        {
            "x": pl.Series([], dtype=pl.Float64),
            "y": pl.Series([], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="session")
def text_df():
    n = 200
    topics = ["science", "art", "history", "technology", "nature", "music"]
    texts = [
        f"A detailed description covering the topic of {topics[i % len(topics)]} "
        f"with multiple words that comfortably exceed the free-text threshold in row {i}"
        for i in range(n)
    ]
    return pl.DataFrame({"review": pl.Series(texts, dtype=pl.Utf8)})


@pytest.fixture(scope="session")
def mixed_df(rng):
    rng = rng(seed=43)
    n = 300

    age = rng.integers(18, 75, size=n)
    income = age * 1200 + rng.normal(0, 5000, size=n)

    salary = rng.normal(50_000, 15_000, size=n).tolist()
    null_mask = rng.random(n) < 0.10
    salary = [None if null_mask[i] else salary[i] for i in range(n)]

    country_choices = ["US", "UK", "CA", "AU", "DE"]
    country = [country_choices[i % len(country_choices)] for i in range(n)]

    names = [f"person_{i}" for i in range(n)]

    is_active = [bool(v) for v in rng.integers(0, 2, size=n)]

    from datetime import date, timedelta
    base = date(2020, 1, 1)
    joined = [base + timedelta(days=int(d)) for d in rng.integers(0, 1460, size=n)]

    return pl.DataFrame({
        "age": pl.Series(age.tolist(), dtype=pl.Int64),
        "income": pl.Series(income.tolist(), dtype=pl.Float64),
        "salary": pl.Series(salary, dtype=pl.Float64),
        "country": pl.Series(country, dtype=pl.Utf8),
        "name": pl.Series(names, dtype=pl.Utf8),
        "is_active": pl.Series(is_active, dtype=pl.Boolean),
        "joined": pl.Series(joined, dtype=pl.Date),
    })
