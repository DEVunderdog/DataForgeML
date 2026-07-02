from datetime import date, datetime, timedelta, timezone

import polars as pl

from dataforge_ml.profiling._datetime_profiler import DatetimeProfiler
from dataforge_ml.profiling._datetime_config import (
    DatetimeFlag,
    DatetimeProfileConfig,
    DatetimeProfileResult,
    InferredGranularity,
)


# ---------------------------------------------------------------------------
# Result type & analysed_columns
# ---------------------------------------------------------------------------


def test_result_type_and_analysed_columns():
    df = pl.DataFrame(
        {
            "event_date": pl.Series(
                [date(2024, 1, i + 1) for i in range(5)], dtype=pl.Date
            ),
            "score": pl.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype=pl.Float64),
        }
    )
    result = DatetimeProfiler().profile(df, ["event_date", "score"])
    assert isinstance(result, DatetimeProfileResult)
    assert "event_date" in result.analysed_columns
    assert "score" not in result.analysed_columns


# ---------------------------------------------------------------------------
# Range: min_date <= max_date
# ---------------------------------------------------------------------------


def test_min_date_lte_max_date():
    dates = [date(2024, 1, 1), date(2024, 6, 15), date(2024, 3, 10)]
    df = pl.DataFrame({"ts": pl.Series(dates, dtype=pl.Date)})
    stats = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert stats.min_date <= stats.max_date


# ---------------------------------------------------------------------------
# Range: range_days non-negative and consistent with min/max
# ---------------------------------------------------------------------------


def test_range_days_non_negative_and_consistent():
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
    df = pl.DataFrame({"ts": pl.Series(dates, dtype=pl.Date)})
    stats = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert stats.date_range_days >= 0
    expected_days = (stats.max_date - stats.min_date).total_seconds() / 86_400.0
    assert abs(stats.date_range_days - expected_days) < 1e-6


# ---------------------------------------------------------------------------
# FutureDates flag present and absent
# ---------------------------------------------------------------------------


def test_future_dates_flag_present_and_absent():
    past_dates = [date(2020, 1, 1), date(2021, 6, 1), date(2022, 12, 31)]
    future_dates = [date(2099, 1, 1), date(2100, 6, 1)]

    df_future = pl.DataFrame(
        {"ts": pl.Series(past_dates + future_dates, dtype=pl.Date)}
    )
    stats_future = DatetimeProfiler().profile(df_future, ["ts"]).columns["ts"]
    assert DatetimeFlag.FutureDates in stats_future.flags

    df_past = pl.DataFrame({"ts": pl.Series(past_dates, dtype=pl.Date)})
    stats_past = DatetimeProfiler().profile(df_past, ["ts"]).columns["ts"]
    assert DatetimeFlag.FutureDates not in stats_past.flags


# ---------------------------------------------------------------------------
# Inferred granularity bands: daily, hourly, monthly
# ---------------------------------------------------------------------------


def test_inferred_granularity_daily_hourly_monthly():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Daily: 1-day gaps → median gap ≈ 86 400 s → Daily band
    daily_ts = [base + timedelta(days=i) for i in range(30)]
    df_daily = pl.DataFrame(
        {"ts": pl.Series(daily_ts, dtype=pl.Datetime("us", "UTC"))}
    )
    stats_daily = DatetimeProfiler().profile(df_daily, ["ts"]).columns["ts"]
    assert stats_daily.inferred_granularity == InferredGranularity.Daily

    # Hourly: 1-hour gaps → median gap ≈ 3 600 s → Hourly band
    hourly_ts = [base + timedelta(hours=i) for i in range(48)]
    df_hourly = pl.DataFrame(
        {"ts": pl.Series(hourly_ts, dtype=pl.Datetime("us", "UTC"))}
    )
    stats_hourly = DatetimeProfiler().profile(df_hourly, ["ts"]).columns["ts"]
    assert stats_hourly.inferred_granularity == InferredGranularity.Hourly

    # Monthly: first day of each month → median gap ≈ 30 days → Monthly band
    monthly_ts = [datetime(2024, m, 1, tzinfo=timezone.utc) for m in range(1, 13)]
    df_monthly = pl.DataFrame(
        {"ts": pl.Series(monthly_ts, dtype=pl.Datetime("us", "UTC"))}
    )
    stats_monthly = DatetimeProfiler().profile(df_monthly, ["ts"]).columns["ts"]
    assert stats_monthly.inferred_granularity == InferredGranularity.Monthly


# ---------------------------------------------------------------------------
# Temporal signals: month and day-of-week vary, year does not
# ---------------------------------------------------------------------------


def test_temporal_signals_month_and_dow_vary_not_year():
    # Six dates in 2024 across distinct months and distinct weekdays, same year.
    # 2024-01-01 = Monday (0), 2024-02-04 = Sunday (6), 2024-03-06 = Wednesday (2)
    # 2024-04-12 = Friday (4), 2024-05-14 = Tuesday (1), 2024-06-22 = Saturday (5)
    dates_2024 = [
        date(2024, 1, 1),
        date(2024, 2, 4),
        date(2024, 3, 6),
        date(2024, 4, 12),
        date(2024, 5, 14),
        date(2024, 6, 22),
    ]
    df = pl.DataFrame({"ts": pl.Series(dates_2024, dtype=pl.Date)})
    stats = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert stats.signals.has_year is False
    assert stats.signals.has_month is True
    assert stats.signals.has_day_of_week is True


# ---------------------------------------------------------------------------
# Config: overriding mnar_null_ratio_threshold changes MnarSuspected flag
# ---------------------------------------------------------------------------


def test_mnar_suspected_fires_when_null_ratio_exceeds_threshold():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 8 non-null values + 2 nulls → null_ratio = 0.20
    ts_values = [base + timedelta(days=i) for i in range(8)] + [None, None]
    df = pl.DataFrame(
        {"ts": pl.Series(ts_values, dtype=pl.Datetime("us", "UTC"))}
    )

    # Default threshold (0.05): 20% > 5% → flag expected
    stats_default = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.MnarSuspected in stats_default.flags

    # Raise threshold above 20%: flag should not fire
    cfg = DatetimeProfileConfig(mnar_null_ratio_threshold=0.25)
    stats_high = DatetimeProfiler(config=cfg).profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.MnarSuspected not in stats_high.flags


def test_mnar_suspected_absent_below_threshold():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # 19 non-null + 1 null → null_ratio = 0.05 (not strictly greater)
    ts_values = [base + timedelta(days=i) for i in range(19)] + [None]
    df = pl.DataFrame(
        {"ts": pl.Series(ts_values, dtype=pl.Datetime("us", "UTC"))}
    )

    # Default threshold (0.05): ratio == threshold, not strictly greater → no flag
    stats = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.MnarSuspected not in stats.flags

    # Lower threshold so 5% triggers
    cfg = DatetimeProfileConfig(mnar_null_ratio_threshold=0.04)
    stats_low = DatetimeProfiler(config=cfg).profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.MnarSuspected in stats_low.flags


# ---------------------------------------------------------------------------
# Config: overriding high_gap_cv_threshold changes HighGapVariance flag
# ---------------------------------------------------------------------------


def test_high_gap_variance_flag_controlled_by_config():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # Irregular gaps: 1d, 1d, 1d, 30d, 1d, 1d → high CV
    offsets = [0, 1, 2, 3, 33, 34, 35]
    ts_values = [base + timedelta(days=d) for d in offsets]
    df = pl.DataFrame(
        {"ts": pl.Series(ts_values, dtype=pl.Datetime("us", "UTC"))}
    )

    # Default threshold (1.0): high CV dataset → flag expected
    stats_default = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.HighGapVariance in stats_default.flags

    # Raise threshold very high: same data → flag absent
    cfg = DatetimeProfileConfig(high_gap_cv_threshold=100.0)
    stats_high = DatetimeProfiler(config=cfg).profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.HighGapVariance not in stats_high.flags


def test_high_gap_variance_absent_for_regular_series():
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    # Perfectly uniform daily series → CV near 0
    ts_values = [base + timedelta(days=i) for i in range(30)]
    df = pl.DataFrame(
        {"ts": pl.Series(ts_values, dtype=pl.Datetime("us", "UTC"))}
    )
    stats = DatetimeProfiler().profile(df, ["ts"]).columns["ts"]
    assert DatetimeFlag.HighGapVariance not in stats.flags


# ---------------------------------------------------------------------------
# Config: round-trip serialisation preserves all fields
# ---------------------------------------------------------------------------


def test_datetime_profile_config_round_trip_defaults():
    cfg = DatetimeProfileConfig()
    assert DatetimeProfileConfig.from_dict(cfg.to_dict()) == cfg


def test_datetime_profile_config_round_trip_custom_values():
    cfg = DatetimeProfileConfig(
        mnar_null_ratio_threshold=0.02,
        high_gap_cv_threshold=1.5,
        recent_window_fraction=0.20,
    )
    restored = DatetimeProfileConfig.from_dict(cfg.to_dict())
    assert restored.mnar_null_ratio_threshold == 0.02
    assert restored.high_gap_cv_threshold == 1.5
    assert restored.recent_window_fraction == 0.20


def test_datetime_profile_config_exported_from_profiling_api():
    from dataforge_ml.profiling import DatetimeProfileConfig as _Exported
    assert _Exported is DatetimeProfileConfig


def test_override_coercion_error_raised_for_total_failure():
    import pytest
    from dataforge_ml.profiling import OverrideCoercionError

    df = pl.DataFrame({"ts": pl.Series(["abc", "def", "ghi"])})
    with pytest.raises(OverrideCoercionError, match="completely failed coercion"):
        DatetimeProfiler().profile(df, ["ts"], user_overrides={"ts"})


def test_override_coercion_error_not_raised_for_partial_failure():
    from dataforge_ml.profiling import OverrideCoercionError
    # 1 valid, 2 invalid
    df = pl.DataFrame({"ts": pl.Series(["2024-01-01", "abc", "def"])})
    result = DatetimeProfiler().profile(df, ["ts"], user_overrides={"ts"})
    assert "ts" in result.analysed_columns


def test_override_coercion_error_not_raised_for_auto_detected_total_failure():
    df = pl.DataFrame({"ts": pl.Series(["abc", "def", "ghi"])})
    result = DatetimeProfiler().profile(df, ["ts"])
    assert "ts" not in result.analysed_columns


# ---------------------------------------------------------------------------
# Epoch Units (ADR-0045)
# ---------------------------------------------------------------------------


def test_datetime_profiler_with_epoch_units():
    df = pl.DataFrame({
        "ts_s": pl.Series([1700000000, 1700000060], dtype=pl.Int64),
        "ts_ms": pl.Series([1700000000000, 1700000060000], dtype=pl.Int64)
    })
    
    from dataforge_ml.profiling._config import EpochUnit
    epoch_units = {"ts_s": EpochUnit("s"), "ts_ms": EpochUnit("ms")}
    
    result = DatetimeProfiler(epoch_units=epoch_units).profile(df, ["ts_s", "ts_ms"])
    
    assert "ts_s" in result.analysed_columns
    assert "ts_ms" in result.analysed_columns
    
    stats_s = result.columns["ts_s"]
    stats_ms = result.columns["ts_ms"]
    
    assert stats_s.min_date is not None
    assert stats_s.min_date.year == 2023
    assert stats_ms.min_date is not None
    assert stats_ms.min_date.year == 2023
    assert stats_s.date_range_days == stats_ms.date_range_days


def test_numeric_override_without_epoch_unit_raises_error():
    import pytest
    from dataforge_ml.profiling import OverrideCoercionError
    
    df = pl.DataFrame({"ts": pl.Series([1700000000, 1700000060], dtype=pl.Int64)})
    
    with pytest.raises(OverrideCoercionError, match="completely failed coercion"):
        DatetimeProfiler().profile(df, ["ts"], user_overrides={"ts"})


# ---------------------------------------------------------------------------
# Declared per-column datetime format (formats constructor arg)
# ---------------------------------------------------------------------------


def test_declared_format_parses_bare_year_column():
    df = pl.DataFrame({"Year": pl.Series(["2013", "2014", "2015"], dtype=pl.Utf8)})
    result = DatetimeProfiler(formats={"Year": "%Y"}).profile(df, ["Year"])
    assert "Year" in result.analysed_columns
    stats = result.columns["Year"]
    assert stats.min_date is not None
    assert stats.min_date.year == 2013
    assert stats.max_date.year == 2015


def test_bare_year_without_declared_format_is_not_analysed():
    df = pl.DataFrame({"Year": pl.Series(["2013", "2014", "2015"], dtype=pl.Utf8)})
    result = DatetimeProfiler().profile(df, ["Year"])
    assert "Year" not in result.analysed_columns


def test_declared_format_partial_match_parses_good_rows():
    # "%Y" matches the bare years; "banana" fails coercion → null (strict=False).
    df = pl.DataFrame(
        {"Year": pl.Series(["2013", "banana", "2015"], dtype=pl.Utf8)}
    )
    result = DatetimeProfiler(formats={"Year": "%Y"}).profile(df, ["Year"])
    assert "Year" in result.analysed_columns
    stats = result.columns["Year"]
    assert stats.min_date.year == 2013
    assert stats.max_date.year == 2015


def test_declared_format_matching_zero_rows_raises_for_override():
    from dataforge_ml.profiling import OverrideCoercionError
    import pytest as _pytest

    df = pl.DataFrame({"Year": pl.Series(["2013", "2014"], dtype=pl.Utf8)})
    with _pytest.raises(OverrideCoercionError, match="set_datetime_format"):
        DatetimeProfiler(formats={"Year": "%H:%M:%S"}).profile(
            df, ["Year"], user_overrides={"Year"}
        )


def test_declared_format_matching_zero_rows_auto_detected_yields_empty():
    df = pl.DataFrame({"Year": pl.Series(["2013", "2014"], dtype=pl.Utf8)})
    result = DatetimeProfiler(formats={"Year": "%H:%M:%S"}).profile(df, ["Year"])
    assert "Year" not in result.analysed_columns


# ---------------------------------------------------------------------------
# FormatMismatch flag (present-but-uncoercible values)
# ---------------------------------------------------------------------------


def test_format_mismatch_flag_fires_on_dirty_inference_column():
    df = pl.DataFrame(
        {"d": pl.Series(["2024-01-01", "banana", "2024-03-01"], dtype=pl.Utf8)}
    )
    result = DatetimeProfiler().profile(df, ["d"])
    stats = result.columns["d"]
    assert stats.has_flag(DatetimeFlag.FormatMismatch)


def test_format_mismatch_flag_absent_on_clean_inference_column():
    df = pl.DataFrame(
        {"d": pl.Series(["2024-01-01", "2024-02-01", "2024-03-01"], dtype=pl.Utf8)}
    )
    result = DatetimeProfiler().profile(df, ["d"])
    stats = result.columns["d"]
    assert not stats.has_flag(DatetimeFlag.FormatMismatch)


def test_format_mismatch_flag_fires_on_declared_format_partial_failure():
    df = pl.DataFrame({"Year": pl.Series(["2013", "banana", "2015"], dtype=pl.Utf8)})
    result = DatetimeProfiler(formats={"Year": "%Y"}).profile(df, ["Year"])
    stats = result.columns["Year"]
    assert stats.has_flag(DatetimeFlag.FormatMismatch)


def test_format_mismatch_flag_absent_on_declared_format_clean_column():
    df = pl.DataFrame({"Year": pl.Series(["2013", "2014", "2015"], dtype=pl.Utf8)})
    result = DatetimeProfiler(formats={"Year": "%Y"}).profile(df, ["Year"])
    stats = result.columns["Year"]
    assert not stats.has_flag(DatetimeFlag.FormatMismatch)


def test_format_mismatch_does_not_alter_null_ratio_signals():
    # Genuine nulls (None) are present alongside dirt; the null-based MNAR
    # signal is computed from the coerced null count exactly as before.
    dates = ["2024-01-01"] * 2 + [None] * 8  # 80% null → MnarSuspected
    df = pl.DataFrame({"d": pl.Series(dates, dtype=pl.Utf8)})
    result = DatetimeProfiler().profile(df, ["d"])
    stats = result.columns["d"]
    assert stats.has_flag(DatetimeFlag.MnarSuspected)
    assert not stats.has_flag(DatetimeFlag.FormatMismatch)
