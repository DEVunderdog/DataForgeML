import polars as pl

_INT_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}

_FLOAT_DTYPES = (pl.Float32, pl.Float64)

_CAT_DTYPES = {pl.Utf8, pl.String, pl.Categorical, pl.Boolean}

_NUMERIC_DTYPES = _INT_DTYPES | {pl.Float32, pl.Float64}

_DATETIME_DTYPES = {pl.Date, pl.Datetime}

# Recognised boolean vocabulary for string-encoded Boolean columns. Lives here
# rather than beside either consumer because both the boolean profiler and the
# dtype-floor normaliser must agree on it exactly: a token the floor cannot map
# becomes an effective null, and a token the profiler cannot map raises
# FormatMismatch.
_TRUE_STRINGS: frozenset[str] = frozenset({"true", "yes", "1", "t", "y"})
_FALSE_STRINGS: frozenset[str] = frozenset({"false", "no", "0", "f", "n"})
