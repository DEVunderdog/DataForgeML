from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Union

import chardet
import polars as pl

from ..records import DataForgeError

PathOrBuffer = Union[str, Path, io.IOBase, io.RawIOBase, io.BufferedIOBase]


def _read_raw(source: PathOrBuffer) -> tuple[bytes, str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise DataForgeError.from_code(
                "input_not_found", f"No file exists at '{path}'.", param="source"
            )

        ext = path.suffix.lower()
        raw = path.read_bytes()

        return raw, ext

    pos = source.tell() if hasattr(source, "tell") else None
    raw = source.read()

    if pos is not None:
        try:
            source.seek(pos)
        except Exception:
            pass

    return raw, None


def _detect_encoding(raw: bytes) -> str:
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def _sniff_csv_delimiter(text: str) -> str:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","



def _load_csv(raw: bytes) -> pl.DataFrame:
    """Load CSV/TSV bytes with auto-detected encoding and delimiter."""
    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")
    delimiter = _sniff_csv_delimiter(text)
    return pl.read_csv(
        io.BytesIO(raw),
        separator=delimiter,
        encoding=encoding,
        infer_schema_length=10_000,
        try_parse_dates=True,
    )


_EXT_LOADERS: dict[str, callable] = {
    ".csv": _load_csv,
    ".tsv": _load_csv,
    ".parquet": lambda raw: pl.read_parquet(io.BytesIO(raw)),
    ".json": lambda raw: pl.read_json(io.BytesIO(raw)),
    ".ndjson": lambda raw: pl.read_ndjson(io.BytesIO(raw)),
    ".jsonl": lambda raw: pl.read_ndjson(io.BytesIO(raw)),
    ".xlsx": lambda raw: pl.read_excel(io.BytesIO(raw)),
    ".xls": lambda raw: pl.read_excel(io.BytesIO(raw)),
    ".arrow": lambda raw: pl.read_ipc(io.BytesIO(raw)),
    ".feather": lambda raw: pl.read_ipc(io.BytesIO(raw)),
}


class DataLoader:
    """Read a tabular file into a polars DataFrame, picking the reader by extension."""

    def load(
        self,
        source: PathOrBuffer,
        fmt: str | None = None,
    ) -> pl.DataFrame:
        """Load ``source`` into a DataFrame.

        Parameters
        ----------
        source : str, Path or file-like
            A path, or an open binary buffer.
        fmt : str, optional
            Accepted but not read: the format comes from the path's extension,
            so a buffer source always fails as ``unsupported_format``.

        Returns
        -------
        polars.DataFrame
            The loaded table.

        Raises
        ------
        InvalidInput
            ``input_not_found`` if a path source does not exist.
            ``unsupported_format`` if the extension has no registered reader.
        """
        raw, ext_from_path = _read_raw(source)

        resolved_fmt = (ext_from_path or "").lower()

        if resolved_fmt not in _EXT_LOADERS:
            label = resolved_fmt if resolved_fmt else "<unknown>"
            raise DataForgeError.from_code(
                "unsupported_format",
                f"Unsupported file format: '{label}'.",
                param="source",
                allowed=sorted(_EXT_LOADERS),
            )

        loader = _EXT_LOADERS[resolved_fmt]
        return loader(raw)
