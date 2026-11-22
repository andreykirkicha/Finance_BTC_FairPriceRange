"""Streaming reader for the PostgreSQL COPY block of the historical dump."""

import gzip
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

__all__ = [
    "DatasetStats",
    "OptionRow",
    "ScanResult",
    "parse_expiry_utc",
    "parse_utc",
    "scan_options_table",
    "timestamp_key",
]

DERIBIT_EXPIRY_HOUR_UTC = 8


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Counts collected while scanning the whole options table once."""

    table: str
    total_rows: int
    btc_rows: int
    btc_calls: int
    btc_puts: int
    eth_rows: int
    other_rows: int
    snapshots: int
    latest_snapshot: datetime | None
    input_bytes: int


@dataclass(frozen=True, slots=True)
class OptionRow:
    """One raw option row from the dump, with numeric fields already parsed."""

    instrument_name: str
    option_type: str
    expiry_raw: str
    snapshot_raw: str
    strike: float | None
    underlying_price: float | None
    bid_price: float | None
    ask_price: float | None
    mid_price: float | None
    exchange_mid_iv: float | None
    interest_rate: float | None

    @property
    def expiry_utc(self) -> datetime:
        return parse_expiry_utc(self.expiry_raw)

    @property
    def snapshot_utc(self) -> datetime:
        return parse_utc(self.snapshot_raw)


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Rows for the selected snapshot plus whole-table statistics."""

    stats: DatasetStats
    rows: tuple[OptionRow, ...]
    snapshot: datetime | None


def timestamp_key(raw: str) -> tuple[str, str]:
    """Return a sortable key without parsing every timestamp in the dump.

    PostgreSQL may print fractional seconds with a variable number of digits;
    pad them to microseconds so string order and numeric order agree.
    """
    base, separator, fraction = raw.strip().partition(".")
    return base, (fraction + "000000")[:6] if separator else "000000"


def parse_utc(raw: str) -> datetime:
    """Parse a naive dump timestamp as UTC."""
    try:
        value = datetime.fromisoformat(raw.strip())
    except ValueError as error:
        raise ValueError(f"Invalid timestamp in dump: {raw!r}") from error
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_expiry_utc(raw: str) -> datetime:
    """Parse expiry and repair the dump's two conventions.

    The source stores 2024 and 2025 as ``0024`` and ``0025`` and keeps only the
    expiry date at midnight. Deribit options expire at 08:00 UTC, which is
    confirmed by matching the exchange ``mid_IV``; a midnight timestamp is
    therefore moved to 08:00. Without both repairs ``T`` is badly wrong.
    """
    value = parse_utc(raw)
    if value.year < 1000:
        value = value.replace(year=value.year + 2000)
    if value.hour == value.minute == value.second == value.microsecond == 0:
        value = value.replace(hour=DERIBIT_EXPIRY_HOUR_UTC)
    return value


def _number(raw: str | None) -> float | None:
    if raw is None or raw.strip() in {"", r"\N"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _parse_copy_columns(line: str) -> tuple[str, ...]:
    start, end = line.find("("), line.find(")")
    if start < 0 or end < start:
        raise ValueError(f"Cannot parse COPY column list: {line[:120]!r}")
    return tuple(column.strip().strip('"') for column in line[start + 1 : end].split(","))


def _column_index(columns: tuple[str, ...], name: str) -> int:
    try:
        return columns.index(name)
    except ValueError as error:
        raise ValueError(f"Required column {name!r} is missing from the COPY block") from error


def _row_from_parts(columns: tuple[str, ...], parts: list[str]) -> OptionRow:
    values = dict(zip(columns, parts))
    return OptionRow(
        instrument_name=values["full_name"],
        option_type=values["type"],
        expiry_raw=values["expiry"],
        snapshot_raw=values["current_time"],
        strike=_number(values.get("strike")),
        underlying_price=_number(values.get("underlying_price")),
        bid_price=_number(values.get("bid_price")),
        ask_price=_number(values.get("ask_price")),
        mid_price=_number(values.get("mid_price")),
        exchange_mid_iv=_number(values.get("mid_IV")),
        interest_rate=_number(values.get("interest_rate")),
    )


def scan_options_table(
    path: Path,
    snapshot: datetime | None = None,
    table: str = "public.options_table",
) -> ScanResult:
    """Scan a gzip/plain pg_dump and return the requested BTC call snapshot.

    The whole table is read exactly once. While scanning, rows belonging to the
    latest BTC call snapshot (or to ``snapshot`` when supplied) are retained;
    all other rows are only counted. This avoids loading 35 million rows.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dump not found: {path}")

    target_key = timestamp_key(snapshot.strftime("%Y-%m-%d %H:%M:%S.%f")) if snapshot else None
    copy_prefix = f"COPY {table} "
    columns: tuple[str, ...] | None = None
    in_copy = False
    total_rows = btc_rows = btc_calls = btc_puts = eth_rows = other_rows = 0
    snapshot_keys: set[tuple[str, str]] = set()
    latest_key: tuple[str, str] | None = None
    latest_raw: str | None = None
    latest_rows: list[OptionRow] = []
    target_rows: list[OptionRow] = []
    name_index = type_index = time_index = -1

    with _open_text(path) as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            if not in_copy:
                if line.startswith(copy_prefix):
                    columns = _parse_copy_columns(line)
                    name_index = _column_index(columns, "full_name")
                    type_index = _column_index(columns, "type")
                    time_index = _column_index(columns, "current_time")
                    in_copy = True
                continue
            if line.startswith("\\."):
                break
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) != len(columns):
                raise ValueError(
                    f"Malformed COPY row at line {line_number}: expected {len(columns)} fields, got {len(parts)}"
                )
            total_rows += 1
            name = parts[name_index]
            if not name.startswith("BTC-"):
                if name.startswith("ETH-"):
                    eth_rows += 1
                else:
                    other_rows += 1
                continue

            btc_rows += 1
            option_type = parts[type_index]
            if option_type == "put":
                btc_puts += 1
                continue
            if option_type != "call":
                continue

            btc_calls += 1
            current_raw = parts[time_index]
            current_key = timestamp_key(current_raw)
            snapshot_keys.add(current_key)

            if latest_key is None or current_key > latest_key:
                latest_key = current_key
                latest_raw = current_raw
                latest_rows = [_row_from_parts(columns, parts)]
            elif current_key == latest_key:
                latest_rows.append(_row_from_parts(columns, parts))

            if target_key is not None and current_key == target_key:
                target_rows.append(_row_from_parts(columns, parts))

    if columns is None:
        raise ValueError(f"COPY block for {table!r} was not found in {path}")

    latest_snapshot = parse_utc(latest_raw) if latest_raw is not None else None
    selected_snapshot = snapshot if target_key is not None else latest_snapshot
    selected_rows = tuple(target_rows if target_key is not None else latest_rows)
    stats = DatasetStats(
        table=table,
        total_rows=total_rows,
        btc_rows=btc_rows,
        btc_calls=btc_calls,
        btc_puts=btc_puts,
        eth_rows=eth_rows,
        other_rows=other_rows,
        snapshots=len(snapshot_keys),
        latest_snapshot=latest_snapshot,
        input_bytes=path.stat().st_size,
    )
    return ScanResult(stats=stats, rows=selected_rows, snapshot=selected_snapshot)
