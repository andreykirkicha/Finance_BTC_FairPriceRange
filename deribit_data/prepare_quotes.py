"""Prepare the fixed research period's BTC call quotes from a historical SQL dump."""

import argparse
import csv
import gzip
import json
import math
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from datetime import time as day_time
from pathlib import Path
from typing import Any

__all__ = [
    "DatasetStats",
    "OptionRow",
    "ScanResult",
    "parse_expiry_utc",
    "parse_utc",
    "prepare_quotes",
    "scan_options_table",
    "timestamp_key",
    "write_table",
]

DERIBIT_EXPIRY_HOUR_UTC = 8
START_DATE = date(2024, 5, 20)
END_DATE = date(2024, 6, 3)  # Exclusive UTC boundary.
EXPIRY_DATE = date(2024, 7, 26)
MAX_LOG_MONEYNESS = 0.15
MAX_RELATIVE_SPREAD = 0.20
FIELDS = (
    "timestamp_utc",
    "instrument_name",
    "expiry_utc",
    "strike_usd",
    "forward_usd",
    "bid_btc",
    "ask_btc",
    "mid_btc",
    "time_to_expiry_years",
    "log_moneyness",
    "relative_spread",
)


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
    exchange_mid_iv: float | None

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
        # Python 3.10's fromisoformat rejects PostgreSQL's variable-length
        # fractional seconds (e.g. .36765); strptime accepts 1–6 digits.
        try:
            value = datetime.strptime(raw.strip(), "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
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
    values = dict(zip(columns, parts, strict=False))
    return OptionRow(
        instrument_name=values["full_name"],
        option_type=values["type"],
        expiry_raw=values["expiry"],
        snapshot_raw=values["current_time"],
        strike=_number(values.get("strike")),
        underlying_price=_number(values.get("underlying_price")),
        bid_price=_number(values.get("bid_price")),
        ask_price=_number(values.get("ask_price")),
        exchange_mid_iv=_number(values.get("mid_IV")),
    )


def _iter_options(path: Path, table: str = "public.options_table") -> Iterator[tuple[tuple[str, ...], list[str]]]:
    """Stream one COPY block, rejecting missing columns, broken rows and truncated files."""
    columns = None
    with _open_text(path) as stream:
        for line_number, raw in enumerate(stream, start=1):
            line = raw.rstrip("\r\n")
            if columns is None:
                if line.startswith(f"COPY {table} "):
                    columns = _parse_copy_columns(line)
                    for field in (
                        "full_name",
                        "type",
                        "expiry",
                        "current_time",
                        "strike",
                        "underlying_price",
                        "bid_price",
                        "ask_price",
                    ):
                        _column_index(columns, field)
                continue
            if line == r"\.":
                return
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(columns):
                raise ValueError(
                    f"Malformed COPY row at line {line_number}: expected {len(columns)} fields, got {len(parts)}"
                )
            yield columns, parts
    if columns is None:
        raise ValueError(f"COPY block for {table!r} was not found in {path}")
    raise ValueError(f"COPY block for {table!r} is incomplete: terminator not found")


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
    total_rows = btc_rows = btc_calls = btc_puts = eth_rows = other_rows = 0
    snapshot_keys: set[tuple[str, str]] = set()
    latest_key: tuple[str, str] | None = None
    latest_raw: str | None = None
    latest_rows: list[OptionRow] = []
    target_rows: list[OptionRow] = []
    name_index = type_index = time_index = -1

    for columns, parts in _iter_options(path, table):
        if name_index == -1:
            name_index = _column_index(columns, "full_name")
            type_index = _column_index(columns, "type")
            time_index = _column_index(columns, "current_time")
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


def write_table(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Replace a CSV only after a successful write; leave no temporary files behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_quotes(
    dump: Path,
    output: Path,
    start: date = START_DATE,
    end: date = END_DATE,
    expiry_day: date = EXPIRY_DATE,
) -> dict[str, Any]:
    """Extract the research quotes, keeping selected rows in memory and no disk cache."""
    if not start < end <= expiry_day:
        raise ValueError("Need start < exclusive end <= expiry")
    if dump.resolve() == output.resolve():
        raise ValueError("Output must not overwrite the source dump")
    expiry = datetime.combine(expiry_day, day_time(8), tzinfo=timezone.utc)
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    indexes: dict[str, int] = {}
    timestamps: dict[str, datetime] = {}
    expiries: dict[str, datetime] = {}
    for columns, parts in _iter_options(dump):
        if not indexes:
            indexes = {key: i for i, key in enumerate(columns)}
        counts["all_option_rows"] += 1
        name = parts[indexes["full_name"]]
        if parts[indexes["type"]] != "call" or not name.startswith("BTC-"):
            continue
        raw_time = parts[indexes["current_time"]]
        if not start.isoformat() <= raw_time[:10] < end.isoformat():
            continue
        if raw_time not in timestamps:
            timestamps[raw_time] = parse_utc(raw_time)
        timestamp = timestamps[raw_time]
        if not start <= timestamp.date() < end:
            continue
        counts["btc_call_rows_in_window"] += 1
        raw_expiry = parts[indexes["expiry"]]
        if raw_expiry not in expiries:
            expiries[raw_expiry] = parse_expiry_utc(raw_expiry)
        if expiries[raw_expiry] != expiry:
            continue
        counts["same_expiry_rows"] += 1
        forward, strike, bid, ask = (
            _number(parts[indexes[key]]) for key in ("underlying_price", "strike", "bid_price", "ask_price")
        )
        if any(value is None or value <= 0 for value in (forward, strike, bid, ask)):
            counts["excluded_invalid_quote_or_parameters"] += 1
            continue
        if bid > ask:
            counts["excluded_crossed_quote"] += 1
            continue
        mid = (bid + ask) / 2
        if not max(1 - strike / forward, 0) < mid < 1:
            counts["excluded_outside_model_bounds"] += 1
            continue
        moneyness = math.log(strike / forward)
        spread = (ask - bid) / mid
        if abs(moneyness) > MAX_LOG_MONEYNESS:
            counts["excluded_outside_atm"] += 1
            continue
        if spread > MAX_RELATIVE_SPREAD:
            counts["excluded_wide_spread"] += 1
            continue
        name = name if name.endswith("-C") else name + "-C"
        timestamp_text = timestamp.isoformat(timespec="microseconds")
        key = timestamp_text, name
        if key in rows:
            raise ValueError(f"Duplicate quote: {timestamp_text}, {name}")
        rows[key] = dict(
            zip(
                FIELDS,
                (
                    timestamp_text,
                    name,
                    expiry.isoformat(),
                    strike,
                    forward,
                    bid,
                    ask,
                    mid,
                    (expiry - timestamp).total_seconds() / (365 * 86400),
                    moneyness,
                    spread,
                ),
                strict=True,
            )
        )
    if not rows:
        raise ValueError("No suitable historical quotes")
    write_table(output, FIELDS, (rows[key] for key in sorted(rows)))
    return {
        "quote_rows": len(rows),
        "quote_contracts": len({key[1] for key in rows}),
        "quote_snapshots": len({key[0] for key in rows}),
        "selection_counts": dict(counts),
    }


def main() -> None:
    """Prepare quotes.csv without a database, checkpoints or separate metadata files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path("data/dump.sql"))
    parser.add_argument("--output", type=Path, default=Path("data/quotes.csv"))
    parser.add_argument("--start", type=date.fromisoformat, default=START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=END_DATE, help="Exclusive UTC end")
    parser.add_argument("--expiry", type=date.fromisoformat, default=EXPIRY_DATE)
    args = parser.parse_args()
    print(json.dumps(prepare_quotes(args.dump, args.output, args.start, args.end, args.expiry), indent=2))


if __name__ == "__main__":
    main()
