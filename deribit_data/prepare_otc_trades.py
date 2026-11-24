"""Подготовка OTC-сделок BTC call.

По умолчанию используется локальный CSV; флаг --download включает загрузку за выбранный период.
"""

import argparse
import csv
import gzip
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime, timedelta, timezone
from http.client import HTTPException
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from deribit_data.prepare_quotes import (
    END_DATE,
    EXPIRY_DATE,
    MAX_DAYS_TO_EXPIRY,
    MIN_DAYS_TO_EXPIRY,
    START_DATE,
    parse_utc,
    write_csv,
)

LOGGER = logging.getLogger(__name__)
HISTORY_URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
PAGE_SIZE = 1_000
TRADE_FIELDS = (
    "trade_id",
    "block_trade_id",
    "trade_seq",
    "timestamp_ms",
    "trade_utc",
    "instrument_name",
    "currency",
    "option_type",
    "expiry_utc",
    "strike_usd",
    "price_btc",
    "amount_btc",
    "contracts",
    "direction",
    "index_price_usd",
    "mark_price_btc",
    "exchange_iv_pct",
    "days_to_expiry",
    "block_trade_leg_count",
    "observed_option_legs_in_block",
    "package_status",
    "block_rfq_id",
    "combo_id",
    "combo_trade_id",
    "block_delayed",
    "source_file",
)


def date_to_timestamp_ms(value: date) -> int:
    return int(datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)


def fetch_complete_range(
    fetch: Callable[[int, int], dict[str, Any]],
    start: int,
    end: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    pending = [(start, end)]
    trades: dict[str, dict[str, Any]] = {}
    audit = []
    requests = 0
    while pending:
        low, high = pending.pop()
        payload = fetch(low, high)
        requests += 1
        if payload.get("testnet") or "error" in payload:
            raise ValueError(f"Invalid historical response: {payload.get('error', 'testnet')}")
        result = payload["result"]
        rows, more = result["trades"], result["has_more"]
        if not isinstance(rows, list) or not isinstance(more, bool):
            raise ValueError("Expected trades list and explicit boolean has_more")
        for row in rows:
            if not low <= row["timestamp"] <= high:
                raise ValueError("Server returned a trade outside the requested time range")
        if more:
            if low == high:
                raise ValueError(f"Truncated millisecond {low}; cannot certify completeness")
            middle = (low + high) // 2
            pending.extend([(middle + 1, high), (low, middle)])
            continue
        audit.append(
            {
                "start_ms": low,
                "end_ms": high,
                "count": len(rows),
                "has_more": False,
                "usIn": payload.get("usIn"),
                "usOut": payload.get("usOut"),
            }
        )
        for row in rows:
            trade_id = str(row["trade_id"])
            if trade_id in trades:
                raise ValueError(f"Duplicate trade ID in disjoint ranges: {trade_id}")
            trades[trade_id] = row
    return sorted(trades.values(), key=lambda row: (row["timestamp"], str(row["trade_id"]))), audit, requests


class HistoryClient:
    def __init__(self, requests_per_second: float = 8.0) -> None:
        self.interval = 1.0 / requests_per_second
        self.next_request = 0.0

    def fetch(self, start: int, end: int) -> dict[str, Any]:
        params = dict(
            currency="BTC", kind="option", start_timestamp=start, end_timestamp=end, count=PAGE_SIZE, sorting="asc"
        )
        request = Request(
            f"{HISTORY_URL}?{urlencode(params)}",
            headers={
                "User-Agent": "btc-option-research/0.1",
                "Accept": "application/json",
            },
        )
        for attempt in range(7):
            delay = max(0.0, self.next_request - time.monotonic())
            self.next_request = time.monotonic() + delay + self.interval
            time.sleep(delay)
            try:
                with urlopen(request, timeout=45) as response:
                    payload = json.load(response)
                if payload.get("error", {}).get("code") == 10028:
                    raise URLError("Deribit rate limit")
                if payload.get("jsonrpc") != "2.0" or "result" not in payload or payload.get("testnet"):
                    raise ValueError(f"Invalid production JSON-RPC response: {payload}")
                return payload
            except HTTPError as error:
                status = error.code
                error.close()
                if status != 429 and not 500 <= status < 600:
                    raise
                failure = f"HTTP {status}"
            except (URLError, TimeoutError, ConnectionError, HTTPException, json.JSONDecodeError) as error:
                failure = str(error)
            if attempt == 6:
                raise RuntimeError(f"Historical request failed: {failure}")
            wait = min(2 ** (attempt + 1), 30)
            LOGGER.warning("%s; retry in %s seconds", failure, wait)
            time.sleep(wait)
        raise AssertionError("Unreachable")


def normalize_trade(row: dict[str, Any], observed_legs: int, source_file: str) -> dict[str, Any]:
    currency, expiry_text, strike_text, option_type = row["instrument_name"].split("-")
    if currency != "BTC" or option_type != "C":
        raise ValueError("Only BTC inverse calls can be normalized")
    # Названия месяцев разбираем независимо от системной локали.
    month_text = expiry_text[-5:-2].upper()
    month = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC").index(month_text) + 1
    expiry = datetime(2000 + int(expiry_text[-2:]), month, int(expiry_text[:-5]), 8, tzinfo=timezone.utc)
    trade_time = datetime.fromtimestamp(row["timestamp"] / 1000, timezone.utc)
    strike = float(strike_text)
    if not math.isfinite(strike) or strike <= 0:
        raise ValueError("Invalid strike")
    reported = row.get("block_trade_leg_count")
    package_status = (
        "multiple_legs"
        if observed_legs > 1 or (reported is not None and reported > 1)
        else ("single_leg_reported" if reported == 1 else "unknown")
    )
    return dict(
        zip(
            TRADE_FIELDS,
            (
                row["trade_id"],
                row["block_trade_id"],
                row["trade_seq"],
                row["timestamp"],
                trade_time.isoformat(),
                row["instrument_name"],
                "BTC",
                "call",
                expiry.isoformat(),
                strike,
                row["price"],
                row["amount"],
                row.get("contracts"),
                row["direction"],
                row["index_price"],
                row.get("mark_price"),
                row.get("iv"),
                (expiry - trade_time).total_seconds() / 86400,
                reported,
                observed_legs,
                package_status,
                row.get("block_rfq_id"),
                row.get("combo_id"),
                row.get("combo_trade_id"),
                row.get("block_delayed"),
                source_file,
            ),
            strict=True,
        )
    )


def _read_annual_trades(source: Path) -> Iterator[dict[str, str]]:
    opener = gzip.open if source.suffix == ".gz" else Path.open
    with opener(source, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != TRADE_FIELDS:
            raise ValueError("OTC source must have the annual export's columns in the original order")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed annual OTC row")
            yield row


def _download_trades(start: date, end: date) -> list[dict[str, Any]]:
    client = HistoryClient()
    blocks = []
    day = start
    while day < end:
        low, high = date_to_timestamp_ms(day), date_to_timestamp_ms(day + timedelta(days=1)) - 1
        trades, _, requests = fetch_complete_range(client.fetch, low, high)
        blocks.extend(row for row in trades if row.get("block_trade_id") not in (None, ""))
        LOGGER.info("%s: %s trades scanned (%s requests)", day, len(trades), requests)
        day += timedelta(days=1)
    if len({str(row["trade_id"]) for row in blocks}) != len(blocks):
        raise ValueError("Duplicate trade IDs across daily time ranges")
    # Считаем call- и put-ноги до фильтрации, чтобы не принять пакет за отдельную сделку.
    observed = Counter(str(row["block_trade_id"]) for row in blocks)
    source = (
        f"{HISTORY_URL}?start_timestamp={date_to_timestamp_ms(start)}&end_timestamp={date_to_timestamp_ms(end) - 1}"
    )
    return [
        normalize_trade(row, observed[str(row["block_trade_id"])], source)
        for row in blocks
        if row["instrument_name"].endswith("-C")
    ]


def _read_quote_contracts(quotes: Path, start: date, end: date, expiry: datetime) -> set[str]:
    with quotes.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"instrument_name", "timestamp_utc", "expiry_utc"} <= set(reader.fieldnames or ()):
            raise ValueError("Quotes must contain instrument_name, timestamp_utc and expiry_utc")
        contracts = set()
        for row in reader:
            if not start <= parse_utc(row["timestamp_utc"]).date() < end or parse_utc(row["expiry_utc"]) != expiry:
                raise ValueError("Quotes have a different research period or expiry")
            contracts.add(row["instrument_name"])
    if not contracts or "" in contracts:
        raise ValueError("No quote contracts available")
    return contracts


def prepare_otc_trades(
    source: Path,
    quotes: Path,
    output: Path,
    start: date = START_DATE,
    end: date = END_DATE,
    expiry_day: date = EXPIRY_DATE,
    *,
    download: bool = False,
) -> dict[str, Any]:
    if not start < end <= expiry_day:
        raise ValueError("Need start < exclusive end <= expiry")
    if output.resolve() in {source.resolve(), quotes.resolve()}:
        raise ValueError("Output must not overwrite the annual source or quotes")
    expiry = datetime(expiry_day.year, expiry_day.month, expiry_day.day, 8, tzinfo=timezone.utc)
    quote_contracts = _read_quote_contracts(quotes, start, end, expiry)
    originals: Iterable[dict[str, Any]] = _download_trades(start, end) if download else _read_annual_trades(source)
    counts: Counter[str] = Counter()
    rows = []
    seen_trade_ids = set()
    for row in originals:
        counts["all_source_rows"] += 1
        if row["currency"] != "BTC" or row["option_type"] != "call":
            continue
        if not start <= parse_utc(row["trade_utc"]).date() < end:
            continue
        counts["btc_call_rows_in_window"] += 1
        if parse_utc(row["expiry_utc"]) != expiry:
            continue
        counts["same_expiry_rows"] += 1
        days_to_expiry = float(row["days_to_expiry"])
        if not MIN_DAYS_TO_EXPIRY <= days_to_expiry <= MAX_DAYS_TO_EXPIRY:
            counts["excluded_outside_horizon"] += 1
            continue
        if row["package_status"] != "single_leg_reported":
            counts["excluded_not_single_leg"] += 1
            continue
        if row["instrument_name"] not in quote_contracts:
            counts["excluded_contract_not_in_quote_universe"] += 1
            continue
        trade_id = str(row["trade_id"])
        if not trade_id or trade_id in seen_trade_ids:
            raise ValueError(f"Missing or duplicate OTC trade ID: {trade_id}")
        seen_trade_ids.add(trade_id)
        rows.append(row)
    rows.sort(key=lambda row: (int(row["timestamp_ms"]), str(row["trade_id"])))
    write_csv(output, TRADE_FIELDS, rows)
    return {
        "otc_rows": len(rows),
        "otc_contracts": len({row["instrument_name"] for row in rows}),
        "package_counts": dict(Counter(row["package_status"] for row in rows)),
        "selection_counts": dict(counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/otc_2024/btc_call_block_trades.csv.gz"))
    parser.add_argument("--quotes", type=Path, default=Path("data/quotes.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/otc_trades.csv"))
    parser.add_argument("--start", type=date.fromisoformat, default=START_DATE)
    parser.add_argument("--end", type=date.fromisoformat, default=END_DATE, help="Exclusive UTC end")
    parser.add_argument("--expiry", type=date.fromisoformat, default=EXPIRY_DATE)
    parser.add_argument(
        "--download", action="store_true", help="Fetch historical API instead of reading the annual CSV"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = prepare_otc_trades(
        args.source, args.quotes, args.output, args.start, args.end, args.expiry, download=args.download
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
