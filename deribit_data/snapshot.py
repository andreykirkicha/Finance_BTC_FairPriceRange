"""Join contract specifications and one bulk market-data response."""

import csv
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from deribit_data.client import API_URL, DeribitClient

__all__ = ["Filters", "build_rows", "collect_snapshot"]

DAY_MS = 86_400_000
YEAR_MS = 365 * DAY_MS


@dataclass(frozen=True)
class Filters:
    """Initial research filters; these thresholds require empirical validation."""

    expiry: date = date(2026, 12, 25)
    max_log_moneyness: float = 0.15
    max_relative_spread: float = 0.20
    max_summary_age_seconds: float = 10.0
    max_snapshot_span_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name != "expiry" and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(value) else None


def _utc(timestamp_ms: float | None) -> str | None:
    if timestamp_ms is None:
        return None
    return datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()


def _index(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {}
    for record in records:
        name = record["instrument_name"]
        if name in indexed:
            raise ValueError(f"Duplicate instrument in response: {name}")
        indexed[name] = record
    return indexed


def build_rows(
    instruments: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    reference_timestamp_ms: float,
    filters: Filters,
) -> list[dict[str, Any]]:
    """Produce one row per BTC inverse call for the requested UTC expiry date.

    Keep rejected rows with explicit reasons. Never substitute mark/last prices
    for missing quotes or exchange IV for the IV to be estimated later.
    """
    books = _index(summaries)
    rows = []
    for name, instrument in _index(instruments).items():
        if not (
            instrument.get("kind") == "option"
            and instrument.get("option_type") == "call"
            and instrument.get("base_currency") == "BTC"
            and instrument.get("settlement_currency") == "BTC"
            and instrument.get("instrument_type") == "reversed"
        ):
            continue
        expiry_ms = _number(instrument.get("expiration_timestamp"))
        if expiry_ms is None or datetime.fromtimestamp(expiry_ms / 1000, timezone.utc).date() != filters.expiry:
            continue
        book = books.get(name, {})
        reasons = []
        if not instrument.get("is_active") or instrument.get("state", "open") != "open":
            reasons.append("inactive_contract")
        if not book:
            reasons.append("missing_summary")

        timestamp_ms = _number(book.get("creation_timestamp"))
        if timestamp_ms is not None and timestamp_ms <= 0:
            timestamp_ms = None
        age = (reference_timestamp_ms - timestamp_ms) / 1000 if timestamp_ms is not None else None
        if age is None:
            reasons.append("missing_summary_timestamp")
        elif age < -1:
            reasons.append("future_summary_timestamp")
        elif age > filters.max_summary_age_seconds:
            reasons.append("stale_summary")

        # Each row uses its own exchange timestamp, not local download time.
        remaining_ms = expiry_ms - timestamp_ms if timestamp_ms is not None else None
        if remaining_ms is not None and remaining_ms <= 0:
            reasons.append("expired_contract")
        forward = _number(book.get("underlying_price"))
        strike = _number(instrument.get("strike"))
        moneyness = None
        if forward is None or forward <= 0:
            reasons.append("invalid_underlying_price")
        if strike is None or strike <= 0:
            reasons.append("invalid_strike")
        if forward is not None and forward > 0 and strike is not None and strike > 0:
            moneyness = math.log(strike / forward)
            if abs(moneyness) > filters.max_log_moneyness:
                reasons.append("outside_atm_band")

        bid = _number(book.get("bid_price"))
        ask = _number(book.get("ask_price"))
        mid = spread = relative_spread = None
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            reasons.append("missing_or_nonpositive_quote")
        elif bid > ask:
            reasons.append("crossed_quotes")
        else:
            mid = (bid + ask) / 2
            spread = ask - bid
            relative_spread = spread / mid
            if relative_spread > filters.max_relative_spread:
                reasons.append("wide_spread")
            if moneyness is not None and not max(1 - strike / forward, 0) < mid < 1:
                # Price range of the inverse-call formula in the project PDF.
                reasons.append("outside_project_model_bounds")

        rows.append(
            {
                "instrument_name": name,
                "instrument_id": instrument.get("instrument_id"),
                "option_type": "call",
                "instrument_type": "reversed",
                "base_currency": "BTC",
                "settlement_currency": "BTC",
                "premium_currency": "BTC",
                "contract_size": instrument.get("contract_size"),
                "tick_size": instrument.get("tick_size"),
                "strike_usd": strike,
                "expiration_timestamp_ms": expiry_ms,
                "expiration_utc": _utc(expiry_ms),
                "summary_timestamp_ms": timestamp_ms,
                "summary_utc": _utc(timestamp_ms),
                "summary_age_seconds": age,
                "days_to_expiry": remaining_ms / DAY_MS if remaining_ms is not None else None,
                "time_to_expiry_years": remaining_ms / YEAR_MS if remaining_ms is not None else None,
                "underlying_price_usd": forward,
                "underlying_index": book.get("underlying_index"),
                "interest_rate_exchange": book.get("interest_rate"),
                "log_moneyness": moneyness,
                "bid_btc": bid,
                "ask_btc": ask,
                "mid_btc": mid,
                "spread_btc": spread,
                "relative_spread": relative_spread,
                "last_btc": book.get("last"),
                "mark_btc": book.get("mark_price"),
                "mark_iv_exchange_pct": book.get("mark_iv"),
                "volume_24h_btc": book.get("volume"),
                "volume_realtime_24h_btc": book.get("volume_realtime"),
                "volume_delayed_24h_btc": book.get("volume_delayed"),
                "open_interest_btc": book.get("open_interest"),
                "selected_for_iv": not reasons,
                "exclusion_reasons": "|".join(reasons),
            }
        )

    selected_times = [row["summary_timestamp_ms"] for row in rows if row["selected_for_iv"]]
    if selected_times and max(selected_times) - min(selected_times) > 1000 * filters.max_snapshot_span_seconds:
        for row in rows:
            if row["selected_for_iv"]:
                row["selected_for_iv"] = False
                row["exclusion_reasons"] = "snapshot_time_span_exceeded"
    return sorted(rows, key=lambda row: (row["strike_usd"] or 0, row["instrument_name"]))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _records(capture: dict[str, Any]) -> list[dict[str, Any]]:
    result = capture["response"]["result"]
    if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
        raise ValueError("API result must be a list of instrument objects")
    return result


def collect_snapshot(client: DeribitClient, output_dir: Path, filters: Filters) -> Path:
    """Save raw API responses, all calls, filtered ATM calls, and provenance.

    Each invocation creates a new directory. A failed run retains its raw data
    and records status=failed; it must not be treated as a completed dataset.
    """
    started = datetime.now(timezone.utc)
    run_dir = output_dir / started.strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "status": "collecting",
        "source": API_URL,
        "started_at_utc": started.isoformat(),
        "filters": {**asdict(filters), "expiry": filters.expiry.isoformat()},
        "time_convention": "ACT/365F; UTC; row summary timestamp",
        "premium_currency": "BTC",
    }
    metadata_path = run_dir / "metadata.json"
    _write_json(metadata_path, metadata)
    try:
        specs = client.get("public/get_instruments", currency="BTC", kind="option", expired="false")
        _write_json(run_dir / "instruments_raw.json", specs)
        markets = client.get("public/get_book_summary_by_currency", currency="BTC", kind="option")
        _write_json(run_dir / "summaries_raw.json", markets)
        server_us = _number(markets["response"].get("usOut"))
        if server_us is not None and server_us > 0:
            reference_ms = server_us / 1000
            metadata["reference_time_source"] = "response.usOut"
        else:
            reference_ms = datetime.fromisoformat(markets["received_at_utc"]).timestamp() * 1000
            metadata["reference_time_source"] = "local_received_at_utc"
        metadata["reference_utc"] = _utc(reference_ms)
        instruments = _records(specs)
        rows = build_rows(instruments, _records(markets), reference_ms, filters)
        if not rows:
            available = sorted(
                {
                    _utc(item["expiration_timestamp"])[:10]
                    for item in instruments
                    if item.get("kind") == "option" and item.get("is_active")
                }
            )
            raise ValueError(f"No BTC inverse calls for {filters.expiry}. Available expiries: {', '.join(available)}")
        selected = [row for row in rows if row["selected_for_iv"]]
        for filename, values in (("all_calls.csv", rows), ("atm_calls.csv", selected)):
            with (run_dir / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(values)
        reasons = Counter(reason for row in rows for reason in row["exclusion_reasons"].split("|") if reason)
        warnings = []
        if len(selected) < 20:
            warnings.append(
                "Fewer than 20 usable ATM contracts: tail quantiles may be unstable, especially after holdout."
            )
        if metadata["reference_time_source"] == "local_received_at_utc":
            warnings.append("No usOut timestamp: freshness checks depend on the local UTC clock.")
        metadata.update(
            status="ok" if selected else "empty",
            total_calls=len(rows),
            selected_calls=len(selected),
            exclusion_counts=dict(reasons),
            warnings=warnings,
        )
    except Exception as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(metadata_path, metadata)
    return run_dir
