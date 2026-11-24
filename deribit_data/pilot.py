"""Пилотный расчёт: holdout, квантили IV, интервалы цены и block trades."""

import argparse
import csv
import json
import math
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from deribit_data.pilot_plots import generate_plots
from deribit_data.prepare_quotes import parse_utc, write_csv
from option_pricing.inverse_call import inverse_call_price
from option_pricing.snapshot import IVFilters, IVRow, OptionQuote, calculate_iv_rows

__all__ = ["main", "render_report", "run_pilot"]

HOLDOUT_FIELDS = ("instrument_name", "expiry_utc", "strike_usd", "holdout")
HOLDOUT_RESULT_FIELDS = (
    "instrument_name",
    "snapshot_utc",
    "expiry_utc",
    "strike_usd",
    "forward_usd",
    "days_to_expiry",
    "log_moneyness",
    "mid_btc",
    "q05",
    "q95",
    "lower_price",
    "upper_price",
    "covered",
    "width_btc",
    "width_relative",
    "peer_count",
    "q10",
    "q90",
    "lower_price_10",
    "upper_price_10",
    "covered_10",
)
BLOCK_RESULT_FIELDS = (
    "trade_id",
    "block_trade_id",
    "trade_utc",
    "snapshot_utc",
    "instrument_name",
    "strike_usd",
    "price_btc",
    "q05",
    "q95",
    "lower_price",
    "upper_price",
    "covered",
    "width_btc",
    "width_relative",
    "peer_count",
    "gap_seconds",
)


def _number(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def read_quotes(path: Path) -> list[OptionQuote]:
    quotes = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_utc", "instrument_name", "expiry_utc", "strike_usd", "forward_usd", "bid_btc", "ask_btc"}
        if not required <= set(reader.fieldnames or ()):
            raise ValueError("quotes.csv has an unexpected schema")
        for row in reader:
            try:
                quote = OptionQuote(
                    instrument_name=row["instrument_name"],
                    expiry_utc=parse_utc(row["expiry_utc"]),
                    snapshot_utc=parse_utc(row["timestamp_utc"]),
                    strike_usd=float(row["strike_usd"]),
                    forward_usd=float(row["forward_usd"]),
                    bid_btc=float(row["bid_btc"]),
                    ask_btc=float(row["ask_btc"]),
                )
            except (TypeError, ValueError):
                continue
            quotes.append(quote)
    if not quotes:
        raise ValueError("No usable quotes")
    return quotes


def assign_holdout(rows: Sequence[IVRow], fraction: float = 0.2) -> dict[str, bool]:
    """Mark a deterministic 20% of instruments inside every expiry."""
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must be in (0, 1)")
    by_expiry: dict[datetime, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_expiry[row.expiry_utc][row.instrument_name] = row.strike_usd
    holdout: dict[str, bool] = {}
    for expiry in sorted(by_expiry):
        instruments = sorted(by_expiry[expiry].items(), key=lambda item: (item[1], item[0]))
        count = max(1, round(len(instruments) * fraction))
        if count == 1:
            indices = {len(instruments) // 2}
        else:
            # Pick central positions instead of the two boundary strikes.
            indices = {
                min(len(instruments) - 1, int((index + 0.5) * len(instruments) / count))
                for index in range(count)
            }
        for index, (name, _) in enumerate(instruments):
            holdout[name] = index in indices
    return holdout


def _group_rows(rows: Sequence[IVRow]) -> dict[tuple[datetime, datetime], list[IVRow]]:
    groups: dict[tuple[datetime, datetime], list[IVRow]] = defaultdict(list)
    for row in rows:
        groups[(row.snapshot_utc, row.expiry_utc)].append(row)
    return groups


def evaluate_holdout(
    rows: Sequence[IVRow],
    holdout: dict[str, bool],
    min_peers: int = 8,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    results = []
    counts: Counter[str] = Counter()
    for (_, _), group in sorted(_group_rows(rows).items()):
        training = [row for row in group if not holdout.get(row.instrument_name, False)]
        targets = [row for row in group if holdout.get(row.instrument_name, False)]
        counts["groups"] += 1
        if not targets:
            continue
        if len(training) < min_peers:
            counts["groups_skipped_small_training"] += 1
            counts["targets_skipped_small_training"] += len(targets)
            continue
        q05 = _quantile([row.iv_brent for row in training], 0.05)
        q95 = _quantile([row.iv_brent for row in training], 0.95)
        q10 = _quantile([row.iv_brent for row in training], 0.10)
        q90 = _quantile([row.iv_brent for row in training], 0.90)
        if None in (q05, q95, q10, q90):
            continue
        for row in targets:
            time_to_expiry = row.days_to_expiry / 365.0
            lower = inverse_call_price(row.forward_usd, row.strike_usd, time_to_expiry, q05)
            upper = inverse_call_price(row.forward_usd, row.strike_usd, time_to_expiry, q95)
            lower10 = inverse_call_price(row.forward_usd, row.strike_usd, time_to_expiry, q10)
            upper10 = inverse_call_price(row.forward_usd, row.strike_usd, time_to_expiry, q90)
            width = upper - lower
            results.append(
                {
                    "instrument_name": row.instrument_name,
                    "snapshot_utc": row.snapshot_utc.isoformat(),
                    "expiry_utc": row.expiry_utc.isoformat(),
                    "strike_usd": row.strike_usd,
                    "forward_usd": row.forward_usd,
                    "days_to_expiry": row.days_to_expiry,
                    "log_moneyness": row.log_moneyness,
                    "mid_btc": row.mid_btc,
                    "q05": q05,
                    "q95": q95,
                    "lower_price": lower,
                    "upper_price": upper,
                    "covered": lower <= row.mid_btc <= upper,
                    "width_btc": width,
                    "width_relative": width / row.mid_btc if row.mid_btc else None,
                    "peer_count": len(training),
                    "q10": q10,
                    "q90": q90,
                    "lower_price_10": lower10,
                    "upper_price_10": upper10,
                    "covered_10": lower10 <= row.mid_btc <= upper10,
                }
            )
    return results, counts


def _summary(results: Sequence[dict[str, Any]], prefix: str = "holdout") -> dict[str, Any]:
    if not results:
        return {"n": 0, f"{prefix}_coverage": None, "mean_width_btc": None, "mean_width_relative": None}
    widths = [float(row["width_btc"]) for row in results]
    relatives = [float(row["width_relative"]) for row in results if row["width_relative"] is not None]
    return {
        "n": len(results),
        f"{prefix}_coverage": sum(bool(row["covered"]) for row in results) / len(results),
        "mean_width_btc": statistics.fmean(widths),
        "median_width_btc": statistics.median(widths),
        "mean_width_relative": statistics.fmean(relatives) if relatives else None,
        "coverage_10_90": sum(bool(row["covered_10"]) for row in results) / len(results)
        if "covered_10" in results[0]
        else None,
    }


def _by_bucket(results: Sequence[dict[str, Any]], key) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        groups[str(key(row))].append(row)
    return {group: _summary(rows) for group, rows in sorted(groups.items())}


def _moneyness_bucket(value: float) -> str:
    absolute = abs(value)
    if absolute <= 0.05:
        return "0-5%"
    if absolute <= 0.10:
        return "5-10%"
    return "10-15%"


def read_otc(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return [row for row in reader if row.get("instrument_name")]


def evaluate_block_trades(
    trades: Sequence[dict[str, str]],
    rows: Sequence[IVRow],
    holdout: dict[str, bool],
    min_peers: int = 8,
    max_gap_seconds: float = 300.0,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    results = []
    counts: Counter[str] = Counter()
    groups = _group_rows(rows)
    snapshot_times = sorted({row.snapshot_utc for row in rows})
    if not snapshot_times:
        return results, counts

    for trade in trades:
        counts["trades"] += 1
        try:
            trade_time = parse_utc(trade["trade_utc"])
            expiry = parse_utc(trade["expiry_utc"])
            strike = float(trade["strike_usd"])
            price = float(trade["price_btc"])
        except (KeyError, TypeError, ValueError):
            counts["skipped_invalid"] += 1
            continue
        index = bisect_right(snapshot_times, trade_time) - 1
        if index < 0:
            counts["skipped_no_previous_snapshot"] += 1
            continue
        snapshot = snapshot_times[index]
        gap = (trade_time - snapshot).total_seconds()
        if gap < 0 or gap > max_gap_seconds:
            counts["skipped_stale_snapshot"] += 1
            continue
        group = groups.get((snapshot, expiry))
        if not group:
            counts["skipped_no_matching_expiry"] += 1
            continue
        training = [
            row
            for row in group
            if not holdout.get(row.instrument_name, False) and row.instrument_name != trade["instrument_name"]
        ]
        if len(training) < min_peers:
            counts["skipped_small_training"] += 1
            continue
        q05 = _quantile([row.iv_brent for row in training], 0.05)
        q95 = _quantile([row.iv_brent for row in training], 0.95)
        if q05 is None or q95 is None:
            counts["skipped_no_quantiles"] += 1
            continue
        time_to_expiry = (expiry - trade_time).total_seconds() / (365 * 86400)
        if time_to_expiry <= 0:
            counts["skipped_expired"] += 1
            continue
        forward = training[0].forward_usd
        lower = inverse_call_price(forward, strike, time_to_expiry, q05)
        upper = inverse_call_price(forward, strike, time_to_expiry, q95)
        width = upper - lower
        results.append(
            {
                "trade_id": trade.get("trade_id"),
                "block_trade_id": trade.get("block_trade_id"),
                "trade_utc": trade_time.isoformat(),
                "snapshot_utc": snapshot.isoformat(),
                "instrument_name": trade["instrument_name"],
                "strike_usd": strike,
                "price_btc": price,
                "q05": q05,
                "q95": q95,
                "lower_price": lower,
                "upper_price": upper,
                "covered": lower <= price <= upper,
                "width_btc": width,
                "width_relative": width / price if price else None,
                "peer_count": len(training),
                "gap_seconds": gap,
            }
        )
    return results, counts


def render_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# Пилот: IV-интервалы для BTC call",
        "",
        f"- Снимков: {metrics['quote_snapshots']}",
        f"- Строк котировок: {metrics['quote_rows']}",
        f"- Успешных IV: {metrics['iv_rows']}",
        f"- Holdout-контрактов: {metrics['holdout_instruments']}",
        f"- Проверок holdout: {metrics['holdout']['n']}",
        f"- Coverage holdout: {metrics['holdout']['holdout_coverage']}",
        f"- Средняя ширина holdout: {metrics['holdout']['mean_width_btc']} BTC",
        f"- Средняя относительная ширина: {metrics['holdout']['mean_width_relative']}",
        "",
        "## Block trades",
        "",
        f"- Сделок в файле: {metrics['block_trade_counts'].get('trades', 0)}",
        f"- Проверок block trades: {metrics['block']['n']}",
        f"- Coverage block trades: {metrics['block'].get('block_coverage')}",
        f"- Средняя ширина block: {metrics['block'].get('mean_width_btc')} BTC",
        "",
    ]
    plots = metrics.get("plots", [])
    if plots:
        lines.extend(["## Графики", ""])
        titles = {
            "holdout_intervals.png": "Рыночная цена и предсказанный интервал",
            "holdout_coverage_by_contract.png": "Покрытие по контрактам",
            "holdout_width_relative.png": "Относительная ширина интервала",
        }
        for plot in plots:
            name = Path(plot).name
            lines.extend([f"### {titles.get(name, name)}", "", f"![{name}]({plot})", ""])
    return "\n".join(lines)


def run_pilot(
    quotes_path: Path,
    otc_path: Path,
    output_dir: Path,
    holdout_fraction: float = 0.2,
    min_peers: int = 8,
    max_gap_seconds: float = 300.0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quotes = read_quotes(quotes_path)
    iv_rows, iv_counts = calculate_iv_rows(quotes, IVFilters(max_log_moneyness=0.15, min_days_to_expiry=1.0))
    if not iv_rows:
        raise ValueError("No IV rows after filters")

    holdout = assign_holdout(iv_rows, holdout_fraction)
    holdout_results, holdout_counts = evaluate_holdout(iv_rows, holdout, min_peers)
    block_results, block_counts = evaluate_block_trades(
        read_otc(otc_path), iv_rows, holdout, min_peers, max_gap_seconds
    )

    assignments = []
    seen = set()
    for row in iv_rows:
        if row.instrument_name in seen:
            continue
        seen.add(row.instrument_name)
        assignments.append(
            {
                "instrument_name": row.instrument_name,
                "expiry_utc": row.expiry_utc.isoformat(),
                "strike_usd": row.strike_usd,
                "holdout": holdout.get(row.instrument_name, False),
            }
        )
    write_csv(output_dir / "holdout_assignments.csv", HOLDOUT_FIELDS, assignments)
    write_csv(output_dir / "holdout_intervals.csv", HOLDOUT_RESULT_FIELDS, holdout_results)
    write_csv(output_dir / "block_intervals.csv", BLOCK_RESULT_FIELDS, block_results)
    plot_paths = generate_plots(output_dir / "holdout_intervals.csv", output_dir / "plots")

    metrics = {
        "quote_snapshots": len({row.snapshot_utc for row in iv_rows}),
        "quote_rows": len(iv_rows),
        "iv_rows": len(iv_rows),
        "iv_counts": iv_counts,
        "holdout_instruments": sum(1 for value in holdout.values() if value),
        "holdout_counts": dict(holdout_counts),
        "holdout": _summary(holdout_results, "holdout"),
        "holdout_by_expiry": _by_bucket(holdout_results, lambda row: row["expiry_utc"][:10]),
        "holdout_by_moneyness": _by_bucket(holdout_results, lambda row: _moneyness_bucket(float(row["log_moneyness"]))),
        "holdout_by_snapshot": _by_bucket(holdout_results, lambda row: row["snapshot_utc"][:10]),
        "block_trade_counts": dict(block_counts),
        "block": _summary(block_results, "block"),
        "plots": [f"plots/{path.name}" for path in plot_paths],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, default=Path("data/quotes.csv"))
    parser.add_argument("--otc", type=Path, default=Path("data/otc_trades.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot"))
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--min-peers", type=int, default=8)
    parser.add_argument("--max-gap-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    metrics = run_pilot(
        args.quotes,
        args.otc,
        args.output_dir,
        args.holdout_fraction,
        args.min_peers,
        args.max_gap_seconds,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
