"""Command-line entry point: python3 -m deribit_data."""

import argparse
import json
import logging
import math
from datetime import date
from pathlib import Path

from deribit_data.client import DeribitClient, DeribitError
from deribit_data.snapshot import Filters, collect_snapshot

__all__ = ["main"]


def _positive(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Укажите положительное конечное число")
    return number


def main() -> int:
    """Collect one snapshot and return a nonzero exit code on failure/empty data."""
    defaults = Filters()
    parser = argparse.ArgumentParser(description="Снимок BTC inverse call Deribit для оценки IV")
    parser.add_argument("--expiry", type=date.fromisoformat, default=defaults.expiry, help="Дата UTC, YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("data"), help="Каталог снимков (по умолчанию data)")
    parser.add_argument(
        "--max-log-moneyness",
        type=_positive,
        default=defaults.max_log_moneyness,
        help=f"Максимальное |ln(K/F)|, по умолчанию {defaults.max_log_moneyness}",
    )
    parser.add_argument(
        "--max-relative-spread",
        type=_positive,
        default=defaults.max_relative_spread,
        help=f"Максимальное (ask-bid)/mid, по умолчанию {defaults.max_relative_spread}",
    )
    parser.add_argument(
        "--max-summary-age-seconds",
        type=_positive,
        default=defaults.max_summary_age_seconds,
        help="Допустимый возраст summary, секунды",
    )
    parser.add_argument(
        "--max-snapshot-span-seconds",
        type=_positive,
        default=defaults.max_snapshot_span_seconds,
        help="Допустимый разброс времён summary, секунды",
    )
    parser.add_argument("--timeout", type=_positive, default=15.0, help="HTTP-таймаут одной попытки, секунды")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    filters = Filters(
        expiry=args.expiry,
        max_log_moneyness=args.max_log_moneyness,
        max_relative_spread=args.max_relative_spread,
        max_summary_age_seconds=args.max_summary_age_seconds,
        max_snapshot_span_seconds=args.max_snapshot_span_seconds,
    )
    try:
        run_dir = collect_snapshot(DeribitClient(timeout=args.timeout), args.output_dir, filters)
    except (DeribitError, ValueError, KeyError, OSError) as error:
        logging.error("Снимок не собран: %s", error)
        return 1
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    print(f"Сохранено в {run_dir.resolve()}")
    print(f"Call-контрактов: {metadata['total_calls']}; прошло фильтры ATM: {metadata['selected_calls']}")
    for warning in metadata["warnings"]:
        logging.warning(warning)
    if metadata["status"] == "empty":
        logging.error("Подходящих наблюдений нет. Причины исключения: %s", metadata["exclusion_counts"])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
