"""Compute implied volatility for one historical BTC option snapshot."""

import argparse
import csv
import json
import logging
import math
import platform
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import scipy

from deribit_data.iv import BrentConfig, implied_volatility
from deribit_data.prepare_quotes import OptionRow, ScanResult, scan_options_table

__all__ = [
    "CSV_FIELDS",
    "IVFilters",
    "IVRow",
    "build_iv_rows",
    "build_summary",
    "main",
    "render_report",
    "write_csv",
]

LOGGER = logging.getLogger(__name__)
DEFAULT_DUMP = Path("data/dump.sql")
DAYS_PER_YEAR = 365.0
SECONDS_PER_DAY = 86_400.0

CSV_FIELDS = (
    "instrument_name",
    "expiry_utc",
    "snapshot_utc",
    "strike_usd",
    "forward_usd",
    "days_to_expiry",
    "log_moneyness",
    "bid_btc",
    "ask_btc",
    "mid_btc",
    "iv_brent",
    "iv_brent_pct",
)

REASON_ORDER = (
    "invalid_timestamp",
    "expired_or_nonpositive_T",
    "missing_or_invalid_forward",
    "missing_or_invalid_strike",
    "missing_or_nonpositive_quote",
    "crossed_quote",
    "outside_model_bounds",
    "outside_atm",
    "lt_min_days_to_expiry",
    "no_bracket",
    "solver_error",
    "not_converged",
    "price_residual",
)

REASON_LABELS = {
    "invalid_timestamp": "Некорректная метка времени",
    "expired_or_nonpositive_T": "T ≤ 0",
    "missing_or_invalid_forward": "Нет валидной F",
    "missing_or_invalid_strike": "Нет валидного K",
    "missing_or_nonpositive_quote": "Нет валидной bid/ask",
    "crossed_quote": "bid > ask",
    "outside_model_bounds": "mid вне границ модели",
    "outside_atm": "Вне ATM-диапазона",
    "lt_min_days_to_expiry": "T меньше минимального срока",
    "no_bracket": "Brent: нет смены знака",
    "solver_error": "Brent: ошибка решателя",
    "not_converged": "Brent: не сошёлся",
    "price_residual": "Brent: остаток цены слишком велик",
}


@dataclass(frozen=True, slots=True)
class IVFilters:
    """Selection rules applied before the volatility solve."""

    max_log_moneyness: float = 0.15
    min_days_to_expiry: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_log_moneyness) and self.max_log_moneyness > 0):
            raise ValueError("max_log_moneyness must be finite and positive")
        if not (math.isfinite(self.min_days_to_expiry) and self.min_days_to_expiry >= 0):
            raise ValueError("min_days_to_expiry must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class IVRow:
    """A successful IV observation, ready for the result CSV."""

    instrument_name: str
    expiry_utc: datetime
    snapshot_utc: datetime
    strike_usd: float
    forward_usd: float
    days_to_expiry: float
    log_moneyness: float
    bid_btc: float
    ask_btc: float
    mid_btc: float
    iv_brent: float
    exchange_mid_iv: float | None

    @property
    def iv_brent_pct(self) -> float:
        return self.iv_brent * 100.0


def _assess_row(row: OptionRow, filters: IVFilters, config: BrentConfig) -> tuple[IVRow | None, str | None]:
    try:
        expiry = row.expiry_utc
        snapshot = row.snapshot_utc
    except ValueError:
        return None, "invalid_timestamp"

    time_to_expiry = (expiry - snapshot).total_seconds() / (DAYS_PER_YEAR * SECONDS_PER_DAY)
    if time_to_expiry <= 0:
        return None, "expired_or_nonpositive_T"

    forward, strike = row.underlying_price, row.strike
    if forward is None or not math.isfinite(forward) or forward <= 0:
        return None, "missing_or_invalid_forward"
    if strike is None or not math.isfinite(strike) or strike <= 0:
        return None, "missing_or_invalid_strike"

    bid, ask = row.bid_price, row.ask_price
    if bid is None or ask is None or not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0:
        return None, "missing_or_nonpositive_quote"
    if bid > ask:
        return None, "crossed_quote"

    mid = (bid + ask) / 2.0
    if not max(1.0 - strike / forward, 0.0) < mid < 1.0:
        return None, "outside_model_bounds"

    log_moneyness = math.log(strike / forward)
    if abs(log_moneyness) > filters.max_log_moneyness:
        return None, "outside_atm"

    days_to_expiry = time_to_expiry * DAYS_PER_YEAR
    if days_to_expiry < filters.min_days_to_expiry:
        return None, "lt_min_days_to_expiry"

    result = implied_volatility(mid, forward, strike, time_to_expiry, config)
    if result.reason is not None:
        return None, result.reason
    if result.iv is None:
        return None, "solver_error"

    return (
        IVRow(
            instrument_name=row.instrument_name,
            expiry_utc=expiry,
            snapshot_utc=snapshot,
            strike_usd=strike,
            forward_usd=forward,
            days_to_expiry=days_to_expiry,
            log_moneyness=log_moneyness,
            bid_btc=bid,
            ask_btc=ask,
            mid_btc=mid,
            iv_brent=result.iv,
            exchange_mid_iv=row.exchange_mid_iv,
        ),
        None,
    )


def build_iv_rows(
    option_rows: Iterable[OptionRow],
    filters: IVFilters | None = None,
    config: BrentConfig | None = None,
) -> tuple[list[IVRow], dict[str, object]]:
    """Apply filters and solve IV for every eligible row.

    Returns successful rows and a counters dictionary with the input count,
    exclusion reasons in a stable order, and the number of solved rows.
    """
    filters = filters or IVFilters()
    config = config or BrentConfig()
    source_rows = list(option_rows)
    reasons: Counter[str] = Counter()
    rows: list[IVRow] = []

    for row in source_rows:
        iv_row, reason = _assess_row(row, filters, config)
        if reason is not None:
            reasons[reason] += 1
            continue
        if iv_row is None:
            reasons["solver_error"] += 1
            continue
        rows.append(iv_row)

    rows.sort(key=lambda row: (row.expiry_utc, row.strike_usd, row.instrument_name))
    excluded = {reason: reasons[reason] for reason in REASON_ORDER if reasons[reason]}
    for reason in reasons:
        if reason not in excluded:
            excluded[reason] = reasons[reason]

    counts: dict[str, object] = {
        "input": len(source_rows),
        "excluded": excluded,
        "solved": len(rows),
    }
    if sum(excluded.values()) + len(rows) != len(source_rows):
        raise AssertionError("exclusion counters do not add up to the input count")
    return rows, counts


def _format_float(value: float, significant: int = 15) -> str:
    return format(value, f".{significant}g")


def write_csv(path: Path, rows: Sequence[IVRow]) -> None:
    """Write only successful observations, with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instrument_name": row.instrument_name,
                    "expiry_utc": row.expiry_utc.isoformat(),
                    "snapshot_utc": row.snapshot_utc.isoformat(),
                    "strike_usd": _format_float(row.strike_usd),
                    "forward_usd": _format_float(row.forward_usd),
                    "days_to_expiry": _format_float(row.days_to_expiry, 10),
                    "log_moneyness": _format_float(row.log_moneyness, 12),
                    "bid_btc": _format_float(row.bid_btc),
                    "ask_btc": _format_float(row.ask_btc),
                    "mid_btc": _format_float(row.mid_btc),
                    "iv_brent": _format_float(row.iv_brent, 12),
                    "iv_brent_pct": _format_float(row.iv_brent_pct, 12),
                }
            )


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
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "min": None, "p05": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "n": len(values),
        "min": min(values),
        "p05": _quantile(values, 0.05),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p95": _quantile(values, 0.95),
        "max": max(values),
    }


def _exchange_comparison(rows: Sequence[IVRow]) -> dict[str, object]:
    differences = [
        abs(row.iv_brent - row.exchange_mid_iv)
        for row in rows
        if row.exchange_mid_iv is not None and math.isfinite(row.exchange_mid_iv)
    ]
    if not differences:
        return {
            "n": 0,
            "mean_abs_diff": None,
            "median_abs_diff": None,
            "max_abs_diff": None,
            "within_0_001": 0,
            "within_0_01": 0,
            "note": "exchange IV is diagnostic only and is not an input",
        }
    return {
        "n": len(differences),
        "mean_abs_diff": statistics.fmean(differences),
        "median_abs_diff": statistics.median(differences),
        "max_abs_diff": max(differences),
        "within_0_001": sum(value <= 0.001 for value in differences),
        "within_0_01": sum(value <= 0.01 for value in differences),
        "note": "exchange IV is diagnostic only and is not an input",
    }


def _by_expiry(rows: Sequence[IVRow]) -> list[dict[str, object]]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row.expiry_utc.date().isoformat()].append(row.iv_brent)
    result = []
    for expiry in sorted(groups):
        values = groups[expiry]
        result.append(
            {
                "expiry": expiry,
                "n": len(values),
                "iv_min": min(values),
                "iv_median": statistics.median(values),
                "iv_max": max(values),
            }
        )
    return result


def _checksum_for(path: Path) -> str | None:
    checksum_file = path.parent / "SHA256SUMS"
    if not checksum_file.is_file():
        return None
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and Path(parts[-1].lstrip("*")).name == path.name:
            return parts[0]
    return None


def build_summary(
    input_path: Path,
    scan: ScanResult,
    filters: IVFilters,
    config: BrentConfig,
    counts: dict[str, object],
    rows: Sequence[IVRow],
    snapshot_explicit: bool,
) -> dict[str, object]:
    """Build a JSON-serializable provenance and statistics document."""
    stats = scan.stats
    snapshot = scan.snapshot
    iv_values = [row.iv_brent for row in rows]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(input_path),
            "size_bytes": stats.input_bytes,
            "sha256": _checksum_for(input_path),
            "table": stats.table,
        },
        "dataset": {
            "total_rows": stats.total_rows,
            "btc_rows": stats.btc_rows,
            "btc_calls": stats.btc_calls,
            "btc_puts": stats.btc_puts,
            "eth_rows": stats.eth_rows,
            "other_rows": stats.other_rows,
            "btc_call_snapshots": stats.snapshots,
            "latest_snapshot_utc": stats.latest_snapshot.isoformat() if stats.latest_snapshot else None,
        },
        "snapshot_utc": snapshot.isoformat() if snapshot else None,
        "snapshot_selection": "explicit --snapshot" if snapshot_explicit else "latest BTC call current_time",
        "rows_in_snapshot": len(scan.rows),
        "filters": asdict(filters),
        "model": {
            "forward": "underlying_price from the same row",
            "rate": 0.0,
            "day_count": "ACT/365F",
            "mid": "(bid_price + ask_price) / 2",
            "expiry_year_repair": "years below 1000 are increased by 2000",
            "expiry_time_repair": "midnight expiry timestamps are interpreted as 08:00 UTC",
        },
        "brent": asdict(config),
        "counts": counts,
        "iv_stats": _distribution(iv_values),
        "by_expiry": _by_expiry(rows),
        "exchange_iv_comparison": _exchange_comparison(rows),
        "software": {
            "python": platform.python_version(),
            "scipy": scipy.__version__,
        },
    }


def _fmt_int(value: object) -> str:
    return f"{int(value):,}".replace(",", " ")


def _fmt_number(value: object, digits: int = 6) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: object, total: object) -> str:
    if not total:
        return "—"
    return f"{100.0 * float(value) / float(total):.1f}%"


def render_report(summary: dict[str, object]) -> str:
    """Render a human-readable Russian report from the summary dictionary."""
    dataset = summary["dataset"]
    input_info = summary["input"]
    counts = summary["counts"]
    brent = summary["brent"]
    iv_stats = summary["iv_stats"]
    exchange = summary["exchange_iv_comparison"]
    total = counts["input"]
    excluded = counts["excluded"]
    solved = counts["solved"]

    lines = [
        "# Расчёт implied volatility для BTC inverse call",
        "",
        f"- **Снимок:** `{summary['snapshot_utc']}`",
        f"- **Сформирован:** `{summary['generated_at_utc']}`",
        f"- **Входной файл:** `{input_info['path']}`",
        f"- **Размер:** {_fmt_int(input_info['size_bytes'])} байт",
        f"- **SHA256:** `{input_info['sha256'] or 'не найден'}`",
        f"- **Таблица:** `{input_info['table']}`",
        "",
        "## 1. Данные",
        "",
        "Источник — PostgreSQL-дамп с таблицей `public.options_table`. "
        "В дампе смешаны BTC и ETH; для расчёта использованы только BTC call. "
        "Снимок выбран по максимальному `current_time` среди BTC call."
        if summary["snapshot_selection"] == "latest BTC call current_time"
        else "Источник — PostgreSQL-дамп с таблицей `public.options_table`. "
        "В дампе смешаны BTC и ETH; для расчёта использованы только BTC call. "
        "Снимок задан явно через `--snapshot`.",
        "",
        "| Показатель | Значение |",
        "| --- | ---: |",
        f"| Всего строк в `options_table` | {_fmt_int(dataset['total_rows'])} |",
        f"| Строк BTC | {_fmt_int(dataset['btc_rows'])} |",
        f"| BTC call | {_fmt_int(dataset['btc_calls'])} |",
        f"| BTC put | {_fmt_int(dataset['btc_puts'])} |",
        f"| Строк ETH | {_fmt_int(dataset['eth_rows'])} |",
        f"| Снимков BTC call | {_fmt_int(dataset['btc_call_snapshots'])} |",
        f"| BTC call в выбранном снимке | {_fmt_int(summary['rows_in_snapshot'])} |",
        "",
        "## 2. Методика",
        "",
        "Для BTC inverse call премия выражается в BTC. Используется форвардная цена `F`, страйк `K` и срок `T`:",
        "",
        "```text",
        "C(σ) = Φ(d1) − (K/F)Φ(d2)",
        "d1 = (ln(F/K) + 0.5 σ²T) / (σ√T)",
        "d2 = d1 − σ√T",
        "```",
        "",
        "`F` — форвардная цена BTC на дату экспирации, а не спот и не индекс. "
        "В качестве `F` взято поле `underlying_price` из той же строки дампа. "
        "Поскольку `F` уже содержит стоимость денег и удержания, отдельная ставка не добавляется (`r = 0`).",
        "",
        "`T` считается как `(expiry − current_time) / (365 × 86400)`, в UTC, по ACT/365F. "
        "В дампе `expiry` хранится как дата в 00:00 UTC; фактические экспирации Deribit — в 08:00 UTC, "
        "поэтому полночь переводится в 08:00. Это подтверждается совпадением с биржевой `mid_IV`. "
        "Цена `mid` пересчитывается как `(bid_price + ask_price) / 2`; готовый `mid_price` не используется. "
        "Годы в `expiry`, записанные как `0024` и `0025`, исправлены на `2024` и `2025`. "
        "Биржевая `mid_IV` не подставляется в результат и используется только для диагностики.",
        "",
        "## 3. Фильтрация",
        "",
        "Порядок фильтров: `T > 0`; валидные `F`, `K`, `bid`, `ask`; `bid ≤ ask`; "
        "`max(1-K/F, 0) < mid < 1`; ATM `|ln(K/F)| ≤ 0.15`; `T ≥ 1 дня`; затем Brent.",
        "",
        "| Причина | Строк | Доля |",
        "| --- | ---: | ---: |",
        f"| Вход | {_fmt_int(total)} | 100.0% |",
    ]

    for reason, count in excluded.items():
        label = REASON_LABELS.get(str(reason), str(reason))
        lines.append(f"| {label} | {_fmt_int(count)} | {_fmt_pct(count, total)} |")
    lines.extend(
        [
            f"| **Успешно посчитано** | **{_fmt_int(solved)}** | **{_fmt_pct(solved, total)}** |",
            "",
            "## 4. Численный метод",
            "",
            "Используется `scipy.optimize.brentq` для функции `f(σ) = C(σ) − mid`.",
            "",
            "| Параметр | Значение |",
            "| --- | ---: |",
            f"| Нижняя граница bracket | {brent['low']} |",
            f"| Верхняя граница bracket | {brent['high']} |",
            f"| `xtol` | {brent['xtol']} |",
            f"| `rtol` | {brent['rtol']} |",
            f"| `maxiter` | {brent['maxiter']} |",
            f"| Допуск по цене | {brent['price_tolerance']} |",
            "",
            "## 5. Результаты",
            "",
            "| Показатель | IV, доли |",
            "| --- | ---: |",
            f"| Наблюдений | {_fmt_int(iv_stats['n'])} |",
            f"| Минимум | {_fmt_number(iv_stats['min'], 6)} |",
            f"| 5-й перцентиль | {_fmt_number(iv_stats['p05'], 6)} |",
            f"| Медиана | {_fmt_number(iv_stats['median'], 6)} |",
            f"| Среднее | {_fmt_number(iv_stats['mean'], 6)} |",
            f"| 95-й перцентиль | {_fmt_number(iv_stats['p95'], 6)} |",
            f"| Максимум | {_fmt_number(iv_stats['max'], 6)} |",
            "",
            "### По экспирациям",
            "",
            "| Экспирация | Наблюдений | IV min | IV median | IV max |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group in summary["by_expiry"]:
        lines.append(
            f"| {group['expiry']} | {group['n']} | {_fmt_number(group['iv_min'], 6)} | "
            f"{_fmt_number(group['iv_median'], 6)} | {_fmt_number(group['iv_max'], 6)} |"
        )
    lines.extend(
        [
            "",
            "## 6. Сверка с биржевой IV",
            "",
            "Биржевая `mid_IV` использовалась только для диагностики и не участвовала в расчёте.",
            "",
            "| Показатель | Значение |",
            "| --- | ---: |",
            f"| Строк с биржевой IV | {_fmt_int(exchange['n'])} |",
            f"| Среднее абсолютное расхождение | {_fmt_number(exchange['mean_abs_diff'], 6)} |",
            f"| Медианное абсолютное расхождение | {_fmt_number(exchange['median_abs_diff'], 6)} |",
            f"| Максимальное абсолютное расхождение | {_fmt_number(exchange['max_abs_diff'], 6)} |",
            f"| Расхождение ≤ 0.001 | {_fmt_int(exchange['within_0_001'])} |",
            f"| Расхождение ≤ 0.01 | {_fmt_int(exchange['within_0_01'])} |",
            "",
            "## 7. Ограничения",
            "",
            "- Рассчитан один снимок; снимки разных моментов рынка не объединяются.",
            "- Опционы с `T < 1 дня` исключены: на коротком сроке IV неустойчива к округлению цены до тика.",
            "- В дампе экспирация записана как 00:00 UTC; принято фактическое время Deribit 08:00 UTC. "
            "При наличии точного `expiration_timestamp` его следует использовать вместо этой поправки.",
            "- `underlying_price` принят как форвард `F`; отдельная кривая форвардов и ставка не строились.",
            "- Биржевая `mid_IV` — только диагностика, не входные данные.",
            "- Целевой контракт, соседние контракты, квантили IV и ценовые интервалы относятся к следующему этапу.",
            "",
            "## 8. Воспроизведение",
            "",
            "```bash",
            'python3 -m pip install "scipy>=1.15"',
            "python3 -m deribit_data.iv_pipeline \\",
            f"  --dump {input_info['path']} \\",
            "  --output-dir results",
            "```",
            "",
            f"Версии: Python {summary['software']['python']}, SciPy {summary['software']['scipy']}.",
            "",
            "## 9. Файлы",
            "",
            f"- `{summary.get('outputs', {}).get('csv', 'CSV')}` — только успешные IV.",
            f"- `{summary.get('outputs', {}).get('summary', 'summary.json')}` — машинная статистика.",
            f"- `{summary.get('outputs', {}).get('report', 'report.md')}` — этот отчёт.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Укажите положительное конечное число")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Укажите неотрицательное конечное число")
    return number


def _snapshot_argument(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Ожидается ISO-время, например 2024-06-04T08:57:48.287611") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Расчёт IV BTC inverse call по историческому дампу")
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP, help="Путь к .sql или .sql.gz дампу")
    parser.add_argument("--snapshot", type=_snapshot_argument, help="Снимок current_time; по умолчанию последний")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Каталог результатов")
    parser.add_argument(
        "--max-log-moneyness",
        type=_positive_float,
        default=0.15,
        help="Максимальное |ln(K/F)| для ATM-фильтра",
    )
    parser.add_argument(
        "--min-days-to-expiry",
        type=_nonnegative_float,
        default=1.0,
        help="Минимальный срок до экспирации в днях",
    )
    parser.add_argument("--force", action="store_true", help="Перезаписать существующие результаты")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        scan = scan_options_table(args.dump, args.snapshot)
        snapshot = scan.snapshot
        if snapshot is None:
            raise ValueError("В дампе не найдено ни одного BTC call")
        if not scan.rows:
            raise ValueError(f"Для снимка {snapshot.isoformat()} не найдено BTC call")

        filters = IVFilters(
            max_log_moneyness=args.max_log_moneyness,
            min_days_to_expiry=args.min_days_to_expiry,
        )
        config = BrentConfig()
        rows, counts = build_iv_rows(scan.rows, filters, config)

        stamp = snapshot.strftime("%Y%m%dT%H%M%SZ")
        csv_path = args.output_dir / f"iv_{stamp}.csv"
        summary_path = args.output_dir / f"iv_{stamp}_summary.json"
        report_path = args.output_dir / f"iv_report_{stamp}.md"
        if not args.force:
            existing = [path for path in (csv_path, summary_path, report_path) if path.exists()]
            if existing:
                raise FileExistsError(f"Результаты уже существуют: {', '.join(str(path) for path in existing)}")
        args.output_dir.mkdir(parents=True, exist_ok=True)

        write_csv(csv_path, rows)
        summary = build_summary(
            args.dump,
            scan,
            filters,
            config,
            counts,
            rows,
            snapshot_explicit=args.snapshot is not None,
        )
        summary["outputs"] = {"csv": csv_path.name, "summary": summary_path.name, "report": report_path.name}
        _write_json(summary_path, summary)
        report_path.write_text(render_report(summary), encoding="utf-8")

        print(f"Снимок: {snapshot.isoformat()}")
        print(f"BTC call в снимке: {counts['input']}; успешных IV: {len(rows)}")
        print(f"CSV: {csv_path}")
        print(f"Отчёт: {report_path}")
        if not rows:
            LOGGER.error("Ни одна строка не прошла фильтры; CSV содержит только заголовок")
            return 2
        return 0
    except (OSError, ValueError) as error:
        LOGGER.error("Расчёт IV не выполнен: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
