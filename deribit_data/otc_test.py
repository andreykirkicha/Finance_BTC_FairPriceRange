"""OTC-тест по готовым CSV без ограничения на давность снимка котировок."""

import argparse
import csv
import json
import math
import re
import statistics
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from deribit_data.pilot import BLOCK_RESULT_FIELDS, _group_rows, _quantile, evaluate_block_trades, read_otc, read_quotes
from deribit_data.prepare_quotes import parse_utc, write_csv
from deribit_data.readable_plots import BLUE, GRAY, INK, Observation, _draw_contract, _save
from option_pricing.inverse_call import inverse_call_price
from option_pricing.snapshot import IVFilters, IVRow, calculate_iv_rows

__all__ = ["generate_combined_plot", "main", "run_otc_test"]

SNAPSHOT_FIELDS = (
    "instrument_name",
    "snapshot_utc",
    "expiry_utc",
    "strike_usd",
    "forward_usd",
    "q05",
    "q95",
    "lower_price",
    "upper_price",
    "peer_count",
)

INTERPOLATED_FIELDS = (
    "trade_id",
    "block_trade_id",
    "trade_utc",
    "instrument_name",
    "strike_usd",
    "price_btc",
    "lower_price",
    "upper_price",
    "covered",
    "width_btc",
    "width_relative",
    "left_snapshot_utc",
    "right_snapshot_utc",
    "interpolation_weight",
)


def _bounds_at(state: dict[str, Any], time: datetime) -> tuple[float, float]:
    """Рассчитать цену по параметрам снимка с учётом срока на заданный момент."""
    if time < state["time"]:
        raise ValueError("Нельзя использовать параметры снимка до его появления")
    remaining = (state["expiry"] - time).total_seconds() / (365 * 86400)
    return (
        inverse_call_price(state["forward"], state["strike"], remaining, state["q05"]),
        inverse_call_price(state["forward"], state["strike"], remaining, state["q95"]),
    )


def _snapshot_bands(
    iv_rows: Sequence[IVRow],
    groups: Sequence[Sequence[Observation]],
    holdout: dict[str, bool],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Рассчитать независимые состояния интервала в каждый момент снимка."""
    states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output = []
    quote_groups = _group_rows(iv_rows)
    for snapshot_group in groups:
        target = snapshot_group[0]
        for (time, expiry), quotes in sorted(quote_groups.items()):
            if expiry != target.expiry:
                continue
            training = [
                row
                for row in quotes
                if not holdout.get(row.instrument_name, False) and row.instrument_name != target.instrument
            ]
            if not training:
                raise ValueError(f"Нет обучающих котировок для {target.instrument} в {time}")
            state = {
                "time": time,
                "expiry": expiry,
                "strike": target.strike,
                "forward": training[0].forward_usd,
                "q05": _quantile([row.iv_brent for row in training], 0.05),
                "q95": _quantile([row.iv_brent for row in training], 0.95),
            }
            lower, upper = _bounds_at(state, time)
            state.update(lower=lower, upper=upper)
            states[target.instrument].append(state)
            output.append(
                dict(
                    zip(
                        SNAPSHOT_FIELDS,
                        (
                            target.instrument,
                            time.isoformat(),
                            expiry.isoformat(),
                            target.strike,
                            state["forward"],
                            state["q05"],
                            state["q95"],
                            lower,
                            upper,
                            len(training),
                        ),
                        strict=True,
                    )
                )
            )
    return states, output


def _read_snapshot_bands(path: Path) -> dict[str, list[dict[str, Any]]]:
    states: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            states[row["instrument_name"]].append(
                {
                    "time": parse_utc(row["snapshot_utc"]),
                    "expiry": parse_utc(row["expiry_utc"]),
                    "strike": float(row["strike_usd"]),
                    "forward": float(row["forward_usd"]),
                    "q05": float(row["q05"]),
                    "q95": float(row["q95"]),
                    "lower": float(row["lower_price"]),
                    "upper": float(row["upper_price"]),
                }
            )
    for group in states.values():
        group.sort(key=lambda state: state["time"])
    return states


def _interpolated_bounds(
    states: Sequence[dict[str, Any]],
    time: datetime,
) -> tuple[float, float, int, int, float]:
    """Линейно интерполировать ценовые границы между соседними снимками."""
    if not states:
        raise ValueError("Нет снимков для интерполяции")
    times = [state["time"] for state in states]
    left = max(0, min(bisect_right(times, time) - 1, len(states) - 1))
    right = min(left + 1, len(states) - 1) if time > times[0] else left
    weight = (
        (time - times[left]).total_seconds() / (times[right] - times[left]).total_seconds() if left != right else 0.0
    )
    lower = states[left]["lower"] + weight * (states[right]["lower"] - states[left]["lower"])
    upper = states[left]["upper"] + weight * (states[right]["upper"] - states[left]["upper"])
    return lower, upper, left, right, weight


def _draw_snapshot_band(
    axis: Any,
    states: Sequence[dict[str, Any]],
    trade_times: Sequence[datetime],
    end: datetime,
    color: str = GRAY,
) -> None:
    """Одна непрерывная кусочно-линейная полоса с узлами в моменты снимков."""
    times = sorted({*(state["time"] for state in states), *trade_times, end})
    bounds = [_interpolated_bounds(states, time) for time in times]
    axis.fill_between(
        times,
        [bound[0] for bound in bounds],
        [bound[1] for bound in bounds],
        color=color,
        alpha=0.12,
        linewidth=0,
        zorder=1,
    )


def generate_combined_plot(intervals_path: Path, output_dir: Path) -> list[Path]:
    """Показать все страйки на одной системе координат по готовым OTC-результатам."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with intervals_path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            lower, upper, price, strike = (
                float(row[field]) for field in ("lower_price", "upper_price", "price_btc", "strike_usd")
            )
            if not all(math.isfinite(value) for value in (lower, upper, price, strike)) or not 0 <= lower <= upper:
                raise ValueError("Некорректные границы или цена в OTC-результатах")
            groups[row["instrument_name"]].append(
                {
                    "time": parse_utc(row["trade_utc"]),
                    "lower": lower,
                    "upper": upper,
                    "price": price,
                    "strike": strike,
                    "covered": lower <= price <= upper,
                }
            )
    if not groups:
        raise ValueError("Нет результатов OTC для общего графика")
    for rows in groups.values():
        rows.sort(key=lambda row: row["time"])
    snapshot_states = _read_snapshot_bands(intervals_path.with_name("otc_snapshot_intervals.csv"))
    for name in groups:
        if not snapshot_states.get(name):
            raise ValueError(f"Нет интервалов снимков для {name}")
    all_rows = [row for rows in groups.values() for row in rows]
    first = min(snapshot_states[name][0]["time"] for name in groups)
    last = max(max(row["time"] for row in all_rows), max(snapshot_states[name][-1]["time"] for name in groups))
    padding = max((last - first) * 0.025, timedelta(minutes=15))
    inside = sum(row["covered"] for row in all_rows)
    coverage = f"{100 * inside / len(all_rows):.1f}".replace(".", ",")
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        height = 7.2
        figure, axis = plt.subplots(figsize=(15, height))
        figure.subplots_adjust(left=0.09, right=0.97, top=1 - 1.85 / height, bottom=0.75 / height)
        figure.text(
            0.09,
            1 - 0.25 / height,
            "Интервалы справедливых цен, тест на OTC-сделках",
            fontsize=17,
            color=INK,
            va="top",
        )
        colors = (BLUE, "#ff7f0e", "#9467bd", "#e377c2")
        for index, name in enumerate(sorted(groups, key=lambda name: (groups[name][0]["strike"], name))):
            rows = groups[name]
            color = colors[index % len(colors)]
            times = [row["time"] for row in rows]
            lower = [min(row["lower"], row["price"]) for row in rows]
            upper = [max(row["upper"], row["price"]) for row in rows]
            _draw_snapshot_band(axis, snapshot_states[name], times, last + padding, color)
            label = re.sub(r"-(\d{1,2}[A-Z]{3})\d{2}-", r"-\1-", name)
            axis.plot(times, [row["price"] for row in rows], color=color, linewidth=1.7, label=label, zorder=3)
            status_colors = ["#2ca02c" if row["covered"] else "#d62728" for row in rows]
            axis.vlines(times, lower, upper, colors=status_colors, linewidth=4, alpha=0.55, zorder=2)
            axis.scatter(
                times,
                [row["price"] for row in rows],
                c=status_colors,
                edgecolors=color,
                linewidths=1.2,
                s=45,
                zorder=4,
            )
        figure.legend(
            handles=axis.lines,
            loc="upper left",
            bbox_to_anchor=(0.085, 1 - 0.67 / height),
            ncol=2,
            frameon=False,
            fontsize=10.5,
            handlelength=3,
            columnspacing=2.6,
        )
        axis.set_title("Все страйки", loc="left", fontsize=11.5, pad=13, color=INK)
        axis.set_title(
            f"Coverage: {inside}/{len(all_rows)} = {coverage}%", loc="right", fontsize=11.5, pad=13, color=INK
        )
        axis.set_xlim(first - padding, last + padding)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10, tz=timezone.utc)
        axis.set_xticks([tick for tick in locator.tick_values(first - padding, last) if tick <= mdates.date2num(last)])
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m", tz=timezone.utc))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}".replace(".", ",")))
        axis.set_xlabel("Дата наблюдения", labelpad=10)
        axis.set_ylabel("Цена опциона, BTC", labelpad=12)
        axis.margins(y=0.12)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e0e0e0", linewidth=0.7)
        axis.grid(axis="x", color="#eeeeee", linewidth=0.5)
        axis.spines[["top", "right"]].set_visible(False)
        for side in ("bottom", "left"):
            axis.spines[side].set_color("#b0b0b0")
        axis.tick_params(colors="#555555", labelsize=10, length=3)
        output_dir.mkdir(parents=True, exist_ok=True)
        return _save(figure, output_dir / "otc_all_strikes")


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    inside = sum(row["covered"] for row in rows)
    return {
        "n": count,
        "inside": inside,
        "outside": count - inside,
        "coverage": inside / count if count else None,
        "mean_width_btc": statistics.fmean(row["width_btc"] for row in rows) if count else None,
        "mean_width_relative": statistics.fmean(row["width_relative"] for row in rows) if count else None,
    }


def _plot(
    groups: Sequence[Sequence[Observation]],
    stem: Path,
    snapshot_states: dict[str, list[dict[str, Any]]],
) -> list[Path]:
    count = len(groups)
    height = 5.8 if count == 1 else 4.5 * count + 1.6
    figure, axes = plt.subplots(count, 1, figsize=(15, height), squeeze=False)
    figure.subplots_adjust(
        left=0.09,
        right=0.97,
        top=1 - 1.45 / height,
        bottom=0.75 / height,
        hspace=0.45,
    )
    figure.text(
        0.09,
        1 - 0.25 / height,
        "Интервалы справедливых цен, тест на OTC-сделках",
        fontsize=17,
        color=INK,
        va="top",
    )
    figure.legend(
        handles=[
            Patch(
                facecolor=GRAY, edgecolor=GRAY, alpha=0.3, hatch="///", linewidth=0, label="Интервал справедливых цен"
            ),
            Line2D([], [], color=BLUE, linewidth=1.7, label="Цена OTC-сделки"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.085, 1 - 0.67 / height),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=3,
        columnspacing=2.6,
    )
    for axis, group in zip(axes.flat, groups, strict=True):
        _draw_contract(axis, group)
        # Заливка строится по снимкам; OTC-точки не добавляют новых изгибов.
        for collection in list(axis.collections):
            if isinstance(collection, PolyCollection):
                collection.remove()
            elif isinstance(collection, LineCollection):
                collection.set_segments(
                    [
                        [
                            (mdates.date2num(row.time), min(row.lower, row.mid)),
                            (mdates.date2num(row.time), max(row.upper, row.mid)),
                        ]
                        for row in group
                    ]
                )
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.4f}".replace(".", ",")))
        states = snapshot_states[group[0].instrument]
        first, last = states[0]["time"], max(states[-1]["time"], group[-1].time)
        span = last - first
        padding = max(span * 0.015, timedelta(minutes=15))
        _draw_snapshot_band(axis, states, [row.time for row in group], last + padding)
        axis.set_xlim(first - padding, last + padding)
        locator = mdates.AutoDateLocator(minticks=4, maxticks=10, tz=timezone.utc)
        axis.set_xticks([tick for tick in locator.tick_values(first - padding, last) if tick <= mdates.date2num(last)])
        date_format = "%d.%m" if span >= timedelta(days=2) else "%d.%m\n%H:%M"
        axis.xaxis.set_major_formatter(mdates.DateFormatter(date_format, tz=timezone.utc))
    return _save(figure, stem)


def run_otc_test(
    quotes_path: Path,
    otc_path: Path,
    assignments_path: Path,
    output_dir: Path,
    strikes: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Построить непрерывные OTC-интервалы и сохранить оба способа проверки.

    Используются только подготовленные CSV. Можно явно выбрать страйки;
    дополнительных фильтров по времени, размеру и денежности нет. Из оценки
    IV исключаются прежние holdout-контракты и контракт проверяемой сделки.
    """
    source_paths = (quotes_path, otc_path, assignments_path)
    for path in source_paths:
        if path.suffix.lower() != ".csv":
            raise ValueError("Входами могут быть только подготовленные CSV")
    if output_dir.resolve() == assignments_path.parent.resolve():
        raise ValueError("Результаты OTC нужно сохранять в отдельной папке")
    with assignments_path.open(encoding="utf-8", newline="") as stream:
        holdout = {row["instrument_name"]: row["holdout"].lower() == "true" for row in csv.DictReader(stream)}
    if not otc_path.is_file():
        raise FileNotFoundError(otc_path)
    trades = read_otc(otc_path)
    source_trade_rows = len(trades)
    if strikes is not None:
        if not strikes or not all(math.isfinite(strike) and strike > 0 for strike in strikes):
            raise ValueError("Укажите положительные страйки")
        selected_strikes = set(strikes)
        trades = [trade for trade in trades if float(trade["strike_usd"]) in selected_strikes]
    if not trades:
        raise ValueError("В OTC CSV нет сделок")
    for trade in trades:
        price = float(trade["price_btc"])
        strike = float(trade["strike_usd"])
        if not all(math.isfinite(value) and value > 0 for value in (price, strike)):
            raise ValueError(f"Некорректные цена или страйк сделки {trade.get('trade_id')}")
        parse_utc(trade["trade_utc"])
        parse_utc(trade["expiry_utc"])
    # Повторный расчёт IV только из существующих quotes.csv. Границы фильтров
    # расширены, чтобы не вводить новый отбор по денежности или горизонту.
    iv_rows, iv_counts = calculate_iv_rows(
        read_quotes(quotes_path),
        IVFilters(max_log_moneyness=1e6, min_days_to_expiry=0),
    )
    results, counts = evaluate_block_trades(
        trades,
        iv_rows,
        holdout,
        min_peers=1,
        max_gap_seconds=float("inf"),
    )
    if not results:
        raise ValueError(f"Не удалось рассчитать интервалы OTC: {dict(counts)}")
    previous_snapshot_results = [dict(row) for row in results]

    by_contract: dict[str, list[dict[str, Any]]] = defaultdict(list)
    trade_lookup = {trade["trade_id"]: trade for trade in trades}
    if len(trade_lookup) != len(trades):
        raise ValueError("Повторные идентификаторы OTC-сделок")
    plot_groups: dict[str, list[Observation]] = defaultdict(list)
    for row in results:
        by_contract[row["instrument_name"]].append(row)
        trade = trade_lookup[row["trade_id"]]
        plot_groups[row["instrument_name"]].append(
            Observation(
                instrument=row["instrument_name"],
                time=parse_utc(row["trade_utc"]),
                expiry=parse_utc(trade["expiry_utc"]),
                strike=row["strike_usd"],
                mid=row["price_btc"],
                lower=row["lower_price"],
                upper=row["upper_price"],
                covered=row["covered"],
            )
        )
    groups = sorted(plot_groups.values(), key=lambda group: (group[0].expiry, group[0].strike))
    for group in groups:
        group.sort(key=lambda row: row.time)
    snapshot_states, snapshot_rows = _snapshot_bands(iv_rows, groups, holdout)
    for row in results:
        states = snapshot_states[row["instrument_name"]]
        lower, upper, left, right, weight = _interpolated_bounds(states, parse_utc(row["trade_utc"]))
        row.update(
            lower_price=lower,
            upper_price=upper,
            covered=lower <= row["price_btc"] <= upper,
            width_btc=upper - lower,
            width_relative=(upper - lower) / row["price_btc"],
            left_snapshot_utc=states[left]["time"].isoformat(),
            right_snapshot_utc=states[right]["time"].isoformat(),
            interpolation_weight=weight,
        )
    for group in groups:
        for index, observation in enumerate(group):
            lower, upper, *_ = _interpolated_bounds(snapshot_states[observation.instrument], observation.time)
            group[index] = replace(
                observation,
                lower=lower,
                upper=upper,
                covered=lower <= observation.mid <= upper,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_metrics = output_dir / "metrics.json"
    previous_plots = (
        json.loads(previous_metrics.read_text(encoding="utf-8")).get("plots", []) if previous_metrics.is_file() else []
    )
    # Прежняя проверка по последнему снимку сохраняется независимо от
    # ретроспективной интерполяции, используемой для непрерывного графика.
    write_csv(output_dir / "otc_intervals.csv", BLOCK_RESULT_FIELDS, previous_snapshot_results)
    interpolated_path = output_dir / "otc_interpolated_intervals.csv"
    write_csv(
        interpolated_path,
        INTERPOLATED_FIELDS,
        [{field: row[field] for field in INTERPOLATED_FIELDS} for row in results],
    )
    write_csv(output_dir / "otc_snapshot_intervals.csv", SNAPSHOT_FIELDS, snapshot_rows)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        paths = _plot(groups, plot_dir / "otc_intervals", snapshot_states)
        for group in groups:
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in group[0].instrument)
            paths.extend(_plot([group], plot_dir / f"otc_{safe_name}", snapshot_states))
    paths.extend(generate_combined_plot(interpolated_path, plot_dir))
    metrics = {
        "inputs": {"quotes": str(quotes_path), "otc": str(otc_path), "holdout": str(assignments_path)},
        "source_trade_rows": source_trade_rows,
        "selected_strikes": sorted(set(strikes)) if strikes is not None else None,
        "package_counts": dict(Counter(trade.get("package_status", "") for trade in trades)),
        "method": {
            "quantiles": [0.05, 0.95],
            "snapshot": "границы цены на снимках соединяются прямыми",
            "plot_band": "непрерывная кусочно-линейная интерполяция ценовых границ по 30 снимкам",
            "coverage": "проверяется интерполированный интервал на время OTC-сделки",
            "interpretation": "ретроспективная интерполяция использует оба соседних снимка",
            "snapshots_per_contract": {name: len(states) for name, states in snapshot_states.items()},
            "price": "фактическая price_btc OTC-сделки, не mid",
            "training": "исключены рыночные holdout-контракты и контракт OTC-сделки",
            "otc_filters": "кроме явно выбранных страйков, дополнительных фильтров по дате, размеру и денежности нет",
            "gap_minutes_min": min(row["gap_seconds"] for row in results) / 60,
            "gap_minutes_max": max(row["gap_seconds"] for row in results) / 60,
        },
        "iv": iv_counts,
        "counts": dict(counts),
        "overall": _summary(results),
        "previous_snapshot_overall": _summary(previous_snapshot_results),
        "tables": {"plot_intervals": interpolated_path.name, "previous_snapshot_intervals": "otc_intervals.csv"},
        "by_contract": {name: _summary(rows) for name, rows in sorted(by_contract.items())},
        "plots": [str(path.relative_to(output_dir)) for path in paths],
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    current_paths = {path.resolve() for path in paths}
    for previous_plot in previous_plots:
        path = output_dir / previous_plot
        if (
            path.resolve().parent == plot_dir.resolve()
            and re.fullmatch(r"otc_(?:BTC-[A-Za-z0-9-]+|intervals|all_strikes)\.png", path.name)
            and path.resolve() not in current_paths
        ):
            path.unlink(missing_ok=True)
    return metrics


def main(argv: Sequence[str] | None = None) -> int:
    """Запустить OTC-тест и графики отдельно от рыночного пилота."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quotes", type=Path, default=Path("data/quotes.csv"))
    parser.add_argument("--otc", type=Path, default=Path("data/otc_trades.csv"))
    parser.add_argument("--holdout", type=Path, default=Path("results/pilot/holdout_assignments.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/otc"))
    parser.add_argument("--strikes", nargs="+", type=float, help="Страйки для тестирования и графиков")
    parser.add_argument("--combined-only", action="store_true", help="Только общий график по готовым результатам OTC")
    parser.add_argument("--intervals", type=Path, default=Path("results/otc/otc_interpolated_intervals.csv"))
    args = parser.parse_args(argv)
    if args.combined_only:
        for path in generate_combined_plot(args.intervals, args.output_dir / "plots"):
            print(path)
        return 0
    metrics = run_otc_test(args.quotes, args.otc, args.holdout, args.output_dir, args.strikes)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
