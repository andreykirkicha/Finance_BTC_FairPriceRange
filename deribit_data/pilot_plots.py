"""Графики пилота: рыночные цены, интервалы, покрытие и ширина."""

import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

__all__ = ["generate_plots"]

GREEN = "#2ca02c"
RED = "#d62728"
BLUE = "#1f77b4"


def _read(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "instrument_name": row["instrument_name"],
                    "snapshot_utc": datetime.fromisoformat(row["snapshot_utc"]),
                    "strike_usd": float(row["strike_usd"]),
                    "mid_btc": float(row["mid_btc"]),
                    "lower_price": float(row["lower_price"]),
                    "upper_price": float(row["upper_price"]),
                    "covered": row["covered"].lower() == "true",
                    "width_relative": float(row["width_relative"]),
                    "log_moneyness": float(row["log_moneyness"]),
                }
            )
    return rows


def _short_name(name: str, strike: float) -> str:
    return f"{strike:,.0f}".replace(",", " ")


def _plot_intervals(rows: Sequence[dict[str, Any]], path: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["instrument_name"]].append(row)
    names = sorted(groups, key=lambda name: groups[name][0]["strike_usd"])
    columns = 1 if len(names) <= 3 else 2
    rows_count = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(rows_count, columns, figsize=(13, 4.2 * rows_count), squeeze=False)
    axes_flat = list(axes.flat)
    for axis, name in zip(axes_flat, names):
        data = sorted(groups[name], key=lambda row: row["snapshot_utc"])
        times = [row["snapshot_utc"] for row in data]
        for row in data:
            color = GREEN if row["covered"] else RED
            axis.vlines(
                row["snapshot_utc"],
                row["lower_price"],
                row["upper_price"],
                color=color,
                alpha=0.55,
                linewidth=3.5,
            )
        axis.scatter(
            times,
            [row["mid_btc"] for row in data],
            c=[GREEN if row["covered"] else RED for row in data],
            s=26,
            zorder=3,
        )
        axis.plot(times, [row["mid_btc"] for row in data], color="#555555", alpha=0.25, linewidth=1)
        covered = sum(row["covered"] for row in data)
        axis.set_title(
            f"{name}  ·  страйк {_short_name(name, data[0]['strike_usd'])}\n"
            f"внутри: {covered}/{len(data)} ({100 * covered / len(data):.0f}%)",
            fontsize=11,
        )
        axis.set_ylabel("Премия, BTC")
        axis.grid(alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        axis.tick_params(axis="x", labelsize=8)
    for axis in axes_flat[len(names) :]:
        axis.axis("off")
    legend = [
        Line2D([0], [0], color=GREEN, linewidth=4, label="mid внутри интервала"),
        Line2D([0], [0], color=RED, linewidth=4, label="mid вне интервала"),
        Line2D([0], [0], marker="o", color="#555555", linestyle="None", label="рыночный mid"),
    ]
    figure.suptitle("Holdout: рыночная цена и предсказанный интервал", fontsize=15, y=0.99)
    figure.legend(handles=legend, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _plot_coverage(rows: Sequence[dict[str, Any]], path: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["instrument_name"]].append(row)
    names = sorted(groups, key=lambda name: groups[name][0]["strike_usd"])
    covered = [sum(row["covered"] for row in groups[name]) for name in names]
    missed = [len(groups[name]) - value for name, value in zip(names, covered)]
    labels = [_short_name(name, groups[name][0]["strike_usd"]) for name in names]

    figure, axis = plt.subplots(figsize=(10, 4.8))
    positions = list(range(len(names)))
    axis.bar(positions, covered, color=GREEN, label="внутри")
    axis.bar(positions, missed, bottom=covered, color=RED, label="вне")
    maximum = max(inside + outside for inside, outside in zip(covered, missed))
    for position, name, inside, outside in zip(positions, names, covered, missed):
        total = inside + outside
        axis.text(
            position,
            total + maximum * 0.02,
            f"{inside}/{total}\n{100 * inside / total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.set_ylim(0, maximum * 1.16)
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Страйк контракта")
    axis.set_ylabel("Число проверок")
    axis.set_title("Покрытие holdout по контрактам", pad=18)
    axis.legend(frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_width(rows: Sequence[dict[str, Any]], path: Path) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["instrument_name"]].append(row)
    names = sorted(groups, key=lambda name: groups[name][0]["strike_usd"])

    figure, axis = plt.subplots(figsize=(11, 4.5))
    for name in names:
        data = sorted(groups[name], key=lambda row: row["snapshot_utc"])
        axis.plot(
            [row["snapshot_utc"] for row in data],
            [100 * row["width_relative"] for row in data],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"страйк {_short_name(name, data[0]['strike_usd'])}",
        )
    axis.set_ylabel("Ширина интервала, % от mid")
    axis.set_title("Относительная ширина интервала во времени")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def generate_plots(intervals_path: Path, output_dir: Path) -> list[Path]:
    rows = _read(intervals_path)
    if not rows:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        output_dir / "holdout_intervals.png",
        output_dir / "holdout_coverage_by_contract.png",
        output_dir / "holdout_width_relative.png",
    ]
    _plot_intervals(rows, paths[0])
    _plot_coverage(rows, paths[1])
    _plot_width(rows, paths[2])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=Path, default=Path("results/pilot/holdout_intervals.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot/plots"))
    args = parser.parse_args(argv)
    paths = generate_plots(args.intervals, args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
