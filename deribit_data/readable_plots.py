"""Новые графики рыночного теста по готовым результатам, без пересчёта модели."""

import argparse
import csv
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

__all__ = ["generate_plots", "main"]

# Сохраняем семантику цветов исходных графиков.
BLUE = "#1f77b4"
GRAY = "#808080"
GREEN = "#2ca02c"
RED = "#d62728"
INK = "#303030"
MUTED = "#555555"


@dataclass(frozen=True)
class Observation:
    """Одна проверка рыночного mid в момент снимка котировок."""

    instrument: str
    time: datetime
    expiry: datetime
    strike: float
    mid: float
    lower: float
    upper: float
    covered: bool


def _read(path: Path) -> list[Observation]:
    required = {
        "instrument_name",
        "snapshot_utc",
        "expiry_utc",
        "strike_usd",
        "mid_btc",
        "lower_price",
        "upper_price",
        "covered",
    }
    observations = []
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or ()):
            raise ValueError(f"{path}: отсутствуют обязательные столбцы результатов holdout")
        for line, row in enumerate(reader, start=2):
            try:
                time = datetime.fromisoformat(row["snapshot_utc"])
                expiry = datetime.fromisoformat(row["expiry_utc"])
                if time.tzinfo is None or expiry.tzinfo is None:
                    raise ValueError("даты должны содержать часовой пояс")
                mid, lower, upper, strike = (
                    float(row[field]) for field in ("mid_btc", "lower_price", "upper_price", "strike_usd")
                )
                if not all(math.isfinite(value) for value in (mid, lower, upper, strike)):
                    raise ValueError("цены и страйк должны быть конечными числами")
                if not 0 <= lower <= upper or mid < 0 or strike <= 0:
                    raise ValueError("некорректные границы, цена или страйк")
                covered = lower <= mid <= upper
                if row["covered"].lower() not in {"true", "false"}:
                    raise ValueError("covered должен быть True или False")
                if covered != (row["covered"].lower() == "true"):
                    raise ValueError("covered не соответствует границам интервала")
                if not row["instrument_name"].strip():
                    raise ValueError("не указано имя контракта")
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}, строка {line}: {error}") from error
            observations.append(
                Observation(
                    instrument=row["instrument_name"],
                    time=time.astimezone(timezone.utc),
                    expiry=expiry.astimezone(timezone.utc),
                    strike=strike,
                    mid=mid,
                    lower=lower,
                    upper=upper,
                    covered=covered,
                )
            )
    if not observations:
        raise ValueError(f"{path}: нет результатов рыночного теста")
    return observations


def _legend() -> list[Line2D | Patch]:
    return [
        Patch(
            facecolor=GRAY,
            edgecolor=GRAY,
            alpha=0.3,
            hatch="///",
            linewidth=0,
            label="Интервал справедливых цен",
        ),
        Line2D([], [], color=BLUE, linewidth=1.7, label="Тестовый опцион (mid)"),
    ]


def _draw_contract(axis: Axes, rows: Sequence[Observation]) -> None:
    times = [row.time for row in rows]
    lower = [row.lower for row in rows]
    upper = [row.upper for row in rows]
    mid = [row.mid for row in rows]
    axis.fill_between(times, lower, upper, color=GRAY, alpha=0.12, linewidth=0, zorder=1)
    axis.vlines(
        times,
        lower,
        upper,
        colors=[GREEN if row.covered else RED for row in rows],
        linewidth=4,
        alpha=0.55,
        zorder=2,
    )
    axis.plot(times, mid, color=BLUE, linewidth=1.7, zorder=4)
    for covered, color in ((True, GREEN), (False, RED)):
        selected = [row for row in rows if row.covered == covered]
        axis.scatter(
            [row.time for row in selected],
            [row.mid for row in selected],
            color=color,
            marker="o",
            s=38,
            linewidths=0.7,
            edgecolors="white",
            zorder=5,
        )

    axis.set_title(
        re.sub(r"-(\d{1,2}[A-Z]{3})\d{2}-", r"-\1-", rows[0].instrument),
        loc="left",
        fontsize=11.5,
        pad=13,
        color=INK,
    )
    inside = sum(row.covered for row in rows)
    coverage = f"{100 * inside / len(rows):.1f}".replace(".", ",")
    axis.set_title(
        f"Coverage: {inside}/{len(rows)} = {coverage}%",
        loc="right",
        fontsize=11.5,
        pad=13,
        color=INK,
    )
    axis.set_ylabel("Цена опциона, BTC", labelpad=12)
    axis.set_xlabel("Дата наблюдения", labelpad=10)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=3))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3f}".replace(".", ",")))
    span = times[-1] - times[0]
    padding = max(span * 0.015, timedelta(minutes=15))
    axis.set_xlim(times[0] - padding, times[-1] + padding)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10, tz=timezone.utc)
    axis.xaxis.set_major_locator(locator)
    axis.set_xticks(
        [
            tick
            for tick in locator.tick_values(times[0] - padding, times[-1] + padding)
            if tick <= mdates.date2num(times[-1])
        ]
    )
    date_format = "%d.%m" if span >= timedelta(days=2) else "%d.%m\n%H:%M"
    axis.xaxis.set_major_formatter(mdates.DateFormatter(date_format, tz=timezone.utc))
    axis.margins(y=0.12)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e0e0e0", linewidth=0.7)
    axis.grid(axis="x", color="#eeeeee", linewidth=0.5)
    axis.spines[["top", "right"]].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#b0b0b0")
    axis.tick_params(colors=MUTED, labelsize=10, length=3)


def _save(figure: Figure, stem: Path) -> list[Path]:
    path = stem.with_suffix(".png")
    try:
        figure.savefig(path, dpi=200, facecolor="white")
    finally:
        plt.close(figure)
    return [path]


def _plot(groups: Sequence[Sequence[Observation]], stem: Path) -> list[Path]:
    count = len(groups)
    height = 5.8 if count == 1 else 4.5 * count + 1.6
    figure, axes = plt.subplots(count, 1, figsize=(15, height), squeeze=False)
    header = 1.45 / height
    footer = 0.75 / height
    figure.subplots_adjust(left=0.09, right=0.97, top=1 - header, bottom=footer, hspace=0.45)
    figure.text(
        0.09,
        1 - 0.25 / height,
        "Интервалы справедливых цен, тест на рыночных сделках",
        fontsize=17,
        color=INK,
        va="top",
    )
    figure.legend(
        handles=_legend(),
        loc="upper left",
        bbox_to_anchor=(0.085, 1 - 0.67 / height),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        handlelength=3,
        columnspacing=2.6,
        labelspacing=1.0,
    )
    for axis, group in zip(axes.flat, groups, strict=True):
        _draw_contract(axis, group)
    return _save(figure, stem)


def generate_plots(intervals_path: Path, output_dir: Path) -> list[Path]:
    """Сохранить общий график и графики контрактов в PNG.

    Читает только готовый CSV результатов. Исходные данные и прежние графики
    не используются для расчётов и не перезаписываются.
    """
    if output_dir.resolve() == (intervals_path.parent / "plots").resolve():
        raise ValueError("Выберите отдельную папку: plots содержит исходные графики")
    observations = _read(intervals_path)
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for row in observations:
        grouped[row.instrument].append(row)
    groups = sorted(grouped.values(), key=lambda group: (group[0].expiry, group[0].strike, group[0].instrument))
    for group in groups:
        group.sort(key=lambda row: row.time)
        if len({row.time for row in group}) != len(group):
            raise ValueError(f"{group[0].instrument}: повторные результаты для одного снимка")
        if len({(row.strike, row.expiry) for row in group}) != 1:
            raise ValueError(f"{group[0].instrument}: страйк или экспирация меняются внутри контракта")
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11}):
        paths = _plot(groups, output_dir / "market_intervals")
        for group in groups:
            safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in group[0].instrument)
            paths.extend(_plot([group], output_dir / f"market_{safe_name}"))
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    """Построить новые графики независимо от исходного пайплайна."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervals", type=Path, default=Path("results/pilot/holdout_intervals.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot/plots_readable"))
    args = parser.parse_args(argv)
    for path in generate_plots(args.intervals, args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
