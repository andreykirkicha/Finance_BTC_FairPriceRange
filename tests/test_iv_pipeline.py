"""Offline tests for filtering, exports, summaries, and the IV CLI."""

import csv
import gzip
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from deribit_data.iv import BrentConfig, inverse_call_price
from deribit_data.iv_pipeline import (
    CSV_FIELDS,
    IVFilters,
    build_iv_rows,
    build_summary,
    main,
    render_report,
    write_csv,
)
from deribit_data.pg_dump import DatasetStats, OptionRow, ScanResult

UNSET = object()
UTC = timezone.utc

COPY_HEADER = (
    'COPY public.options_table (id, full_name, type, open_interest, expiry, "current_time", '
    'strike, interest_rate, volume, price_change, mark_price, bid_price, "bid_IV", '
    'ask_price, "ask_IV", mid_price, "mid_IV", underlying_price) FROM stdin;'
)


def make_row(
    *,
    name: str = "BTC-25DEC24-100000",
    expiry: datetime = datetime(2024, 12, 25, tzinfo=UTC),
    days: float = 30,
    strike: float = 100_000,
    forward: float = 100_000,
    sigma: float = 0.5,
    mid_override: object = UNSET,
    bid_override: object = UNSET,
    ask_override: object = UNSET,
    exchange_mid_iv: float | None = 0.5,
) -> OptionRow:
    # The dump stores expiry at midnight, while the reader maps it to 08:00 UTC.
    snapshot = expiry - timedelta(days=days) + timedelta(hours=8)
    time_to_expiry = days / 365.0
    mid = inverse_call_price(forward, strike, time_to_expiry, sigma) if mid_override is UNSET else mid_override
    bid = mid - 0.001 if bid_override is UNSET else bid_override
    ask = mid + 0.001 if ask_override is UNSET else ask_override
    return OptionRow(
        instrument_name=name,
        option_type="call",
        expiry_raw=expiry.strftime("%Y-%m-%d %H:%M:%S"),
        snapshot_raw=snapshot.strftime("%Y-%m-%d %H:%M:%S"),
        strike=strike,
        underlying_price=forward,
        bid_price=bid,
        ask_price=ask,
        mid_price=mid,
        exchange_mid_iv=exchange_mid_iv,
        interest_rate=0.0,
    )


def stats_for(rows: int) -> DatasetStats:
    return DatasetStats(
        table="public.options_table",
        total_rows=rows,
        btc_rows=rows,
        btc_calls=rows,
        btc_puts=0,
        eth_rows=0,
        other_rows=0,
        snapshots=1,
        latest_snapshot=datetime(2024, 11, 25, 8, tzinfo=UTC),
        input_bytes=123,
    )


def write_cli_dump(path: Path) -> None:
    expiry = datetime(2024, 12, 25, tzinfo=UTC)
    snapshot = datetime(2024, 11, 25, 8, tzinfo=UTC)
    time_to_expiry = 30 / 365.0
    mid = inverse_call_price(100_000, 100_000, time_to_expiry, 0.5)
    fields = [
        "1",
        "BTC-25DEC24-100000",
        "call",
        "10",
        expiry.strftime("%Y-%m-%d %H:%M:%S"),
        snapshot.strftime("%Y-%m-%d %H:%M:%S"),
        "100000",
        "0",
        "1",
        "0",
        "0.05",
        str(mid - 0.001),
        "0.5",
        str(mid + 0.001),
        "0.5",
        str(mid),
        "0.5",
        "100000",
    ]
    text = "\n".join(["--", COPY_HEADER, "\t".join(fields), r"\.", ""])
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(text)


class IVPipelineTests(unittest.TestCase):
    def test_build_iv_rows_applies_filters_in_stable_order(self) -> None:
        rows = [
            make_row(),
            make_row(
                name="outside-atm",
                strike=120_000,
                mid_override=0.02,
                bid_override=0.019,
                ask_override=0.021,
            ),
            make_row(name="short", days=0.5),
            make_row(name="missing-quote", bid_override=None),
            make_row(name="outside-bounds", mid_override=1.2, bid_override=1.19, ask_override=1.21),
        ]
        iv_rows, counts = build_iv_rows(rows, IVFilters(), BrentConfig())

        self.assertEqual(len(iv_rows), 1)
        self.assertEqual(counts["input"], 5)
        self.assertEqual(counts["solved"], 1)
        self.assertEqual(
            counts["excluded"],
            {
                "missing_or_nonpositive_quote": 1,
                "outside_model_bounds": 1,
                "outside_atm": 1,
                "lt_min_days_to_expiry": 1,
            },
        )

    def test_write_csv_contains_only_successful_iv_columns(self) -> None:
        valid = make_row()
        iv_rows, _ = build_iv_rows([valid])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "iv.csv"
            write_csv(path, iv_rows)
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(tuple(reader.fieldnames or ()), CSV_FIELDS)
                exported = list(reader)

        self.assertEqual(len(exported), 1)
        self.assertAlmostEqual(float(exported[0]["iv_brent"]), 0.5, places=6)
        self.assertNotIn("exchange_mid_iv", exported[0])

    def test_summary_and_report_include_counts_and_data_provenance(self) -> None:
        option_rows = [make_row()]
        iv_rows, counts = build_iv_rows(option_rows)
        scan = ScanResult(stats=stats_for(1), rows=tuple(option_rows), snapshot=option_rows[0].snapshot_utc)
        summary = build_summary(
            Path("datasets/dump.sql.gz"),
            scan,
            IVFilters(),
            BrentConfig(),
            counts,
            iv_rows,
            snapshot_explicit=False,
        )
        summary["outputs"] = {"csv": "iv.csv", "summary": "summary.json", "report": "report.md"}
        report = render_report(summary)

        self.assertEqual(summary["counts"]["solved"], 1)
        self.assertEqual(summary["exchange_iv_comparison"]["n"], 1)
        self.assertIn("# Расчёт implied volatility", report)
        self.assertIn("Успешно посчитано", report)
        self.assertIn("datasets/dump.sql.gz", report)
        self.assertIn("iv.csv", report)

    def test_cli_writes_csv_summary_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dump = root / "dump.sql.gz"
            output = root / "results"
            write_cli_dump(dump)

            exit_code = main(["--dump", str(dump), "--output-dir", str(output)])

            self.assertEqual(exit_code, 0)
            csv_files = list(output.glob("iv_*.csv"))
            summary_files = list(output.glob("iv_*_summary.json"))
            report_files = list(output.glob("iv_report_*.md"))
            self.assertEqual((len(csv_files), len(summary_files), len(report_files)), (1, 1, 1))
            with csv_files[0].open(newline="", encoding="utf-8") as stream:
                exported = list(csv.DictReader(stream))
            self.assertEqual(len(exported), 1)
            self.assertAlmostEqual(float(exported[0]["iv_brent"]), 0.5, places=6)
            summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
            self.assertEqual(summary["counts"]["solved"], 1)
            self.assertIn("Расчёт implied volatility", report_files[0].read_text(encoding="utf-8"))

    def test_filters_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            IVFilters(max_log_moneyness=0)
        with self.assertRaises(ValueError):
            IVFilters(min_days_to_expiry=-1)


if __name__ == "__main__":
    unittest.main()
