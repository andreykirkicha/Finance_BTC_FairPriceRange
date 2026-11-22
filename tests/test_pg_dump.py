"""Offline tests for the PostgreSQL COPY-block reader."""

import gzip
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from deribit_data.pg_dump import parse_expiry_utc, scan_options_table, timestamp_key

COPY_HEADER = (
    'COPY public.options_table (id, full_name, type, open_interest, expiry, "current_time", '
    'strike, interest_rate, volume, price_change, mark_price, bid_price, "bid_IV", '
    'ask_price, "ask_IV", mid_price, "mid_IV", underlying_price) FROM stdin;'
)


def copy_row(
    name: str = "BTC-26APR24-43000",
    option_type: str = "call",
    expiry: str = "0024-04-26 00:00:00",
    current_time: str = "2024-03-20 06:52:18.957262",
    strike: str = "43000",
    forward: str = "62874.96",
    bid: str = "0.3235",
    ask: str = "0.327",
    mid: str = "0.32525",
    mid_iv: str = "0.8763",
) -> str:
    return "\t".join(
        [
            "1",
            name,
            option_type,
            "274.8",
            expiry,
            current_time,
            strike,
            "0",
            "0",
            r"\N",
            "0.3250523",
            bid,
            "0.83",
            ask,
            "0.91",
            mid,
            mid_iv,
            forward,
        ]
    )


def write_dump(path: Path, rows: list[str], *, compress: bool = True) -> None:
    text = "\n".join(["--", COPY_HEADER, *rows, r"\.", ""])
    if compress:
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            stream.write(text)
    else:
        path.write_text(text, encoding="utf-8")


class PgDumpTests(unittest.TestCase):
    def test_selects_latest_btc_call_snapshot_and_counts_table(self) -> None:
        latest = "2024-03-21 08:00:00"
        rows = [
            copy_row(),
            copy_row(name="BTC-27DEC24-42000", expiry="0024-12-27 00:00:00", current_time=latest),
            copy_row(name="BTC-27DEC24-43000", expiry="0024-12-27 00:00:00", current_time=latest, bid=r"\N"),
            copy_row(name="BTC-27DEC24-44000", option_type="put", current_time=latest),
            copy_row(name="ETH-21MAR24-4050", current_time=latest, forward="3143.2"),
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dump.sql.gz"
            write_dump(path, rows)
            result = scan_options_table(path)

        self.assertEqual(result.stats.total_rows, 5)
        self.assertEqual(result.stats.btc_rows, 4)
        self.assertEqual(result.stats.btc_calls, 3)
        self.assertEqual(result.stats.btc_puts, 1)
        self.assertEqual(result.stats.eth_rows, 1)
        self.assertEqual(result.stats.snapshots, 2)
        self.assertEqual(result.stats.latest_snapshot, datetime(2024, 3, 21, 8, tzinfo=timezone.utc))
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0].instrument_name, "BTC-27DEC24-42000")
        self.assertAlmostEqual(result.rows[0].underlying_price, 62874.96)
        self.assertAlmostEqual(result.rows[0].exchange_mid_iv, 0.8763)
        self.assertIsNone(result.rows[1].bid_price)

    def test_explicit_snapshot_does_not_change_latest_statistic(self) -> None:
        first = "2024-03-20 06:52:18.957262"
        latest = "2024-03-21 08:00:00"
        rows = [
            copy_row(current_time=first),
            copy_row(name="BTC-27DEC24-42000", current_time=latest),
        ]
        target = datetime(2024, 3, 20, 6, 52, 18, 957262, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dump.sql.gz"
            write_dump(path, rows)
            result = scan_options_table(path, snapshot=target)

        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0].snapshot_utc, target)
        self.assertEqual(result.snapshot, target)
        self.assertEqual(result.stats.latest_snapshot, datetime(2024, 3, 21, 8, tzinfo=timezone.utc))

    def test_repairs_two_and_four_digit_years(self) -> None:
        expiry_2024 = parse_expiry_utc("0024-12-27 00:00:00")
        expiry_2025 = parse_expiry_utc("0025-03-28 00:00:00")
        self.assertEqual(expiry_2024.year, 2024)
        self.assertEqual(expiry_2025.year, 2025)
        self.assertEqual((expiry_2024.hour, expiry_2025.hour), (8, 8))

    def test_variable_fractional_seconds_sort_together(self) -> None:
        self.assertEqual(
            timestamp_key("2024-03-20 06:52:18.1"),
            timestamp_key("2024-03-20 06:52:18.100000"),
        )

    def test_malformed_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dump.sql.gz"
            write_dump(path, ["one\ttwo\tthree"])
            with self.assertRaisesRegex(ValueError, "Malformed COPY row"):
                scan_options_table(path)

    def test_missing_copy_block_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dump.sql.gz"
            write_dump(path, [])
            path.write_bytes(gzip.compress(b"no copy block here\n"))
            with self.assertRaisesRegex(ValueError, "COPY block"):
                scan_options_table(path)

    def test_plain_sql_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dump.sql"
            write_dump(path, [copy_row()], compress=False)
            result = scan_options_table(path)
        self.assertEqual(len(result.rows), 1)


if __name__ == "__main__":
    unittest.main()
