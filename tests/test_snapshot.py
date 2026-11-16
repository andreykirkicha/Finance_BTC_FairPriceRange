"""Offline checks of research-data validity and the complete export pipeline."""

import copy
import csv
import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from deribit_data.snapshot import Filters, build_rows, collect_snapshot

AS_OF_MS = int(datetime(2026, 11, 16, 12, tzinfo=timezone.utc).timestamp() * 1000)
EXPIRY_MS = int(datetime(2026, 12, 25, 8, tzinfo=timezone.utc).timestamp() * 1000)


def instrument(name: str = "BTC-25DEC26-100000-C", **changes: object) -> dict:
    """An explicitly synthetic contract specification, never live market data."""
    return {
        "instrument_name": name,
        "kind": "option",
        "option_type": "call",
        "base_currency": "BTC",
        "settlement_currency": "BTC",
        "instrument_type": "reversed",
        "is_active": True,
        "state": "open",
        "strike": 100_000,
        "expiration_timestamp": EXPIRY_MS,
        "contract_size": 1,
        **changes,
    }


def summary(name: str = "BTC-25DEC26-100000-C", **changes: object) -> dict:
    """Synthetic quote with a deliberately different exchange mark and last."""
    return {
        "instrument_name": name,
        "creation_timestamp": AS_OF_MS,
        "bid_price": 0.049,
        "ask_price": 0.051,
        "underlying_price": 100_000,
        "underlying_index": "BTC-25DEC26",
        "interest_rate": 0,
        "mark_price": 0.07,
        "last": 0.09,
        "mark_iv": 60,
        **changes,
    }


def capture(records: list[dict]) -> dict:
    """Wrap fixtures in the same provenance envelope as the HTTP client."""
    return {
        "received_at_utc": datetime.fromtimestamp(AS_OF_MS / 1000, timezone.utc).isoformat(),
        "response": {"jsonrpc": "2.0", "result": records, "usOut": AS_OF_MS * 1000},
    }


class SnapshotTests(unittest.TestCase):
    def test_one_mid_per_contract_and_exact_remaining_term(self) -> None:
        row = build_rows([instrument()], [summary()], AS_OF_MS, Filters())[0]
        self.assertTrue(row["selected_for_iv"])
        self.assertAlmostEqual(row["mid_btc"], 0.05)
        self.assertAlmostEqual(row["relative_spread"], 0.04)
        self.assertAlmostEqual(row["days_to_expiry"], 38 + 20 / 24)
        self.assertAlmostEqual(row["time_to_expiry_years"], (38 + 20 / 24) / 365)
        self.assertEqual(row["underlying_price_usd"], 100_000)
        self.assertEqual(row["premium_currency"], "BTC")
        self.assertEqual(row["mark_iv_exchange_pct"], 60)

    def test_excludes_put_linear_and_other_expiries(self) -> None:
        specs = [
            instrument(),
            instrument("put", option_type="put"),
            instrument("linear", settlement_currency="USDC", instrument_type="linear"),
            instrument("other", expiration_timestamp=EXPIRY_MS + 86_400_000),
        ]
        rows = build_rows(specs, [summary()], AS_OF_MS, Filters())
        self.assertEqual([row["instrument_name"] for row in rows], [instrument()["instrument_name"]])

    def test_rejects_invalid_quotes_without_mark_or_last_fallback(self) -> None:
        cases = [
            ({"bid_price": None}, "missing_or_nonpositive_quote"),
            ({"ask_price": 0}, "missing_or_nonpositive_quote"),
            ({"bid_price": float("nan")}, "missing_or_nonpositive_quote"),
            ({"bid_price": 0.06}, "crossed_quotes"),
            ({"bid_price": 0.01}, "wide_spread"),
            ({"bid_price": 1.1, "ask_price": 1.2}, "outside_project_model_bounds"),
            ({"underlying_price": None}, "invalid_underlying_price"),
            ({"underlying_price": 125_000}, "outside_atm_band"),
            ({"creation_timestamp": AS_OF_MS - 11_000}, "stale_summary"),
            ({"creation_timestamp": AS_OF_MS + 2_000}, "future_summary_timestamp"),
            ({"creation_timestamp": None}, "missing_summary_timestamp"),
        ]
        for changes, reason in cases:
            with self.subTest(reason=reason, changes=changes):
                row = build_rows([instrument()], [summary(**changes)], AS_OF_MS, Filters())[0]
                self.assertFalse(row["selected_for_iv"])
                self.assertIn(reason, row["exclusion_reasons"])
                if reason in {"missing_or_nonpositive_quote", "crossed_quotes"}:
                    self.assertIsNone(row["mid_btc"])

    def test_missing_book_and_expired_contract_are_rejected(self) -> None:
        row = build_rows([instrument()], [], AS_OF_MS, Filters())[0]
        self.assertFalse(row["selected_for_iv"])
        self.assertIn("missing_summary", row["exclusion_reasons"])
        after_expiry = EXPIRY_MS + 1_000
        row = build_rows([instrument()], [summary(creation_timestamp=after_expiry)], after_expiry, Filters())[0]
        self.assertIn("expired_contract", row["exclusion_reasons"])

    def test_atm_filter_is_symmetric_in_log_moneyness(self) -> None:
        for strike in (96_000, 104_000):
            row = build_rows([instrument(strike=strike)], [summary()], AS_OF_MS, Filters(max_log_moneyness=0.05))[0]
            self.assertTrue(row["selected_for_iv"])
        for strike in (90_000, 110_000):
            row = build_rows([instrument(strike=strike)], [summary()], AS_OF_MS, Filters(max_log_moneyness=0.05))[0]
            self.assertIn("outside_atm_band", row["exclusion_reasons"])

    def test_large_timestamp_span_rejects_incoherent_snapshot(self) -> None:
        specs = [instrument(), instrument("second", strike=101_000)]
        books = [summary(), summary("second", creation_timestamp=AS_OF_MS - 6_000)]
        rows = build_rows(specs, books, AS_OF_MS, Filters())
        self.assertTrue(all(not row["selected_for_iv"] for row in rows))
        self.assertTrue(all("snapshot_time_span_exceeded" in row["exclusion_reasons"] for row in rows))

    def test_rejects_duplicate_instruments_instead_of_weighting_twice(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_rows([instrument()], [summary(), summary()], AS_OF_MS, Filters())

    def test_full_export_retains_rejected_rows_and_raw_responses(self) -> None:
        client = Mock()
        client.get.side_effect = [
            capture([instrument(), instrument("second", strike=120_000)]),
            capture([summary(), summary("second")]),
        ]
        with tempfile.TemporaryDirectory() as temp:
            run_dir = collect_snapshot(client, Path(temp), Filters())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            with (run_dir / "all_calls.csv").open(newline="") as stream:
                all_rows = list(csv.DictReader(stream))
            with (run_dir / "atm_calls.csv").open(newline="") as stream:
                selected = list(csv.DictReader(stream))
            self.assertEqual(len(all_rows), 2)
            self.assertEqual(len(selected), 1)
            self.assertEqual(metadata["status"], "ok")
            self.assertEqual(metadata["selected_calls"], 1)
            self.assertEqual(metadata["reference_time_source"], "response.usOut")
            self.assertTrue((run_dir / "instruments_raw.json").is_file())
            self.assertTrue((run_dir / "summaries_raw.json").is_file())

    def test_empty_atm_export_has_headers_and_explicit_status(self) -> None:
        client = Mock()
        client.get.side_effect = [capture([instrument()]), capture([summary(bid_price=None)])]
        with tempfile.TemporaryDirectory() as temp:
            run_dir = collect_snapshot(client, Path(temp), Filters())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            with (run_dir / "atm_calls.csv").open(newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertIn("mid_btc", reader.fieldnames)
                self.assertEqual(list(reader), [])
            self.assertEqual(metadata["status"], "empty")

    def test_nonexistent_expiry_fails_without_substituting_another(self) -> None:
        client = Mock()
        client.get.side_effect = [capture([instrument()]), capture([summary()])]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Available expiries: 2026-12-25"):
                collect_snapshot(client, Path(temp), Filters(expiry=date(2026, 12, 24)))
            metadata = json.loads(next(Path(temp).glob("*/metadata.json")).read_text())
            self.assertEqual(metadata["status"], "failed")

    def test_transport_failure_retains_first_raw_response(self) -> None:
        client = Mock()
        client.get.side_effect = [capture([instrument()]), RuntimeError("network unavailable")]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(RuntimeError, "network unavailable"):
                collect_snapshot(client, Path(temp), Filters())
            run_dir = next(Path(temp).iterdir())
            self.assertTrue((run_dir / "instruments_raw.json").exists())
            self.assertEqual(json.loads((run_dir / "metadata.json").read_text())["status"], "failed")

    def test_freshness_falls_back_to_local_clock_without_server_time(self) -> None:
        client = Mock()
        market = copy.deepcopy(capture([summary()]))
        del market["response"]["usOut"]
        client.get.side_effect = [capture([instrument()]), market]
        with tempfile.TemporaryDirectory() as temp:
            run_dir = collect_snapshot(client, Path(temp), Filters())
            metadata = json.loads((run_dir / "metadata.json").read_text())
            self.assertEqual(metadata["reference_time_source"], "local_received_at_utc")
            self.assertTrue(any("local UTC clock" in warning for warning in metadata["warnings"]))


if __name__ == "__main__":
    unittest.main()
