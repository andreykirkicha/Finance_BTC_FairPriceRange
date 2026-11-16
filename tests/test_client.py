"""HTTP tests use synthetic responses and never contact the exchange."""

import io
import json
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from deribit_data.client import DeribitClient, DeribitError


def response(payload: object) -> Mock:
    """Build a context-manager response readable by json.load."""
    result = Mock()
    result.__enter__ = Mock(return_value=io.BytesIO(json.dumps(payload).encode()))
    result.__exit__ = Mock(return_value=False)
    return result


class ClientTests(unittest.TestCase):
    @patch("deribit_data.client.urlopen")
    def test_request_uses_public_production_api_and_retains_provenance(self, fetch: Mock) -> None:
        fetch.return_value = response({"jsonrpc": "2.0", "result": []})
        capture = DeribitClient().get("public/get_instruments", currency="BTC", expired="false")
        request = fetch.call_args.args[0]
        self.assertEqual(
            request.full_url, "https://www.deribit.com/api/v2/public/get_instruments?currency=BTC&expired=false"
        )
        self.assertEqual(capture["response"]["result"], [])
        self.assertIn("received_at_utc", capture)

    @patch("deribit_data.client.time.sleep")
    @patch("deribit_data.client.urlopen")
    def test_retries_transport_http_and_rpc_rate_limit_errors(self, fetch: Mock, sleep: Mock) -> None:
        failures = [
            URLError("temporary DNS failure"),
            HTTPError("url", 429, "rate limit", {}, None),
            HTTPError("url", 503, "unavailable", {}, None),
            response({"error": {"code": 10028, "message": "too_many_requests"}}),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                fetch.reset_mock()
                sleep.reset_mock()
                fetch.side_effect = [failure, response({"jsonrpc": "2.0", "result": []})]
                DeribitClient().get("public/get_instruments")
                self.assertEqual(fetch.call_count, 2)
                sleep.assert_called_once_with(1)

    @patch("deribit_data.client.time.sleep")
    @patch("deribit_data.client.urlopen")
    def test_permanent_errors_are_not_retried(self, fetch: Mock, sleep: Mock) -> None:
        failures = [
            HTTPError("url", 400, "bad request", {}, None),
            response({"error": {"code": -32602, "message": "invalid params"}}),
            response({"jsonrpc": "2.0", "result": [], "testnet": True}),
            response({"unexpected": []}),
            response([]),
        ]
        for failure in failures:
            with self.subTest(failure=failure):
                fetch.reset_mock()
                fetch.side_effect = None
                if isinstance(failure, Exception):
                    fetch.side_effect = failure
                else:
                    fetch.return_value = failure
                with self.assertRaises(DeribitError):
                    DeribitClient().get("public/get_instruments")
                self.assertEqual(fetch.call_count, 1)
        sleep.assert_not_called()

    @patch("deribit_data.client.time.sleep")
    @patch("deribit_data.client.urlopen", side_effect=URLError("offline"))
    def test_retry_budget_is_bounded(self, fetch: Mock, sleep: Mock) -> None:
        with self.assertRaisesRegex(DeribitError, "after 3 attempts"):
            DeribitClient().get("public/get_instruments")
        self.assertEqual(fetch.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_private_methods_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            DeribitClient().get("private/buy")


if __name__ == "__main__":
    unittest.main()
