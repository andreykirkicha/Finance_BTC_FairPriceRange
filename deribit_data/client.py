"""Read-only access to the public production API, using the standard library."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

__all__ = ["API_URL", "DeribitClient", "DeribitError"]

API_URL = "https://www.deribit.com/api/v2"
LOGGER = logging.getLogger(__name__)


class DeribitError(RuntimeError):
    """A public API request failed or returned an invalid response."""


class DeribitClient:
    """Fetch public market data without authentication or trading permissions."""

    def __init__(self, timeout: float = 15, attempts: int = 3) -> None:
        """Set the timeout per attempt and the maximum number of attempts."""
        if timeout <= 0 or attempts < 1:
            raise ValueError("timeout must be positive; attempts must be at least 1")
        self.timeout = timeout
        self.attempts = attempts

    def get(self, method: str, **params: str) -> dict[str, Any]:
        """Return the full response with request parameters and local UTC times.

        Retry transport failures, HTTP 429/5xx, and JSON-RPC rate-limit errors.
        All other API errors fail immediately. Only public GETs are allowed.
        """
        if not method.startswith("public/"):
            raise ValueError("Only public methods are supported")
        request = Request(
            f"{API_URL}/{method}?{urlencode(params)}",
            headers={"Accept": "application/json", "User-Agent": "btc-option-research/0.1"},
        )
        for attempt in range(self.attempts):
            started = datetime.now(timezone.utc)
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                received = datetime.now(timezone.utc)
            except HTTPError as error:
                status = error.code
                error.close()
                if status != 429 and not 500 <= status < 600:
                    raise DeribitError(f"{method}: HTTP {status}") from error
                failure = f"HTTP {status}"
            except (URLError, TimeoutError, ConnectionError) as error:
                failure = str(error)
            except (ValueError, UnicodeError) as error:
                raise DeribitError(f"{method}: response is not valid JSON") from error
            else:
                if not isinstance(payload, dict):
                    raise DeribitError(f"{method}: expected a JSON object")
                if payload.get("testnet"):
                    raise DeribitError("Production request returned testnet data")
                if "error" not in payload:
                    if payload.get("jsonrpc") != "2.0" or "result" not in payload:
                        raise DeribitError(f"{method}: invalid JSON-RPC response")
                    return {
                        "method": method,
                        "params": params,
                        "request_started_at_utc": started.isoformat(),
                        "received_at_utc": received.isoformat(),
                        "response": payload,
                    }
                error_data = payload["error"]
                if not isinstance(error_data, dict) or error_data.get("code") != 10028:
                    raise DeribitError(f"{method}: API error {error_data}")
                failure = f"API rate limit: {error_data}"
            if attempt + 1 == self.attempts:
                raise DeribitError(f"{method}: failed after {self.attempts} attempts: {failure}")
            delay = min(2**attempt, 8)
            LOGGER.warning("%s: %s; retry in %s s", method, failure, delay)
            time.sleep(delay)
        raise AssertionError("Unreachable")
