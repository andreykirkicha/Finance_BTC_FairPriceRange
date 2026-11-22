"""Offline tests for inverse-call pricing and Brent IV recovery."""

import math
import unittest

from deribit_data.iv import BrentConfig, implied_volatility, inverse_call_price


class InverseCallPriceTests(unittest.TestCase):
    def test_zero_volatility_returns_intrinsic_value(self) -> None:
        self.assertAlmostEqual(inverse_call_price(100_000, 80_000, 0.5, 0), 0.2)
        self.assertEqual(inverse_call_price(100_000, 120_000, 0.5, 0), 0.0)

    def test_matches_pdf_example(self) -> None:
        price = inverse_call_price(68_862.57, 42_000, 0.7718, 0.8031)
        self.assertAlmostEqual(price, 0.4660016, places=7)

    def test_price_is_monotonic_in_volatility(self) -> None:
        prices = [inverse_call_price(100_000, 95_000, 0.25, sigma) for sigma in (0.1, 0.3, 0.6, 1.0)]
        self.assertEqual(prices, sorted(prices))
        self.assertGreater(prices[-1], prices[0])

    def test_rejects_invalid_inputs(self) -> None:
        for arguments in ((0, 100, 1, 0.5), (100, 0, 1, 0.5), (100, 100, 0, 0.5), (100, 100, 1, -0.1)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                inverse_call_price(*arguments)


class ImpliedVolatilityTests(unittest.TestCase):
    def test_round_trip_recovers_volatility(self) -> None:
        cases = (
            (100_000, 100_000, 30 / 365, 0.65),
            (68_862.57, 42_000, 0.7718, 0.8031),
            (68_957.59, 76_000, 2 / 365, 0.60),
        )
        for forward, strike, time_to_expiry, expected in cases:
            with self.subTest(expected=expected):
                mid = inverse_call_price(forward, strike, time_to_expiry, expected)
                result = implied_volatility(mid, forward, strike, time_to_expiry)
                self.assertTrue(result.converged)
                self.assertIsNone(result.reason)
                self.assertIsNotNone(result.iv)
                self.assertAlmostEqual(result.iv, expected, places=7)
                self.assertLessEqual(result.price_error, 1e-10)

    def test_rejects_quote_outside_model_bounds(self) -> None:
        result = implied_volatility(1.1, 100_000, 120_000, 0.25)
        self.assertEqual(result.reason, "outside_model_bounds")
        self.assertIsNone(result.iv)

    def test_reports_missing_bracket(self) -> None:
        forward = strike = 100_000
        time_to_expiry = 0.25
        mid = inverse_call_price(forward, strike, time_to_expiry, 0.5)
        result = implied_volatility(mid, forward, strike, time_to_expiry, BrentConfig(high=0.01))
        self.assertEqual(result.reason, "no_bracket")
        self.assertFalse(result.converged)

    def test_reports_price_residual(self) -> None:
        forward = strike = 100_000
        time_to_expiry = 0.25
        mid = inverse_call_price(forward, strike, time_to_expiry, 0.5)
        result = implied_volatility(
            mid,
            forward,
            strike,
            time_to_expiry,
            BrentConfig(xtol=1e-3, price_tolerance=1e-15),
        )
        self.assertEqual(result.reason, "price_residual")
        self.assertIsNotNone(result.iv)

    def test_brent_config_is_validated(self) -> None:
        for changes in ({"low": 0}, {"low": 2, "high": 1}, {"xtol": 0}, {"rtol": 1}, {"maxiter": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                BrentConfig(**changes)

    def test_price_error_is_finite(self) -> None:
        mid = inverse_call_price(100_000, 100_000, 0.25, 0.5)
        result = implied_volatility(mid, 100_000, 100_000, 0.25)
        self.assertTrue(math.isfinite(result.price_error))


if __name__ == "__main__":
    unittest.main()
