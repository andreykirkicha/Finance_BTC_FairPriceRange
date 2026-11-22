"""Расчёт цены обратного call-опциона и восстановление волатильности методом Брента.

Цена выражена в BTC, волатильность — в долях, срок — в годах.
Используется форвард F: C = N(d1) - (K/F) N(d2), без отдельного дисконтирования.
"""

import math
from dataclasses import dataclass

from scipy.optimize import brentq

__all__ = ["BrentConfig", "IVResult", "implied_volatility", "inverse_call_price"]


@dataclass(frozen=True, slots=True)
class BrentConfig:
    low: float = 1e-8
    high: float = 10.0
    xtol: float = 1e-10
    rtol: float = 1e-12
    maxiter: int = 100
    price_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not (math.isfinite(self.low) and math.isfinite(self.high) and 0 < self.low < self.high):
            raise ValueError("Brent bracket must satisfy 0 < low < high")
        if not (math.isfinite(self.xtol) and self.xtol > 0):
            raise ValueError("xtol must be finite and positive")
        if not (math.isfinite(self.rtol) and 0 <= self.rtol < 1):
            raise ValueError("rtol must be finite and in [0, 1)")
        if self.maxiter < 1:
            raise ValueError("maxiter must be at least 1")
        if not (math.isfinite(self.price_tolerance) and self.price_tolerance > 0):
            raise ValueError("price_tolerance must be finite and positive")


@dataclass(frozen=True, slots=True)
class IVResult:
    iv: float | None
    price_error: float | None
    iterations: int
    function_calls: int
    converged: bool
    reason: str | None = None


def _normal_cdf(value: float) -> float:
    return 0.5 * math.erfc(-value / math.sqrt(2.0))


def inverse_call_price(forward: float, strike: float, time_to_expiry: float, volatility: float) -> float:
    values = (forward, strike, time_to_expiry, volatility)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("forward, strike, time_to_expiry, and volatility must be finite")
    if forward <= 0 or strike <= 0 or time_to_expiry <= 0:
        raise ValueError("forward, strike, and time_to_expiry must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    intrinsic = max(1.0 - strike / forward, 0.0)
    if volatility == 0:
        return intrinsic

    sqrt_t = math.sqrt(time_to_expiry)
    vol_sqrt_t = volatility * sqrt_t
    d1 = (math.log(forward / strike) + 0.5 * vol_sqrt_t * vol_sqrt_t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return _normal_cdf(d1) - (strike / forward) * _normal_cdf(d2)


def _failure(reason: str, *, error: float | None = None) -> IVResult:
    return IVResult(
        iv=None,
        price_error=error,
        iterations=0,
        function_calls=0,
        converged=False,
        reason=reason,
    )


def implied_volatility(
    mid_price: float,
    forward: float,
    strike: float,
    time_to_expiry: float,
    config: BrentConfig | None = None,
) -> IVResult:
    config = config or BrentConfig()
    if not all(math.isfinite(value) for value in (mid_price, forward, strike, time_to_expiry)):
        raise ValueError("mid_price, forward, strike, and time_to_expiry must be finite")
    if forward <= 0 or strike <= 0 or time_to_expiry <= 0:
        raise ValueError("forward, strike, and time_to_expiry must be positive")

    intrinsic = max(1.0 - strike / forward, 0.0)
    if not intrinsic < mid_price < 1.0:
        return _failure("outside_model_bounds")

    def price_difference(volatility: float) -> float:
        return inverse_call_price(forward, strike, time_to_expiry, volatility) - mid_price

    low, high = config.low, config.high
    f_low, f_high = price_difference(low), price_difference(high)
    if f_low == 0:
        return IVResult(iv=low, price_error=0.0, iterations=0, function_calls=0, converged=True)
    if f_high == 0:
        return IVResult(iv=high, price_error=0.0, iterations=0, function_calls=0, converged=True)
    if f_low * f_high > 0:
        return _failure("no_bracket", error=min(abs(f_low), abs(f_high)))

    try:
        root, details = brentq(
            price_difference,
            low,
            high,
            xtol=config.xtol,
            rtol=config.rtol,
            maxiter=config.maxiter,
            full_output=True,
            disp=False,
        )
    except ValueError as error:
        if "no sign change" in str(error):
            return _failure("no_bracket")
        return _failure("solver_error")
    except RuntimeError:
        return _failure("solver_error")

    price_error = abs(price_difference(root))
    result = IVResult(
        iv=float(root),
        price_error=price_error,
        iterations=int(details.iterations),
        function_calls=int(details.function_calls),
        converged=bool(details.converged),
    )
    if not details.converged:
        return IVResult(
            iv=result.iv,
            price_error=price_error,
            iterations=result.iterations,
            function_calls=result.function_calls,
            converged=False,
            reason="not_converged",
        )
    if price_error > config.price_tolerance:
        return IVResult(
            iv=result.iv,
            price_error=price_error,
            iterations=result.iterations,
            function_calls=result.function_calls,
            converged=True,
            reason="price_residual",
        )
    return result
