"""Расчёт подразумеваемой волатильности набора котировок и учёт причин исключения."""

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from option_pricing.inverse_call import BrentConfig, implied_volatility

__all__ = ["IVFilters", "IVRow", "OptionQuote", "calculate_iv_rows"]


DAYS_PER_YEAR = 365.0
SECONDS_PER_DAY = 86_400.0
REASON_ORDER = (
    "invalid_timestamp",
    "expired_or_nonpositive_T",
    "missing_or_invalid_forward",
    "missing_or_invalid_strike",
    "missing_or_nonpositive_quote",
    "crossed_quote",
    "outside_model_bounds",
    "outside_atm",
    "lt_min_days_to_expiry",
    "no_bracket",
    "solver_error",
    "not_converged",
    "price_residual",
)


@dataclass(frozen=True, slots=True)
class OptionQuote:
    instrument_name: str
    expiry_utc: datetime | None
    snapshot_utc: datetime | None
    strike_usd: float | None
    forward_usd: float | None
    bid_btc: float | None
    ask_btc: float | None
    exchange_mid_iv: float | None = None


@dataclass(frozen=True, slots=True)
class IVFilters:
    max_log_moneyness: float = 0.15
    min_days_to_expiry: float = 1.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.max_log_moneyness) and self.max_log_moneyness > 0):
            raise ValueError("max_log_moneyness must be finite and positive")
        if not (math.isfinite(self.min_days_to_expiry) and self.min_days_to_expiry >= 0):
            raise ValueError("min_days_to_expiry must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class IVRow:
    instrument_name: str
    expiry_utc: datetime
    snapshot_utc: datetime
    strike_usd: float
    forward_usd: float
    days_to_expiry: float
    log_moneyness: float
    bid_btc: float
    ask_btc: float
    mid_btc: float
    iv_brent: float
    exchange_mid_iv: float | None

    @property
    def iv_brent_pct(self) -> float:
        return self.iv_brent * 100.0


def _calculate_row_iv(row: OptionQuote, filters: IVFilters, config: BrentConfig) -> tuple[IVRow | None, str | None]:
    expiry = row.expiry_utc
    snapshot = row.snapshot_utc
    if expiry is None or snapshot is None:
        return None, "invalid_timestamp"

    time_to_expiry = (expiry - snapshot).total_seconds() / (DAYS_PER_YEAR * SECONDS_PER_DAY)
    if time_to_expiry <= 0:
        return None, "expired_or_nonpositive_T"

    forward, strike = row.forward_usd, row.strike_usd
    if forward is None or not math.isfinite(forward) or forward <= 0:
        return None, "missing_or_invalid_forward"
    if strike is None or not math.isfinite(strike) or strike <= 0:
        return None, "missing_or_invalid_strike"

    bid, ask = row.bid_btc, row.ask_btc
    if bid is None or ask is None or not math.isfinite(bid) or not math.isfinite(ask) or bid <= 0 or ask <= 0:
        return None, "missing_or_nonpositive_quote"
    if bid > ask:
        return None, "crossed_quote"

    mid = (bid + ask) / 2.0
    if not max(1.0 - strike / forward, 0.0) < mid < 1.0:
        return None, "outside_model_bounds"

    log_moneyness = math.log(strike / forward)
    if abs(log_moneyness) > filters.max_log_moneyness:
        return None, "outside_atm"

    days_to_expiry = time_to_expiry * DAYS_PER_YEAR
    if days_to_expiry < filters.min_days_to_expiry:
        return None, "lt_min_days_to_expiry"

    result = implied_volatility(mid, forward, strike, time_to_expiry, config)
    if result.reason is not None:
        return None, result.reason
    if result.iv is None:
        return None, "solver_error"

    return (
        IVRow(
            instrument_name=row.instrument_name,
            expiry_utc=expiry,
            snapshot_utc=snapshot,
            strike_usd=strike,
            forward_usd=forward,
            days_to_expiry=days_to_expiry,
            log_moneyness=log_moneyness,
            bid_btc=bid,
            ask_btc=ask,
            mid_btc=mid,
            iv_brent=result.iv,
            exchange_mid_iv=row.exchange_mid_iv,
        ),
        None,
    )


def calculate_iv_rows(
    option_rows: Iterable[OptionQuote],
    filters: IVFilters | None = None,
    config: BrentConfig | None = None,
) -> tuple[list[IVRow], dict[str, object]]:
    filters = filters or IVFilters()
    config = config or BrentConfig()
    source_rows = list(option_rows)
    reasons: Counter[str] = Counter()
    rows: list[IVRow] = []

    for row in source_rows:
        iv_row, reason = _calculate_row_iv(row, filters, config)
        if reason is not None:
            reasons[reason] += 1
            continue
        if iv_row is None:
            reasons["solver_error"] += 1
            continue
        rows.append(iv_row)

    rows.sort(key=lambda row: (row.expiry_utc, row.strike_usd, row.instrument_name))
    excluded = {reason: reasons[reason] for reason in REASON_ORDER if reasons[reason]}
    for reason in reasons:
        if reason not in excluded:
            excluded[reason] = reasons[reason]

    counts: dict[str, object] = {
        "input": len(source_rows),
        "excluded": excluded,
        "solved": len(rows),
    }
    if sum(excluded.values()) + len(rows) != len(source_rows):
        raise AssertionError("exclusion counters do not add up to the input count")
    return rows, counts
