"""
factors/momentum.py -- Price and earnings-revision momentum factors
Part of alpha-factor-research. Pure Python, no external dependencies.

Implements:
  - Cross-sectional price momentum (12-1, 6-1, 3-1, 1m)
  - Skip-1-month convention to avoid short-term reversal
  - Earnings revision momentum (SUE, analyst revision)
  - Z-score normalisation and rank normalisation
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Date = str         # ISO date string "YYYY-MM-DD"
Ticker = str
PriceMatrix = Dict[Date, Dict[Ticker, float]]   # date -> {ticker: price}
FactorScores = Dict[Date, Dict[Ticker, float]]  # date -> {ticker: score}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _rank_normalise(scores: Dict[Ticker, float]) -> Dict[Ticker, float]:
    """Convert raw scores to cross-sectional percentile ranks in [-1, +1]."""
    tickers = [t for t, s in scores.items() if s is not None and not math.isnan(s)]
    if not tickers:
        return {}
    sorted_t = sorted(tickers, key=lambda t: scores[t])
    n = len(sorted_t)
    return {t: 2.0 * i / (n - 1) - 1.0 if n > 1 else 0.0
            for i, t in enumerate(sorted_t)}


def _zscore_normalise(scores: Dict[Ticker, float]) -> Dict[Ticker, float]:
    """Cross-sectional z-score: (score - mean) / std, winsorised at ±3."""
    vals = [v for v in scores.values() if v is not None and not math.isnan(v)]
    if not vals:
        return {}
    m, s = _mean(vals), _std(vals)
    if s == 0:
        return {t: 0.0 for t in scores}
    result = {}
    for t, v in scores.items():
        if v is None or math.isnan(v):
            continue
        z = (v - m) / s
        result[t] = max(-3.0, min(3.0, z))
    return result


def _log_return(p_start: float, p_end: float) -> Optional[float]:
    if p_start is None or p_end is None or p_start <= 0 or p_end <= 0:
        return None
    return math.log(p_end / p_start)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PriceSeries:
    """Sorted list of (date, price) for one ticker."""
    ticker: Ticker
    data: List[Tuple[Date, float]] = field(default_factory=list)  # sorted asc

    def price_at_offset(self, idx: int, offset: int) -> Optional[float]:
        """Price at index idx - offset (negative offset = look back)."""
        target = idx + offset
        if 0 <= target < len(self.data):
            return self.data[target][1]
        return None

    def __len__(self) -> int:
        return len(self.data)


# ---------------------------------------------------------------------------
# MomentumFactor
# ---------------------------------------------------------------------------

@dataclass
class MomentumFactor:
    """
    Cross-sectional price momentum factor.

    Standard specification: 12-month look-back, skip most-recent month.
      lookback=252 trading days, skip=21 trading days

    Score for ticker i at date t:
      mom_i = log(P_{t-skip} / P_{t-lookback})
    """
    lookback: int = 252    # trading days for cumulative return window
    skip: int = 21         # days to skip (short-term reversal avoidance)
    normalise: str = "zscore"  # "zscore" | "rank" | "raw"

    def compute(self, prices: PriceMatrix) -> FactorScores:
        """
        Compute momentum scores for every date.

        Parameters
        ----------
        prices : {date: {ticker: price}} -- full price history

        Returns
        -------
        {date: {ticker: momentum_score}}
        """
        dates = sorted(prices.keys())
        tickers = sorted({t for p in prices.values() for t in p})

        # Build per-ticker price series indexed by position
        series: Dict[Ticker, List[Tuple[Date, float]]] = {t: [] for t in tickers}
        for d in dates:
            for t in tickers:
                if t in prices[d]:
                    series[t].append((d, prices[d][t]))

        # For each date, compute look-back return for each ticker
        result: FactorScores = {}
        for i, date in enumerate(dates):
            raw: Dict[Ticker, float] = {}
            for t in tickers:
                ts = series[t]
                # Find index of current date in this ticker's series
                # (simple linear scan — fine for small universes)
                t_idx = next((j for j, (d, _) in enumerate(ts) if d == date), None)
                if t_idx is None:
                    continue
                p_end = ts[t_idx - self.skip][1] if t_idx >= self.skip else None
                p_start = ts[t_idx - self.lookback][1] if t_idx >= self.lookback else None
                r = _log_return(p_start, p_end)
                if r is not None:
                    raw[t] = r

            if not raw:
                continue

            if self.normalise == "zscore":
                result[date] = _zscore_normalise(raw)
            elif self.normalise == "rank":
                result[date] = _rank_normalise(raw)
            else:
                result[date] = raw

        return result

    def short_term_reversal(self, prices: PriceMatrix) -> FactorScores:
        """1-month reversal factor (negative of 1m momentum)."""
        reversal = MomentumFactor(lookback=self.skip, skip=1, normalise=self.normalise)
        scores = reversal.compute(prices)
        # Negate: reversal is the opposite of recent momentum
        return {d: {t: -s for t, s in smap.items()} for d, smap in scores.items()}


# ---------------------------------------------------------------------------
# Earnings revision momentum
# ---------------------------------------------------------------------------

@dataclass
class EarningsRevisionFactor:
    """
    Earnings revision momentum: standardised unexpected earnings (SUE).

    SUE_i = (EPS_i - mean(EPS_i_hist)) / std(EPS_i_hist)

    Parameters
    ----------
    window : number of quarters for historical mean/std
    """
    window: int = 8   # quarters

    def compute(
        self,
        eps_history: Dict[Ticker, List[float]],   # ticker -> list of quarterly EPS (ascending)
        surprise_actual: Dict[Ticker, float],      # ticker -> most-recent EPS
    ) -> Dict[Ticker, float]:
        """
        Compute SUE scores for one cross-section.

        Parameters
        ----------
        eps_history : historical EPS per ticker (most recent last)
        surprise_actual : actual EPS just announced

        Returns
        -------
        {ticker: sue_score}
        """
        raw: Dict[Ticker, float] = {}
        for t, actual in surprise_actual.items():
            hist = eps_history.get(t, [])
            if len(hist) < 2:
                continue
            recent = hist[-self.window:]
            m = _mean(recent)
            s = _std(recent)
            if s < 1e-9:
                continue
            raw[t] = (actual - m) / s

        return _zscore_normalise(raw)


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random
    rng = random.Random(42)

    # Simulate 300 days of prices for 5 tickers
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    prices: PriceMatrix = {}
    current = {t: 100.0 + rng.uniform(-20, 20) for t in tickers}
    dates = [f"2024-{(i//21+1):02d}-{(i%21+1):02d}" for i in range(300)]

    for d in dates:
        prices[d] = {}
        for t in tickers:
            current[t] *= math.exp(rng.gauss(0.0003, 0.015))
            prices[d][t] = round(current[t], 4)

    factor = MomentumFactor(lookback=252, skip=21, normalise="zscore")
    scores = factor.compute(prices)

    last_date = dates[-1]
    if last_date in scores:
        print("=" * 45)
        print("  12-1 Momentum Factor Scores (last date)")
        print("=" * 45)
        for t, s in sorted(scores[last_date].items(), key=lambda x: -x[1]):
            bar = "#" * int(abs(s) * 10)
            sign = "+" if s >= 0 else "-"
            print(f"  {t:<8} {sign}{bar:<12} {s:+.3f}")
        print("=" * 45)
