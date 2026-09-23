# alpha-factor-research

Cross-sectional alpha factor construction, IC analysis, and portfolio tilts — pure Python.

## Features

- **Momentum factors**: 1m, 3m, 6m, 12m price momentum with skip-1-month convention
- **Value factors**: earnings yield, book-to-market, sales-to-price, cash flow yield
- **Quality factors**: ROE, ROA, gross margin stability, accruals ratio
- **Low-volatility factors**: realised vol, idiosyncratic vol, beta
- **IC analysis**: information coefficient vs. forward returns (Spearman rank)
- **IC decay**: rolling IC over multiple holding periods
- **Factor combination**: equal-weight, IC-weighted, and regression-based blending
- **Turnover analysis**: factor rank stability across rebalance dates

## Quick Start

```python
from factors.momentum import MomentumFactor
from factors.value import ValueFactor
from analysis.ic import information_coefficient, ic_summary

# Build 12-1 price momentum factor
mom = MomentumFactor(lookback=252, skip=21)
scores = mom.compute(prices)          # {date: {ticker: score}}

# Evaluate predictiveness
fwd_returns = compute_forward_returns(prices, holding_period=21)
ic_series = information_coefficient(scores, fwd_returns)
print(ic_summary(ic_series))
# IC mean: 0.042 | IC std: 0.089 | IR: 0.47 | Hit rate: 56.2%
```

## Project Structure

```
alpha-factor-research/
├── factors/
│   ├── momentum.py     # 1/3/6/12m price momentum, earnings revision momentum
│   ├── value.py        # E/P, B/P, S/P, CF/P, dividend yield
│   ├── quality.py      # ROE, ROA, gross margin, accruals, leverage
│   └── low_vol.py      # realized vol, idiosyncratic vol, market beta
├── analysis/
│   ├── ic.py           # Spearman IC, IC decay, t-stat, hit rate
│   └── turnover.py     # rank correlation across periods, trading cost
└── portfolio/
    └── tilt.py         # z-score normalisation, factor blending, long/short construction
```

## References

- Barra Risk Model Handbook (MSCI)
- Asness, Frazzini, Pedersen (2019) — *Quality Minus Junk*
- Novy-Marx (2013) — *The Other Side of Value*
- Frazzini & Pedersen (2014) — *Betting Against Beta*
