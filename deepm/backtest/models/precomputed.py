"""Backtest adapters for signals computed outside the training framework, and capital-weight benchmarks.

The backtester's convention (see ``SignalBacktestJob.calc_returns``): a position p earns
``p * target`` with ``target = r_{t+1} * vs_factor`` (``vs_factor = 1 / daily vol``), pays
``|d(p * vs_factor)| * half_spread_bp * 1e-4``, and the portfolio is the mean over tickers.

PrecomputedSignal
    Reads a wide (date x ticker) parquet of positions already in those risk units, e.g. an ensemble
    written by ``scripts/make_ensemble.py``.
CapitalWeights
    Long-only capital-weight benchmarks (equal weight, inverse vol, fixed weights such as 60/40).
    A capital weight w is converted to ``p = N * w / vs_factor`` so that the backtester's
    ``mean_i(p * target) = sum_i w_i * r_i`` and its cost ``|d(p * vs_factor)| * tc = N * |dw| * tc``
    (i.e. capital turnover). Weights are renormalised over the tickers available that day.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from deepm.backtest.models.traditional import (
    AbstractBenchmark,
    _long_to_wide,
    _prepare_long_panel,
)

ROOT = Path(__file__).resolve().parents[3]


class PrecomputedSignal(AbstractBenchmark):
    """Positions read from ``signal_file`` (path relative to the repo root)."""

    def __init__(self, signal_file: str, **kwargs):
        super().__init__(**kwargs)
        self.signal_file = ROOT / signal_file

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        pos = pd.read_parquet(self.signal_file)
        pos.index = pd.to_datetime(pos.index)
        missing = sorted(set(tickers) - set(pos.columns))
        if missing:
            raise KeyError(f"{self.signal_file} has no positions for {missing}")
        pos = pos.loc[pos.index >= pd.to_datetime(start_date), list(tickers)].fillna(0.0)
        pos.index.name = "date"
        pos.columns.name = "ticker"
        return pos


class CapitalWeights(AbstractBenchmark):
    """Long-only capital weights: ``mode`` in {"equal", "inverse_vol", "fixed"} (fixed uses ``weights``)."""

    def __init__(self, mode: str = "equal", weights: Optional[Dict[str, float]] = None, **kwargs):
        super().__init__(**kwargs)
        if mode not in ("equal", "inverse_vol", "fixed"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "fixed" and not weights:
            raise ValueError("mode='fixed' needs weights")
        self.mode = mode
        self.weights = weights or {}

    def predict(self, data: pd.DataFrame, tickers: List[str], start_date: str, **kwargs) -> pd.DataFrame:
        df = _prepare_long_panel(data, tickers, start_date, required_cols=["vs_factor"])
        vs = _long_to_wide(df, "vs_factor", tickers).replace(0.0, np.nan)
        # A ticker counts as available over its whole listed life: days without a feature row (unchanged close,
        # dropped by build_features) keep the previous vol and weight instead of renormalising the others
        # (renormalising on those days would create spurious turnover). Those days have no target, so no PnL.
        last = vs.apply(lambda s: s.last_valid_index())
        vs = vs.ffill()
        for t, d in last.items():
            if d is not None:
                vs.loc[vs.index > d, t] = np.nan
        avail = vs > 0
        if self.mode == "equal":
            w = avail.astype(float)
        elif self.mode == "inverse_vol":
            w = vs.where(avail, 0.0)  # vs_factor = 1 / daily vol
        else:
            fixed = pd.Series(self.weights, dtype=float).reindex(tickers).fillna(0.0)
            w = avail.astype(float).mul(fixed, axis=1)
        w = w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
        pos = (len(tickers) * w / vs.where(avail)).fillna(0.0)
        pos.columns.name = "ticker"
        return pos
