"""Unit tests for the ETF data layer and the capital-weight benchmark, on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

from deepm.backtest.models.precomputed import CapitalWeights
from deepm.data.etf import close_panel, excess_prices, next_open_target

DATES = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08"])  # Fri -> Mon gap


def rate(value):
    return pd.Series(value, index=pd.date_range("2019-12-01", "2020-02-01"), dtype=float)


def test_zero_cash_rate_keeps_returns():
    px = pd.DataFrame({"A": [100.0, 101.0, 99.0, 102.0, 103.0]}, index=DATES)
    xs = excess_prices(px, rate(0.0))
    pd.testing.assert_series_equal(xs["A"].pct_change(), px["A"].pct_change(), check_names=False)


def test_cash_rate_subtracted_act360_with_weekend():
    px = pd.DataFrame({"A": [100.0, 101.0, 99.0, 102.0, 103.0]}, index=DATES)
    xs = excess_prices(px, rate(3.6))  # 3.6% a year = 0.01% per calendar day on Act/360
    days = np.array([1, 3, 1, 1])  # Jan 3, Jan 6 (over the weekend), Jan 7, Jan 8
    expected = px["A"].pct_change().iloc[1:].to_numpy() - 0.036 * days / 360
    np.testing.assert_allclose(xs["A"].pct_change().iloc[1:].to_numpy(), expected, rtol=0, atol=1e-12)


def test_unchanged_close_day_charges_cash_on_next_move():
    px = pd.DataFrame({"A": [100.0, 101.0, 101.0, 102.0, 103.0]}, index=DATES)  # Jan 6 unchanged
    xs = excess_prices(px, rate(3.6))["A"]
    assert xs.iloc[2] == xs.iloc[1]  # stays unchanged, so the row is dropped later like the raw one
    r = xs.iloc[3] / xs.iloc[2] - 1
    assert r == pytest.approx(102 / 101 - 1 - 0.036 * (3 + 1) / 360, abs=1e-12)


def test_excess_prices_keep_missing_before_listing():
    px = pd.DataFrame({"A": [100.0, 101.0, 99.0, 102.0, 103.0], "B": [np.nan, np.nan, 50.0, 51.0, 52.0]}, index=DATES)
    xs = excess_prices(px, rate(1.0))
    assert xs["B"].iloc[:2].isna().all() and xs["B"].iloc[2] == 50.0


def _prices(opens):
    return pd.DataFrame({"date": DATES, "ticker": "A", "open": opens, "close": [100.0, 101.0, 102.0, 103.0, 104.0]})


def _feats(dates, vs=2.0):
    return pd.DataFrame({"ticker": "A", "target": 0.5, "vs_factor": vs}, index=pd.DatetimeIndex(dates, name="date"))


def test_next_open_target_one_day():
    opens = [10.0, 11.0, 12.1, 12.0, 13.0]
    out = next_open_target(_feats(DATES[:3]), _prices(opens), rate(0.0))
    # decision at the close of day t: enter at open t+1, exit at open t+2; scaled by vs_factor
    assert out["target"].iloc[0] == pytest.approx((12.1 / 11.0 - 1) * 2.0)
    assert out["target"].iloc[1] == pytest.approx((12.0 / 12.1 - 1) * 2.0)
    assert out["target_close"].iloc[0] == 0.5


def test_next_open_target_holds_until_next_feature_row():
    opens = [10.0, 11.0, 12.1, 12.0, 13.0]
    feats = _feats([DATES[0], DATES[2], DATES[3]], vs=1.0)  # Jan 3 had no feature row (unchanged close)
    out = next_open_target(feats, _prices(opens), rate(0.0))
    assert out["target"].iloc[0] == pytest.approx(12.0 / 11.0 - 1)  # open Jan 3 -> open Jan 7


def test_next_open_target_is_clipped_and_last_row_missing():
    opens = [10.0, 11.0, 100.0, 100.0, 100.0]
    out = next_open_target(_feats(DATES, vs=50.0), _prices(opens), rate(0.0))
    assert out["target"].iloc[0] == 20.0
    assert np.isnan(out["target"].iloc[-1])


def test_close_panel_is_wide_and_sorted():
    long = pd.DataFrame({"date": list(DATES[::-1]) * 1, "ticker": "A", "open": 1.0, "close": [5.0, 4.0, 3.0, 2.0, 1.0]})
    wide = close_panel(long)
    assert list(wide.columns) == ["A"] and wide.index.is_monotonic_increasing and wide["A"].iloc[0] == 1.0


def test_capital_weights_equal_reproduce_average_return():
    idx = pd.DatetimeIndex(DATES, name="date")
    vs = {"A": 50.0, "B": 100.0}
    rets = {"A": [0.01, -0.02, 0.005, 0.0, 0.01], "B": [0.002, 0.001, -0.004, 0.003, 0.0]}
    data = pd.concat([pd.DataFrame({"ticker": t, "vs_factor": vs[t], "target": np.array(rets[t]) * vs[t]}, index=idx)
                      for t in vs])
    pos = CapitalWeights(mode="equal").predict(data, ["A", "B"], "2020-01-01")
    target = data.pivot_table(index=data.index, columns="ticker", values="target")
    pnl = (pos * target).mean(axis=1)  # the backtester's portfolio return
    expected = (pd.DataFrame(rets, index=idx)).mean(axis=1)
    np.testing.assert_allclose(pnl.to_numpy(), expected.to_numpy(), atol=1e-12)
