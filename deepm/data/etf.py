"""ETF dataset construction: excess-of-cash prices and next-open execution targets.

Futures returns (what the DeePM paper trades) are excess returns, while ETF total returns also contain the
cash rate earned on the fund's collateral. To trade ETFs on the same footing this module

1. subtracts the overnight cash rate from the ETF total-return series (``excess_prices``), and
2. replaces the close-to-close training target with an open-to-open one (``next_open_target``): a decision
   made at the close of day t trades in the opening auction of t+1, which is what a real order placed after
   the close can achieve.

Input prices are long format with columns ``date, ticker, open, close`` where ``close`` is adjusted for
splits and dividends (a total-return index) and ``open`` is adjusted by the same factor. The cash rate is an
annualised percentage (e.g. FRED ``DFF``) indexed by date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_MAX_ABS = 20.0  # same clip as deepm/data/build_features.py


def load_prices(path: str) -> pd.DataFrame:
    """Read the long price file (.parquet or .csv) and check its columns."""
    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    missing = {"date", "ticker", "open", "close"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)} (need date, ticker, open, close)")
    df = df[["date", "ticker", "open", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["ticker"] = df["ticker"].astype(str)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def load_cash_rate(path: str) -> pd.Series:
    """Read a two-column CSV (date, annual rate in percent); FRED downloads work as they are."""
    rate = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return pd.to_numeric(rate, errors="coerce").dropna().sort_index()


def close_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide adjusted-close panel (date x ticker), NaN before a ticker's first price."""
    px = prices[prices["close"] > 0]
    wide = px.pivot(index="date", columns="ticker", values="close").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def excess_prices(prices: pd.DataFrame, cash_rate: pd.Series) -> pd.DataFrame:
    """Excess-of-cash price index: r_xs_t = P_t / P_{t-1} - 1 - rf_t with rf_t = rate_{t-1} / 100 * days / 360.

    The rate is the last published value, lagged one trading day (known when the position is held); accrual is
    Act/360 over the calendar days since the previous trading day. Days with an unchanged close are dropped later
    by ``prepare_features``; they stay unchanged here and the cash accrued over them is charged on the next day
    with a real price move, so the same rows survive and no carry is lost.
    """
    prices = prices.sort_index()
    idx = prices.index
    rate = cash_rate.reindex(cash_rate.index.union(idx)).ffill().reindex(idx)
    days = pd.Series(idx, index=idx).diff().dt.days
    rf = (rate.shift(1) / 100.0 * days / 360.0).fillna(0.0)

    rets = prices.pct_change(fill_method=None)
    rf_cum = rf.cumsum()

    out = pd.DataFrame(index=idx, columns=prices.columns, dtype=float)
    for t in prices.columns:
        s = prices[t]
        first = s.first_valid_index()
        raw = rets[t].loc[first:]
        moved = raw.notna() & (raw != 0)
        moved.iloc[0] = True
        c = rf_cum.loc[first:]
        rf_eff = c - c.where(moved).ffill().shift(1).fillna(c.iloc[0])
        r = (raw - rf_eff).where(moved, 0.0).fillna(0.0)
        r.iloc[0] = 0.0
        out.loc[first:, t] = s.loc[first] * (1.0 + r).cumprod()
        out.loc[s.isna(), t] = np.nan
    return out


def next_open_target(feats: pd.DataFrame, prices: pd.DataFrame, cash_rate: pd.Series) -> pd.DataFrame:
    """Replace ``target`` (close t -> close t+1) with the next-open excess return, scaled and clipped like deepm.

    For each feature row t of a ticker the position is entered at the open after t and held until the open after
    the ticker's next feature row t' (rows with an unchanged close are dropped by ``prepare_features``, and the
    backtester holds the position through them). Steps follow each ticker's own trading calendar. The original
    close-to-close target is kept as ``target_close``.
    """
    feats = feats.copy()
    feats.index = pd.to_datetime(feats.index)
    feats.index.name = "date"
    px = prices[(prices["open"] > 0) & (prices["close"] > 0)].sort_values(["ticker", "date"])
    feat_dates = {t: pd.DatetimeIndex(g.index).sort_values() for t, g in feats.groupby("ticker")}

    def per_ticker(g):
        fd = feat_dates.get(g.name)
        if fd is None:
            return pd.DataFrame({"date": pd.DatetimeIndex([]), "r_open": []})
        dates = pd.DatetimeIndex(g["date"].to_numpy())
        o = g["open"].to_numpy()
        rate = cash_rate.reindex(cash_rate.index.union(dates)).ffill().reindex(dates).to_numpy()
        step = np.r_[0.0, rate[:-1] / 100.0 * np.diff(dates.values).astype("timedelta64[D]").astype(float) / 360.0]
        accr = np.cumsum(step)
        pos = pd.Series(np.arange(len(dates)), index=dates)
        i_t = pos.reindex(fd).to_numpy(dtype=float)
        i_entry = i_t + 1
        i_exit = np.r_[i_t[1:], np.nan] + 1
        ok = ~np.isnan(i_entry) & ~np.isnan(i_exit) & (i_exit < len(dates))
        r = np.full(len(fd), np.nan)
        a, b = i_entry[ok].astype(int), i_exit[ok].astype(int)
        r[ok] = o[b] / o[a] - 1.0 - (accr[b] - accr[a])
        return pd.DataFrame({"date": fd, "r_open": r})

    r = px.groupby("ticker", group_keys=True)[["date", "open"]].apply(per_ticker).reset_index(level=0)
    out = feats.reset_index().merge(r, on=["date", "ticker"], how="left")
    out["target_close"] = out["target"]
    out["target"] = (out["r_open"] * out["vs_factor"]).clip(-TARGET_MAX_ABS, TARGET_MAX_ABS)
    return out.drop(columns=["r_open"]).set_index("date")
