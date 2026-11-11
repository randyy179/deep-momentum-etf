#!/usr/bin/env python
"""Hold a strategy next to SPY: each leg scaled to the same volatility, then 50/50, rebalanced daily.

Each leg is scaled to `--vol` annualised volatility with an EWMA (span 63 days) of its own past daily returns,
lagged one day, so the scaling only uses information available at the time. Rebalancing between the legs is
assumed free (only the size of each leg changes).

Legs: `bt:<backtest name>` (net daily returns from backtest_diagnostics/<name>/pnl_curves.csv) or `spy`
(buy-and-hold SPY excess return from data/<dataset>_xs.parquet).

Usage: python scripts/combine_with_spy.py bt:bt-etf-deepm-ens spy [--start 2015-01-01] [--vol 0.10]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPAN = 63


def leg(spec, dataset):
    if spec == "spy":
        px = pd.read_parquet(ROOT / "data" / f"{dataset}_xs.parquet")["SPY"]
        return px.pct_change(fill_method=None).rename("spy")
    name = spec.removeprefix("bt:")
    pnl = pd.read_csv(ROOT / "backtest_diagnostics" / name / "pnl_curves.csv", index_col=0, parse_dates=True)
    return pnl["net_return_rescaled"].shift(1).rename(name)  # pnl_curves labels by decision day; move to realisation day


def vol_target(r, target):
    vol = r.ewm(span=SPAN, min_periods=SPAN).std().shift(1) * np.sqrt(252)
    return (r * target / vol).rename(r.name)


def stats(r):
    r = r.dropna()
    ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
    w = (1 + r).cumprod()
    return {"SR": ann / vol, "ann": ann, "vol": vol, "MDD": (w / w.cummax() - 1).min()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("legs", nargs="+", help="bt:<backtest name> or spy")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--vol", type=float, default=0.10, help="volatility each leg is scaled to")
    ap.add_argument("--dataset", default="etf", help="dataset name used by build_etf_dataset.py")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    raw = pd.concat([leg(s, args.dataset) for s in args.legs], axis=1, sort=True).loc[start - pd.DateOffset(months=7):]
    scaled = pd.concat([vol_target(raw[c], args.vol) for c in raw], axis=1).dropna().loc[start:]
    series = {**{f"{c} (scaled)": scaled[c] for c in scaled}, "50/50 combination": scaled.mean(axis=1)}
    if "spy" in raw:
        series["SPY (unscaled)"] = raw["spy"].reindex(scaled.index)
    rows = []
    for lbl, r in series.items():
        s = stats(r)
        rows.append({"series": lbl, "Sharpe": s["SR"], "return": s["ann"], "vol": s["vol"], "max drawdown": s["MDD"]})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).set_index("series").round(3).to_string())
    if scaled.shape[1] == 2:
        print(f"correlation of the two legs: {scaled.corr().iloc[0, 1]:.2f}")
    print(f"{scaled.index.min().date()}..{scaled.index.max().date()}, {len(scaled)} days")


if __name__ == "__main__":
    main()
