#!/usr/bin/env python
"""Build the model-ready ETF dataset from your own prices and a cash rate.

Steps: adjusted closes -> excess-of-cash price index -> DeePM features (+ cross-sectional) -> next-open target.

Usage:
    python scripts/build_etf_dataset.py --prices data/prices.parquet --cash-rate data/DFF.csv --name etf

Writes data/<name>_xs.parquet (wide excess-of-cash prices) and data/feats-<name>.parquet; training and backtest
configs then use ``data_parquet: <name>.parquet``. See docs/DATA.md for the input format.
"""

import argparse

from deepm._paths import DATA_DIR
from deepm.data.build_features import add_cross_sectional_features, prepare_features
from deepm.data.etf import close_panel, excess_prices, load_cash_rate, load_prices, next_open_target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prices", required=True, help="long file with date, ticker, open, close (adjusted)")
    ap.add_argument("--cash-rate", required=True, help="CSV of date, annual cash rate in percent (e.g. FRED DFF)")
    ap.add_argument("--name", default="etf", help="dataset name; output data/feats-<name>.parquet")
    ap.add_argument("--tickers", nargs="*", help="only these tickers (default: all in the price file)")
    args = ap.parse_args()

    prices = load_prices(args.prices)
    if args.tickers:
        prices = prices[prices["ticker"].isin(args.tickers)]
    rate = load_cash_rate(args.cash_rate)

    closes = close_panel(prices)
    if args.tickers:
        closes = closes[[t for t in args.tickers if t in closes.columns]]
    xs = excess_prices(closes, rate)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    xs.to_parquet(DATA_DIR / f"{args.name}_xs.parquet")
    print(f"excess prices: {xs.shape[1]} tickers, {xs.index.min().date()}..{xs.index.max().date()}")
    if rate.index.max() < xs.index.max():
        print(f"note: cash rate ends {rate.index.max().date()}, carried forward to the last price date")

    feats = prepare_features(xs).drop(columns="date", errors="ignore")
    feats = add_cross_sectional_features(feats)
    feats = next_open_target(feats, prices, rate)
    out = DATA_DIR / f"feats-{args.name}.parquet"
    feats.to_parquet(out)
    n_missing = int((feats["target_close"].notna() & feats["target"].isna()).sum())
    print(f"saved {out}: {len(feats)} rows; rows without a next-open target: {n_missing}")
    print(f"use  data_parquet: {args.name}.parquet  in your configs")


if __name__ == "__main__":
    main()
