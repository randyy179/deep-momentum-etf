#!/usr/bin/env python
"""Average the daily positions of several backtests (equal weight, per date and ETF) into one signal file.

Used for the DeePM ensemble: the gamma 0.5 and gamma 2 models trade the same signals with different cost
weights, and their average trades less than either while keeping most of the return.

Usage: python scripts/make_ensemble.py <out_name> <bt_name> <bt_name> [...]
  reads backtest_results/<bt_name>/<TICKER>.parquet, writes models_torch/ensembles/<out_name>/positions.parquet,
  which a backtest config reads with deepm.backtest.models.precomputed.PrecomputedSignal.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def positions(name):
    files = sorted((ROOT / "backtest_results" / name).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no positions in backtest_results/{name}; run its backtest first")
    p = pd.concat([pd.read_parquet(f) for f in files], axis=1)
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def main():
    out, names = sys.argv[1], sys.argv[2:]
    if len(names) < 2:
        raise SystemExit(__doc__)
    members = [positions(n) for n in names]
    for n, p in zip(names[1:], members[1:]):
        if not (p.index.equals(members[0].index) and sorted(p.columns) == sorted(members[0].columns)):
            raise SystemExit(f"{n}: dates/tickers differ from {names[0]}")
    ens = sum(p[members[0].columns] for p in members) / len(members)
    dst = ROOT / "models_torch" / "ensembles" / out / "positions.parquet"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ens.to_parquet(dst)
    corr = pd.concat([p.stack() for p in members], axis=1, keys=names).corr().round(3)
    print(f"saved {dst}: {ens.shape}, {ens.index.min().date()}..{ens.index.max().date()}")
    print(f"position correlation between members:\n{corr.to_string()}")


if __name__ == "__main__":
    main()
