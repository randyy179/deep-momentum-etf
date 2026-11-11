#!/usr/bin/env python
"""Draw the README figures from the backtests written by `dmetf backtest`.

Usage: python scripts/make_readme_figures.py   ->  docs/figures/{growth,sharpe}.png
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "backtest_diagnostics"
OUT = ROOT / "docs" / "figures"
START = pd.Timestamp("2015-01-01")

# reference categorical palette (validated light-mode steps) + neutral benchmark
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "text.color": INK, "legend.frameon": False, "axes.axisbelow": True,
})


def _combine_module():
    spec = importlib.util.spec_from_file_location("combine_with_spy", ROOT / "scripts" / "combine_with_spy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def combination():
    c = _combine_module()
    raw = pd.concat([c.leg("bt:bt-etf-deepm-ens", "etf"), c.leg("spy", "etf")], axis=1, sort=True)
    raw = raw.loc[START - pd.DateOffset(months=7):]
    scaled = pd.concat([c.vol_target(raw[k], 0.10) for k in raw], axis=1).dropna().loc[START:]
    combo = scaled.mean(axis=1)
    spy = raw["spy"].reindex(scaled.index)
    leverage = spy.std() / combo.std()
    return combo, combo * leverage, spy, leverage, c.stats


def net_returns(name):
    pnl = pd.read_csv(DIAG / name / "pnl_curves.csv", index_col=0, parse_dates=True)
    return pnl["net_return_rescaled"].loc[START:]


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(252)


def end_label(ax, series, text, color, dy=0):
    ax.annotate(text, (series.index[-1], series.iloc[-1]), xytext=(6, dy), textcoords="offset points",
                va="center", color=INK, fontsize=10, fontweight="bold")
    ax.plot(series.index[-1], series.iloc[-1], "o", ms=5, color=color, mec=SURFACE, mew=1.5)


def growth(combo, lev, spy, leverage, stats):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    curves = [(lev, f"DeePM + SPY, at SPY's risk", BLUE), (spy, "SPY buy and hold", GRAY),
              (combo, "DeePM + SPY, 50/50", ORANGE)]
    for r, label, color in curves:
        w = (1 + r).cumprod()
        s = stats(r)
        ax.plot(w, color=color, lw=2, label=f"{label}  ({s['ann']:.1%}/yr, max drawdown {s['MDD']:.0%})")
        end_label(ax, w, f"{w.iloc[-1]:.1f}x", color)
    ax.set_title("Growth of $1 above cash, net of trading costs, 2015-2026")
    ax.legend(loc="upper left")
    ax.set_xlim(right=lev.index[-1] + pd.Timedelta(days=330))
    fig.text(0.01, 0.01, f"DeePM ensemble and SPY each scaled to equal volatility from their own past returns, held 50/50. "
             f"'At SPY's risk' applies a constant {leverage:.1f}x so its volatility matches SPY's.",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "growth.png", dpi=160)


def sharpe_bars(combo):
    rows = [("DeePM + SPY, 50/50", sharpe(combo), BLUE)]
    for name, label in [("bt-etf-4etf-equal", "SPY / IEF / DBC / GLD, equal weight"),
                        ("bt-etf-spy", "SPY buy and hold"), ("bt-etf-deepm-ens", "DeePM ensemble alone"),
                        ("bt-etf-tsmom", "Trend rule (12-month momentum)"),
                        ("bt-etf-longonly", "All 33 ETFs long, equal risk")]:
        rows.append((label, sharpe(net_returns(name)), GRAY))
    rows = rows[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
    for i, (_, v, _) in enumerate(rows):
        ax.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.15)
    ax.set_title("Sharpe ratio net of trading costs, 2015-2026")
    fig.tight_layout()
    fig.savefig(OUT / "sharpe.png", dpi=160)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    combo, lev, spy, leverage, stats = combination()
    growth(combo, lev, spy, leverage, stats)
    sharpe_bars(combo)
    for lbl, r in [("50/50", combo), ("at SPY risk", lev), ("SPY", spy)]:
        s = stats(r)
        print(f"{lbl:12s} Sharpe {s['SR']:.2f}  {s['ann']:.1%}/yr  vol {s['vol']:.1%}  max DD {s['MDD']:.0%}")
    print(f"leverage {leverage:.2f}; {combo.index.min().date()}..{combo.index.max().date()}; saved {OUT}")


if __name__ == "__main__":
    main()
