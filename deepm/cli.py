"""`dmetf`: build the ETF dataset, train DeePM, backtest it against reference strategies, and report.

    dmetf data --prices data/prices.parquet --cash-rate data/DFF.csv
    dmetf train                      # DeePM at cost weights 0.5 and 2 (50 trials per window each)
    dmetf backtest                   # DeePM, ensemble and reference strategies from 2015
    dmetf report                     # table + equity chart in reports/

Each step only reads what the previous one wrote, so they can be rerun independently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from deepm._paths import PROJECT_ROOT

DIAG = PROJECT_ROOT / "backtest_diagnostics"
REPORTS = PROJECT_ROOT / "reports"
DEEPM = {"0.5": "etf-deepm-g05", "2": "etf-deepm-g2"}
REFERENCES = {
    "bt-etf-deepm-ens": "DeePM ensemble",
    "bt-etf-deepm-g05": "DeePM, cost weight 0.5",
    "bt-etf-deepm-g2": "DeePM, cost weight 2",
    "bt-etf-tsmom": "Trend rule (12-month momentum)",
    "bt-etf-longonly": "All 33 ETFs long, equal risk",
    "bt-etf-4etf-equal": "SPY/IEF/DBC/GLD equal weight",
    "bt-etf-spy": "SPY buy and hold",
}
LEARNED = {"bt-etf-linear": "Linear model", "bt-etf-mlp": "Small neural net (MLP)"}


def _run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    full_env = {**os.environ, "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"), **(env or {})}
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=full_env, check=True)


def cmd_data(a):
    _run([sys.executable, "scripts/build_etf_dataset.py", "--prices", a.prices, "--cash-rate", a.cash_rate,
          "--name", "etf"])


def cmd_train(a):
    for g in a.gamma:
        cfg = DEEPM[g]
        if a.shards > 1:
            _run(["bash", "scripts/run_sharded.sh", cfg, "DeePM", str(a.shards)], env={"PY": sys.executable})
        else:
            _run([sys.executable, "-m", "deepm.training", "-r", cfg, "-a", "DeePM"])
    for arch in a.baselines:
        _run([sys.executable, "-m", "deepm.training", "-r", f"etf-{arch}", "-a", arch.upper()])


def _backtest(name, start):
    _run([sys.executable, "-m", "deepm.backtest", "--name", name, "--start_date", start, "--diagnostics"])


def cmd_backtest(a):
    for g in ("g05", "g2"):
        _backtest(f"bt-etf-deepm-{g}", a.start)
    _run([sys.executable, "scripts/make_ensemble.py", "etf-deepm-ens", "bt-etf-deepm-g05", "bt-etf-deepm-g2"])
    for name in ["bt-etf-deepm-ens", "bt-etf-tsmom", "bt-etf-longonly", "bt-etf-4etf-equal", "bt-etf-spy"]:
        _backtest(name, a.start)
    for name, arch in [("bt-etf-linear", "LINEAR"), ("bt-etf-mlp", "MLP")]:
        if (PROJECT_ROOT / "models_torch" / name.removeprefix("bt-") / arch).exists():
            _backtest(name, a.start)


def _row(name, label, start):
    pnl = pd.read_csv(DIAG / name / "pnl_curves.csv", index_col=0, parse_dates=True).loc[start:]
    m = json.loads((DIAG / name / "metrics.json").read_text())["net"]
    r, g = pnl["net_return_rescaled"], pnl["gross_return_rescaled"]
    w = (1 + r).cumprod()
    return {
        "strategy": label,
        "net Sharpe": r.mean() / r.std() * np.sqrt(252),
        "gross Sharpe": g.mean() / g.std() * np.sqrt(252),
        "max drawdown": (w / w.cummax() - 1).min(),
        "turnover / year": m["Annualized Turnover (Scaled)"],
    }


def cmd_report(a):
    REPORTS.mkdir(exist_ok=True)
    names = {**REFERENCES, **LEARNED}
    rows = [_row(n, lbl, a.start) for n, lbl in names.items() if (DIAG / n / "pnl_curves.csv").exists()]
    if not rows:
        raise SystemExit("no backtests found; run `dmetf backtest` first")
    table = pd.DataFrame(rows).set_index("strategy")
    table.round(3).to_csv(REPORTS / "summary.csv")
    with pd.option_context("display.width", 200):
        print(f"\nNet of costs, all strategies scaled to 10% volatility, from {a.start}:\n")
        print(table.round(2).to_string())

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    for n in ["bt-etf-deepm-ens", "bt-etf-spy", "bt-etf-tsmom"]:
        if (DIAG / n / "pnl_curves.csv").exists():
            r = pd.read_csv(DIAG / n / "pnl_curves.csv", index_col=0, parse_dates=True)["net_return_rescaled"].loc[a.start:]
            ax.plot((1 + r).cumprod(), label=names[n])
    ax.set_title("Growth of $1, net of costs, each strategy at 10% volatility")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "equity.png", dpi=150)
    print(f"\nsaved {REPORTS / 'summary.csv'} and {REPORTS / 'equity.png'}")

    if (DIAG / "bt-etf-deepm-ens").exists():
        print("\nDeePM ensemble held 50/50 with SPY (same-volatility legs):\n", flush=True)
        _run([sys.executable, "scripts/combine_with_spy.py", "bt:bt-etf-deepm-ens", "spy", "--start", a.start])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="dmetf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("data", help="build data/feats-etf.parquet from your prices and a cash rate (docs/DATA.md)")
    p.add_argument("--prices", required=True)
    p.add_argument("--cash-rate", required=True)
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("train", help="train DeePM (and optionally the linear / MLP baselines)")
    p.add_argument("--gamma", nargs="+", choices=list(DEEPM), default=list(DEEPM), help="cost weights to train")
    p.add_argument("--shards", type=int, default=1, help="parallel processes sharing one GPU")
    p.add_argument("--baselines", nargs="*", choices=["linear", "mlp"], default=[], help="also train these")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("backtest", help="backtest DeePM, the ensemble and the reference strategies")
    p.add_argument("--start", default="2015-01-01")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("report", help="summary table and equity chart in reports/")
    p.add_argument("--start", default="2015-01-01")
    p.set_defaults(func=cmd_report)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
