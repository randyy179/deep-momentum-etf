# Bringing your own data

The pipeline needs two files. Neither is shipped with this repository: price data is usually licensed, and it
never goes into git (`/data/` is ignored).

## 1. Prices: `data/prices.parquet` (or `.csv`)

Long format, one row per ticker and trading day:

| column | meaning |
|---|---|
| `date` | trading day |
| `ticker` | ETF symbol, matching the transaction-cost file |
| `close` | close adjusted for splits **and dividends** (a total-return index) |
| `open` | open adjusted by the **same factor** as the close (`open * adj_close / raw_close`) |

Missing or non-positive opens are allowed (`NaN`); those days cannot be used as entry or exit points for the
next-open target. Start each ticker at its first listing day; no back-filling.

Any vendor works as long as the close is a total-return series. Free sources typically provide `Adj Close`
(already dividend adjusted) together with raw `Open` and `Close`, from which the adjusted open follows.

## 2. Cash rate: `data/DFF.csv`

Two columns, date and the annualised overnight rate in percent. The FRED series
[DFF](https://fred.stlouisfed.org/series/DFF) (effective federal funds rate) downloads in exactly this shape.

ETF total returns include the interest earned on the fund's cash and collateral; futures returns do not. The
rate is subtracted so that the model learns from excess returns, as in the DeePM paper.

## 3. Transaction costs: `configs/tcost/etf.csv`

One row per ticker with its group (`EQUITY`, `BOND`, `COMDTY`, `FX`) and `transaction_cost`, the **half
spread in basis points** charged on every unit of turnover, both in training and in the backtest. The shipped
values are issuer-published median bid-ask spreads for the 33 default ETFs (sources in
`configs/tcost/etf_spread_sources.csv`). To use other ETFs, add rows here and list the tickers in the configs.

## 4. Build the dataset

```bash
python scripts/build_etf_dataset.py --prices data/prices.parquet --cash-rate data/DFF.csv --name etf
```

This writes `data/feats-etf.parquet`; configs refer to it as `data_parquet: etf.parquet`. The steps:

1. **Excess-of-cash prices.** Daily return minus the previous day's cash rate, accrued Act/360 over calendar days.
2. **Features.** The original DeePM feature set (volatility-scaled returns from 1 day to 1 year, MACD signals,
   price z-scores, volatility) plus cross-sectional features, computed by the original code. The default ETF
   configs use six of them: `r1d, r1m, r3m, r1y, z_price_21d, z_price_252d`.
3. **Next-open target.** A signal computed at the close of day t is traded at the open of t+1 and held to the
   open after the next signal. Training on this target, rather than close-to-close, means the model never
   learns from a fill it could not have had.
