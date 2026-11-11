# deep-momentum-etf

Run the DeePM deep momentum model on ETFs: bring your own price data, then train and backtest with realistic trading costs.

> **Status: work in progress.** The data layer, baselines, ensemble tools and command-line interface are being added step by step. Results will be published here once the full pipeline has been run end to end from this repository.

## What this is

[DeePM](https://arxiv.org/abs/2601.05975) (Wood, Roberts and Zohren, 2026) is a deep-learning portfolio manager that the paper evaluates on 50 futures contracts. This project applies it to exchange-traded funds, which anyone can trade, and adds what that takes:

- runs on a laptop (Apple Silicon, CPU or a single GPU), with no experiment-tracking server required
- returns measured in excess of cash, real bid-ask spreads, and execution at the next day's open
- simple reference strategies (trend following, buy and hold) evaluated under the same rules
- one command line for data preparation, training, backtesting and reporting

## Built on

This repository starts from the official DeePM code by Kieran Wood, <https://github.com/kieranjwood/deepm>, released under the MIT license. The first three commits are that code, unchanged; its original README is kept at [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md). Everything after those commits is the ETF adaptation.

If you use the model, please cite the paper:

```bibtex
@article{wood2026deepm,
  title={DeePM: Regime-Robust Deep Learning for Systematic Macro Portfolio Management},
  author={Wood, Kieran and Roberts, Stephen J and Zohren, Stefan},
  journal={arXiv preprint arXiv:2601.05975},
  year={2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The original copyright notice is retained.
