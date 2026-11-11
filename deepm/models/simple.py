"""Simple learned baselines: per-timestep linear model and small MLP.

Same inputs, Sharpe + turnover loss and training loop as the deep models (they reuse
``DeepMomentumNetwork``), but no temporal memory, no attention, no ticker embedding:
the position at day t is a function of that day's features only (which already include
1d/1m/3m/1y returns and price z-scores), pooled across all assets — i.e. the original
DMN-style MLP. LINEAR is position = tanh(w . x_t + b); MLP adds two hidden ELU layers.
"""

from torch import nn

from deepm.models.base import DeepMomentumNetwork


class _PerStepBaseline(DeepMomentumNetwork):
    def __init__(self, input_dim: int, num_tickers: int, hidden_dim: int, dropout: float, **kwargs):
        super().__init__(
            input_dim=input_dim, num_tickers=num_tickers,
            hidden_dim=hidden_dim, dropout=dropout, **kwargs,
        )
        # base.forward appends the vol-scaling (cost) channel when tcost_inputs is on
        self.n_in = self.input_dim + self.extra_tcost_channels
        self.is_cross_section = False

    def forward_candidate_arch(self, target_x, target_tickers, **kwargs):
        if target_x.dim() == 2:
            target_x = target_x.unsqueeze(-1)
        return self.rep(target_x)  # [Batch, Seq, Hidden]

    def variable_importance(self, target_x, target_tickers, **kwargs):
        raise NotImplementedError("No variable selection network in per-step baselines")


class LinearBaseline(_PerStepBaseline):
    """position_t = tanh(head(W x_t)) — linear in features (hidden layer has no activation)."""

    def __init__(self, input_dim, num_tickers, hidden_dim, dropout, **kwargs):
        super().__init__(input_dim, num_tickers, hidden_dim, dropout, **kwargs)
        self.rep = nn.Linear(self.n_in, self.hidden_dim)


class MlpBaseline(_PerStepBaseline):
    """position_t = tanh(head(MLP(x_t))) with two ELU hidden layers."""

    def __init__(self, input_dim, num_tickers, hidden_dim, dropout, **kwargs):
        super().__init__(input_dim, num_tickers, hidden_dim, dropout, **kwargs)
        h = self.hidden_dim
        self.rep = nn.Sequential(
            nn.Linear(self.n_in, h), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(h, h), nn.ELU(), nn.Dropout(dropout),
        )
