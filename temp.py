"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Portfolio Optimizer — Inference Module  v3                                 ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Mirrors the EXACT architecture of drl_portfolio_v3.py                     ║
║  Changes vs v2:                                                              ║
║    • N_FEATURES: 6 → 9  (+ xs_rank_ret5, xs_rank_ret20, xs_rank_vol)        ║
║    • SACActor: Normal+tanh+softmax → SACActorDirichlet                      ║
║    • SignalGen: MLP/CNN → CrossAssetTransformer (cross-stock attention)     ║
║    • Model files: *_final_v2.pt  → *_final_v3.pt                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, math, copy, warnings
from typing import Optional

import numpy  as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  — must match training
# ──────────────────────────────────────────────────────────────────────────────
N_FEATURES = 9   # norm_px, log_ret, mom5, mom20, vol20, rsi14,
                  # xs_rank_ret5, xs_rank_ret20, xs_rank_vol


# ──────────────────────────────────────────────────────────────────────────────
# A.  PRIMITIVES
# ──────────────────────────────────────────────────────────────────────────────
def to_simplex(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Clamp to (eps, 1-eps) then L1-normalise → valid Dirichlet support."""
    w = w.clamp(min=eps, max=1.0 - eps)
    return w / w.sum(dim=-1, keepdim=True)


class MaskingLayer(nn.Module):
    """Zero inactive stocks and renormalise weights to sum = 1."""
    def forward(self, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = weights * mask.float()
        s = w.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return w / s


# ──────────────────────────────────────────────────────────────────────────────
# B.  SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────────────────────────
class CrossAssetTransformer(nn.Module):
    """
    Each stock is a token; self-attention compares stocks cross-sectionally.
    Input : (B, N, T, F)  →  Output: (B, N, out_dim)

    This is the key architectural fix vs v2: MLP/CNN treated every stock
    independently — no cross-sectional comparison was possible.
    """
    def __init__(self, T: int, F: int,
                 d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 ffn_mult: int = 2, dropout: float = 0.10, out_dim: int = 32):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(T * F, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = d_model * ffn_mult,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, out_dim)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F = x.shape
        tokens = self.input_proj(x.reshape(B * N, T * F)).reshape(B, N, -1)
        out    = self.transformer(tokens)
        return self.output_proj(out)


class SignalGeneratorMLP(nn.Module):
    """Fallback MLP — used when checkpoint was trained with signal.type='mlp'."""
    def __init__(self, T: int, F_dim: int, hidden: int = 64, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(T * F_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F_dim = x.shape
        return self.net(x.reshape(B * N, T * F_dim)).reshape(B, N, -1)


class SignalGeneratorCNN(nn.Module):
    """Fallback CNN — used when checkpoint was trained with signal.type='cnn'."""
    def __init__(self, T: int, F_dim: int,
                 filters: int = 32, kernel: int = 4, out_dim: int = 16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(F_dim, filters, kernel_size=kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(filters, filters, kernel_size=kernel, padding=kernel // 2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(filters, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F_dim = x.shape
        h = self.conv(x.reshape(B * N, T, F_dim).permute(0, 2, 1)).squeeze(-1)
        return self.head(h).reshape(B, N, -1)


def _build_signal_gen(cfg: dict, T: int, F: int) -> nn.Module:
    s = cfg["signal"]
    if s["type"] == "transformer":
        return CrossAssetTransformer(T, F,
                                     d_model=s["d_model"],
                                     n_heads=s["n_heads"],
                                     n_layers=s["n_layers"],
                                     ffn_mult=s["ffn_mult"],
                                     dropout=s["dropout"],
                                     out_dim=s["out_dim"])
    if s["type"] == "cnn":
        return SignalGeneratorCNN(T, F, s["cnn_filters"], s["cnn_kernel"], s["out_dim"])
    return SignalGeneratorMLP(T, F, s.get("hidden_dim", 64), s["out_dim"])


# ──────────────────────────────────────────────────────────────────────────────
# C.  NETWORKS
# ──────────────────────────────────────────────────────────────────────────────
def _make_trunk(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
        nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
    )


# ── PPO ───────────────────────────────────────────────────────────────────────
class PPOActor(nn.Module):
    def __init__(self, in_dim: int, N: int, hidden: int = 256):
        super().__init__()
        self.trunk      = _make_trunk(in_dim, hidden)
        self.alpha_head = nn.Linear(hidden, N)
        nn.init.xavier_uniform_(self.alpha_head.weight, gain=0.3)
        nn.init.constant_(self.alpha_head.bias, 0.5)

    def alphas(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.alpha_head(self.trunk(x))) + 1.0

    def get_dist(self, x: torch.Tensor) -> Dirichlet:
        return Dirichlet(self.alphas(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alphas(x)


class PPOCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(*_make_trunk(in_dim, hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ── SAC ───────────────────────────────────────────────────────────────────────
class SACActorDirichlet(nn.Module):
    """
    Dirichlet policy — the correct distribution for portfolio weights.

    v2 used Normal + tanh + softmax which collapses to uniform 1/N because:
      • softmax([0, …, 0]) = 1/N at initialisation (Xavier → μ ≈ 0)
      • Gradients of softmax are antisymmetric — pushing one stock up
        forces ALL others down equally → can never escape uniform

    Dirichlet with αᵢ > 1 has a well-defined mode at
        wᵢ = (αᵢ - 1) / Σ(αⱼ - 1)
    so distinct concentrations per stock emerge naturally from training.
    """
    def __init__(self, in_dim: int, N: int, hidden: int = 256):
        super().__init__()
        self.trunk      = _make_trunk(in_dim, hidden)
        self.alpha_head = nn.Linear(hidden, N)
        nn.init.xavier_uniform_(self.alpha_head.weight, gain=0.3)
        nn.init.constant_(self.alpha_head.bias, 0.5)
        self.N = N

    def alphas(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.alpha_head(self.trunk(x))) + 1.0

    def forward(self, x: torch.Tensor,
                deterministic: bool = False):
        alpha = self.alphas(x)
        dist  = Dirichlet(alpha)

        if deterministic:
            # Mode of Dirichlet (valid since all αᵢ > 1)
            w = (alpha - 1.0).clamp(min=0.0)
            s = w.sum(dim=-1, keepdim=True)
            w = torch.where(s > 1e-8, w / s,
                            torch.full_like(w, 1.0 / self.N))
        else:
            w = dist.rsample()

        w_valid  = to_simplex(w)
        log_prob = dist.log_prob(w_valid)
        return w_valid, log_prob

    def get_dist(self, x: torch.Tensor) -> Dirichlet:
        return Dirichlet(self.alphas(x))


class SACCritic(nn.Module):
    def __init__(self, in_dim: int, N: int, hidden: int = 256):
        super().__init__()
        d = in_dim + N
        def _q():
            return nn.Sequential(
                nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1))
        self.q1 = _q(); self.q2 = _q()

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        xa = torch.cat([x, a], dim=-1)
        return self.q1(xa).squeeze(-1), self.q2(xa).squeeze(-1)

    def q_min(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(x, a); return torch.min(q1, q2)


# ──────────────────────────────────────────────────────────────────────────────
# D.  AGENT WRAPPERS
# ──────────────────────────────────────────────────────────────────────────────
class PPOAgent(nn.Module):
    def __init__(self, N: int, T: int, F_dim: int, cfg: dict):
        super().__init__()
        sig_out = cfg["signal"]["out_dim"]
        in_dim  = N * T * F_dim + sig_out * N
        self.signal_gen = _build_signal_gen(cfg, T, F_dim)
        self.masking    = MaskingLayer()
        self.actor      = PPOActor(in_dim, N)
        self.critic     = PPOCritic(in_dim)
        self.N = N

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        B   = obs.shape[0]
        sig = self.signal_gen(obs)
        return torch.cat([obs.reshape(B, -1), sig.reshape(B, -1)], dim=-1)

    def act(self, obs_np: np.ndarray,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = True) -> np.ndarray:
        obs = torch.tensor(obs_np[None], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            x    = self._encode(obs)
            dist = self.actor.get_dist(x)
            if deterministic:
                a = dist.concentration
                w = (a - 1.0).clamp(min=0.0)
                s = w.sum(-1, keepdim=True)
                w = torch.where(s > 1e-8, w / s,
                                torch.full_like(w, 1.0 / self.N))
            else:
                w = dist.rsample()
            w = to_simplex(w)
            if mask is not None:
                w = self.masking(w, torch.tensor(mask, device=DEVICE))
        return w[0].cpu().numpy()


class SACAgent(nn.Module):
    def __init__(self, N: int, T: int, F_dim: int, cfg: dict):
        super().__init__()
        sig_out = cfg["signal"]["out_dim"]
        in_dim  = N * T * F_dim + sig_out * N
        self.signal_gen  = _build_signal_gen(cfg, T, F_dim)
        self.masking     = MaskingLayer()
        self.actor       = SACActorDirichlet(in_dim, N)
        self.critic      = SACCritic(in_dim, N)
        self.critic_targ = copy.deepcopy(self.critic)
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))
        self.N = N

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        B   = obs.shape[0]
        sig = self.signal_gen(obs)
        return torch.cat([obs.reshape(B, -1), sig.reshape(B, -1)], dim=-1)

    def act(self, obs_np: np.ndarray,
            mask: Optional[np.ndarray] = None,
            deterministic: bool = True) -> np.ndarray:
        obs = torch.tensor(obs_np[None], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            w, _ = self.actor(self._encode(obs), deterministic)
            if mask is not None:
                w = self.masking(w, torch.tensor(mask, device=DEVICE))
        return w[0].cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# E.  FEATURE ENGINEERING  (must stay in sync with training)
# ──────────────────────────────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def _xs_rank(arr: np.ndarray) -> np.ndarray:
    """
    Cross-sectional percentile rank at each time step.
    arr : (T, N) → returns same shape in [0, 1]

    v3 NEW: this feature lets the model compare stocks to each other
    (e.g. which stocks have the best 5-day momentum THIS DAY).
    """
    out = np.zeros_like(arr, dtype=np.float32)
    N   = arr.shape[1]
    for t in range(len(arr)):
        order  = arr[t].argsort().argsort()
        out[t] = order.astype(np.float32) / max(N - 1, 1)
    return out


def build_features_from_prices(prices: pd.DataFrame, window: int = 20) -> np.ndarray:
    """
    Build observation array of shape  (N, window, N_FEATURES=9)
    from a prices DataFrame.

    Features (in order):
        0: z-score normalised price
        1: log return
        2: 5-day momentum
        3: 20-day momentum
        4: 20-day rolling volatility
        5: RSI-14 / 100
        6: cross-sectional percentile rank of 5d momentum   [NEW v3]
        7: cross-sectional percentile rank of 20d momentum  [NEW v3]
        8: cross-sectional percentile rank of volatility    [NEW v3]

    Parameters
    ----------
    prices : pd.DataFrame
        Rows = dates (sorted ascending), columns = tickers.
        Must have at least window + 20 rows.
    window : int
        Lookback window used at training time (default 20).

    Returns
    -------
    np.ndarray
        Shape (N, window, 9), clipped to [-10, 10].
    """
    prices = prices.copy().sort_index()
    prices.ffill(inplace=True); prices.bfill(inplace=True)

    log_ret = np.log(prices / prices.shift(1)).fillna(0)
    mu      = prices.rolling(window).mean()
    sigma   = prices.rolling(window).std().fillna(1)
    norm_p  = ((prices - mu) / sigma).fillna(0)
    mom5    = prices.pct_change(5).fillna(0)
    mom20   = prices.pct_change(20).fillna(0)
    vol20   = log_ret.rolling(20).std().fillna(0)
    rsi14   = (prices.apply(_rsi).fillna(50) / 100.0)

    # Convert to numpy (T_all, N)
    mom5_np  = mom5.values.astype(np.float32)
    mom20_np = mom20.values.astype(np.float32)
    vol20_np = vol20.values.astype(np.float32)

    xs_r5  = _xs_rank(mom5_np)
    xs_r20 = _xs_rank(mom20_np)
    xs_rv  = _xs_rank(vol20_np)

    # Stack → (T_all, N, 9)
    stack = np.stack([
        norm_p.values.astype(np.float32),
        log_ret.values.astype(np.float32),
        mom5_np,
        mom20_np,
        vol20_np,
        rsi14.values.astype(np.float32),
        xs_r5,
        xs_r20,
        xs_rv,
    ], axis=-1)

    # Keep only last `window` rows → (window, N, 9) → (N, window, 9)
    obs = stack[-window:].transpose(1, 0, 2)
    return np.clip(obs, -10, 10).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# F.  PORTFOLIO OPTIMIZER  — main inference class
# ──────────────────────────────────────────────────────────────────────────────
class PortfolioOptimizer:
    """
    Single entry-point for v3 inference.

    Parameters
    ----------
    model_path : str
        Path to .pt file saved by drl_portfolio_v3.py
        (e.g. ``ppo_final_v3.pt`` or ``sac_final_v3.pt``).
    deterministic : bool
        Use MAP / mode action (recommended for production).

    Quick start
    -----------
    >>> opt    = PortfolioOptimizer("ppo_final_v3.pt")
    >>> result = opt.predict(tickers=["TCS","INFY","WIPRO"], price_data=df)
    >>> # result["allocations"] → {"TCS": 18.4, "INFY": 12.1, ..., "CASH": 2.0}
    """

    def __init__(self, model_path: str, deterministic: bool = True):
        self.deterministic = deterministic
        self._load_model(model_path)

    # ── internals ──────────────────────────────────────────────────────────
    def _load_model(self, path: str):
        ck  = torch.load(path, map_location=DEVICE, weights_only=False)
        self.cfg             = ck["cfg"]
        self.master_tickers  = ck["tickers"]
        self.N_master        = ck["N"]
        self.T               = ck["T"]
        self.F_dim           = ck.get("F_dim", N_FEATURES)
        self.window          = self.cfg["window"]
        self.cash_buffer_pct = self.cfg.get("cash_buffer", 0.02) * 100
        self.version         = ck.get("version", "unknown")

        # ── Detect algorithm from state-dict keys ──────────────────────────
        # WHY NOT "alpha_head": v3 PPO and SAC both use Dirichlet actors,
        # so both have "actor.alpha_head" keys → can no longer distinguish.
        #
        # SAC-UNIQUE keys (PPO never has these):
        #   • "log_alpha"       — learnable entropy temperature
        #   • "critic_targ.*"   — target Q-network for soft updates
        #   • "critic.q1.*"     — twin Q-network heads
        #
        # PPO-UNIQUE keys (SAC never has these):
        #   • "critic.net.*"    — simple value head (single MLP)
        state   = ck["model_state"]
        is_sac  = any(
            k.startswith("log_alpha") or k.startswith("critic_targ")
            for k in state.keys()
        )

        if is_sac:
            self.agent = SACAgent(self.N_master, self.T, self.F_dim, self.cfg)
            self.algo  = "SAC"
        else:
            self.agent = PPOAgent(self.N_master, self.T, self.F_dim, self.cfg)
            self.algo  = "PPO"

        self.agent.load_state_dict(state, strict=True)
        self.agent.eval().to(DEVICE)

        # ── Sanity: confirm F_dim matches v3 expectation ──────────────────
        if self.F_dim != N_FEATURES:
            raise RuntimeError(
                f"[ERROR] Model was trained with F_dim={self.F_dim} features, "
                f"but this inference module expects {N_FEATURES}.\n"
                f"Make sure you are loading a v3 model (drl_portfolio_v3.py).")

        print(f"[PortfolioOptimizer v3] {self.algo} loaded from '{path}'")
        print(f"  Version           : {self.version}")
        print(f"  Master universe   : {self.N_master} stocks")
        print(f"  Observation shape : ({self.N_master}, {self.window}, {self.F_dim})")

    def _validate_tickers(self, tickers: list) -> list:
        unknown = [t for t in tickers if t not in self.master_tickers]
        if unknown:
            raise ValueError(
                f"[ERROR] {len(unknown)} ticker(s) not in master universe: {unknown}\n"
                f"Supported tickers: {self.master_tickers}")
        return tickers

    def _build_mask(self, user_tickers: list) -> np.ndarray:
        mask = np.zeros(self.N_master, dtype=bool)
        for t in user_tickers:
            mask[self.master_tickers.index(t)] = True
        return mask

    def _prepare_obs(self, user_tickers: list,
                     price_data: pd.DataFrame) -> np.ndarray:
        required_rows = self.window + 20
        if len(price_data) < required_rows:
            raise ValueError(
                f"price_data needs at least {required_rows} rows "
                f"(window={self.window} + 20 warm-up).  Got {len(price_data)}.")

        # Build full master-universe price frame; inactive stocks → flat 100
        full_prices = pd.DataFrame(
            index    = price_data.index,
            columns  = self.master_tickers,
            dtype    = np.float32)

        for ticker in self.master_tickers:
            if ticker in user_tickers and ticker in price_data.columns:
                full_prices[ticker] = price_data[ticker].values
            else:
                full_prices[ticker] = 100.0

        return build_features_from_prices(full_prices, window=self.window)

    # ── public API ─────────────────────────────────────────────────────────
    def predict(self,
                tickers    : list,
                price_data : pd.DataFrame,
                top_n      : Optional[int]   = None,
                min_weight : float = 0.005) -> dict:
        """
        Generate portfolio allocation.

        Parameters
        ----------
        tickers : list[str]
            1–30 tickers from the master universe.
        price_data : pd.DataFrame
            Close prices.
            - Index  : dates (DatetimeIndex)
            - Columns: one per ticker in ``tickers``
            - Minimum rows: window + 20 = 40  (252 recommended)
        top_n : int, optional
            Keep only the top-N holdings; zero out the rest.
        min_weight : float
            Threshold; stocks below this are zeroed and weight redistributed.

        Returns
        -------
        dict
            {
              "allocations": {"TCS": 18.4, "INFY": 12.1, ..., "CASH": 2.0},
              "summary": {
                  "n_stocks": 7,
                  "invested_pct": 98.0,
                  "cash_pct": 2.0,
                  "top_holding": "TCS",
                  "algo": "SAC",
                  "hhi": 0.0821,         # concentration metric
                  "top5_share": 72.3,    # % in top-5 holdings
              }
            }
        """
        # 1. Validate
        if not isinstance(tickers, list) or len(tickers) == 0:
            raise ValueError("`tickers` must be a non-empty list.")
        tickers = [t.upper().strip() for t in tickers]
        self._validate_tickers(tickers)

        # 2. Normalise price_data index
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data = price_data.copy()
            price_data.index = pd.to_datetime(price_data.index)
        price_data = price_data.sort_index()

        # Accept single-ticker OHLCV with "Close" / "Adj Close" column
        if len(tickers) == 1:
            for alias in ("Close", "Adj Close", "close", "adj_close"):
                if alias in price_data.columns and tickers[0] not in price_data.columns:
                    price_data = price_data.rename(columns={alias: tickers[0]})
                    break

        missing = [t for t in tickers if t not in price_data.columns]
        if missing:
            raise ValueError(f"price_data missing columns: {missing}")

        # 3. Build obs + mask
        obs  = self._prepare_obs(tickers, price_data)
        mask = self._build_mask(tickers)

        # 4. Model forward
        raw_w = self.agent.act(obs, mask=mask, deterministic=self.deterministic)

        # 5. Collect user-ticker weights
        user_w = {t: float(raw_w[self.master_tickers.index(t)]) for t in tickers}

        # 6. Apply min-weight threshold
        user_w = {t: (w if w >= min_weight else 0.0) for t, w in user_w.items()}

        # 7. Normalise to invested ratio (1 - cash_buffer)
        invested_ratio = 1.0 - self.cfg.get("cash_buffer", 0.02)
        total          = sum(user_w.values())
        if total > 1e-8:
            user_w = {t: w / total * invested_ratio for t, w in user_w.items()}

        # 8. Optional top-N filter
        if top_n is not None and top_n > 0:
            top_items = dict(sorted(user_w.items(), key=lambda x: x[1],
                                    reverse=True)[:top_n])
            top_total = sum(top_items.values())
            user_w    = ({t: w / top_total * invested_ratio for t, w in top_items.items()}
                         if top_total > 1e-8 else top_items)

        # 9. Build final allocation dict (percentages)
        alloc_pct = {t: round(w * 100, 2) for t, w in user_w.items()}
        alloc_pct = dict(sorted(alloc_pct.items(), key=lambda x: x[1], reverse=True))
        alloc_pct["CASH"] = round(self.cash_buffer_pct, 2)

        # 10. Diagnostics (HHI, top-5)
        weights_arr = np.array([user_w.get(t, 0.0) for t in tickers])
        hhi         = float((weights_arr ** 2).sum())
        top5_share  = float(np.sort(weights_arr)[-5:].sum()) * 100
        n_active    = sum(1 for t, v in alloc_pct.items() if t != "CASH" and v > 0)
        top_stock   = max((k for k in alloc_pct if k != "CASH"),
                          key=lambda k: alloc_pct[k], default="N/A")

        return {
            "allocations": alloc_pct,
            "summary": {
                "n_stocks"    : n_active,
                "invested_pct": round(sum(v for k, v in alloc_pct.items() if k != "CASH"), 2),
                "cash_pct"    : alloc_pct["CASH"],
                "top_holding" : top_stock,
                "algo"        : self.algo,
                "hhi"         : round(hhi, 4),
                "top5_share"  : round(top5_share, 2),
                "uniform_hhi" : round(1.0 / max(len(tickers), 1), 4),
            }
        }

    def available_tickers(self) -> list:
        return self.master_tickers.copy()

    def is_supported(self, ticker: str) -> bool:
        return ticker.upper().strip() in self.master_tickers


# ──────────────────────────────────────────────────────────────────────────────
# G.  CONVENIENCE LOADER
# ──────────────────────────────────────────────────────────────────────────────
def load_both_models(model_dir: str):
    """Load PPO and SAC v3 optimizers. Returns (ppo_opt, sac_opt)."""
    ppo_path = os.path.join(model_dir, "ppo_final_v3.pt")
    sac_path = os.path.join(model_dir, "sac_final_v3.pt")
    ppo_opt  = PortfolioOptimizer(ppo_path) if os.path.exists(ppo_path) else None
    sac_opt  = PortfolioOptimizer(sac_path) if os.path.exists(sac_path) else None
    if ppo_opt is None: print("[WARN] ppo_final_v3.pt not found")
    if sac_opt is None: print("[WARN] sac_final_v3.pt not found")
    return ppo_opt, sac_opt


# ──────────────────────────────────────────────────────────────────────────────
# H.  PRETTY-PRINT HELPER
# ──────────────────────────────────────────────────────────────────────────────
def print_allocation(result: dict, title: str = "Portfolio Allocation"):
    alloc   = result["allocations"]
    summary = result["summary"]
    width   = 54

    print(f"\n{'═'*width}")
    print(f"  {title}  [{summary['algo']} v3]".center(width))
    print(f"{'═'*width}")
    print(f"  {'Ticker':<12}  {'Weight (%)':>10}  {'Bar'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*18}")

    for ticker, pct in alloc.items():
        bar = "█" * int(pct / 2)
        print(f"  {ticker:<12}  {pct:>9.2f}%  {bar}")

    print(f"{'─'*width}")
    print(f"  Active stocks  : {summary['n_stocks']}")
    print(f"  Invested       : {summary['invested_pct']:.2f}%")
    print(f"  Cash buffer    : {summary['cash_pct']:.2f}%")
    print(f"  Top holding    : {summary['top_holding']}")
    print(f"  HHI            : {summary['hhi']:.4f}  "
          f"(uniform={summary['uniform_hhi']:.4f} — higher = more selective)")
    print(f"  Top-5 share    : {summary['top5_share']:.1f}%")
    print(f"{'═'*width}\n")


# ──────────────────────────────────────────────────────────────────────────────
# I.  MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, glob

    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Working directory: {current_dir}\n")

    # ── Load models ──────────────────────────────────────────────────────
    optimizers = {}
    for name, fname in [("PPO", "ppo_final_v3.pt"), ("SAC", "sac_final_v3.pt")]:
        path = os.path.join(current_dir, fname)
        if os.path.exists(path):
            optimizers[name] = PortfolioOptimizer(path)
        else:
            print(f"[WARN] {path} not found")

    if not optimizers:
        print("[ERROR] No v3 models found. Train with drl_portfolio_v3.py first.")
        exit(1)

    # ── Load CSV data ────────────────────────────────────────────────────
    price_data_all = {}
    for csv_file in sorted(glob.glob(os.path.join(current_dir, "*_daily.csv"))):
        ticker = os.path.basename(csv_file).replace("_daily.csv", "").upper()
        try:
            df = pd.read_csv(csv_file)
            if "close" not in df.columns:
                print(f"  [SKIP] {ticker}: no 'close' column"); continue
            df = df[["timestamp", "close"]].copy()
            df.columns = ["Date", ticker]
            df["Date"]  = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            price_data_all[ticker] = df[ticker]
            print(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    prices_combined = pd.DataFrame(price_data_all)
    print(f"\nCombined: {prices_combined.shape}  |  "
          f"{prices_combined.index[0].date()} → {prices_combined.index[-1].date()}\n")

    # ── Run predictions ──────────────────────────────────────────────────
    first_opt       = next(iter(optimizers.values()))
    supported       = first_opt.available_tickers()
    tickers_to_use  = [t for t in supported if t in prices_combined.columns]

    print(f"Using {len(tickers_to_use)} tickers: {tickers_to_use}\n")
    prices_subset = prices_combined[tickers_to_use].copy()

    for algo_name, optimizer in optimizers.items():
        try:
            result = optimizer.predict(
                tickers    = tickers_to_use,
                price_data = prices_subset,
                min_weight = 0.005,
            )
            print_allocation(result, title=f"Portfolio Allocation [{algo_name}]")
            print(f"JSON ({algo_name}):\n{json.dumps(result['allocations'], indent=2)}\n")
        except Exception as e:
            import traceback
            print(f"[ERROR] {algo_name}: {e}"); traceback.print_exc()