"""
Marketing Mix Model (MMM)
=========================
An end-to-end, interpretable marketing mix model in Python:

  spend (by channel)  --adstock-->  --saturation-->  regression  -->  sales
                                                           |
                                      channel contribution decomposition
                                                           |
                                                channel ROI + holdout validation

Pipeline
--------
1. Weekly data: spend by media channel + a sales (revenue) series.
   By default this script SIMULATES a realistic 3-year weekly dataset with a
   known ground truth (base + trend + seasonality + adstocked/saturated media),
   so the model can be checked against the truth. Swap `simulate_data()` for a
   CSV loader to run it on your own data (see load_csv()).
2. Adstock (carryover): geometric decay per channel, decay tuned by validation.
3. Saturation (diminishing returns): Hill transform per channel.
4. Base model: OLS of sales on trend + yearly seasonality (Fourier terms).
5. Media model: non-negative least squares (NNLS) of the transformed media on
   the base residual, so media contributions are non-negative (as in practice).
6. Decomposition: per-week contribution of each channel; ROI = contributed
   revenue / spend.
7. Validation: time-based train/holdout split; report R^2 and MAPE on holdout.

Run
---
    pip install -r requirements.txt
    python mmm.py

Outputs: console summary, `mmm_results.csv`, and PNG charts in ./outputs/.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import lsq_linear

RNG = np.random.default_rng(42)
CHANNELS = ["tv", "search", "social", "display", "video"]
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
def geometric_adstock(x: np.ndarray, decay: float) -> np.ndarray:
    """Geometric carryover: a_t = x_t + decay * a_{t-1}."""
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t in range(len(x)):
        carry = x[t] + decay * carry
        out[t] = carry
    return out


def hill_saturation(x: np.ndarray, half: float, slope: float = 1.0) -> np.ndarray:
    """Hill diminishing-returns curve in [0, 1); `half` is the half-saturation point."""
    half = max(half, 1e-9)
    xs = np.clip(x, 0, None)
    return xs ** slope / (xs ** slope + half ** slope)


def transform_channel(spend: np.ndarray, decay: float, slope: float = 1.0) -> np.ndarray:
    adstocked = geometric_adstock(spend, decay)
    pos = adstocked[adstocked > 0]
    # half-saturation set high (1.5x mean) so we operate on the informative,
    # non-plateau part of the curve and channels stay distinguishable
    half = 1.5 * pos.mean() if pos.size else 1.0
    return hill_saturation(adstocked, half, slope)


# --------------------------------------------------------------------------- #
# Base design matrix (trend + yearly seasonality via Fourier terms)
# --------------------------------------------------------------------------- #
def base_design(n_weeks: int, n_fourier: int = 3) -> np.ndarray:
    t = np.arange(n_weeks)
    cols = [np.ones(n_weeks), t / n_weeks]  # intercept + linear trend
    for k in range(1, n_fourier + 1):
        cols.append(np.sin(2 * np.pi * k * t / 52.0))
        cols.append(np.cos(2 * np.pi * k * t / 52.0))
    return np.column_stack(cols)


# --------------------------------------------------------------------------- #
# Simulated ground-truth data
# --------------------------------------------------------------------------- #
def _independent_spend(level: float, n_weeks: int) -> np.ndarray:
    """Independent weekly spend: AR(1) log-noise + occasional promo bursts."""
    z = np.zeros(n_weeks)
    for t in range(1, n_weeks):
        z[t] = 0.6 * z[t - 1] + RNG.normal(0, 0.35)
    spend = level * np.exp(z - z.var() / 2)
    bursts = RNG.random(n_weeks) < 0.06
    spend = spend * np.where(bursts, RNG.uniform(1.4, 2.0, n_weeks), 1.0)
    return np.clip(spend, 0, None)


def simulate_data(n_weeks: int = 156) -> tuple[pd.DataFrame, dict]:
    dates = pd.date_range("2022-01-03", periods=n_weeks, freq="W-MON")
    t = np.arange(n_weeks)

    # True media behaviour (recovered later as a sanity check)
    true_decay = {"tv": 0.7, "search": 0.1, "social": 0.4, "display": 0.5, "video": 0.6}
    true_roi = {"tv": 2.8, "search": 4.5, "social": 3.2, "display": 1.6, "video": 2.1}
    base_spend = {"tv": 60_000, "search": 40_000, "social": 30_000,
                  "display": 20_000, "video": 25_000}

    # Each channel gets an INDEPENDENT spend pattern -> identifiable model
    spends = {ch: _independent_spend(base_spend[ch], n_weeks) for ch in CHANNELS}

    contributions = {}
    for ch in CHANNELS:
        sat = transform_channel(spends[ch], true_decay[ch])
        beta = true_roi[ch] * spends[ch].sum() / max(sat.sum(), 1e-9)  # so ROI == true_roi
        contributions[ch] = beta * sat

    base = 1_200_000 + 4_000 * t                       # trend
    seasonality = 180_000 * np.sin(2 * np.pi * t / 52.0)   # base seasonality, independent of media
    promo = 120_000 * (RNG.random(n_weeks) < 0.08)     # occasional promo spikes
    noise = RNG.normal(0, 40_000, n_weeks)

    sales = base + seasonality + promo + noise + sum(contributions.values())

    df = pd.DataFrame({"date": dates, "sales": sales})
    for ch in CHANNELS:
        df[f"{ch}_spend"] = spends[ch]

    truth = {"decay": true_decay, "roi": true_roi,
             "contribution": {c: contributions[c].sum() for c in CHANNELS}}
    return df, truth


def load_csv(path: str) -> pd.DataFrame:
    """Swap-in for real data: needs a 'date', a 'sales' column, and '<channel>_spend' columns."""
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# Fit
# --------------------------------------------------------------------------- #
def tune_and_fit(df: pd.DataFrame, train_idx: np.ndarray):
    """
    Joint base + media fit with non-negativity on media coefficients only, plus
    a greedy per-channel adstock-decay search. Base (trend + seasonality) and
    media are estimated TOGETHER, so the base can't absorb the media.
    """
    n = len(df)
    y = df["sales"].to_numpy(float)
    X_base = base_design(n)
    n_base = X_base.shape[1]

    decays = {ch: 0.4 for ch in CHANNELS}
    grid = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]

    def media_matrix(decay_map):
        """Transformed, column-normalized media matrix (so coefficients are comparable)."""
        M = np.column_stack(
            [transform_channel(df[f"{ch}_spend"].to_numpy(float), decay_map[ch]) for ch in CHANNELS]
        )
        scale = M.mean(axis=0)
        scale[scale == 0] = 1.0
        return M / scale

    def joint_fit(decay_map):
        M = media_matrix(decay_map)
        X = np.hstack([X_base, M])
        lb = np.r_[np.full(n_base, -np.inf), np.zeros(len(CHANNELS))]  # media >= 0
        ub = np.full(X.shape[1], np.inf)
        res = lsq_linear(X[train_idx], y[train_idx], bounds=(lb, ub), method="bvls")
        coef = res.x
        pred_tr = X[train_idx] @ coef
        ss_res = np.sum((y[train_idx] - pred_tr) ** 2)
        ss_tot = np.sum((y[train_idx] - y[train_idx].mean()) ** 2)
        return coef, M, 1 - ss_res / ss_tot

    # greedy adstock-decay search on train R^2
    _, _, best_r2 = joint_fit(decays)
    for _ in range(2):
        for ch in CHANNELS:
            for d in grid:
                trial = dict(decays)
                trial[ch] = d
                _, _, r2 = joint_fit(trial)
                if r2 > best_r2 + 1e-6:
                    best_r2, decays[ch] = r2, d

    coef, M, _ = joint_fit(decays)
    base_coef, media_coef = coef[:n_base], coef[n_base:]
    base_pred_full = X_base @ base_coef

    return {
        "base_coef": base_coef, "base_pred": base_pred_full,
        "decays": decays, "media_coef": media_coef, "M": M, "y": y,
    }


# --------------------------------------------------------------------------- #
# Decompose, validate, report
# --------------------------------------------------------------------------- #
def decompose(df, fit):
    contrib = pd.DataFrame({"date": df["date"]})
    contrib["base"] = fit["base_pred"]
    for i, ch in enumerate(CHANNELS):
        contrib[ch] = fit["M"][:, i] * fit["media_coef"][i]
    return contrib


def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-9, None))) * 100
    return r2, mape


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df, truth = simulate_data()
    n = len(df)

    split = int(n * 0.8)
    train_idx = np.arange(split)
    test_idx = np.arange(split, n)

    fit = tune_and_fit(df, train_idx)
    contrib = decompose(df, fit)

    y = df["sales"].to_numpy(float)
    pred = contrib[["base"] + CHANNELS].sum(axis=1).to_numpy()

    r2_tr, mape_tr = metrics(y[train_idx], pred[train_idx])
    r2_te, mape_te = metrics(y[test_idx], pred[test_idx])

    # ROI per channel
    rows = []
    for ch in CHANNELS:
        spend = df[f"{ch}_spend"].sum()
        c = contrib[ch].sum()
        rows.append({"channel": ch, "spend": spend, "contribution": c,
                     "roi": c / spend, "adstock_decay": fit["decays"][ch]})
    roi_df = pd.DataFrame(rows).sort_values("roi", ascending=False)

    # ---- report ----
    print("\n=== Marketing Mix Model ===")
    print(f"Weeks: {n}  |  Train: {len(train_idx)}  Holdout: {len(test_idx)}")
    print(f"Train   R^2={r2_tr:6.3f}  MAPE={mape_tr:5.2f}%")
    print(f"Holdout R^2={r2_te:6.3f}  MAPE={mape_te:5.2f}%")
    print("\nChannel ROI (contributed revenue per $ spent):")
    print(roi_df.to_string(index=False,
          formatters={"spend": "{:,.0f}".format, "contribution": "{:,.0f}".format,
                      "roi": "{:.2f}".format, "adstock_decay": "{:.1f}".format}))
    print("\nRecovered vs true ROI (sanity check against the simulation):")
    for ch in CHANNELS:
        est = roi_df.set_index("channel").loc[ch, "roi"]
        print(f"  {ch:8s} est={est:4.2f}  true={truth['roi'][ch]:4.2f}")

    roi_df.to_csv(os.path.join(OUT_DIR, "mmm_results.csv"), index=False)

    # ---- charts ----
    # 1) actual vs predicted
    plt.figure(figsize=(11, 4))
    plt.plot(df["date"], y, label="Actual sales", color="#222", lw=1.3)
    plt.plot(df["date"], pred, label="Predicted", color="#d1495b", lw=1.3)
    plt.axvline(df["date"].iloc[split], color="#888", ls="--", lw=1, label="Holdout start")
    plt.title("MMM fit: actual vs predicted weekly sales")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mmm_fit.png"), dpi=150)
    plt.close()

    # 2) stacked contribution decomposition
    plt.figure(figsize=(11, 4))
    plt.stackplot(df["date"], contrib["base"], *[contrib[c] for c in CHANNELS],
                  labels=["base"] + CHANNELS,
                  colors=["#cccccc", "#2e4057", "#33a1fd", "#f6ae2d", "#f26419", "#06a77d"])
    plt.title("Sales decomposition: base + media channel contributions")
    plt.legend(loc="upper left", ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mmm_contributions.png"), dpi=150)
    plt.close()

    # 3) ROI bar
    plt.figure(figsize=(7, 4))
    plt.bar(roi_df["channel"], roi_df["roi"], color="#2e4057")
    plt.ylabel("ROI (revenue per $ spent)")
    plt.title("Channel ROI")
    for i, v in enumerate(roi_df["roi"]):
        plt.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mmm_roi.png"), dpi=150)
    plt.close()

    print(f"\nSaved charts and mmm_results.csv to {OUT_DIR}/")


if __name__ == "__main__":
    main()
