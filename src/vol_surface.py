"""
Module E — implied volatility smile / skew and the 3D vol surface.

Pipeline:
    1. Load the option chain CSV (source-agnostic schema from fetch_data.py).
    2. Compute time-to-expiry T in YEARS and moneyness K/S.
    3. Filter to clean, liquid, vega-meaningful quotes (the spec's anti-garbage
       rules): two-sided quotes, a moneyness band, minimum liquidity.
    4. Keep the OTM wing of each strike (puts below spot, calls above spot) --
       the most liquid, least early-exercise-contaminated quotes -- and back out
       Black-Scholes implied vol from the MID price.
    5. Plot IV vs moneyness for one expiry (the SMILE/SKEW) and, across expiries,
       a 3D IV surface.

Interpretation (written up in the README): equity-index options show a DOWNWARD
SKEW -- OTM puts trade at higher IV than OTM calls -- because the market pays up
for crash protection. The smile is the market CORRECTING Black-Scholes' wrong
flat-volatility / normal-returns assumption, not BS predicting the prices.

Conventions: r and q below are stated assumptions for SPY (US 3M T-bill and SPY
dividend yield). SPY options are American; we use short-dated OTM contracts where
the early-exercise premium is negligible and treat them as European for the BS
inversion. State this in the README.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from .implied_vol import implied_vol
except ImportError:  # allow running as a script
    from implied_vol import implied_vol

# --- Stated market assumptions for SPY (see README) -------------------------
RISK_FREE_RATE = 0.043     # US 3-month T-bill, continuously-compounded approx.
DIVIDEND_YIELD = 0.012     # SPY trailing dividend yield, approx.

# --- Cleaning / filtering parameters ----------------------------------------
MONEYNESS_LO = 0.80        # keep 0.80 <= K/S <= 1.20 (drop deep wings: vega~0)
MONEYNESS_HI = 1.20
MIN_OI = 1                 # require some open interest
MIN_PRICE = 0.05           # drop sub-nickel prices (one tick of noise = huge IV)

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "option_chain.csv")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def load_chain(path=DATA_PATH):
    """Load the chain CSV and add time-to-expiry (years) and moneyness."""
    df = pd.read_csv(path)
    snap = pd.to_datetime(df["snapshot_utc"].iloc[0], utc=True)
    exp = pd.to_datetime(df["expiry"], utc=True) + pd.Timedelta(hours=20)  # ~US close
    df["T"] = (exp - snap).dt.total_seconds() / (365.0 * 24 * 3600)
    df["moneyness"] = df["strike"] / df["spot"]
    return df


def clean_chain(df):
    """Apply the anti-garbage filters from the spec's pitfalls section."""
    n0 = len(df)
    m = (
        (df["bid"] > 0) & (df["ask"] > 0) &        # genuine two-sided market
        (df["ask"] >= df["bid"]) &                 # sane quote
        (df["mid"] >= MIN_PRICE) &                 # not sub-tick noise
        (df["T"] > 0) &                            # not expired
        (df["open_interest"].fillna(0) >= MIN_OI) &
        (df["moneyness"].between(MONEYNESS_LO, MONEYNESS_HI))
    )
    out = df[m].copy()
    print(f"clean_chain: kept {len(out)}/{n0} rows after filtering")
    return out


def select_otm(df):
    """Keep the out-of-the-money wing: puts for K<S, calls for K>=S.

    These are the liquid, early-exercise-clean quotes and they stitch into one
    continuous smile across strikes.
    """
    is_otm_put = (df["option_type"] == "put") & (df["strike"] < df["spot"])
    is_otm_call = (df["option_type"] == "call") & (df["strike"] >= df["spot"])
    return df[is_otm_put | is_otm_call].copy()


def compute_iv(df, r=RISK_FREE_RATE, q=DIVIDEND_YIELD):
    """Back out implied vol from the MID price for each row; drop failures."""
    ivs = []
    for row in df.itertuples(index=False):
        iv = implied_vol(
            market_price=row.mid, S=row.spot, K=row.strike,
            T=row.T, r=r, option_type=row.option_type, q=q,
        )
        ivs.append(iv)
    df = df.assign(iv=ivs)
    n_before = len(df)
    df = df[np.isfinite(df["iv"])].copy()
    print(f"compute_iv: {len(df)}/{n_before} rows produced a finite IV")
    return df


def plot_smile(df, expiry, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[df["expiry"] == expiry].sort_values("moneyness")
    if sub.empty:
        print(f"plot_smile: no data for expiry {expiry}")
        return
    T_days = sub["T"].iloc[0] * 365.0

    fig, ax = plt.subplots(figsize=(9, 5.5))
    puts = sub[sub["option_type"] == "put"]
    calls = sub[sub["option_type"] == "call"]
    ax.plot(puts["moneyness"], puts["iv"] * 100, "o-", ms=5, color="crimson",
            label="OTM puts (downside)")
    ax.plot(calls["moneyness"], calls["iv"] * 100, "o-", ms=5, color="#1f77b4",
            label="OTM calls (upside)")
    ax.axvline(1.0, color="gray", ls=":", lw=1, label="at-the-money (K=S)")
    ax.set_xlabel("Moneyness  K / S")
    ax.set_ylabel("Implied volatility (%)")
    ax.set_title(f"SPY implied-volatility smile / skew\n"
                 f"expiry {expiry}  (~{T_days:.0f} days)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved smile plot -> {out_path}")


def plot_surface(df, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    x = df["moneyness"].values
    y = (df["T"] * 365.0).values
    z = (df["iv"] * 100).values
    surf = ax.plot_trisurf(x, y, z, cmap=cm.viridis, linewidth=0.2,
                           edgecolor="gray", alpha=0.9)
    ax.set_xlabel("Moneyness K/S")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied vol (%)")
    ax.set_title("SPY implied-volatility surface")
    ax.view_init(elev=22, azim=-60)
    fig.colorbar(surf, shrink=0.5, aspect=12, label="IV (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Saved 3D surface plot -> {out_path}")


def skew_report(df):
    """Quantify the skew: compare OTM-put IV vs OTM-call IV per expiry."""
    print("\n=== Skew analysis (per expiry) ===")
    print(f"  {'expiry':>12} | {'~days':>5} | {'ATM IV':>7} | "
          f"{'25d-ish put':>11} | {'25d-ish call':>12} | {'put-call skew':>13}")
    for exp, g in df.groupby("expiry"):
        g = g.sort_values("moneyness")
        T_days = g["T"].iloc[0] * 365.0
        # ATM ~ closest to moneyness 1.0
        atm = g.iloc[(g["moneyness"] - 1.0).abs().argmin()]["iv"]
        # OTM put wing ~ moneyness 0.95, OTM call wing ~ 1.05 (proxy for 25-delta)
        puts = g[g["option_type"] == "put"]
        calls = g[g["option_type"] == "call"]
        if puts.empty or calls.empty:
            continue
        put_w = puts.iloc[(puts["moneyness"] - 0.95).abs().argmin()]["iv"]
        call_w = calls.iloc[(calls["moneyness"] - 1.05).abs().argmin()]["iv"]
        skew = put_w - call_w
        print(f"  {exp:>12} | {T_days:>5.0f} | {atm*100:>6.2f}% | "
              f"{put_w*100:>10.2f}% | {call_w*100:>11.2f}% | "
              f"{skew*100:>+12.2f}%")
    print("\n  Positive put-minus-call skew => OTM puts richer than OTM calls")
    print("  => market pays up for crash protection (downward / negative skew).")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=== Module E: implied-vol smile & surface ===")
    print(f"Assumptions: r={RISK_FREE_RATE:.3f} (US 3M), q={DIVIDEND_YIELD:.3f} (SPY div)\n")

    raw = load_chain()
    print(f"loaded {len(raw)} rows; spot={raw['spot'].iloc[0]:.2f}; "
          f"expiries={sorted(raw['expiry'].unique())}")
    print(f"T (years) per expiry: "
          f"{ {e: round(g['T'].iloc[0],4) for e,g in raw.groupby('expiry')} }\n")

    clean = clean_chain(raw)
    otm = select_otm(clean)
    print(f"select_otm: kept {len(otm)} OTM-wing rows")
    iv_df = compute_iv(otm)

    print("\nIV summary by expiry:")
    print(iv_df.groupby("expiry")["iv"].describe()[["count", "mean", "min", "max"]]
          .to_string())

    # Smile for the nearest expiry; 3D surface across all expiries.
    expiries = sorted(iv_df["expiry"].unique())
    plot_smile(iv_df, expiries[0],
               os.path.join(RESULTS_DIR, "vol_smile.png"))
    if len(expiries) >= 2:
        plot_surface(iv_df, os.path.join(RESULTS_DIR, "vol_surface_3d.png"))

    skew_report(iv_df)

    # Persist the computed IV table for the notebook / inspection.
    iv_out = os.path.join(os.path.dirname(DATA_PATH), "implied_vols.csv")
    iv_df.to_csv(iv_out, index=False)
    print(f"\nSaved IV table -> {iv_out}")
