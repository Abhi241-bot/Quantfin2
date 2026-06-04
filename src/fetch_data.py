"""
Data acquisition for the empirical half.

Downloads a real SPY (S&P 500 ETF) option chain via yfinance and writes it to
data/option_chain.csv in a SOURCE-AGNOSTIC schema so the rest of the pipeline
reads a plain CSV (swap in an NSE export later without touching the code).

Output columns:
    expiry        : option expiry date (YYYY-MM-DD)
    option_type   : 'call' or 'put'
    strike        : strike price
    bid, ask      : top-of-book quotes
    last          : last traded price
    mid           : (bid + ask) / 2   <- use THIS for implied vol (less stale)
    volume        : contracts traded
    open_interest : open contracts
    spot          : underlying spot at snapshot time
    snapshot_utc  : when this snapshot was taken

We choose SPY because (a) NSE blocks programmatic scraping and (b) SPY's deep
liquidity gives a clean, well-defined skew. This is the spec's recommended
fallback and is stated explicitly in the README.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import pandas as pd

TICKER = "SPY"
# Target horizons (calendar days). We pick the available expiry closest to each
# so the chain SPANS the term structure -- a proper 3D surface needs spread in
# time, and it dodges the degenerate ~0-day-to-expiry (0DTE) contracts whose IVs
# are noisy and extreme.
TARGET_DAYS = [7, 30, 60, 90]
MIN_DAYS = 3            # never include an expiry closer than this (T ~ 0 is junk)
OUT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "option_chain.csv")


def _select_expiries(expiries, target_days=TARGET_DAYS, min_days=MIN_DAYS):
    """Pick the available expiries closest to each target horizon.

    Returns a sorted, de-duplicated list. Falls back to the nearest valid
    expiries if too few clear the min_days filter.
    """
    today = datetime.now(timezone.utc).date()
    dated = []
    for e in expiries:
        d = (datetime.strptime(e, "%Y-%m-%d").date() - today).days
        if d >= min_days:
            dated.append((e, d))
    if not dated:
        # Market data too sparse / all near-dated: take whatever is furthest out.
        return sorted(expiries)[-len(target_days):]

    chosen = []
    for tgt in target_days:
        best = min(dated, key=lambda ed: abs(ed[1] - tgt))
        if best[0] not in chosen:
            chosen.append(best[0])
    return sorted(chosen)


def _get_spot(tk) -> float:
    """Best-effort spot from fast_info, falling back to last close."""
    try:
        px = float(tk.fast_info["last_price"])
        if px > 0:
            return px
    except Exception:
        pass
    hist = tk.history(period="1d")
    return float(hist["Close"].iloc[-1])


def fetch_chain(ticker=TICKER, out_path=OUT_PATH):
    import yfinance as yf

    tk = yf.Ticker(ticker)
    expiries = tk.options
    if not expiries:
        raise RuntimeError(f"No expiries returned for {ticker} "
                           "(market data unavailable / network issue).")
    expiries = _select_expiries(list(expiries))

    spot = _get_spot(tk)
    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"{ticker} spot = {spot:.2f}   snapshot = {snapshot}")
    print(f"Pulling {len(expiries)} expiries: {expiries}")

    frames = []
    for exp in expiries:
        chain = tk.option_chain(exp)
        for ot, df in (("call", chain.calls), ("put", chain.puts)):
            sub = df[["strike", "bid", "ask", "lastPrice",
                      "volume", "openInterest"]].copy()
            sub.columns = ["strike", "bid", "ask", "last",
                           "volume", "open_interest"]
            sub.insert(0, "option_type", ot)
            sub.insert(0, "expiry", exp)
            frames.append(sub)

    chain_df = pd.concat(frames, ignore_index=True)
    chain_df["mid"] = (chain_df["bid"] + chain_df["ask"]) / 2.0
    chain_df["spot"] = spot
    chain_df["snapshot_utc"] = snapshot

    chain_df = chain_df[[
        "expiry", "option_type", "strike", "bid", "ask", "last", "mid",
        "volume", "open_interest", "spot", "snapshot_utc",
    ]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    chain_df.to_csv(out_path, index=False)
    return chain_df, out_path


def sanity_report(df: pd.DataFrame) -> None:
    print("\n=== Data sanity checks ===")
    print(f"shape            : {df.shape}")
    print(f"expiries         : {sorted(df['expiry'].unique())}")
    print(f"option types     : {df['option_type'].value_counts().to_dict()}")
    print(f"strike range     : {df['strike'].min()} .. {df['strike'].max()}")
    print(f"spot             : {df['spot'].iloc[0]:.2f}")
    print("\nnull counts per column:")
    print(df.isnull().sum().to_string())
    print("\nrows with bid<=0 or ask<=0 (no real mid):",
          int(((df['bid'] <= 0) | (df['ask'] <= 0)).sum()))
    print("\nhead():")
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(df.head(8).to_string(index=False))


if __name__ == "__main__":
    try:
        df, path = fetch_chain()
    except Exception as e:
        print(f"ERROR fetching live chain: {e}", file=sys.stderr)
        print("Check network / market hours, or supply data/option_chain.csv "
              "manually (NSE export).", file=sys.stderr)
        sys.exit(1)
    sanity_report(df)
    print(f"\nSaved chain -> {path}")
