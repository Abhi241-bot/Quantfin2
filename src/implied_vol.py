"""
Module D — implied volatility by root-finding.

Black-Scholes maps (sigma -> price) but is NOT invertible in closed form for
sigma: the unknown sits inside N(d1), N(d2) where d1, d2 themselves depend on
sigma, so there is no algebraic rearrangement to isolate it. We therefore solve
the inverse problem numerically -- find the sigma that reprices the option:

    f(sigma) = bs_price(S, K, T, r, sigma, type) - market_price = 0

This is a clean job for a bracketing root-finder. The BS price is STRICTLY
INCREASING in sigma (vega > 0), so on any interval [lo, hi] where f changes sign
there is exactly one root. We use scipy.optimize.brentq -- bracketing + inverse
quadratic interpolation, guaranteed to converge given a valid sign-change.

Edge cases handled (return NaN rather than garbage):
  * Price below intrinsic value / above no-arbitrage upper bound -> no real IV.
  * Deep ITM/OTM where vega ~ 0: the price is nearly flat in sigma, so IV is
    numerically unidentifiable; we flag these so the caller can filter them.
  * Non-positive T, K, S, or price.

All times T in YEARS, rates continuously-compounded decimals.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

try:
    from .black_scholes import bs_price, vega
except ImportError:  # allow running as a script
    from black_scholes import bs_price, vega

__all__ = ["implied_vol", "no_arbitrage_bounds"]


def _validate_option_type(option_type: str) -> str:
    ot = option_type.lower().strip()
    if ot in ("c", "call"):
        return "call"
    if ot in ("p", "put"):
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def no_arbitrage_bounds(S, K, T, r, option_type, q=0.0):
    """Return (lower, upper) no-arbitrage price bounds for a European option.

    Any market price outside [lower, upper] cannot correspond to a real,
    non-negative implied volatility under Black-Scholes.

        call: max(S e^{-qT} - K e^{-rT}, 0)  <= C <= S e^{-qT}
        put : max(K e^{-rT} - S e^{-qT}, 0)  <= P <= K e^{-rT}
    """
    ot = _validate_option_type(option_type)
    fwd_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if ot == "call":
        return max(fwd_S - disc_K, 0.0), fwd_S
    return max(disc_K - fwd_S, 0.0), disc_K


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0,
                lo=1e-6, hi=5.0, vega_floor=1e-6):
    """Back out Black-Scholes implied volatility from a market price.

    Returns the implied volatility (decimal, annualised) or np.nan if the price
    is unreachable / the option is numerically vega-dead.

    Parameters
    ----------
    market_price : observed option price (use mid of bid/ask)
    lo, hi : volatility search bracket
    vega_floor : if vega at the solved sigma is below this, the IV is considered
        unreliable (deep ITM/OTM, price insensitive to sigma) -> return NaN.
    """
    ot = _validate_option_type(option_type)

    # Basic input sanity.
    if not np.isfinite(market_price) or market_price <= 0:
        return np.nan
    if S <= 0 or K <= 0 or T <= 0:
        return np.nan

    # No-arbitrage bound check: a price outside the bounds has no real IV.
    lower, upper = no_arbitrage_bounds(S, K, T, r, ot, q)
    # Small tolerance for prices sitting essentially at the bounds.
    tol = 1e-8 * max(1.0, S)
    if market_price < lower - tol or market_price > upper + tol:
        return np.nan

    def objective(sigma):
        return bs_price(S, K, T, r, sigma, ot, q) - market_price

    f_lo = objective(lo)
    f_hi = objective(hi)
    # Need a sign change to bracket the root. If both ends have the same sign
    # the target price lies outside [price(lo), price(hi)] -- unreachable here.
    if np.sign(f_lo) == np.sign(f_hi):
        return np.nan

    try:
        iv = brentq(objective, lo, hi, xtol=1e-8, rtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan

    # Vega gate: if the option is essentially insensitive to sigma at the
    # solution, the IV is not trustworthy -- filter it out.
    if vega(S, K, T, r, iv, q) < vega_floor:
        return np.nan

    return float(iv)


if __name__ == "__main__":
    from black_scholes import bs_price as _bsp

    S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.0
    print("=== Implied volatility self-test ===")
    print(f"S={S} K={K} T={T} r={r} q={q}\n")

    # 1) Round-trip: price at a known sigma, then recover it.
    print("Round-trip recovery (price at true sigma -> invert -> compare):")
    print(f"  {'type':>4} | {'true sigma':>10} | {'price':>9} | {'recovered':>10} | {'err':>9}")
    for ot in ("call", "put"):
        for true_sig in (0.10, 0.20, 0.35, 0.60):
            px = _bsp(S, K, T, r, true_sig, ot, q)
            iv = implied_vol(px, S, K, T, r, ot, q)
            print(f"  {ot:>4} | {true_sig:>10.4f} | {px:>9.4f} | "
                  f"{iv:>10.6f} | {abs(iv - true_sig):>9.2e}")
    print()

    # 2) Across strikes (constant input vol -> IV should come back flat).
    print("Recover flat IV across strikes (true sigma=0.25, call):")
    true_sig = 0.25
    print(f"  {'K':>6} | {'moneyness K/S':>13} | {'price':>9} | {'IV':>9}")
    for Kx in (70, 85, 100, 115, 130):
        px = _bsp(S, Kx, T, r, true_sig, "call", q)
        iv = implied_vol(px, S, Kx, T, r, "call", q)
        iv_str = f"{iv:.6f}" if np.isfinite(iv) else "NaN"
        print(f"  {Kx:>6} | {Kx / S:>13.3f} | {px:>9.4f} | {iv_str:>9}")
    print()

    # 3) Edge cases that must return NaN.
    print("Edge cases (expect NaN):")
    intrinsic_call = max(S - K * np.exp(-r * T), 0.0)
    cases = [
        ("price below intrinsic", implied_vol(intrinsic_call - 1.0, S, K, T, r, "call", q)),
        ("zero price", implied_vol(0.0, S, K, T, r, "call", q)),
        ("price above spot (call)", implied_vol(S + 5.0, S, K, T, r, "call", q)),
        ("negative time", implied_vol(5.0, S, K, -0.1, r, "call", q)),
    ]
    for label, val in cases:
        print(f"  {label:>30} -> {val}")
    print()

    # 4) Vega gate demonstration on a genuinely vega-thin deep-OTM option.
    #    We price a far OTM call at a real sigma, recover it normally, then show
    #    that raising vega_floor above the option's actual vega makes the gate
    #    fire (because here a tiny price wobble would swing the IV wildly).
    from black_scholes import vega as _vega
    K_far, sig_far = 200.0, 0.20
    px_far = _bsp(S, K_far, T, r, sig_far, "call", q)
    v_far = _vega(S, K_far, T, r, sig_far, q)
    iv_default = implied_vol(px_far, S, K_far, T, r, "call", q)
    iv_gated = implied_vol(px_far, S, K_far, T, r, "call", q, vega_floor=v_far * 2)
    print("Vega gate (deep OTM K=200 call, true sigma=0.20):")
    print(f"  price={px_far:.6f}  vega={v_far:.6f}")
    print(f"  IV (default floor 1e-6)        -> {iv_default:.6f}  (recovered fine)")
    print(f"  IV (floor set above this vega) -> {iv_gated}  (gate fires -> NaN)")
