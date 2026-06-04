"""
Module A — Black-Scholes pricing and Greeks (from scratch).

The Black-Scholes-Merton model prices a European option under the assumptions:
    - The underlying follows geometric Brownian motion with constant volatility sigma.
    - Prices are log-normally distributed (log-returns are normal).
    - No arbitrage, continuous frictionless trading / hedging.
    - Constant risk-free rate r and constant dividend yield q.

For a non-dividend-paying-adjusted underlying:
    d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Call = S e^{-qT} N(d1) - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

All times T are in YEARS, all rates are continuously-compounded decimals (0.065, not 6.5).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "d1_d2",
    "bs_price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "put_call_parity_check",
]


def _validate_option_type(option_type: str) -> str:
    ot = option_type.lower().strip()
    if ot in ("c", "call"):
        return "call"
    if ot in ("p", "put"):
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def d1_d2(S, K, T, r, sigma, q=0.0):
    """Return the Black-Scholes d1 and d2 terms.

    Kept as a shared helper because every price/Greek formula reuses these two
    numbers; computing them once keeps the math visible and avoids drift.
    """
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """European option price under Black-Scholes-Merton.

    Parameters
    ----------
    S : spot price of the underlying
    K : strike price
    T : time to expiry in YEARS
    r : continuously-compounded risk-free rate (decimal)
    sigma : volatility (annualised, decimal)
    option_type : 'call' or 'put'
    q : continuous dividend yield (decimal), default 0
    """
    ot = _validate_option_type(option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    if ot == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dS. Call delta in (0, 1), put delta in (-1, 0).

    Derivation: differentiating the price w.r.t. S, the terms in N'(d1) cancel
    (the classic identity S e^{-qT} N'(d1) = K e^{-rT} N'(d2)), leaving the
    clean e^{-qT} N(d1) form.
    """
    ot = _validate_option_type(option_type)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    if ot == "call":
        return disc_q * norm.cdf(d1)
    return disc_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """d^2 Price/dS^2 -- same for calls and puts.

    gamma = e^{-qT} N'(d1) / (S sigma sqrt(T)). It is highest at-the-money and
    near expiry, because that is where delta changes most rapidly with spot.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return disc_q * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """dPrice/dsigma -- same for calls and puts.

    Derivation (common interview question): the explicit dependence on sigma
    appears through d1 and d2. When you differentiate, the chain-rule terms
    multiplying dd1/dsigma and dd2/dsigma again collapse via the identity
    S e^{-qT} N'(d1) = K e^{-rT} N'(d2), and since d1 - d2 = sigma sqrt(T) you are
    left with vega = S e^{-qT} N'(d1) sqrt(T).

    Returned in PRICE-per-1.00-of-vol. Divide by 100 for per-1-vol-point.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = np.exp(-q * T)
    return S * disc_q * norm.pdf(d1) * np.sqrt(T)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dT_calendar (per YEAR). Usually negative -- time decay.

    Divide by 365 for per-calendar-day theta.
    """
    ot = _validate_option_type(option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    disc_q = np.exp(-q * T)
    term1 = -(S * disc_q * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if ot == "call":
        return (
            term1
            - r * K * disc_r * norm.cdf(d2)
            + q * S * disc_q * norm.cdf(d1)
        )
    return (
        term1
        + r * K * disc_r * norm.cdf(-d2)
        - q * S * disc_q * norm.cdf(-d1)
    )


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    """dPrice/dr (per 1.00 of rate). Divide by 100 for per-1%-rate."""
    ot = _validate_option_type(option_type)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_r = np.exp(-r * T)
    if ot == "call":
        return K * T * disc_r * norm.cdf(d2)
    return -K * T * disc_r * norm.cdf(-d2)


def put_call_parity_check(S, K, T, r, sigma, q=0.0, tol=1e-8):
    """Verify C - P = S e^{-qT} - K e^{-rT}.

    Returns (ok, lhs, rhs, abs_error). This is a model-free no-arbitrage
    relationship, so it is a clean sanity check on the pricing code.
    """
    call = bs_price(S, K, T, r, sigma, "call", q)
    put = bs_price(S, K, T, r, sigma, "put", q)
    lhs = call - put
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    err = float(np.abs(lhs - rhs))
    return bool(err < tol), float(lhs), float(rhs), err


if __name__ == "__main__":
    # Self-test with a textbook setup.
    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    print("=== Black-Scholes self-test ===")
    print(f"S={S} K={K} T={T} r={r} sigma={sigma} q={q}\n")

    c = bs_price(S, K, T, r, sigma, "call", q)
    p = bs_price(S, K, T, r, sigma, "put", q)
    print(f"Call price : {c:.6f}   (Hull textbook ATM ~10.45)")
    print(f"Put  price : {p:.6f}\n")

    print("Greeks (call):")
    print(f"  delta : {delta(S, K, T, r, sigma, 'call', q):.6f}")
    print(f"  gamma : {gamma(S, K, T, r, sigma, q):.6f}")
    print(f"  vega  : {vega(S, K, T, r, sigma, q):.6f}  (per 1.00 vol; /100 = per vol-pt)")
    print(f"  theta : {theta(S, K, T, r, sigma, 'call', q):.6f}  (per year)")
    print(f"  rho   : {rho(S, K, T, r, sigma, 'call', q):.6f}\n")

    print("Greeks (put):")
    print(f"  delta : {delta(S, K, T, r, sigma, 'put', q):.6f}")
    print(f"  theta : {theta(S, K, T, r, sigma, 'put', q):.6f}")
    print(f"  rho   : {rho(S, K, T, r, sigma, 'put', q):.6f}\n")

    ok, lhs, rhs, err = put_call_parity_check(S, K, T, r, sigma, q)
    print("Put-call parity:  C - P = S e^{-qT} - K e^{-rT}")
    print(f"  LHS (C-P) = {lhs:.8f}")
    print(f"  RHS       = {rhs:.8f}")
    print(f"  abs error = {err:.2e}   -> {'PASS' if ok else 'FAIL'}\n")

    # Finite-difference check that the analytic Greeks match numeric derivatives.
    h = 1e-4
    fd_delta = (bs_price(S + h, K, T, r, sigma, "call", q)
                - bs_price(S - h, K, T, r, sigma, "call", q)) / (2 * h)
    fd_vega = (bs_price(S, K, T, r, sigma + h, "call", q)
               - bs_price(S, K, T, r, sigma - h, "call", q)) / (2 * h)
    print("Finite-difference Greek validation (call):")
    print(f"  delta analytic={delta(S, K, T, r, sigma, 'call', q):.6f}  fd={fd_delta:.6f}")
    print(f"  vega  analytic={vega(S, K, T, r, sigma, q):.6f}  fd={fd_vega:.6f}")
