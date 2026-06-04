"""
Module B — Cox-Ross-Rubinstein (CRR) binomial tree pricer.

Idea: discretise the option's life into N steps of length dt = T/N. Over each
step the underlying moves up by factor u or down by factor d, chosen so the
tree's log-return variance matches the Black-Scholes volatility:

    u = exp(sigma * sqrt(dt)),   d = 1/u

Under the RISK-NEUTRAL measure the up-probability is

    p = (exp((r - q) * dt) - d) / (u - d)

We then roll the option payoff backwards through the tree, discounting one step
at a time at the risk-free rate. For EUROPEAN options we just discount the
expected continuation value; for AMERICAN options we take max(continuation,
immediate exercise) at every node -- which is exactly the early-exercise feature
Black-Scholes cannot capture.

As N -> infinity the CRR price converges to Black-Scholes for European options.
That convergence is a strong correctness signal and is plotted in __main__.

All times T in YEARS, rates continuously-compounded decimals.
"""

from __future__ import annotations

import numpy as np

try:
    from .black_scholes import bs_price
except ImportError:  # allow running as a script
    from black_scholes import bs_price

__all__ = ["crr_price"]


def _validate_option_type(option_type: str) -> str:
    ot = option_type.lower().strip()
    if ot in ("c", "call"):
        return "call"
    if ot in ("p", "put"):
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def crr_price(S, K, T, r, sigma, option_type="call", q=0.0, N=500, american=False):
    """Price a European or American option on an N-step CRR binomial tree.

    Parameters
    ----------
    S, K, T, r, sigma, q : standard BS inputs (T in years, rates decimal)
    option_type : 'call' or 'put'
    N : number of time steps (more steps -> closer to continuous / BS)
    american : if True, allow early exercise at every node
    """
    ot = _validate_option_type(option_type)
    if N < 1:
        raise ValueError("N must be >= 1")

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 < p < 1.0):
        # Arbitrage / parameters too coarse for these step sizes.
        raise ValueError(
            f"Risk-neutral prob p={p:.4f} outside (0,1); "
            "check inputs or increase N."
        )

    # Terminal underlying prices: S * u^j * d^(N-j) for j = 0..N up-moves.
    j = np.arange(N + 1)
    ST = S * u**j * d**(N - j)

    if ot == "call":
        values = np.maximum(ST - K, 0.0)
    else:
        values = np.maximum(K - ST, 0.0)

    # Roll backwards through the tree.
    for i in range(N, 0, -1):
        values = disc * (p * values[1:i + 1] + (1.0 - p) * values[0:i])
        if american:
            # Underlying prices at step i-1 (i nodes).
            j = np.arange(i)
            S_nodes = S * u**j * d**((i - 1) - j)
            if ot == "call":
                exercise = np.maximum(S_nodes - K, 0.0)
            else:
                exercise = np.maximum(K - S_nodes, 0.0)
            values = np.maximum(values, exercise)

    return float(values[0])


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0

    print("=== CRR binomial tree self-test ===")
    print(f"S={S} K={K} T={T} r={r} sigma={sigma} q={q}\n")

    bs_call = bs_price(S, K, T, r, sigma, "call", q)
    bs_put = bs_price(S, K, T, r, sigma, "put", q)

    print("European convergence to Black-Scholes:")
    print(f"  {'N':>5} | {'tree call':>11} | {'abs err':>10}")
    for N in (10, 50, 100, 500, 1000, 2000):
        tc = crr_price(S, K, T, r, sigma, "call", q, N=N)
        print(f"  {N:>5} | {tc:>11.6f} | {abs(tc - bs_call):>10.2e}")
    print(f"  Black-Scholes call = {bs_call:.6f}\n")

    # American vs European put: early-exercise premium should be > 0.
    euro_put = crr_price(S, K, T, r, sigma, "put", q, N=1000, american=False)
    amer_put = crr_price(S, K, T, r, sigma, "put", q, N=1000, american=True)
    print("Early-exercise premium (puts, N=1000):")
    print(f"  European put (tree) = {euro_put:.6f}  (BS put = {bs_put:.6f})")
    print(f"  American  put (tree) = {amer_put:.6f}")
    print(f"  early-exercise premium = {amer_put - euro_put:.6f}  "
          f"(should be >= 0)\n")

    # American call on a NON-dividend stock should equal the European call
    # (never optimal to exercise early) -- a classic correctness check.
    amer_call = crr_price(S, K, T, r, sigma, "call", q, N=1000, american=True)
    euro_call = crr_price(S, K, T, r, sigma, "call", q, N=1000, american=False)
    print("American vs European call, q=0 (should be ~equal):")
    print(f"  European call = {euro_call:.6f}")
    print(f"  American call = {amer_call:.6f}")
    print(f"  difference    = {abs(amer_call - euro_call):.2e}\n")

    # Convergence plot.
    Ns = np.arange(5, 405, 5)
    tree_prices = [crr_price(S, K, T, r, sigma, "call", q, N=int(n)) for n in Ns]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(Ns, tree_prices, marker="o", ms=3, lw=1,
            label="CRR binomial tree", color="#1f77b4")
    ax.axhline(bs_call, color="crimson", ls="--", lw=1.5,
               label=f"Black-Scholes = {bs_call:.4f}")
    ax.set_xlabel("Number of tree steps N")
    ax.set_ylabel("European call price")
    ax.set_title("Binomial tree converges to Black-Scholes\n"
                 "(note the characteristic odd/even oscillation that damps as N grows)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "results", "method_convergence.png")
    fig.savefig(out, dpi=130)
    print(f"Saved convergence plot -> {out}")
