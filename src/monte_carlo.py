"""
Module C — Monte Carlo option pricing under geometric Brownian motion (GBM).

Risk-neutral pricing intuition: the arbitrage-free price of a European option is
the DISCOUNTED EXPECTED PAYOFF under the risk-neutral measure, where the
underlying drifts at the risk-free rate (not its real-world expected return):

    Price = e^{-rT} * E_Q[ payoff(S_T) ]

We don't know that expectation in closed form for arbitrary payoffs, so we
estimate it by averaging the payoff over many simulated terminal prices. Under
GBM the terminal price has an exact one-step solution (no time-stepping needed
for a European payoff):

    S_T = S * exp( (r - q - sigma^2 / 2) * T + sigma * sqrt(T) * Z ),   Z ~ N(0,1)

Because it's a sample mean, the estimator carries a STANDARD ERROR = s / sqrt(N)
that shrinks like 1/sqrt(N): to halve the error you need 4x the paths. We report
that SE and a 95% confidence interval so the estimate is honest about its noise.

Variance reduction (stretch): ANTITHETIC VARIATES pairs each draw Z with -Z. The
two payoffs are negatively correlated, so their average has lower variance than
two independent draws -- same accuracy for fewer effective samples.

All times T in YEARS, rates continuously-compounded decimals.
"""

from __future__ import annotations

import numpy as np

try:
    from .black_scholes import bs_price
    from .binomial_tree import crr_price
except ImportError:  # allow running as a script
    from black_scholes import bs_price
    from binomial_tree import crr_price

__all__ = ["mc_price"]


def _validate_option_type(option_type: str) -> str:
    ot = option_type.lower().strip()
    if ot in ("c", "call"):
        return "call"
    if ot in ("p", "put"):
        return "put"
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def _payoff(ST, K, ot):
    if ot == "call":
        return np.maximum(ST - K, 0.0)
    return np.maximum(K - ST, 0.0)


def mc_price(S, K, T, r, sigma, option_type="call", q=0.0, n_paths=100_000,
             antithetic=False, seed=None):
    """Monte Carlo price of a European option under GBM.

    Returns a dict with:
        price    : discounted mean payoff (the estimate)
        stderr   : standard error of the estimate
        ci95     : (low, high) 95% confidence interval
        n_paths  : number of simulated paths actually used

    With antithetic=True we draw n_paths/2 normals and append their negatives,
    so the total sample count stays n_paths but the draws are paired.
    """
    ot = _validate_option_type(option_type)
    rng = np.random.default_rng(seed)
    disc = np.exp(-r * T)
    drift = (r - q - 0.5 * sigma**2) * T
    diffusion = sigma * np.sqrt(T)

    if antithetic:
        # Draw n_paths/2 normals and pair each Z with -Z. The two payoffs are
        # negatively correlated, so the PAIR AVERAGE is the natural i.i.d. unit:
        # we must compute the standard error across pair-averages (not across all
        # individual payoffs), otherwise we throw away the variance reduction.
        half = n_paths // 2
        Z_half = rng.standard_normal(half)
        ST_pos = S * np.exp(drift + diffusion * Z_half)
        ST_neg = S * np.exp(drift + diffusion * (-Z_half))
        pay_pos = disc * _payoff(ST_pos, K, ot)
        pay_neg = disc * _payoff(ST_neg, K, ot)
        pair_means = 0.5 * (pay_pos + pay_neg)          # one i.i.d. sample / pair

        price = float(pair_means.mean())
        stderr = float(pair_means.std(ddof=1) / np.sqrt(half))
        n_used = 2 * half
    else:
        Z = rng.standard_normal(n_paths)
        ST = S * np.exp(drift + diffusion * Z)
        discounted = disc * _payoff(ST, K, ot)

        price = float(discounted.mean())
        stderr = float(discounted.std(ddof=1) / np.sqrt(len(discounted)))
        n_used = len(discounted)

    ci95 = (price - 1.96 * stderr, price + 1.96 * stderr)

    return {
        "price": price,
        "stderr": stderr,
        "ci95": ci95,
        "n_paths": n_used,
    }


if __name__ == "__main__":
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    SEED = 12345

    print("=== Monte Carlo self-test ===")
    print(f"S={S} K={K} T={T} r={r} sigma={sigma} q={q}\n")

    bs_call = bs_price(S, K, T, r, sigma, "call", q)

    # 1) Standard error shrinks like 1/sqrt(N).
    print("Standard error scaling (plain MC, call):")
    print(f"  {'N':>9} | {'price':>10} | {'stderr':>9} | {'true err':>9}")
    Ns = [1_000, 10_000, 100_000, 1_000_000]
    for n in Ns:
        res = mc_price(S, K, T, r, sigma, "call", q, n_paths=n, seed=SEED)
        print(f"  {n:>9} | {res['price']:>10.5f} | {res['stderr']:>9.5f} | "
              f"{abs(res['price'] - bs_call):>9.5f}")
    print(f"  Black-Scholes call = {bs_call:.5f}")
    print("  (stderr should roughly 1/sqrt(10)=3.16x shrink per 10x paths)\n")

    # 2) Antithetic variates: lower stderr at the same path count.
    n = 100_000
    plain = mc_price(S, K, T, r, sigma, "call", q, n_paths=n, seed=SEED)
    anti = mc_price(S, K, T, r, sigma, "call", q, n_paths=n, antithetic=True,
                    seed=SEED)
    print(f"Variance reduction at N={n:,} (call):")
    print(f"  plain      : price={plain['price']:.5f}  stderr={plain['stderr']:.5f}")
    print(f"  antithetic : price={anti['price']:.5f}  stderr={anti['stderr']:.5f}")
    print(f"  stderr ratio (anti/plain) = {anti['stderr'] / plain['stderr']:.3f}\n")

    # 3) THREE-WAY AGREEMENT TABLE -- the project's core correctness proof.
    print("=== Three-way agreement: BS vs binomial tree vs Monte Carlo ===")
    print(f"  {'option':>5} | {'Black-Scholes':>14} | {'CRR tree':>12} | "
          f"{'Monte Carlo (95% CI)':>30}")
    for ot in ("call", "put"):
        bs = bs_price(S, K, T, r, sigma, ot, q)
        tree = crr_price(S, K, T, r, sigma, ot, q, N=2000)
        mc = mc_price(S, K, T, r, sigma, ot, q, n_paths=1_000_000,
                      antithetic=True, seed=SEED)
        lo, hi = mc["ci95"]
        in_ci = "OK" if lo <= bs <= hi else "OUT"
        print(f"  {ot:>5} | {bs:>14.5f} | {tree:>12.5f} | "
              f"{mc['price']:>8.5f} [{lo:.4f}, {hi:.4f}] {in_ci}")
    print("\n  All three agree to within MC noise -> this is the unit test.\n")

    # Convergence plot: |MC - BS| with +/-1.96*SE band vs N.
    Ns_plot = np.unique(np.logspace(2.5, 6.5, 25).astype(int))
    errs, ses = [], []
    for n in Ns_plot:
        res = mc_price(S, K, T, r, sigma, "call", q, n_paths=int(n), seed=SEED)
        errs.append(abs(res["price"] - bs_call))
        ses.append(1.96 * res["stderr"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(Ns_plot, errs, marker="o", ms=4, lw=1, color="#1f77b4",
              label="|MC price - BS price|")
    ax.loglog(Ns_plot, ses, ls="--", color="crimson",
              label="1.96 x standard error")
    ref = errs[0] * np.sqrt(Ns_plot[0] / Ns_plot.astype(float))
    ax.loglog(Ns_plot, ref, ls=":", color="gray", label=r"$\propto 1/\sqrt{N}$ reference")
    ax.set_xlabel("Number of paths N")
    ax.set_ylabel("Pricing error (log scale)")
    ax.set_title("Monte Carlo error shrinks as 1/sqrt(N)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "results", "mc_convergence.png")
    fig.savefig(out, dpi=130)
    print(f"Saved MC convergence plot -> {out}")
