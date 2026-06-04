"""
Module F (capstone) — delta-hedging simulation and hedging error.

What this demonstrates: the Black-Scholes price is the COST OF REPLICATION, not a
forecast. If you sell an option for its BS price and then continuously hold
delta shares of the underlying (financing the rest at the risk-free rate), the
hedge portfolio's value at expiry exactly matches the option payoff -- your P&L
is zero. You can do this WITHOUT knowing the real drift mu: that is the whole
point of risk-neutral pricing.

In reality you can only rebalance at discrete times, so a residual HEDGING ERROR
remains. Theory (Boyle-Emanuel) says its standard deviation scales like
1/sqrt(n_rebalances): rebalance 4x as often -> halve the error. As frequency ->
infinity the error variance -> 0, recovering the continuous-hedging ideal behind
Black-Scholes. We simulate this and plot it.

----------------------------------------------------------------------------
Discipline checkpoint (this is the first module with paths / positions / P&L):
  * NO LOOK-AHEAD: the hedge ratio at each rebalance is delta(S_t, tau_t) using
    only the spot and the remaining time KNOWN AT THAT INSTANT. The terminal
    payoff uses S_T only at expiry. No future price ever informs a present trade.
  * COSTS: a per-trade proportional transaction cost is applied on EVERY position
    change -- the initial hedge, every rebalance, and the final liquidation. The
    headline experiment is frictionless (tc=0) to isolate discretization error;
    we also report a with-cost run so the effect is explicit and honest.
  * The hedge trades at the contemporaneous observed price (correct for a hedge,
    which is not a predictive signal that needs lagging).
  * Drift-independence: paths are simulated under a REAL-WORLD drift mu that need
    not equal r, yet the mean hedging error stays ~0 -- the replication result.
----------------------------------------------------------------------------

All times T in YEARS, rates continuously-compounded decimals.
"""

from __future__ import annotations

import os

import numpy as np

try:
    from .black_scholes import bs_price, delta as bs_delta
except ImportError:  # allow running as a script
    from black_scholes import bs_price, delta as bs_delta


def simulate_delta_hedge(S0, K, T, r, sigma, option_type="call", q=0.0,
                         n_rebalance=21, n_paths=20_000, mu=None, tc=0.0,
                         seed=None):
    """Simulate selling one option and delta-hedging it to expiry.

    Returns the array of per-path hedging errors (final P&L of: short option +
    dynamic stock hedge + interest-bearing cash account). Ideal continuous hedge
    => all zeros; discrete hedge => mean ~0, spread shrinking with n_rebalance.

    Parameters
    ----------
    n_rebalance : number of hedge-adjustment intervals over the option's life
    n_paths     : number of simulated GBM paths
    mu          : real-world drift for the simulation (defaults to r). Choosing
                  mu != r is a deliberate test that hedging is drift-independent.
    tc          : proportional transaction cost per unit notional traded (e.g.
                  0.0005 = 5 bps). Applied on every position change.
    """
    if mu is None:
        mu = r
    rng = np.random.default_rng(seed)
    dt = T / n_rebalance
    disc_step = np.exp(r * dt)

    # Simulate GBM paths under the REAL-WORLD measure: shape (n_paths, n_reb+1).
    Z = rng.standard_normal((n_paths, n_rebalance))
    log_increments = (mu - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    S = S0 * np.exp(log_paths)

    # t=0: sell option (receive premium), set up initial hedge.
    premium = bs_price(S0, K, T, r, sigma, option_type, q)
    delta_prev = np.full(n_paths, bs_delta(S0, K, T, r, sigma, option_type, q))
    # Cash = premium received, minus cost of buying delta_0 shares, minus the
    # transaction cost on that initial trade.
    cash = premium - delta_prev * S0 - tc * np.abs(delta_prev) * S0

    # Interior rebalances at times 1 .. n_rebalance-1 (NOT expiry).
    for i in range(1, n_rebalance):
        cash *= disc_step                      # cash accrues interest over dt
        tau = T - i * dt                       # remaining time -> known now
        Si = S[:, i]                           # spot observed now
        delta_new = bs_delta(Si, K, tau, r, sigma, option_type, q)
        trade = delta_new - delta_prev         # shares to buy(+)/sell(-)
        cash -= trade * Si                     # pay for the trade
        cash -= tc * np.abs(trade) * Si        # transaction cost on the trade
        delta_prev = delta_new

    # Expiry: final interest accrual, liquidate the stock hedge, settle option.
    cash *= disc_step
    S_T = S[:, n_rebalance]
    cash += delta_prev * S_T                   # sell remaining shares
    cash -= tc * np.abs(delta_prev) * S_T      # cost to liquidate
    if option_type.lower().startswith("c"):
        payoff = np.maximum(S_T - K, 0.0)
    else:
        payoff = np.maximum(K - S_T, 0.0)
    cash -= payoff                             # pay what we owe on the short

    return cash                                # hedging error (P&L) per path


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.20, 0.0
    MU = 0.10          # real-world drift != r, to prove drift-independence
    N_PATHS = 40_000
    SEED = 7
    premium = bs_price(S0, K, T, r, sigma, "call", q)

    print("=== Delta-hedging simulation (short 1 ATM call) ===")
    print(f"S0={S0} K={K} T={T} r={r} sigma={sigma} q={q}")
    print(f"BS premium received = {premium:.4f}")
    print(f"Path drift mu={MU} (deliberately != r={r} to test drift-independence)\n")

    freqs = [1, 2, 5, 10, 21, 50, 100, 252]
    print("Frictionless hedging error vs rebalance frequency:")
    print(f"  {'n_rebalance':>11} | {'mean err':>9} | {'std err':>9} | "
          f"{'std / premium':>13}")
    stds = []
    errors_by_freq = {}
    for n in freqs:
        err = simulate_delta_hedge(S0, K, T, r, sigma, "call", q,
                                   n_rebalance=n, n_paths=N_PATHS, mu=MU,
                                   tc=0.0, seed=SEED)
        stds.append(err.std())
        errors_by_freq[n] = err
        print(f"  {n:>11} | {err.mean():>+9.4f} | {err.std():>9.4f} | "
              f"{err.std() / premium:>12.1%}")
    print("\n  mean ~ 0 (drift-independent replication); std shrinks ~ 1/sqrt(n).")

    # Check the 1/sqrt(n) scaling: std * sqrt(n) should be roughly constant.
    print("\n  scaling check (std * sqrt(n), should be ~flat):")
    for n, s in zip(freqs, stds):
        print(f"    n={n:>3}: std*sqrt(n) = {s * np.sqrt(n):.4f}")

    # With transaction costs: error mean turns negative (costs bleed P&L), and
    # MORE rebalancing now COSTS more -- the realistic tension.
    print("\nWith transaction costs (5 bps per trade):")
    print(f"  {'n_rebalance':>11} | {'mean err':>9} | {'std err':>9}")
    for n in [5, 21, 100, 252]:
        err = simulate_delta_hedge(S0, K, T, r, sigma, "call", q,
                                   n_rebalance=n, n_paths=N_PATHS, mu=MU,
                                   tc=0.0005, seed=SEED)
        print(f"  {n:>11} | {err.mean():>+9.4f} | {err.std():>9.4f}")
    print("  -> costs push mean P&L negative and grow with rebalancing: the")
    print("     real-world tradeoff between hedging error and transaction cost.")

    # --- Plots: error distributions + std-vs-frequency scaling ---------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    for n in [1, 10, 100]:
        ax1.hist(errors_by_freq[n], bins=80, alpha=0.5, density=True,
                 label=f"n={n} (std={errors_by_freq[n].std():.3f})")
    ax1.axvline(0, color="k", ls=":", lw=1)
    ax1.set_xlabel("Hedging error (P&L at expiry)")
    ax1.set_ylabel("Density")
    ax1.set_title("Hedging-error distribution tightens\nas rebalancing frequency rises")
    ax1.legend()
    ax1.grid(alpha=0.3)

    freqs_arr = np.array(freqs, dtype=float)
    stds_arr = np.array(stds)
    ax2.loglog(freqs_arr, stds_arr, "o-", ms=5, color="#1f77b4",
               label="simulated std of hedging error")
    ref = stds_arr[0] * np.sqrt(freqs_arr[0] / freqs_arr)
    ax2.loglog(freqs_arr, ref, "--", color="crimson",
               label=r"$\propto 1/\sqrt{n}$ reference")
    ax2.set_xlabel("Number of rebalances n")
    ax2.set_ylabel("Std of hedging error (log)")
    ax2.set_title("Hedging error std scales as 1/sqrt(n)\n(Boyle-Emanuel)")
    ax2.legend()
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       "results", "hedging_error.png")
    fig.savefig(out, dpi=130)
    print(f"\nSaved hedging-error plot -> {out}")
