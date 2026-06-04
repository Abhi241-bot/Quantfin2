# Options Pricing & the Implied Volatility Surface

A from-scratch options-pricing library (Black-Scholes, binomial tree, Monte Carlo) with full
Greeks, plus an empirical study that backs implied volatilities out of a **real SPY option chain**
and analyses the volatility smile/skew, capped with a **delta-hedging simulation** that quantifies
hedging error.

## Summary

Implemented three independent pricing methods that **agree to within Monte-Carlo error** (this
three-way agreement is the correctness proof — closed-form, lattice, and random-sampling errors
are unrelated, so agreement is strong evidence). Backed implied vols out of a real SPY chain with a
Brent root-finder and documented the characteristic **downward equity skew** (OTM puts richer than
OTM calls — the market pricing crash risk). Simulated delta hedging and showed the hedging-error
standard deviation shrinks as **1/√n** with rebalancing frequency, demonstrating that the BS price
is the **cost of replication**, not a forecast.

## Pricing engine

- **Black-Scholes + full Greeks** (`src/black_scholes.py`) — delta, gamma, vega, theta, rho, all
  derived in the docstrings. Validated by put-call parity (exact) and finite-difference checks.
- **Binomial tree** (`src/binomial_tree.py`) — Cox-Ross-Rubinstein, European *and* American.
  Converges to Black-Scholes as steps grow (with the classic odd/even oscillation); American put
  shows a positive early-exercise premium; American call = European call when q=0 (as it must).
- **Monte Carlo** (`src/monte_carlo.py`) — GBM terminal-price simulation with reported standard
  error (shrinks as 1/√N) and **antithetic-variates** variance reduction (~30% lower std at equal
  cost).
- **Three-way agreement table** — the unit test. BS / tree / MC land on the same call and put
  prices, with BS inside the MC 95% confidence interval.

## Empirical volatility study

- **Data source:** real **SPY** (S&P 500 ETF) option chain via `yfinance`, snapshotted to
  `data/option_chain.csv`. SPY is chosen because NSE blocks programmatic scraping and SPY's deep
  liquidity gives a clean, well-defined skew (the spec's recommended fallback). The loader reads a
  plain CSV, so an NSE export can be swapped in without code changes.
- **Stated assumptions:** spot from the snapshot; risk-free rate `r = 4.3%` (US 3-month T-bill,
  continuously compounded); dividend yield `q = 1.2%` (SPY). Time to expiry in years, ACT/365.
- **Method:** mid (bid-ask average) prices; clean filters drop options without two-sided quotes,
  below a minimum price, outside a 0.80–1.20 moneyness band, or with no open interest. The OTM wing
  (puts below spot, calls above spot) is used because those quotes are the most liquid and least
  contaminated by early exercise. Implied vol via `scipy.optimize.brentq`.
- **American-vs-European caveat:** SPY options are American. We use short-dated OTM contracts where
  the early-exercise premium is negligible and treat them as European for the BS inversion.
- **Findings:** downward skew at every expiry (OTM-put minus OTM-call IV is positive and steepens
  at the short end), and an upward-sloping vol term structure (calm-market contango). See
  `results/vol_smile.png` and `results/vol_surface_3d.png`.
- **Framing:** the smile *is the evidence Black-Scholes is wrong* — the market corrects BS's flat-
  vol / normal-returns assumption. We do not claim BS "predicts" the prices.

## Delta-hedging experiment

- Sell one ATM call at its BS price, then delta-hedge to expiry along simulated GBM paths,
  rebalancing at a chosen frequency with an interest-bearing cash account.
- **Hedging error** = final P&L of (short option + dynamic stock hedge + cash). Frictionless mean
  ≈ 0 even when the path drift `mu ≠ r` (replication is drift-independent), and the error standard
  deviation scales as **1/√n** (Boyle-Emanuel). See `results/hedging_error.png`.
- **Transaction costs** (5 bps per trade) push mean P&L negative and grow with rebalancing — the
  real-world tension between discretization error and trading cost.
- **Interpretation:** the BS price is the cost of the replicating portfolio, not a market forecast.

## Repo structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── option_chain.csv      # SPY snapshot (source-agnostic schema)
│   └── implied_vols.csv      # computed IV table
├── src/
│   ├── black_scholes.py      # Module A
│   ├── binomial_tree.py      # Module B
│   ├── monte_carlo.py        # Module C
│   ├── implied_vol.py        # Module D
│   ├── vol_surface.py        # Module E
│   ├── delta_hedging.py      # Module F
│   └── fetch_data.py         # SPY chain downloader
└── results/
    ├── method_convergence.png   # tree -> BS
    ├── mc_convergence.png       # MC error ~ 1/sqrt(N)
    ├── vol_smile.png            # the skew
    ├── vol_surface_3d.png       # IV surface
    └── hedging_error.png        # hedging error ~ 1/sqrt(n)
```

## Limitations

- Black-Scholes assumes constant volatility — the observed smile is exactly the failure of that
  assumption. No stochastic-vol (Heston) or local-vol model is calibrated here (a stretch goal).
- SPY options are American; we approximate with European BS on short-dated OTM contracts.
- The hedging simulation assumes the true vol is known and constant, and (in the headline run)
  ignores transaction costs; the cost-aware run is reported separately.
- A single chain snapshot is used; no time-series of the surface (variance risk premium) is studied.

## How to run

```bash
pip install -r requirements.txt

# Each module self-tests and writes its plot(s) to results/ when run directly:
python src/black_scholes.py     # prices, Greeks, put-call parity, FD checks
python src/binomial_tree.py     # tree -> BS convergence plot
python src/monte_carlo.py       # stderr scaling, antithetic, 3-way agreement
python src/implied_vol.py       # round-trip IV recovery + edge cases

# Empirical half (needs internet for the SPY download):
python src/fetch_data.py        # download SPY chain -> data/option_chain.csv
python src/vol_surface.py       # smile, 3D surface, skew report

# Capstone:
python src/delta_hedging.py     # hedging error vs rebalance frequency
```
