# Project 2 — Options Pricing & Implied Volatility Surface

> **Build target:** A from-scratch options pricing library (Black-Scholes, binomial tree,
> Monte Carlo) with Greeks, plus an empirical study that backs out implied volatilities from a
> real NSE option chain and plots/analyzes the volatility smile and skew. Optional capstone:
> a delta-hedging simulation that quantifies hedging error.

---

## 0. Intent (read this first)

This project signals **mathematical maturity** — the thing quant *research* roles weight most
heavily. ML projects show you can call libraries; this shows you understand the math underneath
derivatives. The two halves matter equally:

1. **Theory half:** implement three pricing methods that agree with each other. Agreement
   across methods is your correctness proof.
2. **Empirical half:** take real market option prices, invert Black-Scholes to recover implied
   volatility, and show you understand *why the smile/skew exists* (it's the market's correction
   for Black-Scholes' wrong assumption that returns are normal/constant-vol).

The empirical half is what makes it stand out — anyone can code Black-Scholes; connecting it to
real market structure shows you understand markets.

---

## 1. Learning objectives (defend these in an interview)

- Derive/state the Black-Scholes formula and the assumptions it makes (log-normal prices,
  constant volatility, no arbitrage, continuous hedging, constant rates).
- Explain each Greek intuitively: delta, gamma, vega, theta, rho.
- Explain put-call parity and use it as a sanity check.
- Explain *why* the implied volatility smile/skew exists and what a steep put skew implies about
  market fear (crash risk pricing / fat left tail).
- Explain why you need a numerical root-finder to get implied vol (BS isn't invertible in closed form for σ).
- Explain risk-neutral pricing intuition (why we discount expected payoff at the risk-free rate).

---

## 2. Tech stack

```
python >= 3.10
numpy
scipy            # norm.cdf/pdf, brentq root-finder for implied vol
pandas
matplotlib
mpl_toolkits / plotly   # 3D vol surface
```

No heavy dependencies needed. Keep the pricing core pure NumPy/SciPy so the math is visible.

---

## 3. Data acquisition (the empirical half)

NSE actively blocks programmatic scraping of its option chain, so plan for this:

**Recommended approach (most reliable):** manually download an NSE option chain snapshot as CSV.
Go to the NSE option chain page for an index (NIFTY / BANKNIFTY) or a liquid stock, export the
chain for one or two expiries, and save as `data/option_chain.csv`. One good snapshot is enough
for a compelling smile plot.

**Fallbacks:**
- A Kaggle "NIFTY / BANKNIFTY options" dataset.
- `yfinance` option chains work well for **US** tickers (`yf.Ticker("SPY").option_chain(...)`).
  Using SPY is perfectly acceptable and avoids the NSE access problem — the skew analysis is
  arguably cleaner on SPY. State your choice in the README either way.

Your code should read the chain from a local CSV so the source is swappable. Required columns:
strike, expiry, option type (call/put), market price (mid of bid/ask is best), underlying spot.

You also need: current **spot price**, **risk-free rate** (use the Indian 91-day T-bill / repo
rate ~6.5%, or US 3-month yield for SPY — be explicit), and **time to expiry** in years.

---

## 4. Implementation plan (modules)

### Module A — `black_scholes.py`
- `bs_price(S, K, T, r, sigma, option_type, q=0)` — European call/put price (include dividend
  yield `q` for completeness).
- Greeks as separate functions: `delta`, `gamma`, `vega`, `theta`, `rho`. Derive vega and gamma
  carefully (common interview question).
- `put_call_parity_check(...)` — verify `C - P = S·e^{-qT} - K·e^{-rT}` holds for your prices.

### Module B — `binomial_tree.py`
- Cox-Ross-Rubinstein binomial tree pricer for European options (and **American** options as a
  stretch — American puts are where the tree earns its keep, since BS can't price early exercise).
- Show that as the number of steps → large, the tree price **converges to Black-Scholes**. Plot
  this convergence. This convergence plot is a strong correctness signal.

### Module C — `monte_carlo.py`
- Simulate terminal prices under geometric Brownian motion:
  `S_T = S·exp((r - q - σ²/2)T + σ√T·Z)`, `Z ~ N(0,1)`.
- Price = `e^{-rT}·mean(payoff)`. Report the **standard error** of the estimate and show it
  shrinks as `1/√N`.
- (Stretch) Add antithetic variates or control variates for variance reduction — name-drops well.
- Confirm all three methods (BS, tree, MC) agree to within MC noise. **This three-way agreement is your unit test.**

### Module D — `implied_vol.py`
- `implied_vol(market_price, S, K, T, r, option_type)` using `scipy.optimize.brentq` to solve
  `bs_price(σ) - market_price = 0` for σ.
- Handle edge cases: prices below intrinsic value, deep ITM/OTM where vega ≈ 0 (root-finder
  struggles), and filter those out rather than producing garbage.

### Module E — `vol_surface.py`
- Apply `implied_vol` across all strikes (and expiries if you have them) in the chain.
- Plot **implied vol vs strike** (or vs moneyness K/S) for a single expiry → the **smile/skew**.
- If you have multiple expiries, build a **3D implied volatility surface** (strike × expiry × IV).
- Write the analysis: equity index options typically show a **downward skew** (OTM puts have
  higher IV than OTM calls) because the market prices crash risk / demand for downside protection.

### Module F — `delta_hedging.py` (capstone — high impact)
- Simulate selling one option and **delta-hedging** it over its life along a simulated GBM path.
- Rebalance the hedge at a chosen frequency (daily). Track the **hedging error** = final P&L of
  (short option + dynamic stock hedge + cash account).
- Show that as rebalancing frequency increases, hedging error variance shrinks toward zero (the
  continuous-hedging idealization behind Black-Scholes). Plot error distribution vs rebalance frequency.
- This demonstrates you understand that the BS price is the cost of *replication*, not a forecast.

---

## 5. Pitfalls to avoid

- **Time units**: T must be in **years**. Mixing days and years silently breaks everything. (e.g. 30 days = 30/365.)
- **Rate units**: use a continuously-compounded decimal rate (0.065, not 6.5).
- **Mid vs last price**: use mid (bid-ask average) for implied vol; last-traded can be stale and produce noisy IV.
- **Illiquid strikes**: deep OTM options with tiny prices give unstable IV — filter by minimum volume/OI or by |moneyness| range.
- **American vs European**: NSE stock options are American; index options are European. BS is for
  European. If you use stock options, acknowledge the (small) American premium or stick to index/European.
- **Vega ≈ 0 regions**: don't trust implied vol from deep ITM/OTM options where the price is
  insensitive to σ.
- **Claiming BS is "right"**: the smile *is the evidence BS is wrong*. Frame your analysis as
  "the market corrects BS's flat-vol assumption," not "BS predicts the prices."

---

## 6. Deliverables & repo structure

```
options-pricing-vol-surface/
├── README.md
├── requirements.txt
├── data/
│   └── option_chain.csv
├── src/
│   ├── black_scholes.py
│   ├── binomial_tree.py
│   ├── monte_carlo.py
│   ├── implied_vol.py
│   ├── vol_surface.py
│   └── delta_hedging.py
├── notebooks/
│   └── analysis.ipynb      # narrative: pricing agreement → real chain → smile → hedging
└── results/
    ├── method_convergence.png
    ├── vol_smile.png
    ├── vol_surface_3d.png
    └── hedging_error.png
```

---

## 7. README template

```
# Options Pricing & the Implied Volatility Surface

## Summary
One paragraph: implemented three pricing methods (BS / binomial / MC) that agree to within MC
error; backed out implied vols from a real <NIFTY / SPY> chain; documented the volatility skew;
simulated delta hedging and quantified hedging error vs rebalance frequency.

## Pricing engine
- Black-Scholes + full Greeks, validated by put-call parity
- Binomial tree converging to BS (convergence plot)
- Monte Carlo with standard error and variance reduction
- Three-way agreement table (this is the correctness proof)

## Empirical volatility study
- Data source, spot, rate, expiry assumptions
- Implied vol method (Brent root-finder)
- Smile / skew plot + interpretation (what the skew says about market fear)
- 3D surface (if multi-expiry)

## Delta hedging experiment
- Setup, rebalancing frequencies tested
- Hedging-error distribution and how it shrinks with frequency
- Interpretation: BS price = cost of replication

## Limitations
- Constant-vol assumption, transaction costs ignored in hedging sim, American vs European, etc.

## How to run
```

---

## 8. Stretch goals

- Calibrate a simple **local volatility** or **Heston (stochastic vol)** model and compare its
  smile to the market — this is genuinely impressive for a 2nd-year.
- **Greeks surface**: plot delta/gamma across strikes and time.
- **Implied vs realized volatility** comparison over time (the variance risk premium).

---

## 9. Interview talking points

- "Derive vega." / "Why is gamma highest at the money?"
- "Why does the implied volatility smile exist?"
- "What does a steep put skew tell you about the market?"
- "Why can't you solve Black-Scholes for sigma in closed form?"
- "What is the Black-Scholes price actually the price *of*?" (answer: the replicating portfolio)
- "Your three methods agree — how do you know they're all not wrong the same way?" (they make
  different assumptions/errors, so agreement across them is strong evidence)
