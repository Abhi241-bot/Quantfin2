# Session Summary — Options Pricing & Implied Volatility Surface

Date: 2026-06-04/05  |  Repo: https://github.com/Abhi241-bot/Quantfin2 (branch `main`)

## 1. What was built (all six modules complete + data layer)

| Module | File | Status |
|--------|------|--------|
| A — Black-Scholes + Greeks | `src/black_scholes.py` | ✅ done, pushed |
| B — CRR binomial tree (Euro + American) | `src/binomial_tree.py` | ✅ done, pushed |
| C — Monte Carlo (stderr + antithetic) | `src/monte_carlo.py` | ✅ done, pushed |
| D — Implied vol (Brent root-finder) | `src/implied_vol.py` | ✅ done, pushed |
| E — Vol smile / skew / 3D surface | `src/vol_surface.py` | ✅ done, pushed |
| F — Delta-hedging simulation | `src/delta_hedging.py` | ✅ done, pushed |
| Data acquisition | `src/fetch_data.py` | ✅ done, pushed |
| README | `README.md` | ✅ done |

Plots in `results/`: `method_convergence.png`, `mc_convergence.png`, `vol_smile.png`,
`vol_surface_3d.png`, `hedging_error.png`. IV table in `data/implied_vols.csv`.

Each module was run, verified, committed, and pushed before moving to the next (per operating
rules). Two genuine bugs were caught and fixed mid-build (see §3).

## 2. Current key numbers

**Pricing correctness (S=100, K=100, T=1, r=0.05, σ=0.20, q=0):**
- BS call = **10.4506** (matches Hull ~10.45); put-call parity abs error **0.00e+00**.
- Tree → BS: abs err **1.0e-03** at N=2000; American-put early-exercise premium **+0.518**;
  American call = European call (diff **0.00e+00**) at q=0.
- Three-way agreement (call): BS **10.4506** / tree **10.4496** / MC **10.436** (BS inside MC 95% CI).
- MC standard error scales ~1/√N; antithetic variates cut std by **~30%** (ratio 0.704).

**Empirical (real SPY chain, spot ≈ 757.7, snapshot 2026-06-04):**
- Expiries at ~7/28/57/88 days; 568 OTM quotes all produced finite IV.
- Downward skew (OTM-put minus OTM-call IV): **+9.56% (7d) → +5.11% (88d)** — steepens at short end.
- ATM IV term structure: **10.6% → 12.3% → 14.0% → 15.2%** (upward / contango).

**Delta hedging (short ATM call, mu=0.10 ≠ r=0.05):**
- Mean hedging error ≈ 0 at all frequencies (drift-independent replication).
- Std: **6.12 (n=1) → 0.426 (n=252)**; `std·√n` ≈ 6.1–6.8 (confirms 1/√n, Boyle-Emanuel).
- With 5 bps costs: mean P&L goes negative and worsens with rebalancing (−0.138 → −0.314).

## 3. Assumptions / deviations from spec (and why)

- **Data source = SPY, not NSE.** Chosen explicitly via the spec's stated fallback (NSE blocks
  scraping; SPY's liquidity gives a cleaner skew). User approved this choice.
- **r = 4.3% (US 3M T-bill), q = 1.2% (SPY dividend yield).** Stated assumptions for the US
  underlying; q included so call/put IVs are consistent at the same strike.
- **OTM-wing convention** for the smile (puts below spot, calls above) to get one clean, liquid,
  early-exercise-clean curve. SPY options are American — acknowledged; short-dated OTM premium is
  negligible so BS (European) inversion is used.
- **Bug fixed — antithetic stderr:** initially computed over individual payoffs (showed no variance
  reduction). Fixed to compute over pair-means, which is where the negative correlation lives;
  reduction then correctly appeared (~30%).
- **Bug fixed — IV vega-gate test:** an edge-case test expected NaN for a price that was actually
  reachable (not vega-dead). Rewrote the demo to show the gate firing honestly with the real vega.
- **Expiry selection:** `fetch_data.py` picks expiries near 7/30/60/90 days instead of the nearest
  four, because yfinance returns 0DTE-heavy near-dated expiries (T≈0 is degenerate / noisy).

## 4. Where to pick up next session

**The core build is DONE.** Remaining items, in priority order:

1. **`notebooks/analysis.ipynb`** (spec deliverable, not yet built) — the narrative notebook
   stitching the story: pricing agreement → real chain → smile → hedging. This is the main
   outstanding deliverable. Start here. (NotebookEdit tool, or jupytext.)
2. **Optional stretch goals** (spec §8), any of which would strengthen a 2nd-year resume:
   - Calibrate **Heston** or a **local-vol** model and compare its smile to the SPY market smile.
   - **Greeks surface** — plot delta/gamma across strike × time.
   - **Implied vs realized vol** over time (the variance risk premium).
3. **Optional polish:** add a `tests/` folder formalising the self-test asserts (currently the
   `__main__` blocks act as runnable checks); pin the SPY snapshot date in the README.

**Nothing is broken.** Every module runs clean and is pushed to `origin/main` (latest commit
`e61e9d0`). To resume: `python src/<module>.py` re-runs any module's self-test and regenerates its
plots; `python src/fetch_data.py` refreshes the SPY snapshot (will shift the numbers above to the
new market date).
