import numpy as np
import numpy_financial as npf
from typing import Dict, List, Tuple
from financial_model import (
    ASSUMPTIONS, YEARS, build_sampled_params, simulate_cash_flows, compute_metrics
)

# ============================================================================
# RISK METRICS (VaR & CVaR)
# ============================================================================

def calculate_risk_metrics(npv_array: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR) from a simulated
    NPV distribution.

    Parameters
    ----------
    npv_array : np.ndarray
        Array of simulated NPVs produced by run_monte_carlo or
        run_advanced_simulation.
    confidence_level : float
        Tail confidence level (default 0.95 → 5th-percentile cutoff).

    Returns
    -------
    tuple of (var, cvar)
        var  : NPV at the lower-tail percentile.
        cvar : Mean NPV of all outcomes at or below var (Expected Shortfall).
    """
    percentile = (1.0 - confidence_level) * 100
    var = np.percentile(npv_array, percentile)
    
    # CVaR is the average of all NPVs worse than or equal to VaR
    worst_cases = npv_array[npv_array <= var]
    cvar = np.mean(worst_cases) if len(worst_cases) > 0 else var
    
    return var, cvar

# ============================================================================
# REAL OPTIONS (ABANDON / EXPAND)
# ============================================================================

def apply_real_options(
    scalars: Dict,
    per_year: Dict,
    years: List[int],
    assumptions: Dict
) -> Tuple[List[float], str]:
    """
    Classify a simulated cash-flow path into a real-options scenario and apply
    the corresponding action.  All rules are derived purely from the modeled
    path; no external thresholds or uplift factors are introduced.

    Classification (based on operational CFs, i.e. terminal value stripped):
    - Abandon : Year-1 operational CF < 0 and cumulative operational CF never
                recovers above zero.  Cash flows from Year 2 onwards are zeroed.
    - Expand  : Cumulative operational CF is non-negative in every year.
                Cash flows are left as modelled (no synthetic uplift added).
    - Base    : All other paths.

    Returns
    -------
    tuple of (adjusted_cash_flows, scenario_label)
    """
    # Simulate base cash flows for this path
    cf = simulate_cash_flows(scalars, per_year, years, assumptions)

    scenario = "Base"
    adjusted_cf = list(cf)

    if len(adjusted_cf) == 0:
        return adjusted_cf, scenario

    # Strip terminal value from the last year so that scenario classification
    # reflects only the underlying operating economics, not valuation mechanics.
    # terminal_value_multiple is a fixed model parameter (default 10).
    terminal_multiple = scalars.get(
        "terminal_value_multiple",
        assumptions["terminal_value_multiple"]["value"]
    )
    operational_cf = list(adjusted_cf)
    if terminal_multiple != 0:
        # Reverse the terminal-value addition: CF_last = op_CF_last * (1 + mult)
        # so op_CF_last = CF_last / (1 + mult)
        operational_cf[-1] = adjusted_cf[-1] / (1.0 + terminal_multiple)

    cum_op_cf = np.cumsum(operational_cf)

    if operational_cf[0] < 0 and np.all(cum_op_cf[1:] < 0):
        # Path never recovers: abandon after Year 1.
        scenario = "Abandon"
        for t in range(1, len(years)):
            adjusted_cf[t] = 0.0

    elif np.all(cum_op_cf >= 0):
        # Expand: exercise the growth option by re-simulating with adoption_ramp
        # at its maximum values (already defined in ASSUMPTIONS["adoption_ramp"]["max"]).
        # All other sampled parameters are held constant — no new assumptions added.
        expanded_per_year = dict(per_year)
        expanded_per_year["adoption_ramp"] = list(assumptions["adoption_ramp"]["max"])
        adjusted_cf = list(simulate_cash_flows(scalars, expanded_per_year, years, assumptions))
        scenario = "Expand"

    # Base: no adjustment needed.
    return adjusted_cf, scenario


# ============================================================================
# INVESTOR RETURN METRICS
# ============================================================================

def compute_investor_returns(
    cash_flows: List[float],
    discount_rate: float,
    years: List[int]
) -> Tuple[List[float], float, float, float, float]:
    """
    Compute project-level IRR, MOIC, Discounted Payback Period, and
    Profitability Index directly from modeled cash flows.

    No synthetic financing layer is applied; all metrics derive from the
    path produced by simulate_cash_flows and apply_real_options, discounted
    at the path-sampled discount_rate from ASSUMPTIONS.

    Parameters
    ----------
    cash_flows    : project cash flows for this simulation path.
    discount_rate : sampled from ASSUMPTIONS["discount_rate"] for this path.
    years         : model year labels (e.g. [1..7]); used to label DPP output.

    Returns
    -------
    tuple of (cash_flows, irr, moic, dpp, pi)
        irr  : Internal rate of return; -1.0 if indeterminate.
        moic : Total positive CFs / |total negative CFs|.
        dpp  : First year where cumulative *discounted* CF >= 0; np.inf if never.
        pi   : Profitability Index = 1 + NPV / |capital deployed|.
    """
    if len(cash_flows) == 0:
        return [], 0.0, 0.0, np.inf, np.nan

    # --- IRR ---
    # Solve sum(CF_t / (1+r*)^t) = 0 numerically.
    try:
        irr = npf.irr(cash_flows)
        if np.isnan(irr):
            irr = -1.0 if sum(cash_flows) < 0 else 0.0
    except Exception:
        irr = -1.0

    # --- Capital deployed (denominator for MOIC and PI) ---
    total_invested = abs(sum(cf for cf in cash_flows if cf < 0))
    total_returned = sum(cf for cf in cash_flows if cf > 0)
    if total_invested == 0:
        return cash_flows, irr, 0.0, np.inf, np.nan

    # --- MOIC ---
    # Multiple on Invested Capital: total cash returned / total cash deployed.
    # Ignores time value; complements IRR by showing magnitude of return.
    moic = total_returned / total_invested

    # --- Discounted cash flows (index 0 = period 0, consistent with npf.npv) ---
    # CF_t_discounted = CF_t / (1 + r)^t
    discounted_cfs = [
        cf / (1.0 + discount_rate) ** t
        for t, cf in enumerate(cash_flows)
    ]
    cum_discounted = np.cumsum(discounted_cfs)

    # --- Discounted Payback Period (DPP) ---
    # DPP = min{ year : sum_{s=0}^{year} CF_s/(1+r)^s >= 0 }
    # Unlike simple breakeven, respects the time cost of waiting for returns.
    dpp = np.inf
    for year, cum_cf in zip(years, cum_discounted):
        if cum_cf >= 0:
            dpp = float(year)
            break

    # --- Profitability Index (PI) ---
    # PI = 1 + NPV / |capital deployed|
    # PI > 1.0 : value-creating path
    # PI = 1.0 : breaks even in present value terms
    # PI < 1.0 : value-destroying path
    # Normalises for scale, making Abandon vs Expand paths comparable.
    npv_val = float(np.sum(discounted_cfs))
    pi = 1.0 + npv_val / total_invested

    return cash_flows, irr, moic, dpp, pi

# ============================================================================
# ADVANCED SIMULATION
# ============================================================================

def run_advanced_simulation(
    assumptions: Dict = ASSUMPTIONS,
    years: List[int] = YEARS,
    n_paths: int = 5_000,
    seed: int = 42
) -> Dict:
    """
    Monte Carlo simulation with real-options scenario classification and
    project-level investor return metrics.

    Each path:
    1. Samples parameters from existing ASSUMPTIONS distributions.
    2. Classifies the path (Abandon / Base / Expand) from modeled CFs only.
    3. Computes NPV using the sampled discount_rate.
    4. Computes IRR and MOIC from the same modeled CFs.

    Outputs are aggregated into tail-risk statistics (VaR / CVaR), scenario
    frequencies, and average investor return metrics.
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)

    npvs = []
    scenarios = {"Abandon": 0, "Base": 0, "Expand": 0}
    irrs, moics, dpps, pis = [], [], [], []

    for _ in range(n_paths):
        scalars, per_year = build_sampled_params(assumptions, years)
        discount_rate = scalars["discount_rate"]

        # 1. Classify path and apply scenario action (e.g., zero post-Y1 CFs on Abandon,
        #    re-simulate at max adoption ramp on Expand).
        adj_cf, scenario = apply_real_options(scalars, per_year, years, assumptions)
        scenarios[scenario] += 1

        # 2. NPV using sampled discount rate from ASSUMPTIONS.
        npv, _, _ = compute_metrics(adj_cf, discount_rate, years)
        npvs.append(npv)

        # 3. Investor return metrics from the same modeled cash flows.
        _, irr, moic, dpp, pi = compute_investor_returns(adj_cf, discount_rate, years)
        irrs.append(irr)
        moics.append(moic)
        dpps.append(dpp)
        pis.append(pi)

    npvs_arr = np.array(npvs)
    var_95, cvar_95 = calculate_risk_metrics(npvs_arr, confidence_level=0.95)

    # DPP: report mean over paths that achieve discounted breakeven, plus probability.
    finite_dpps = [d for d in dpps if np.isfinite(d)]
    # PI: exclude paths with no capital deployed (undefined).
    valid_pis = [p for p in pis if not np.isnan(p)]

    return {
        "npv_array": npvs_arr,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "scenarios": scenarios,
        "avg_irr": np.mean(irrs),
        "avg_moic": np.mean(moics),
        "avg_dpp": np.mean(finite_dpps) if finite_dpps else np.inf,
        "dpp_prob": len(finite_dpps) / n_paths,
        "avg_pi": np.mean(valid_pis) if valid_pis else np.nan,
        "n_paths": n_paths,
    }
