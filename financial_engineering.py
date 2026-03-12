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
    Calculate Value at Risk (VaR) and Conditional VaR (CVaR) on NPV distribution.
    
    Parameters
    ----------
    npv_array : np.ndarray
        Array of simulated NPVs.
    confidence_level : float
        Confidence level for VaR (e.g., 0.95 for 95% confidence).
        
    Returns
    -------
    tuple of (var, cvar)
        var: Value at Risk (the 5th percentile NPV)
        cvar: Expected Shortfall (average NPV of the worst 5% of outcomes)
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
    assumptions: Dict,
    yield_metric: float
) -> Tuple[List[float], str]:
    """
    Run cash flow simulation with path-dependent Real Options (Abandon / Expand).
    
    Stop-Loss (Abandon): If Yield < 5%, project is abandoned after Year 1.
    Expansion (National SPV): If Yield >= 7.5%, terminal value multiple increases and revenue scales up.
    
    Returns
    -------
    tuple of (cash_flows, scenario_label)
    """
    # 1. Simulate base cash flows for this path
    cf = simulate_cash_flows(scalars, per_year, years, assumptions)
    
    scenario = "Base"
    adjusted_cf = list(cf)
    
    if yield_metric < 0.05:
        scenario = "Abandon"
        # Zero out cash flows after Year 1
        for t in range(1, len(years)):
            adjusted_cf[t] = 0.0
            
    elif yield_metric >= 0.075:
        scenario = "Expand"
        # In Expansion, we assume active users/revenues scale up significantly (e.g., +50% from Year 2 onwards)
        # For simplicity, we model this as a 50% increase in net cash flows from Year 2 to Year 7
        # and a higher terminal value (which is embedded in Year 7 cf in base model, so we must adjust it)
        
        # Base terminal value is in adjusted_cf[-1].
        # Let's extract base CFs by running without terminal value logic to scale them accurately:
        cf_no_term = simulate_cash_flows(scalars, per_year, years, assumptions)
        # Strip terminal value from base model's last year for accurate scaling
        term_mult = scalars.get("terminal_value_multiple", assumptions["terminal_value_multiple"]["value"])
        cf_no_term[-1] -= (cf_no_term[-1] * term_mult / (1 + term_mult)) if cf_no_term[-1] > 0 else 0
        
        for t in range(1, len(years)):
            if cf_no_term[t] > 0:
                adjusted_cf[t] = cf_no_term[t] * 1.5  # 50% uplift on positive operational CFs
            else:
                adjusted_cf[t] = cf_no_term[t]  # Do not penalize losses more
                
        # Apply enhanced terminal value multiple for SPV
        enhanced_mult = term_mult * 1.5  # E.g., 10x becomes 15x
        adjusted_cf[-1] += adjusted_cf[-1] * enhanced_mult

    return adjusted_cf, scenario


# ============================================================================
# BLENDED FINANCING STRUCTURE
# ============================================================================

def apply_capital_structure(
    unlevered_cfs: List[float], 
    scenario: str,
    coupon_rate: float = 0.08,
    conversion_discount: float = 0.20
) -> Tuple[List[float], float, float]:
    """
    Apply Blended Financing (Tranche A Grants + Tranche B Equity).
    
    Tranche A (Grants): 25% of negative CFs in Year 1.
    Tranche B (Institutional Equity): 75% of negative CFs in Year 1.
    
    Tranche B gets an 8% cumulative coupon.
    If 'Expand' scenario, Tranche B converts to National SPV equity at a discount.
    
    Returns
    -------
    tuple of (tranche_b_cfs, tranche_b_irr, tranche_b_moic)
    """
    # Identify capital required (sum of all negative cash flows in early years, assumed mostly Year 1)
    capital_req = 0.0
    for cf in unlevered_cfs:
        if cf < 0:
            capital_req += abs(cf)
        else:
            break
            
    if capital_req == 0:
        return unlevered_cfs, 0.0, 0.0
        
    tranche_a_inv = capital_req * 0.25
    tranche_b_inv = capital_req * 0.75
    
    tranche_b_cfs = [0.0] * len(unlevered_cfs)
    # Tranche B cash outflow
    tranche_b_cfs[0] = -tranche_b_inv
    
    if scenario == "Abandon":
        # Total loss for Tranche B
        pass 
    elif scenario == "Expand":
        # Conversion Event: Tranche B converts at a discount. 
        # Value of Tranche B = (Tranche B Inv * (1+coupon)^T) / (1 - discount)
        # We assume exit in the final year.
        T = len(unlevered_cfs) - 1
        accrued_value = tranche_b_inv * ((1 + coupon_rate) ** T)
        exit_value = accrued_value / (1.0 - conversion_discount)
        
        # Optional: Cap exit value at a percentage of total enterprise value (last year CF)
        # For simplicity, we assume SPV can fully cover the conversion.
        tranche_b_cfs[-1] = exit_value
    else:
        # Base scenario: standard return of accrued value + some equity share
        # We assume Tranche B just gets their accrued value back from the cash flows in final year
        T = len(unlevered_cfs) - 1
        accrued_value = tranche_b_inv * ((1 + coupon_rate) ** T)
        # Only pay if there is enough cash
        final_cf = unlevered_cfs[-1]
        tranche_b_cfs[-1] = min(accrued_value, max(final_cf, 0))
        
    # Calculate IRR and MOIC for Tranche B
    try:
        tb_irr = npf.irr(tranche_b_cfs)
        if np.isnan(tb_irr):
            tb_irr = -1.0 if sum(tranche_b_cfs) < 0 else 0.0
    except:
        tb_irr = -1.0
        
    tb_moic = sum(cf for cf in tranche_b_cfs if cf > 0) / tranche_b_inv if tranche_b_inv > 0 else 0.0
    
    return tranche_b_cfs, tb_irr, tb_moic

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
    Run full Monte Carlo with Real Options and Blended Financing.
    """
    if seed is not None:
        np.random.seed(seed)
        import random
        random.seed(seed)
        
    npvs = []
    scenarios = {"Abandon": 0, "Base": 0, "Expand": 0}
    tb_irrs = []
    tb_moics = []
    
    for _ in range(n_paths):
        scalars, per_year = build_sampled_params(assumptions, years)
        discount_rate = scalars["discount_rate"]
        
        # Diagnostic Yield = Screening Participation % (as discussed in plan)
        yield_metric = scalars["screening_participation_pct"]
        
        # Real Options
        adj_cf, scenario = apply_real_options(scalars, per_year, years, assumptions, yield_metric)
        scenarios[scenario] += 1
        
        # Calculate Levered/Project NPV
        npv, _, _ = compute_metrics(adj_cf, discount_rate, years)
        npvs.append(npv)
        
        # Blended Financing
        _, tb_irr, tb_moic = apply_capital_structure(adj_cf, scenario)
        tb_irrs.append(tb_irr)
        tb_moics.append(tb_moic)
        
    npvs_arr = np.array(npvs)
    var_95, cvar_95 = calculate_risk_metrics(npvs_arr, confidence_level=0.95)
    
    return {
        "npv_array": npvs_arr,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "scenarios": scenarios,
        "tranche_b_avg_irr": np.mean(tb_irrs),
        "tranche_b_avg_moic": np.mean(tb_moics),
        "n_paths": n_paths
    }
