"""
Financial Model for Digital Health Subscription Business
========================================================

Core calculation engine for Monte Carlo simulation and sensitivity analysis.

Author: Wasu Mekniran
Date: 09.12.2025
"""

import random
import numpy as np
import numpy_financial as npf
import pandas as pd
from typing import Dict, List, Tuple, Optional


# ============================================================================
# MODEL ASSUMPTIONS
# ============================================================================

ASSUMPTIONS = {
    # Population & Market
    "adult_population": {"value": 6_500_000, "type": "fixed"},
    "screening_participation_pct": {
        "min": 0.05, "most_likely": 0.10, "max": 0.15,
        "distribution": "triangular",
    },
    
    # Adoption Ramp (year-dependent)
    "adoption_ramp": {
        "min":         [0.005, 0.01, 0.03, 0.05, 0.08, 0.10, 0.12],
        "most_likely": [0.015, 0.03, 0.07, 0.10, 0.15, 0.18, 0.22],
        "max":         [0.02,  0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "distribution": "triangular",
    },
    
    # Funnel Mechanics
    "conv_screen_to_paid": {
        "min": 0.20, "most_likely": 0.40, "max": 0.60,
        "distribution": "triangular"
    },
    
    # Epidemiology
    "p_USPSTF_eligible": {
        "min": 0.30, "most_likely": 0.40, "max": 0.50,
        "distribution": "triangular"
    },
    "p_IFG": {
        "min": 0.10, "most_likely": 0.11, "max": 0.12,
        "distribution": "triangular"
    },
    "p_undiag": {
        "min": 0.11, "most_likely": 0.30, "max": 0.80,
        "distribution": "triangular"
    },
    
    # Pricing & Retention
    "price_per_month": {
        "min": 20, "most_likely": 40, "max": 60,
        "distribution": "triangular"
    },
    "annual_churn_rate": {
        "min": 0.40, "most_likely": 0.55, "max": 0.70,
        "distribution": "triangular"
    },
    
    # Customer Acquisition
    "cac_per_user": {
        "min": 150, "most_likely": 220, "max": 350,
        "distribution": "triangular"
    },
    
    # Staffing
    "technician_ratio": {
        "min": 500, "most_likely": 700, "max": 900,
        "distribution": "triangular"
    },
    "manager_ratio": {
        "min": 10, "most_likely": 15, "max": 20,
        "distribution": "triangular"
    },
    "technician_salary": {
        "min": 85_000, "most_likely": 95_000, "max": 110_000,
        "distribution": "triangular"
    },
    "manager_salary": {
        "min": 100_000, "most_likely": 120_000, "max": 140_000,
        "distribution": "triangular"
    },
    
    # Operational Costs
    "app_dev": {
        "min":         [250_000, 40_000, 40_000, 40_000, 40_000, 40_000, 40_000],
        "most_likely": [300_000, 50_000, 50_000, 50_000, 50_000, 50_000, 50_000],
        "max":         [400_000, 60_000, 60_000, 60_000, 60_000, 60_000, 60_000],
        "distribution": "triangular",
    },
    "office_rent_per_worker": {
        "min": 8_000, "most_likely": 10_000, "max": 12_000,
        "distribution": "triangular"
    },
    "office_supplies_per_worker": {
        "min": 800, "most_likely": 1200, "max": 1500,
        "distribution": "triangular"
    },
    "call_center_per_user": {
        "min": 15, "most_likely": 20, "max": 30,
        "distribution": "triangular"
    },
    "backend_services_per_user": {
        "min": 25, "most_likely": 35, "max": 45,
        "distribution": "triangular"
    },
    "ce_mdr_certification_cost": {
        "min":         [10_000, 0, 0, 0, 0, 0, 0],
        "most_likely": [20_000, 0, 0, 0, 0, 0, 0],
        "max":         [30_000, 0, 0, 0, 0, 0, 0],
        "distribution": "triangular",
    },
    "rev_fadp_compliance_cost": {
        "min": 15_000, "most_likely": 25_000, "max": 40_000,
        "distribution": "triangular"
    },
    
    # Finance
    "discount_rate": {
        "min": 0.12, "most_likely": 0.18, "max": 0.30,
        "distribution": "triangular"
    },
    "terminal_value_multiple": {"value": 10, "type": "fixed"},
}

YEARS = [1, 2, 3, 4, 5, 6, 7]


# ============================================================================
# PARAMETER SAMPLING
# ============================================================================

def triangular_sample(lo: float, mode: float, hi: float) -> float:
    """Sample from triangular distribution."""
    return random.triangular(lo, hi, mode)


def sample_param(name: str, details: Dict, years: List[int] = YEARS):
    """
    Sample parameter value from distribution.
    
    Parameters
    ----------
    name : str
        Parameter name
    details : dict
        Parameter specification with min/most_likely/max
    years : list
        Model time horizon
        
    Returns
    -------
    float or list
        Sampled parameter value(s)
    """
    if details.get("type") == "fixed":
        return details["value"]
    
    dist = details.get("distribution", "triangular").lower()
    
    # Year-dependent parameter
    if isinstance(details.get("min"), list):
        if dist == "triangular":
            return [
                triangular_sample(
                    details["min"][i],
                    details["most_likely"][i],
                    details["max"][i]
                )
                for i in range(len(years))
            ]
        return details["most_likely"]
    
    # Scalar parameter
    if dist == "triangular":
        return triangular_sample(
            details["min"],
            details["most_likely"],
            details["max"]
        )
    
    return details.get("most_likely", details.get("value"))


def build_sampled_params(
    assumptions: Dict,
    years: List[int] = YEARS
) -> Tuple[Dict, Dict]:
    """
    Sample all parameters once for a simulation path.
    
    Returns
    -------
    tuple of (scalars, per_year)
        scalars : dict of time-invariant parameters
        per_year : dict of year-dependent parameters
    """
    scalars, per_year = {}, {}
    
    for name, details in assumptions.items():
        value = sample_param(name, details, years)
        if isinstance(value, list):
            per_year[name] = value
        else:
            scalars[name] = value
    
    return scalars, per_year


def get_value(
    name: str,
    year_idx: int,
    scalars: Dict,
    per_year: Dict,
    assumptions: Dict
):
    """Retrieve parameter value for given year."""
    if name in per_year:
        return per_year[name][year_idx]
    if name in scalars:
        return scalars[name]
    
    # Fallback to most_likely
    details = assumptions[name]
    base = details.get("most_likely", details.get("value"))
    if isinstance(base, list):
        return base[year_idx]
    return base


# ============================================================================
# CORE FINANCIAL MODEL
# ============================================================================

def simulate_cash_flows(
    scalars: Dict,
    per_year: Dict,
    years: List[int] = YEARS,
    assumptions: Dict = ASSUMPTIONS,
    return_details: bool = False
) -> List[float]:
    """
    Simulate annual cash flows for given parameters.
    
    Parameters
    ----------
    scalars : dict
        Time-invariant parameters
    per_year : dict
        Year-dependent parameters
    years : list
        Model horizon
    assumptions : dict
        Full parameter specification
    return_details : bool
        If True, return detailed breakdown
        
    Returns
    -------
    list of float
        Annual cash flows (including terminal value in final year)
    dict (optional)
        Detailed yearly metrics if return_details=True
    """
    # Initialize epidemiology
    adult_pop = scalars.get(
        "adult_population",
        assumptions["adult_population"]["value"]
    )
    p_eligible = scalars["p_USPSTF_eligible"]
    p_ifg = scalars["p_IFG"]
    p_undiag = scalars["p_undiag"]
    
    n_risk = adult_pop * p_eligible * (p_ifg + p_undiag)
    
    # Long-run subscriber potential
    long_run_max_subscribers = (
        n_risk
        * scalars["screening_participation_pct"]
        * scalars["conv_screen_to_paid"]
    )
    
    churn = scalars["annual_churn_rate"]
    
    # State variables
    active_users = 0.0
    cumulative_subscribers = 0.0
    
    # Results storage
    cash_flows = []
    details = {
        "year": [],
        "active_users": [],
        "new_subscribers": [],
        "revenue": [],
        "cac_cost": [],
        "tech_cost": [],
        "mgr_cost": [],
        "call_cost": [],
        "backend_cost": [],
        "rent_cost": [],
        "supplies_cost": [],
        "app_dev_cost": [],
        "ce_mdr_cost": [],
        "fadp_cost": [],
        "total_cost": [],
        "net_cf": [],
    }
    
    for t, year in enumerate(years):
        # Adoption dynamics
        adoption_factor = get_value(
            "adoption_ramp", t, scalars, per_year, assumptions
        )
        target_cum_subs = long_run_max_subscribers * adoption_factor
        new_subscribers = max(target_cum_subs - cumulative_subscribers, 0.0)
        
        # Update user base
        active_users = active_users * (1.0 - churn) + new_subscribers
        cumulative_subscribers += new_subscribers
        
        # Revenue
        price_per_month = get_value(
            "price_per_month", t, scalars, per_year, assumptions
        )
        revenue = active_users * price_per_month * 12
        
        # CAC (only on new subscribers)
        cac_per_user = get_value(
            "cac_per_user", t, scalars, per_year, assumptions
        )
        cac_cost = new_subscribers * cac_per_user
        
        # Staffing costs
        tech_ratio = get_value(
            "technician_ratio", t, scalars, per_year, assumptions
        )
        tech_salary = get_value(
            "technician_salary", t, scalars, per_year, assumptions
        )
        mgr_ratio = get_value(
            "manager_ratio", t, scalars, per_year, assumptions
        )
        mgr_salary = get_value(
            "manager_salary", t, scalars, per_year, assumptions
        )
        
        technicians = (
            int(np.ceil(active_users / tech_ratio)) if active_users > 0 else 0
        )
        managers = (
            int(np.ceil(technicians / mgr_ratio)) if technicians > 0 else 0
        )
        
        tech_cost = technicians * tech_salary
        mgr_cost = managers * mgr_salary
        
        # Variable operational costs
        call_cost = active_users * get_value(
            "call_center_per_user", t, scalars, per_year, assumptions
        )
        backend_cost = active_users * get_value(
            "backend_services_per_user", t, scalars, per_year, assumptions
        )
        
        # Fixed operational costs
        total_staff = technicians + managers
        rent_cost = total_staff * get_value(
            "office_rent_per_worker", t, scalars, per_year, assumptions
        )
        supplies_cost = total_staff * get_value(
            "office_supplies_per_worker", t, scalars, per_year, assumptions
        )
        
        # Regulatory and development costs
        app_dev_cost = get_value(
            "app_dev", t, scalars, per_year, assumptions
        )
        ce_mdr_cost = get_value(
            "ce_mdr_certification_cost", t, scalars, per_year, assumptions
        )
        fadp_cost = get_value(
            "rev_fadp_compliance_cost", t, scalars, per_year, assumptions
        )
        
        # Total costs and cash flow
        total_cost = (
            cac_cost + tech_cost + mgr_cost + call_cost + backend_cost
            + rent_cost + supplies_cost + app_dev_cost + ce_mdr_cost
            + fadp_cost
        )
        net_cf = revenue - total_cost
        
        cash_flows.append(net_cf)
        
        if return_details:
            details["year"].append(year)
            details["active_users"].append(active_users)
            details["new_subscribers"].append(new_subscribers)
            details["revenue"].append(revenue)
            details["cac_cost"].append(cac_cost)
            details["tech_cost"].append(tech_cost)
            details["mgr_cost"].append(mgr_cost)
            details["call_cost"].append(call_cost)
            details["backend_cost"].append(backend_cost)
            details["rent_cost"].append(rent_cost)
            details["supplies_cost"].append(supplies_cost)
            details["app_dev_cost"].append(app_dev_cost)
            details["ce_mdr_cost"].append(ce_mdr_cost)
            details["fadp_cost"].append(fadp_cost)
            details["total_cost"].append(total_cost)
            details["net_cf"].append(net_cf)
    
    # Add terminal value to final year
    if cash_flows:
        terminal_multiple = scalars.get(
            "terminal_value_multiple",
            assumptions["terminal_value_multiple"]["value"]
        )
        terminal_value = cash_flows[-1] * terminal_multiple
        cash_flows[-1] += terminal_value
    
    if return_details:
        return cash_flows, details
    return cash_flows


def compute_metrics(
    cash_flows: List[float],
    discount_rate: float,
    years: List[int] = YEARS
) -> Tuple[float, float, float]:
    """
    Compute NPV, IRR, and breakeven year.
    
    Returns
    -------
    tuple of (npv, irr, breakeven_year)
    """
    npv = npf.npv(discount_rate, cash_flows)
    
    try:
        irr = npf.irr(cash_flows)
    except (ValueError, RuntimeError):
        irr = np.nan
    
    # Find breakeven year
    cum_cf = np.cumsum(cash_flows)
    breakeven_year = np.inf
    for t, cf in zip(years, cum_cf):
        if cf >= 0:
            breakeven_year = t
            break
    
    return npv, irr, breakeven_year


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo(
    assumptions: Dict = ASSUMPTIONS,
    years: List[int] = YEARS,
    n_paths: int = 5_000,
    seed: Optional[int] = None
) -> Dict:
    """
    Run Monte Carlo simulation.
    
    Parameters
    ----------
    assumptions : dict
        Model assumptions
    years : list
        Model horizon
    n_paths : int
        Number of simulation paths
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Results with keys: npv, irr, breakeven_year, cash_flows
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    npvs, irrs, breakevens, paths = [], [], [], []
    
    for _ in range(n_paths):
        scalars, per_year = build_sampled_params(assumptions, years)
        cash_flows = simulate_cash_flows(scalars, per_year, years, assumptions)
        discount_rate = scalars["discount_rate"]
        
        npv, irr, breakeven = compute_metrics(
            cash_flows, discount_rate, years
        )
        
        npvs.append(npv)
        irrs.append(irr)
        breakevens.append(breakeven)
        paths.append(cash_flows)
    
    return {
        "npv": np.array(npvs),
        "irr": np.array(irrs),
        "breakeven_year": np.array(breakevens),
        "cash_flows": paths,
    }


# ============================================================================
# SENSITIVITY ANALYSIS
# ============================================================================

def build_base_case_params(
    assumptions: Dict,
    years: List[int] = YEARS
) -> Tuple[Dict, Dict]:
    """Build deterministic base-case parameters (most_likely values)."""
    scalars, per_year = {}, {}
    
    for name, details in assumptions.items():
        if details.get("type") == "fixed":
            base = details["value"]
        else:
            base = details.get("most_likely", details.get("value"))
        
        if isinstance(base, (list, tuple, np.ndarray)):
            per_year[name] = list(base)
        else:
            scalars[name] = base
    
    return scalars, per_year


def run_deterministic(
    overrides: Dict = None,
    assumptions: Dict = ASSUMPTIONS,
    years: List[int] = YEARS,
    return_details: bool = False
):
    """
    Run deterministic pro-forma with optional parameter overrides.
    
    Parameters
    ----------
    overrides : dict, optional
        Parameter overrides on top of base case
    assumptions : dict
        Model assumptions
    years : list
        Model horizon
    return_details : bool
        Return detailed breakdown
        
    Returns
    -------
    tuple of (npv, irr, breakeven) or (npv, irr, breakeven, details)
    """
    if overrides is None:
        overrides = {}
    
    scalars, per_year = {}, {}
    n_years = len(years)
    
    # Build base case with overrides
    for name, details in assumptions.items():
        if name in overrides:
            override_val = overrides[name]
            if isinstance(override_val, (list, tuple, np.ndarray)):
                per_year[name] = list(override_val)
            else:
                base = details.get("most_likely", details.get("value"))
                if isinstance(base, (list, tuple, np.ndarray)):
                    per_year[name] = [override_val] * n_years
                else:
                    scalars[name] = override_val
            continue
        
        # Use base-case value
        if details.get("type") == "fixed":
            base = details["value"]
        else:
            base = details.get("most_likely", details.get("value"))
        
        if isinstance(base, (list, tuple, np.ndarray)):
            per_year[name] = list(base)
        else:
            scalars[name] = base
    
    # Simulate
    result = simulate_cash_flows(
        scalars, per_year, years, assumptions, return_details=return_details
    )
    
    if return_details:
        cash_flows, details = result
    else:
        cash_flows = result
    
    # Compute metrics
    dr_details = assumptions["discount_rate"]
    discount_rate = dr_details.get("most_likely", dr_details.get("value"))
    
    npv, irr, breakeven = compute_metrics(cash_flows, discount_rate, years)
    
    if return_details:
        return npv, irr, breakeven, details
    return npv, irr, breakeven


def build_tornado_data(
    assumptions: Dict = ASSUMPTIONS,
    years: List[int] = YEARS
) -> Tuple[pd.DataFrame, float, float, float]:
    """
    One-way sensitivity analysis for tornado diagram.
    
    Returns
    -------
    tuple of (tornado_df, base_npv, base_irr, base_breakeven)
    """
    base_npv, base_irr, base_breakeven = run_deterministic({}, assumptions, years)
    
    rows = []
    skip = {"terminal_value_multiple"}
    
    for name, details in assumptions.items():
        if details.get("type") == "fixed" or name in skip:
            continue
        
        lo = details.get("min")
        hi = details.get("max")
        if lo is None or hi is None:
            continue
        
        for label, val in [("Min", lo), ("Max", hi)]:
            npv, irr, breakeven = run_deterministic(
                {name: val}, assumptions, years
            )
            rows.append({
                "Variable": name,
                "Scenario": label,
                "NPV": npv,
                "IRR": irr,
                "Breakeven Year": breakeven,
            })
    
    df = pd.DataFrame(rows)
    df["NPV_Diff"] = df["NPV"] - base_npv
    
    return df, base_npv, base_irr, base_breakeven


def run_elasticity_grid(
    param1_name: str,
    param1_grid: np.ndarray,
    param2_name: str,
    param2_grid: np.ndarray,
    assumptions: Dict = ASSUMPTIONS,
    years: List[int] = YEARS,
    n_paths: int = 1_000
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-way sensitivity analysis on a grid.
    
    Returns
    -------
    tuple of (npv_matrix, breakeven_prob_matrix)
    """
    npv_matrix = np.zeros((len(param2_grid), len(param1_grid)))
    be_matrix = np.zeros_like(npv_matrix)
    
    for i, val2 in enumerate(param2_grid):
        for j, val1 in enumerate(param1_grid):
            # Run mini Monte Carlo at this point
            npvs, breakevens = [], []
            
            for _ in range(n_paths):
                scalars, per_year = build_sampled_params(assumptions, years)
                
                # Override parameters
                if isinstance(val1, (list, tuple, np.ndarray)):
                    per_year[param1_name] = list(val1)
                else:
                    scalars[param1_name] = float(val1)
                
                if isinstance(val2, (list, tuple, np.ndarray)):
                    per_year[param2_name] = list(val2)
                else:
                    scalars[param2_name] = float(val2)
                
                cash_flows = simulate_cash_flows(
                    scalars, per_year, years, assumptions
                )
                discount_rate = scalars["discount_rate"]
                npv, _, breakeven = compute_metrics(
                    cash_flows, discount_rate, years
                )
                
                npvs.append(npv)
                breakevens.append(breakeven)
            
            npv_matrix[i, j] = np.mean(npvs)
            be_matrix[i, j] = np.mean(np.array(breakevens) != np.inf)
    
    return npv_matrix, be_matrix


def make_param_grid(
    param_name: str,
    n_steps: int = 5,
    assumptions: Dict = ASSUMPTIONS
) -> np.ndarray:
    """Create evenly-spaced grid for a parameter."""
    details = assumptions[param_name]
    if isinstance(details.get("min"), list):
        raise ValueError(f"{param_name} is year-dependent")
    return np.linspace(details["min"], details["max"], n_steps)
