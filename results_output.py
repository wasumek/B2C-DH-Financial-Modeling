"""
Results Output Module
=====================

Functions for printing model results in structured formats.

Author: Wasu Mekniran
Date: 09.12.2025
"""

import numpy as np
import pandas as pd
from typing import Dict

# Parameter labels for output
PARAM_LABELS = {
    "screening_participation_pct": "Screening Participation",
    "adoption_ramp": "Adoption Ramp Factor",
    "conv_screen_to_paid": "Conversion Screen to Paid",
    "p_USPSTF_eligible": "USPSTF Eligible Pop. %",
    "p_IFG": "IFG Prevalence %",
    "p_undiag": "Undiagnosed Prevalence %",
    "price_per_month": "Price per Month",
    "annual_churn_rate": "Annual Churn Rate",
    "cac_per_user": "CAC per User",
    "technician_ratio": "Technician Ratio (Users/Tech)",
    "manager_ratio": "Manager Ratio (Tech/Manager)",
    "technician_salary": "Technician Salary",
    "manager_salary": "Manager Salary",
    "app_dev": "App Development Cost",
    "office_rent_per_worker": "Office Rent per Worker",
    "office_supplies_per_worker": "Office Supplies per Worker",
    "call_center_per_user": "Call Center Cost per User",
    "backend_services_per_user": "Backend Services per User",
    "ce_mdr_certification_cost": "CE MDR Cert. Cost",
    "rev_fadp_compliance_cost": "Rev FADP Compliance Cost",
    "discount_rate": "Discount Rate",
}


def print_assumptions_summary(
    assumptions: Dict,
    years: list,
    scalars: Dict,
    per_year: Dict
):
    """Print summary of model assumptions."""
    print("=" * 80)
    print("MODEL ASSUMPTIONS SUMMARY")
    print("=" * 80)
    print(f"\nTime Horizon: {years[0]} to {years[-1]} ({len(years)} years)")
    print(f"\nTotal Parameters: {len(assumptions)}")
    
    # Scalar parameters
    print("\n" + "-" * 80)
    print("SCALAR PARAMETERS (Base Case)")
    print("-" * 80)
    
    for name in sorted(scalars.keys()):
        label = PARAM_LABELS.get(name, name)
        value = scalars[name]
        
        if "pct" in name or "rate" in name or name == "discount_rate":
            print(f"{label:.<50} {value:.2%}")
        elif "ratio" in name:
            print(f"{label:.<50} {value:,.1f}")
        elif "population" in name or "salary" in name or name.endswith("_user"):
            print(f"{label:.<50} {value:,.0f}")
        else:
            print(f"{label:.<50} {value:,.2f}")
    
    # Year-dependent parameters
    print("\n" + "-" * 80)
    print("YEAR-DEPENDENT PARAMETERS (Base Case)")
    print("-" * 80)
    
    for name in sorted(per_year.keys()):
        label = PARAM_LABELS.get(name, name)
        print(f"\n{label}:")
        values = per_year[name]
        for i, (year, val) in enumerate(zip(years, values)):
            if "ramp" in name or "pct" in name:
                print(f"  Year {year}: {val:.2%}")
            elif "_cost" in name or name == "app_dev":
                print(f"  Year {year}: {val:>12,.0f} CHF")
            else:
                print(f"  Year {year}: {val:>12,.2f}")


def print_monte_carlo_summary(mc_results: Dict):
    """Print Monte Carlo simulation summary statistics."""
    print("\n" + "=" * 80)
    print("MONTE CARLO SIMULATION SUMMARY")
    print("=" * 80)
    
    npv = mc_results["npv"]
    irr = mc_results["irr"]
    breakeven = mc_results["breakeven_year"]
    
    valid_irr = irr[~np.isnan(irr)]
    
    print(f"\nSimulation Paths: {len(npv):,}")
    
    # NPV statistics
    print("\n" + "-" * 80)
    print("NET PRESENT VALUE (CHF)")
    print("-" * 80)
    print(f"Mean:             {np.mean(npv):>15,.0f}")
    print(f"Median:           {np.median(npv):>15,.0f}")
    print(f"Std Dev:          {np.std(npv):>15,.0f}")
    print(f"2.5th Percentile: {np.percentile(npv, 2.5):>15,.0f}")
    print(f"97.5th Percentile:{np.percentile(npv, 97.5):>15,.0f}")
    print(f"Prob(NPV > 0):    {np.mean(npv > 0):>15.1%}")
    
    # IRR statistics
    print("\n" + "-" * 80)
    print("INTERNAL RATE OF RETURN")
    print("-" * 80)
    if valid_irr.size > 0:
        print(f"Mean:             {np.mean(valid_irr):>15.2%}")
        print(f"Median:           {np.median(valid_irr):>15.2%}")
        print(f"Std Dev:          {np.std(valid_irr):>15.2%}")
        print(f"2.5th Percentile: {np.percentile(valid_irr, 2.5):>15.2%}")
        print(f"97.5th Percentile:{np.percentile(valid_irr, 97.5):>15.2%}")
    else:
        print("No valid IRR values computed")
    
    # Breakeven statistics
    print("\n" + "-" * 80)
    print("BREAKEVEN YEAR")
    print("-" * 80)
    print(f"Prob(Breakeven):  {np.mean(breakeven != np.inf):>15.1%}")
    
    finite_be = breakeven[breakeven != np.inf]
    if finite_be.size > 0:
        print(f"Mean Year:        {np.mean(finite_be):>15.1f}")
        print(f"Median Year:      {np.median(finite_be):>15.1f}")


def print_base_case_projection(details: Dict):
    """Print base-case financial projection."""
    print("\n" + "=" * 80)
    print("BASE-CASE FINANCIAL PROJECTION")
    print("=" * 80)
    
    df = pd.DataFrame({
        "Year": details["year"],
        "Active Users": details["active_users"],
        "New Subs": details["new_subscribers"],
        "Revenue": details["revenue"],
        "Total Cost": details["total_cost"],
        "Net CF": details["net_cf"],
        "Cum. CF": np.cumsum(details["net_cf"]),
    })
    
    # Format output
    pd.options.display.float_format = '{:,.0f}'.format
    print("\n" + df.to_string(index=False))
    
    # Cost breakdown for final year
    print("\n" + "-" * 80)
    print(f"COST BREAKDOWN - YEAR {details['year'][-1]}")
    print("-" * 80)
    
    idx = -1  # Last year
    total = details["total_cost"][idx]
    
    cost_items = [
        ("CAC", details["cac_cost"][idx]),
        ("Technician Salaries", details["tech_cost"][idx]),
        ("Manager Salaries", details["mgr_cost"][idx]),
        ("Call Center", details["call_cost"][idx]),
        ("Backend Services", details["backend_cost"][idx]),
        ("Office Rent", details["rent_cost"][idx]),
        ("Office Supplies", details["supplies_cost"][idx]),
        ("App Development", details["app_dev_cost"][idx]),
        ("CE MDR Certification", details["ce_mdr_cost"][idx]),
        ("FADP Compliance", details["fadp_cost"][idx]),
    ]
    
    for label, cost in cost_items:
        pct = cost / total * 100 if total > 0 else 0
        print(f"{label:.<40} {cost:>12,.0f}  ({pct:>5.1f}%)")
    
    print(f"{'TOTAL':.<40} {total:>12,.0f}  (100.0%)")


def print_tornado_summary(tornado_df: pd.DataFrame, base_npv: float, top_n: int = 10):
    """Print tornado diagram data."""
    print("\n" + "=" * 80)
    print("ONE-WAY SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"\nBase Case NPV: {base_npv:,.0f} CHF")
    
    # Compute impact for each variable
    summary = (
        tornado_df
        .pivot(index="Variable", columns="Scenario", values="NPV")
        .reset_index()
    )
    summary["Delta_Min"] = summary["Min"] - base_npv
    summary["Delta_Max"] = summary["Max"] - base_npv
    summary["Impact"] = summary[["Delta_Min", "Delta_Max"]].abs().max(axis=1)
    summary = summary.sort_values("Impact", ascending=False).head(top_n)
    
    print(f"\nTop {top_n} Most Influential Parameters:")
    print("-" * 80)
    
    for _, row in summary.iterrows():
        var = row["Variable"]
        label = PARAM_LABELS.get(var, var)
        impact = row["Impact"]
        delta_min = row["Delta_Min"]
        delta_max = row["Delta_Max"]
        
        print(f"\n{label}")
        print(f"  Min Scenario: {delta_min:>+15,.0f}  (NPV: {row['Min']:,.0f})")
        print(f"  Max Scenario: {delta_max:>+15,.0f}  (NPV: {row['Max']:,.0f})")
        print(f"  Max Impact:   {impact:>15,.0f}")


def print_elasticity_grid(
    param1_name: str,
    param1_grid: np.ndarray,
    param2_name: str,
    param2_grid: np.ndarray,
    npv_matrix: np.ndarray,
    be_matrix: np.ndarray
):
    """Print elasticity analysis results."""
    print("\n" + "=" * 80)
    print(f"TWO-WAY SENSITIVITY: {param1_name.upper()} vs {param2_name.upper()}")
    print("=" * 80)
    
    # NPV matrix
    print("\nAverage NPV (CHF):")
    print("-" * 80)
    
    npv_df = pd.DataFrame(
        npv_matrix,
        index=[f"{v:.2f}" for v in param2_grid],
        columns=[f"{v:.0f}" for v in param1_grid]
    )
    npv_df.index.name = PARAM_LABELS.get(param2_name, param2_name)
    npv_df.columns.name = PARAM_LABELS.get(param1_name, param1_name)
    
    pd.options.display.float_format = '{:,.0f}'.format
    print(npv_df.to_string())
    
    # Breakeven probability matrix
    print("\n\nBreakeven Probability:")
    print("-" * 80)
    
    be_df = pd.DataFrame(
        be_matrix,
        index=[f"{v:.2f}" for v in param2_grid],
        columns=[f"{v:.0f}" for v in param1_grid]
    )
    be_df.index.name = PARAM_LABELS.get(param2_name, param2_name)
    be_df.columns.name = PARAM_LABELS.get(param1_name, param1_name)
    
    pd.options.display.float_format = '{:.2%}'.format
    print(be_df.to_string())


def export_results_to_excel(
    mc_results: Dict,
    base_case_details: Dict,
    tornado_df: pd.DataFrame,
    filename: str = "model_results.xlsx"
):
    """Export all results to Excel workbook."""
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        # Monte Carlo summary
        mc_summary = pd.DataFrame({
            "NPV": mc_results["npv"],
            "IRR": mc_results["irr"],
            "Breakeven Year": mc_results["breakeven_year"],
        })
        mc_summary.to_excel(writer, sheet_name="Monte Carlo", index=False)
        
        # Base case projection
        base_df = pd.DataFrame(base_case_details)
        base_df["Cumulative CF"] = np.cumsum(base_df["net_cf"])
        base_df.to_excel(writer, sheet_name="Base Case", index=False)
        
        # Tornado data
        tornado_df.to_excel(writer, sheet_name="Sensitivity", index=False)
    
    print(f"\nResults exported to: {filename}")
