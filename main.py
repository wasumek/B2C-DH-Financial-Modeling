"""
Main Analysis Script
====================

Execute complete financial model analysis with Monte Carlo simulation,
sensitivity analysis, and elasticity testing.

Usage:
    python main.py

Author: Wasu Mekniran
Date: 09.12.2025
"""

import numpy as np
from financial_model import (
    ASSUMPTIONS, YEARS,
    run_monte_carlo,
    run_deterministic,
    build_tornado_data,
    run_elasticity_grid,
    make_param_grid
)
from financial_engineering import run_advanced_simulation
from results_output import (
    print_assumptions_summary,
    print_monte_carlo_summary,
    print_base_case_projection,
    print_tornado_summary,
    print_elasticity_grid,
    export_results_to_excel,
    print_advanced_simulation_summary
)
from visualization import create_all_plots


def main():
    """Execute complete financial analysis."""
    
    print("\n" + "=" * 80)
    print("FINANCIAL MODEL ANALYSIS")
    print("=" * 80)
    
    # ========================================================================
    # 1. MONTE CARLO SIMULATION
    # ========================================================================
    print("\n[1/5] Running Monte Carlo simulation...")
    mc_results = run_monte_carlo(
        assumptions=ASSUMPTIONS,
        years=YEARS,
        n_paths=5_000,
        seed=42  # For reproducibility
    )
    print("✓ Monte Carlo complete")
    
    # ========================================================================
    # 2. BASE CASE ANALYSIS
    # ========================================================================
    print("\n[2/5] Computing base case projection...")
    _, _, _, base_case_details = run_deterministic(
        overrides={},
        assumptions=ASSUMPTIONS,
        years=YEARS,
        return_details=True
    )
    print("✓ Base case complete")
    
    # ========================================================================
    # 3. ONE-WAY SENSITIVITY ANALYSIS
    # ========================================================================
    print("\n[3/5] Running one-way sensitivity analysis...")
    tornado_df, base_npv, base_irr, base_breakeven = build_tornado_data(
        assumptions=ASSUMPTIONS,
        years=YEARS
    )
    print("✓ Tornado analysis complete")
    
    # ========================================================================
    # 4. TWO-WAY ELASTICITY ANALYSIS
    # ========================================================================
    print("\n[4/5] Running two-way elasticity analysis...")
    
    # Price × Churn
    price_grid = make_param_grid("price_per_month", n_steps=5, assumptions=ASSUMPTIONS)
    churn_grid = make_param_grid("annual_churn_rate", n_steps=5, assumptions=ASSUMPTIONS)
    
    npv_pc, be_pc = run_elasticity_grid(
        "price_per_month", price_grid,
        "annual_churn_rate", churn_grid,
        assumptions=ASSUMPTIONS,
        years=YEARS,
        n_paths=1_000
    )
    
    # CAC × Screening
    cac_grid = make_param_grid("cac_per_user", n_steps=5, assumptions=ASSUMPTIONS)
    screen_grid = make_param_grid("screening_participation_pct", n_steps=5, assumptions=ASSUMPTIONS)
    
    npv_cs, be_cs = run_elasticity_grid(
        "cac_per_user", cac_grid,
        "screening_participation_pct", screen_grid,
        assumptions=ASSUMPTIONS,
        years=YEARS,
        n_paths=1_000
    )
    
    elasticity_results = {
        "price_grid": price_grid,
        "churn_grid": np.sort(churn_grid),
        "npv_price_churn": npv_pc,
        "be_price_churn": be_pc,
        "cac_grid": cac_grid,
        "screen_grid": np.sort(screen_grid),
        "npv_cac_screen": npv_cs,
        "be_cac_screen": be_cs,
    }
    print("✓ Elasticity analysis complete")
    
    # ========================================================================
    # 5. ADVANCED FINANCIAL ENGINEERING
    # ========================================================================
    print("\n[5/6] Running advanced financial engineering simulation...")
    advanced_results = run_advanced_simulation(
        assumptions=ASSUMPTIONS,
        years=YEARS,
        n_paths=5_000,
        seed=42
    )
    print("✓ Advanced simulation complete")
    
    # ========================================================================
    # 6. GENERATE OUTPUTS
    # ========================================================================
    print("\n[6/6] Generating outputs...")
    
    # Print results to console
    from financial_model import build_base_case_params
    scalars, per_year = build_base_case_params(ASSUMPTIONS, YEARS)
    
    print_assumptions_summary(ASSUMPTIONS, YEARS, scalars, per_year)
    print_monte_carlo_summary(mc_results)
    print_base_case_projection(base_case_details)
    print_tornado_summary(tornado_df, base_npv, top_n=10)
    
    print_elasticity_grid(
        "price_per_month", price_grid,
        "annual_churn_rate", churn_grid,
        npv_pc, be_pc
    )
    
    print_elasticity_grid(
        "cac_per_user", cac_grid,
        "screening_participation_pct", screen_grid,
        npv_cs, be_cs
    )
    
    # Print advanced simulation summary
    print_advanced_simulation_summary(advanced_results)
    
    # Export to Excel
    export_results_to_excel(
        mc_results,
        base_case_details,
        tornado_df,
        filename="model_results.xlsx"
    )
    
    # Generate all plots
    create_all_plots(
        mc_results,
        base_case_details,
        tornado_df,
        base_npv,
        elasticity_results,
        YEARS,
        output_dir="plots"
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOutputs:")
    print("  • Console: Detailed results and statistics")
    print("  • Excel: model_results.xlsx")
    print("  • PDF plots: plots/ directory")
    print("\n")


if __name__ == "__main__":
    main()
