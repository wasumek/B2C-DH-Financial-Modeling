import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_model import (
    ASSUMPTIONS, YEARS,
    run_monte_carlo,
    run_deterministic,
    build_base_case_params
)
from results_output import (
    print_assumptions_summary,
    print_monte_carlo_summary,
    print_base_case_projection,
    export_results_to_excel
)


# ============================================================================
# TRANCHE STRUCTURING FUNCTIONS
# ============================================================================

def calculate_liquidity_gap(
    mc_results: dict,
    target_year: int = 7,
    var_percentile: float = 0.95,
    verbose: bool = True
) -> dict:
    """
    Calculate the Liquidity Gap using Value at Risk (VaR) on cumulative cash flows.
    
    The Liquidity Gap represents the worst-case cumulative cash flow deficit at the
    target investment horizon. This is the amount of "patient capital" needed to
    absorb timing risk and protect senior investors.
    
    Parameters
    ----------
    mc_results : dict
        Output from run_monte_carlo() containing cash_flows paths
    target_year : int
        Investment horizon / hard stop year (default: 7 years)
    var_percentile : float
        VaR percentile for worst-case analysis (0.95 = 95th, 0.99 = 99th)
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    dict
        Liquidity gap analysis including VaR, distribution stats, and percentiles
    """
    cash_flow_paths = mc_results["cash_flows"]
    n_simulations = len(cash_flow_paths)
    
    # Terminal value multiple (used in Monte Carlo, need to back it out)
    terminal_multiple = ASSUMPTIONS.get("terminal_value_multiple", {}).get("value", 10)
    
    # Calculate cumulative cash flow at target year for each simulation
    # EXCLUDING terminal value (which is notional, not actual liquidity)
    cumulative_cf_at_target = []
    
    for path in cash_flow_paths:
        # Copy path to avoid mutation
        adjusted_path = list(path)
        
        # Back out terminal value from final year
        # In simulate_cash_flows: cash_flows[-1] += cash_flows[-1] * terminal_multiple
        # So: final_with_TV = final_operating * (1 + terminal_multiple)
        # Therefore: final_operating = final_with_TV / (1 + terminal_multiple)
        if len(adjusted_path) > 0:
            adjusted_path[-1] = adjusted_path[-1] / (1 + terminal_multiple)
        
        # Cumulative cash flow up to target year (index = target_year - 1 for 0-indexed)
        year_idx = min(target_year - 1, len(adjusted_path) - 1)
        cumulative_cf = np.sum(adjusted_path[:year_idx + 1])
        cumulative_cf_at_target.append(cumulative_cf)
    
    cumulative_cf_array = np.array(cumulative_cf_at_target)
    
    # Calculate deficits (negative cumulative CF = liquidity gap)
    deficits = np.where(cumulative_cf_array < 0, -cumulative_cf_array, 0)
    
    # VaR: The deficit at the specified percentile
    # We want the worst-case deficit, so we look at the upper percentile of deficits
    var_95 = np.percentile(deficits, 95)
    var_99 = np.percentile(deficits, 99)
    var_selected = np.percentile(deficits, var_percentile * 100)
    
    # Probability of any deficit
    prob_deficit = np.mean(cumulative_cf_array < 0)
    
    # Expected Shortfall (CVaR) - average of losses beyond VaR
    cvar_mask = deficits >= var_selected
    expected_shortfall = np.mean(deficits[cvar_mask]) if cvar_mask.any() else 0
    
    if verbose:
        print(f"\n[LIQUIDITY GAP ANALYSIS] Investment Horizon: Year {target_year}")
        print(f"  Simulations: {n_simulations:,}")
        print(f"\n  Cumulative Cash Flow Distribution at Year {target_year}:")
        print(f"    Mean:   CHF {np.mean(cumulative_cf_array):>15,.0f}")
        print(f"    Median: CHF {np.median(cumulative_cf_array):>15,.0f}")
        print(f"    Std:    CHF {np.std(cumulative_cf_array):>15,.0f}")
        print(f"\n  Deficit Analysis:")
        print(f"    P(Deficit): {prob_deficit:.1%}")
        print(f"    VaR 95%:    CHF {var_95:>15,.0f}")
        print(f"    VaR 99%:    CHF {var_99:>15,.0f}")
        print(f"    CVaR:       CHF {expected_shortfall:>15,.0f}")
    
    return {
        "target_year": target_year,
        "cumulative_cf_distribution": cumulative_cf_array,
        "deficit_distribution": deficits,
        "var_95": var_95,
        "var_99": var_99,
        "var_selected": var_selected,
        "var_percentile": var_percentile,
        "expected_shortfall": expected_shortfall,
        "prob_deficit": prob_deficit,
        "mean_cf": np.mean(cumulative_cf_array),
        "median_cf": np.median(cumulative_cf_array),
    }


def size_tranches_from_liquidity_gap(
    liquidity_gap_results: dict,
    total_fund_size: float = 100_000_000,
    use_var_99: bool = False,
    include_mezzanine: bool = True,
    mezzanine_buffer_pct: float = 0.10,
    verbose: bool = True
) -> dict:
    """
    Size tranches based on the calculated Liquidity Gap.
    
    The Junior Tranche (patient capital / government) is sized to cover the
    Liquidity Gap. The Senior Tranche (VC/institutional) is the remaining capital,
    now protected because the Junior absorbs timing risk.
    
    Parameters
    ----------
    liquidity_gap_results : dict
        Output from calculate_liquidity_gap()
    total_fund_size : float
        Total fund capital to structure
    use_var_99 : bool
        Use 99% VaR instead of 95% for conservative sizing
    include_mezzanine : bool
        Include a mezzanine tranche as additional buffer
    mezzanine_buffer_pct : float
        Mezzanine as percentage of fund (acts as cushion above junior)
    verbose : bool
        Print tranche structure
        
    Returns
    -------
    dict
        Tranche structure with sizes and risk metrics
    """
    # Select VaR level
    liquidity_gap = liquidity_gap_results["var_99"] if use_var_99 else liquidity_gap_results["var_95"]
    var_label = "99%" if use_var_99 else "95%"
    
    # Junior Tranche = Liquidity Gap (first-loss buffer)
    junior_size = liquidity_gap
    junior_pct = junior_size / total_fund_size
    
    # Optional Mezzanine (additional buffer)
    if include_mezzanine:
        mezzanine_size = total_fund_size * mezzanine_buffer_pct
    else:
        mezzanine_size = 0
    mezzanine_pct = mezzanine_size / total_fund_size
    
    # Senior Tranche = Remainder (protected capital)
    senior_size = total_fund_size - junior_size - mezzanine_size
    senior_pct = senior_size / total_fund_size
    
    # Validate structure
    if senior_size < 0:
        # Liquidity gap exceeds fund - need to adjust
        if verbose:
            print(f"\n  ⚠ WARNING: Liquidity Gap ({liquidity_gap:,.0f}) exceeds available capital")
            print(f"    Adjusting to maximum junior allocation...")
        junior_size = total_fund_size * 0.60  # Cap at 60%
        junior_pct = 0.60
        mezzanine_size = total_fund_size * 0.15
        mezzanine_pct = 0.15
        senior_size = total_fund_size - junior_size - mezzanine_size
        senior_pct = senior_size / total_fund_size
    
    # Calculate expected senior protection
    # Senior only loses if cumulative loss exceeds junior + mezzanine
    total_buffer = junior_size + mezzanine_size
    deficits = liquidity_gap_results["deficit_distribution"]
    senior_loss_prob = np.mean(deficits > total_buffer)
    
    if verbose:
        print(f"\n[TRANCHE STRUCTURE] Based on VaR {var_label}")
        print(f"  Liquidity Gap (VaR {var_label}): CHF {liquidity_gap:>12,.0f}")
        print(f"\n  ┌{'─' * 58}┐")
        print(f"  │ {'Tranche':<20} {'Size (CHF)':>15} {'% of Fund':>10} {'Role':<8} │")
        print(f"  ├{'─' * 58}┤")
        print(f"  │ {'Junior (Gov)':<19} {junior_size:>15,.0f} {junior_pct:>9.1%} {'First-Loss':<10} │")
        if include_mezzanine:
            print(f"  │ {'Mezzanine':<19} {mezzanine_size:>15,.0f} {mezzanine_pct:>9.1%} {'Buffer':<10} │")
        print(f"  │ {'Senior (VC)':<19} {senior_size:>15,.0f} {senior_pct:>9.1%} {'Protected':<10} │")
        print(f"  ├{'─' * 58}┤")
        print(f"  │ {'TOTAL':<18} {total_fund_size:>15,.0f} {'100.0%':>10} {'':<10} │")
        print(f"  └{'─' * 58}┘")
        print(f"\n  Senior Tranche Protection:")
        print(f"    Buffer Size:        CHF {total_buffer:>12,.0f}")
        print(f"    P(Senior Loss):     {senior_loss_prob:.2%}")
        print(f"    Protection Level:   {1 - senior_loss_prob:.1%}")
    
    return {
        "junior_size": junior_size,
        "junior_pct": junior_pct,
        "mezzanine_size": mezzanine_size,
        "mezzanine_pct": mezzanine_pct,
        "senior_size": senior_size,
        "senior_pct": senior_pct,
        "total_fund_size": total_fund_size,
        "liquidity_gap": liquidity_gap,
        "var_level": var_label,
        "senior_loss_prob": senior_loss_prob,
        "target_year": liquidity_gap_results["target_year"],
    }


def plot_liquidity_gap_analysis(
    liquidity_gap_results: dict,
    tranche_structure: dict,
    output_dir: str = "plots"
):
    """
    Visualize liquidity gap analysis and tranche structure.
    
    Creates a 2-panel figure:
    1. Cumulative cash flow distribution with VaR markers
    2. Tranche structure bar chart
    
    Parameters
    ----------
    liquidity_gap_results : dict
        Output from calculate_liquidity_gap()
    tranche_structure : dict
        Output from size_tranches_from_liquidity_gap()
    output_dir : str
        Directory to save plots
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Panel 1: Cumulative Cash Flow Distribution ---
    ax1 = axes[0]
    cf_dist = liquidity_gap_results["cumulative_cf_distribution"]
    target_year = liquidity_gap_results["target_year"]
    var_95 = liquidity_gap_results["var_95"]
    var_99 = liquidity_gap_results["var_99"]
    
    # Histogram
    n, bins, patches = ax1.hist(
        cf_dist / 1e6,  # Convert to millions
        bins=50,
        color='steelblue',
        alpha=0.7,
        edgecolor='white',
        linewidth=0.5
    )
    
    # Color negative bins red
    for i, patch in enumerate(patches):
        if bins[i] < 0:
            patch.set_facecolor('indianred')
    
    # VaR lines
    ax1.axvline(0, color='black', linestyle='-', linewidth=2, label='Breakeven')
    ax1.axvline(
        -var_95 / 1e6,
        color='orange',
        linestyle='--',
        linewidth=2,
        label=f'VaR 95%: CHF {var_95/1e6:.1f}M'
    )
    ax1.axvline(
        -var_99 / 1e6,
        color='red',
        linestyle='--',
        linewidth=2,
        label=f'VaR 99%: CHF {var_99/1e6:.1f}M'
    )
    
    ax1.set_xlabel("Cumulative Cash Flow at Year 7 (CHF Millions)", fontsize=11)
    ax1.set_ylabel("Frequency", fontsize=11)
    ax1.set_title(
        f"Cash Flow Distribution at Investment Horizon (T={target_year})",
        fontsize=12,
        fontweight='bold'
    )
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add annotation for deficit probability
    prob_deficit = liquidity_gap_results["prob_deficit"]
    ax1.annotate(
        f'P(Deficit) = {prob_deficit:.1%}',
        xy=(0, max(n) * 0.9),
        xytext=(-var_95 / 1e6 - 5, max(n) * 0.9),
        fontsize=10,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    )
    
    # --- Panel 2: Tranche Structure ---
    ax2 = axes[1]
    
    tranches = ['Junior\n(First-Loss)', 'Mezzanine\n(Buffer)', 'Senior\n(Protected)']
    sizes = [
        tranche_structure["junior_size"] / 1e6,
        tranche_structure["mezzanine_size"] / 1e6,
        tranche_structure["senior_size"] / 1e6
    ]
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    
    bars = ax2.bar(tranches, sizes, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels on bars
    for bar, size, pct in zip(bars, sizes, [
        tranche_structure["junior_pct"],
        tranche_structure["mezzanine_pct"],
        tranche_structure["senior_pct"]
    ]):
        height = bar.get_height()
        ax2.annotate(
            f'CHF {size:.1f}M\n({pct:.1%})',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )
    
    ax2.set_ylabel("Capital (CHF Millions)", fontsize=11)
    ax2.set_title(
        f"Tranche Structure (VaR {tranche_structure['var_level']})",
        fontsize=12,
        fontweight='bold'
    )
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add total fund size annotation
    total = tranche_structure["total_fund_size"] / 1e6
    ax2.axhline(total, color='gray', linestyle=':', linewidth=1)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "liquidity_gap_analysis.pdf")
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"  → Saved: {filepath}")
    plt.close(fig)


def main():
    """Execute complete financial analysis with VaR-based tranche structuring."""

    print("\n" + "=" * 80)
    print("FINANCIAL MODEL: VaR-BASED TRANCHE STRUCTURING")
    print("=" * 80)

    # ========================================================================
    # 1. MONTE CARLO SIMULATION
    # ========================================================================
    print("\n[1/4] Running Monte Carlo simulation...")
    mc_results = run_monte_carlo(
        assumptions=ASSUMPTIONS,
        years=YEARS,
        n_paths=5_000,
        seed=42
    )
    print("✓ Monte Carlo complete")

    # ========================================================================
    # 2. BASE CASE ANALYSIS
    # ========================================================================
    print("\n[2/4] Computing base case projection...")
    _, _, _, base_case_details = run_deterministic(
        overrides={},
        assumptions=ASSUMPTIONS,
        years=YEARS,
        return_details=True
    )
    print("✓ Base case complete")

    # ========================================================================
    # 3. LIQUIDITY GAP ANALYSIS (VaR-Based)
    # ========================================================================
    print("\n[3/4] Calculating Liquidity Gap and Fund Size...")
    
    # Step 1: Calculate Liquidity Gap using VaR on cumulative cash flows
    liquidity_gap_results = calculate_liquidity_gap(
        mc_results=mc_results,
        target_year=7,  # Investment horizon: 7 years
        var_percentile=0.95,
        verbose=True
    )

    # Step 2: Determine Total Fund Size from existing NPV calculation
    # The Monte Carlo already computed NPV for each path using compute_metrics()
    # We use the expected (mean) NPV as the fund valuation basis
    npv_distribution = mc_results["npv"]
    
    # For fund sizing, use the median positive NPV or a capital requirement metric
    # Option A: Use expected NPV (mean of all paths)
    # Option B: Use peak capital requirement from cumulative cash flows
    
    # Use VaR 99% of peak deficit as total capital needed
    cash_flow_paths = mc_results["cash_flows"]
    terminal_multiple = ASSUMPTIONS.get("terminal_value_multiple", {}).get("value", 10)
    
    peak_deficits = []
    for path in cash_flow_paths:
        adjusted_path = list(path)
        if len(adjusted_path) > 0:
            adjusted_path[-1] = adjusted_path[-1] / (1 + terminal_multiple)
        cumulative = np.cumsum(adjusted_path)
        peak_deficit = min(0, np.min(cumulative))
        peak_deficits.append(-peak_deficit)
    
    total_capital_required = np.percentile(peak_deficits, 99)
    
    # Also show NPV for context
    mean_npv = np.mean(npv_distribution)
    median_npv = np.median(npv_distribution)
    
    print(f"\n  NPV from Monte Carlo (includes {terminal_multiple}x terminal value):")
    print(f"    Mean NPV:   CHF {mean_npv:>12,.0f}")
    print(f"    Median NPV: CHF {median_npv:>12,.0f}")
    print(f"    Note: NPV = PV of cash flows + terminal value (exit multiple)")
    print(f"\n  Peak Capital Requirement (VaR 99%, excludes terminal value):")
    print(f"    CHF {total_capital_required:>12,.0f}")
    print(f"    Note: Max cumulative deficit = actual cash needed before exit")
    
    # Step 3: Size tranches based on Liquidity Gap
    # Junior = Liquidity Gap at Year 7 (timing risk buffer)
    # Mezzanine = Additional buffer
    # Senior = Remainder (protected)
    tranche_structure = size_tranches_from_liquidity_gap(
        liquidity_gap_results=liquidity_gap_results,
        total_fund_size=total_capital_required,  # Dynamic from model!
        use_var_99=False,  # Use 95% VaR for junior sizing
        include_mezzanine=True,
        mezzanine_buffer_pct=0.15,  # 15% mezzanine
        verbose=True
    )
    print("\n✓ Tranche structuring complete")

    # ========================================================================
    # 4. GENERATE OUTPUTS
    # ========================================================================
    print("\n[4/4] Generating outputs...")

    # Generate visualization
    plot_liquidity_gap_analysis(
        liquidity_gap_results,
        tranche_structure,
        output_dir="plots"
    )

    # Print summary outputs
    scalars, per_year = build_base_case_params(ASSUMPTIONS, YEARS)
    print_assumptions_summary(ASSUMPTIONS, YEARS, scalars, per_year)
    print_monte_carlo_summary(mc_results)
    print_base_case_projection(base_case_details)

    print("\n" + "=" * 80)
    print("TRANCHE ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Results:")
    print(f"  • Investment Horizon: {liquidity_gap_results['target_year']} years")
    print(f"  • Liquidity Gap (VaR 95%): CHF {liquidity_gap_results['var_95']:,.0f}")
    print(f"  • Liquidity Gap (VaR 99%): CHF {liquidity_gap_results['var_99']:,.0f}")
    print(f"  • Junior Tranche: CHF {tranche_structure['junior_size']:,.0f} ({tranche_structure['junior_pct']:.1%})")
    print(f"  • Senior Protection: {1 - tranche_structure['senior_loss_prob']:.1%}")
    print("\nOutputs:")
    print("  • PDF: plots/liquidity_gap_analysis.pdf")
    print("\n")


if __name__ == "__main__":
    main()