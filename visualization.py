"""
Visualization Module
====================

Publication-ready plotting functions for financial model results.

Author: Wasu Mekniran
Date: 09.12.2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.patches import Patch
from typing import Dict, Optional
import os

# ============================================================================
# PLOTTING STYLE CONFIGURATION
# ============================================================================

# Set global style
mpl.rcParams.update({
    "figure.dpi": 120,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

# Color palette
TEXT_COLOR = "#1F2933"
GRID_COLOR = "#D0D4DA"
COLOR_POSITIVE = sns.color_palette("Blues")[4]
COLOR_NEGATIVE = sns.color_palette("Reds")[4]
COLOR_ACCENT = sns.color_palette("Purples")[4]
COLOR_NEUTRAL = "#888888"

# Parameter labels for plots
PARAM_LABELS = {
    "screening_participation_pct": "Screening Participation",
    "adoption_ramp": "Adoption Ramp",
    "conv_screen_to_paid": "Conversion Rate",
    "p_USPSTF_eligible": "Eligible Population %",
    "p_IFG": "IFG Prevalence %",
    "p_undiag": "Undiagnosed %",
    "price_per_month": "Price per Month",
    "annual_churn_rate": "Annual Churn",
    "cac_per_user": "CAC per User",
    "technician_ratio": "Users per Tech",
    "manager_ratio": "Tech per Manager",
    "technician_salary": "Tech Salary",
    "manager_salary": "Manager Salary",
}


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def fmt_millions(x, pos=None):
    """Format axis labels in millions."""
    return f"{x/1e6:,.0f}m"


def fmt_percent(x, pos=None):
    """Format axis labels as percentages."""
    return f"{x:.0%}"


def save_or_show(fig, filename: Optional[str], output_dir: str = "plots"):
    """Save figure to PDF or display."""
    if filename:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved: {filepath}")
    else:
        plt.show()


# ============================================================================
# MONTE CARLO DISTRIBUTION PLOTS
# ============================================================================

def plot_npv_distribution(
    mc_results: Dict,
    filename: Optional[str] = None,
    figsize: tuple = (8, 5)
):
    """Plot NPV distribution with statistics."""
    npv = mc_results["npv"] / 1e6  # Convert to millions
    
    median_npv = np.median(npv)
    low_npv = np.percentile(npv, 2.5)
    high_npv = np.percentile(npv, 97.5)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram with KDE
    sns.histplot(
        npv, kde=True, ax=ax,
        color=COLOR_POSITIVE, alpha=0.75, edgecolor="white"
    )
    
    # Reference lines
    ax.axvline(median_npv, color=COLOR_NEUTRAL, linestyle="--", linewidth=2)
    ax.axvline(low_npv, color=COLOR_NEGATIVE, linestyle=":", linewidth=1.5)
    ax.axvline(high_npv, color=COLOR_NEGATIVE, linestyle=":", linewidth=1.5)
    
    # Statistics box
    stats_text = (
        f"Median: {median_npv:.1f}m CHF\n"
        f"95% CI: [{low_npv:.1f}m, {high_npv:.1f}m]"
    )
    ax.text(
        0.98, 0.88, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID_COLOR, alpha=0.9)
    )
    
    ax.set_xlabel("Net Present Value (CHF, millions)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo NPV Distribution", weight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    
    fig.tight_layout()
    save_or_show(fig, filename)


def plot_irr_distribution(
    mc_results: Dict,
    filename: Optional[str] = None,
    figsize: tuple = (8, 5)
):
    """Plot IRR distribution with statistics."""
    irr = mc_results["irr"]
    valid_irr = irr[~np.isnan(irr)] * 100  # Convert to percentage
    
    if valid_irr.size == 0:
        print("Warning: No valid IRR values to plot")
        return
    
    median_irr = np.median(valid_irr)
    low_irr = np.percentile(valid_irr, 2.5)
    high_irr = np.percentile(valid_irr, 97.5)
    
    # Focus on central mass
    x_min = np.percentile(valid_irr, 1)
    x_max = np.percentile(valid_irr, 99)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(
        valid_irr, kde=True, ax=ax,
        color=COLOR_ACCENT, alpha=0.75, edgecolor="white"
    )
    ax.set_xlim(x_min, x_max)
    
    # Reference lines
    ax.axvline(median_irr, color=COLOR_NEUTRAL, linestyle="--", linewidth=2)
    ax.axvline(low_irr, color=COLOR_NEGATIVE, linestyle=":", linewidth=1.5)
    ax.axvline(high_irr, color=COLOR_NEGATIVE, linestyle=":", linewidth=1.5)
    
    # Statistics box
    stats_text = (
        f"Median: {median_irr:.1f}%\n"
        f"95% CI: [{low_irr:.1f}%, {high_irr:.1f}%]"
    )
    ax.text(
        0.98, 0.88, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=GRID_COLOR, alpha=0.9)
    )
    
    ax.set_xlabel("Internal Rate of Return (%)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo IRR Distribution", weight="bold")
    ax.grid(True, linestyle=":", alpha=0.7)
    
    fig.tight_layout()
    save_or_show(fig, filename)


def plot_breakeven_probability(
    mc_results: Dict,
    years: list,
    filename: Optional[str] = None,
    figsize: tuple = (8.5, 5)
):
    """Plot breakeven probability by year."""
    breakeven = mc_results["breakeven_year"]
    
    # Calculate probabilities
    probs = [np.mean(breakeven == y) for y in years]
    x_labels = [str(y) for y in years]
    x_pos = np.arange(len(x_labels))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(
        x_pos, probs,
        color=COLOR_ACCENT, edgecolor="black", linewidth=0.7
    )
    
    # Labels on bars
    for rect, p in zip(bars, probs):
        if p <= 0:
            continue
        height = rect.get_height()
        
        if height > 0.15:
            y_text = height - 0.03
            va = "top"
            color = "white"
        else:
            y_text = height + 0.02
            va = "bottom"
            color = TEXT_COLOR
        
        ax.text(
            rect.get_x() + rect.get_width() / 2, y_text,
            f"{p:.0%}", ha="center", va=va,
            fontsize=10, color=color, fontweight="medium"
        )
    
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability of Breakeven")
    ax.set_title("Breakeven Probability by Year", weight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_percent))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    
    fig.tight_layout()
    save_or_show(fig, filename)


# ============================================================================
# BASE CASE PROJECTION PLOT
# ============================================================================

def plot_base_case_projection(
    details: Dict,
    filename: Optional[str] = None,
    figsize: tuple = (10, 5.5)
):
    """Plot base-case revenue, cost, and cumulative profit."""
    years = details["year"]
    revenue = np.array(details["revenue"])
    cost = np.array(details["total_cost"])
    cum_profit = np.cumsum(details["net_cf"])
    
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Primary axis: Revenue and Cost
    line_rev, = ax1.plot(
        years, revenue, marker="o", linewidth=2,
        color=COLOR_POSITIVE, label="Annual Revenue", zorder=3
    )
    line_cost, = ax1.plot(
        years, cost, marker="s", linewidth=2,
        color=COLOR_NEGATIVE, label="Annual Total Cost", zorder=3
    )
    ax1.axhline(0, color=GRID_COLOR, linestyle=":", linewidth=1)
    
    # Find first profitable year
    profitable_year = next(
        (int(y) for y, r, c in zip(years, revenue, cost) if r >= c),
        None
    )
    if profitable_year:
        ax1.axvline(
            profitable_year, color="#BBBBBB", linestyle="--", linewidth=1
        )
        ax1.text(
            profitable_year + 0.1, ax1.get_ylim()[1] * 0.9,
            f"Annual Profitability: Y{profitable_year}",
            fontsize=10, ha='left', zorder=4
        )
    
    # Secondary axis: Cumulative Profit
    ax2 = ax1.twinx()
    line_cum, = ax2.plot(
        years, cum_profit, marker="^", linewidth=2,
        color=COLOR_ACCENT, label="Cumulative Profit/Loss", zorder=3
    )
    ax2.axhline(0, color=GRID_COLOR, linestyle=":", linewidth=1)
    
    # Find breakeven year
    breakeven_year = next(
        (int(y) for y, v in zip(years, cum_profit) if v >= 0),
        None
    )
    if breakeven_year:
        ax2.axvline(
            breakeven_year, color="#BBBBBB", linestyle=":", linewidth=1
        )
        ax2.text(
            breakeven_year - 0.1, ax2.get_ylim()[0] * 0.9,
            f"Cumulative Breakeven: Y{breakeven_year}",
            fontsize=10, ha='right', zorder=4
        )
    
    # Formatting
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Revenue / Cost (CHF)")
    ax2.set_ylabel("Cumulative Profit/Loss (CHF)", color=COLOR_ACCENT)
    
    ax1.ticklabel_format(style="plain", axis="y")
    ax2.ticklabel_format(style="plain", axis="y")
    
    # Legend
    lines = [line_rev, line_cost, line_cum]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", frameon=True, facecolor="white")
    
    ax1.grid(True, linestyle=":", color=GRID_COLOR, alpha=0.6)
    ax2.yaxis.grid(False)
    ax1.set_axisbelow(True)
    
    plt.title("Financial Projection – Base Case", weight="bold")
    fig.tight_layout()
    save_or_show(fig, filename)


# ============================================================================
# TORNADO DIAGRAM
# ============================================================================

def plot_tornado_diagram(
    tornado_df: pd.DataFrame,
    base_npv: float,
    top_n: int = 10,
    min_impact_millions: float = 0.1,
    filename: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plot tornado diagram for sensitivity analysis."""
    # Prepare data
    df = (
        tornado_df
        .pivot(index="Variable", columns="Scenario", values="NPV")
        .dropna(subset=["Min", "Max"])
        .reset_index()
    )
    
    df["Delta_Min"] = df["Min"] - base_npv
    df["Delta_Max"] = df["Max"] - base_npv
    df["Impact"] = df[["Delta_Min", "Delta_Max"]].abs().max(axis=1)
    
    # Filter by minimum impact
    cutoff = min_impact_millions * 1e6
    df = df[df["Impact"] >= cutoff]
    
    if df.empty:
        print("No variables exceed minimum impact threshold")
        return
    
    # Sort and take top N
    df = df.sort_values("Impact", ascending=True).tail(top_n)
    
    # Reshape for plotting
    long = df.melt(
        id_vars=["Variable"],
        value_vars=["Delta_Min", "Delta_Max"],
        var_name="Scenario",
        value_name="Delta_NPV"
    )
    long["Scenario"] = long["Scenario"].map({
        "Delta_Min": "Min Value",
        "Delta_Max": "Max Value"
    })
    long["Label"] = long["Variable"].map(PARAM_LABELS).fillna(long["Variable"])
    
    # Set categorical order
    var_order = df["Variable"].tolist()
    label_order = [PARAM_LABELS.get(v, v) for v in var_order]
    long["Label"] = pd.Categorical(
        long["Label"], categories=label_order, ordered=True
    )
    
    # Assign colors
    long["Color"] = [
        COLOR_POSITIVE if delta > 0 else COLOR_NEGATIVE
        for delta in long["Delta_NPV"]
    ]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.barh(
        y=long["Label"], width=long["Delta_NPV"],
        color=long["Color"], edgecolor="black",
        linewidth=0.5, alpha=0.9
    )
    
    # Symmetric axis
    max_dev = np.max(np.abs(long["Delta_NPV"]))
    x_lim = 1.35 * max_dev
    ax.set_xlim(-x_lim, x_lim)
    
    # Zero line
    ax.axvline(0, color="#555", linestyle="--", linewidth=1.2)
    
    # Value labels outside bars
    for _, row in long.iterrows():
        delta = row["Delta_NPV"]
        y_val = row["Label"]
        delta_m = delta / 1e6
        
        label = f"{delta_m:+.1f}m"
        
        if delta > 0:
            x_pos = delta + (0.03 * x_lim)
            ha = "left"
        else:
            x_pos = delta - (0.03 * x_lim)
            ha = "right"
        
        ax.text(
            x_pos, y_val, label,
            ha=ha, va="center", fontsize=10, color=TEXT_COLOR
        )
    
    # Formatting
    ax.set_xlabel("Change in NPV vs Base (CHF)")
    ax.set_title("Tornado Diagram: One-way Sensitivity of NPV", weight="bold")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_millions))
    ax.grid(axis="x", linestyle=":", color=GRID_COLOR, alpha=0.7)
    ax.set_axisbelow(True)
    
    # Legend
    legend_handles = [
        Patch(facecolor=COLOR_POSITIVE, edgecolor="black",
              label="Higher NPV than base"),
        Patch(facecolor=COLOR_NEGATIVE, edgecolor="black",
              label="Lower NPV than base"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True,
              facecolor="white")
    
    fig.tight_layout()
    save_or_show(fig, filename)


# ============================================================================
# HEATMAP PLOTS
# ============================================================================

def annotate_heatmap(ax, data, fmt_func, fontsize: int = 10):
    """Annotate heatmap with formatted values."""
    n_rows, n_cols = data.shape
    vmax_local = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            
            # Choose text color for contrast
            txt_color = "white" if abs(val) > 0.6 * vmax_local else TEXT_COLOR
            
            ax.text(
                j + 0.5, i + 0.5, fmt_func(val),
                ha="center", va="center",
                fontsize=fontsize, color=txt_color
            )


def plot_elasticity_heatmap(
    param1_name: str,
    param1_grid: np.ndarray,
    param2_name: str,
    param2_grid: np.ndarray,
    matrix: np.ndarray,
    metric: str = "NPV",
    filename: Optional[str] = None,
    figsize: tuple = (9, 4.5)
):
    """
    Plot elasticity heatmap for two parameters.
    
    Parameters
    ----------
    metric : str
        Either "NPV" or "Breakeven"
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Format tick labels
    if "churn" in param2_name or "pct" in param2_name:
        yticklabels = [f"{v:.0%}" for v in param2_grid]
    else:
        yticklabels = [f"{v:,.0f}" for v in param2_grid]
    
    xticklabels = [f"{v:,.0f}" for v in param1_grid]
    
    # Choose colormap and normalization
    if metric == "NPV":
        vmax = np.percentile(np.abs(matrix), 98)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cmap = "RdBu"
        cbar_label = "Average NPV (CHF)"
        fmt_func = lambda x: f"{x/1e6:,.1f}"
        cbar_format = FuncFormatter(fmt_millions)
    else:  # Breakeven
        norm = None
        cmap = "Blues"
        cbar_label = "Breakeven Probability"
        fmt_func = lambda x: f"{x:.0%}"
        cbar_format = FuncFormatter(fmt_percent)
    
    # Create heatmap
    hm = sns.heatmap(
        matrix,
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        cmap=cmap,
        norm=norm,
        cbar_kws={"label": cbar_label, "format": cbar_format},
        annot=False,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    
    # Annotate cells
    annotate_heatmap(ax, matrix, fmt_func)
    
    # Labels
    param1_label = PARAM_LABELS.get(param1_name, param1_name)
    param2_label = PARAM_LABELS.get(param2_name, param2_name)
    
    ax.set_xlabel(param1_label)
    ax.set_ylabel(param2_label)
    ax.set_title(
        f"Elasticity Heatmap: {metric} by {param1_label} and {param2_label}",
        weight="bold"
    )
    
    fig.tight_layout()
    save_or_show(fig, filename)


# ============================================================================
# CONVENIENCE FUNCTION FOR ALL PLOTS
# ============================================================================

def create_all_plots(
    mc_results: Dict,
    base_case_details: Dict,
    tornado_df: pd.DataFrame,
    base_npv: float,
    elasticity_results: Dict,
    years: list,
    output_dir: str = "plots"
):
    """Generate all standard plots and save to directory."""
    print(f"\nGenerating plots in: {output_dir}/")
    
    # Monte Carlo distributions
    plot_npv_distribution(
        mc_results, filename="mc_npv_distribution.pdf"
    )
    plot_irr_distribution(
        mc_results, filename="mc_irr_distribution.pdf"
    )
    plot_breakeven_probability(
        mc_results, years, filename="breakeven_probability.pdf"
    )
    
    # Base case projection
    plot_base_case_projection(
        base_case_details, filename="base_case_projection.pdf"
    )
    
    # Tornado diagram
    plot_tornado_diagram(
        tornado_df, base_npv, filename="tornado_diagram.pdf"
    )
    
    # Elasticity heatmaps
    if elasticity_results:
        # Price × Churn
        plot_elasticity_heatmap(
            "price_per_month",
            elasticity_results["price_grid"],
            "annual_churn_rate",
            elasticity_results["churn_grid"],
            elasticity_results["npv_price_churn"],
            metric="NPV",
            filename="elasticity_npv_price_churn.pdf"
        )
        
        plot_elasticity_heatmap(
            "price_per_month",
            elasticity_results["price_grid"],
            "annual_churn_rate",
            elasticity_results["churn_grid"],
            elasticity_results["be_price_churn"],
            metric="Breakeven",
            filename="elasticity_be_price_churn.pdf"
        )
        
        # CAC × Screening
        plot_elasticity_heatmap(
            "cac_per_user",
            elasticity_results["cac_grid"],
            "screening_participation_pct",
            elasticity_results["screen_grid"],
            elasticity_results["npv_cac_screen"],
            metric="NPV",
            filename="elasticity_npv_cac_screen.pdf"
        )
        
        plot_elasticity_heatmap(
            "cac_per_user",
            elasticity_results["cac_grid"],
            "screening_participation_pct",
            elasticity_results["screen_grid"],
            elasticity_results["be_cac_screen"],
            metric="Breakeven",
            filename="elasticity_be_cac_screen.pdf"
        )
    
    print("All plots generated successfully!")
