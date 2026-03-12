"""
CE/BIA Bridge Module
====================

Translate Cost-Effectiveness (CE) and Budget Impact Analysis (BIA) outputs
into model-ready assumption patches for the financial simulation engine.

Design goals:
- Deterministic mapping with explicit provenance (trace)
- No hidden constants; only direct CE/BIA values or algebraic transforms
- Guardrails that keep values in original model bounds where available
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from financial_model import ASSUMPTIONS, YEARS
from financial_engineering import run_advanced_simulation


def _first_present(data: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(min(max(value, lo), hi))


def _to_float_list(values: Any) -> Optional[List[float]]:
    if values is None:
        return None
    if not isinstance(values, (list, tuple, np.ndarray)):
        return None
    return [float(v) for v in values]


def _scalar_tri_from_payload(payload: Dict[str, Any], prefix: str) -> Optional[Tuple[float, float, float, str]]:
    lo = _first_present(payload, [f"{prefix}_min", f"{prefix}_low", f"{prefix}_p10"])
    ml = _first_present(payload, [f"{prefix}_most_likely", f"{prefix}_base", f"{prefix}_median", f"{prefix}"])
    hi = _first_present(payload, [f"{prefix}_max", f"{prefix}_high", f"{prefix}_p90"])

    if lo is None and ml is None and hi is None:
        return None

    # If only a point estimate is provided, create a degenerate triangular value.
    if ml is not None and lo is None and hi is None:
        lo = ml
        hi = ml
    elif ml is None and lo is not None and hi is not None:
        ml = (float(lo) + float(hi)) / 2.0
    elif lo is None and ml is not None and hi is not None:
        lo = ml
    elif hi is None and lo is not None and ml is not None:
        hi = ml

    return float(lo), float(ml), float(hi), "triangular"


def _ordered_tri(lo: float, ml: float, hi: float) -> Tuple[float, float, float]:
    ordered = sorted([float(lo), float(ml), float(hi)])
    return ordered[0], ordered[1], ordered[2]


def _apply_scalar_bounds(
    name: str,
    tri: Tuple[float, float, float, str],
    base_assumptions: Dict[str, Dict[str, Any]],
    guards: List[Dict[str, Any]],
) -> Tuple[float, float, float, str]:
    lo, ml, hi, dist = tri
    lo, ml, hi = _ordered_tri(lo, ml, hi)

    base = base_assumptions.get(name, {})
    if "min" in base and "max" in base and not isinstance(base.get("min"), list):
        base_lo = float(base["min"])
        base_hi = float(base["max"])
        clamped = (_clamp(lo, base_lo, base_hi), _clamp(ml, base_lo, base_hi), _clamp(hi, base_lo, base_hi))
        if clamped != (lo, ml, hi):
            guards.append(
                {
                    "assumption": name,
                    "rule": "clamped_to_base_bounds",
                    "before": {"min": lo, "most_likely": ml, "max": hi},
                    "after": {"min": clamped[0], "most_likely": clamped[1], "max": clamped[2]},
                }
            )
        lo, ml, hi = _ordered_tri(*clamped)

    return lo, ml, hi, dist


def _apply_list_bounds(
    name: str,
    values: List[float],
    base_assumptions: Dict[str, Dict[str, Any]],
    guards: List[Dict[str, Any]],
) -> List[float]:
    base = base_assumptions.get(name, {})
    base_min = base.get("min")
    base_max = base.get("max")

    bounded = list(values)
    if isinstance(base_min, list) and isinstance(base_max, list) and len(base_min) == len(values):
        before = list(bounded)
        bounded = [_clamp(v, float(base_min[i]), float(base_max[i])) for i, v in enumerate(values)]
        if before != bounded:
            guards.append(
                {
                    "assumption": name,
                    "rule": "clamped_to_base_bounds_per_year",
                    "before": before,
                    "after": bounded,
                }
            )

    return bounded


def derive_assumption_patch(
    ce_result: Dict[str, Any],
    bia_result: Dict[str, Any],
    base_assumptions: Dict[str, Dict[str, Any]] = ASSUMPTIONS,
    policy_mode: str = "hybrid",
) -> Dict[str, Any]:
    """
    Derive a patch that can be merged into ASSUMPTIONS.

    Returns
    -------
    dict with keys:
      - assumption_patch : dict
      - trace            : list of mapping records
      - guards           : list of clamp/order corrections
      - policy_mode      : echo of selected mode
    """
    patch: Dict[str, Dict[str, Any]] = {}
    trace: List[Dict[str, Any]] = []
    guards: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1) CE -> Price and churn dynamics
    # ------------------------------------------------------------------
    price_tri = _scalar_tri_from_payload(ce_result, "price_per_month")
    if price_tri is not None:
        lo, ml, hi, dist = _apply_scalar_bounds("price_per_month", price_tri, base_assumptions, guards)
        patch["price_per_month"] = {"min": lo, "most_likely": ml, "max": hi, "distribution": dist}
        trace.append(
            {
                "target": "price_per_month",
                "source": "ce_result.price_per_month_*",
                "transform": "direct_triangular",
            }
        )

    churn_tri = _scalar_tri_from_payload(ce_result, "annual_churn_rate")
    if churn_tri is not None:
        lo, ml, hi, dist = _apply_scalar_bounds("annual_churn_rate", churn_tri, base_assumptions, guards)
        patch["annual_churn_rate"] = {"min": lo, "most_likely": ml, "max": hi, "distribution": dist}
        trace.append(
            {
                "target": "annual_churn_rate",
                "source": "ce_result.annual_churn_rate_*",
                "transform": "direct_triangular",
            }
        )
    else:
        retention_tri = _scalar_tri_from_payload(ce_result, "annual_retention_rate")
        if retention_tri is not None:
            r_lo, r_ml, r_hi, _ = retention_tri
            # churn = 1 - retention; reorder to ensure min<=ml<=max
            c1, c2, c3 = (1.0 - r_hi), (1.0 - r_ml), (1.0 - r_lo)
            c_lo, c_ml, c_hi, dist = _apply_scalar_bounds(
                "annual_churn_rate",
                _ordered_tri(c1, c2, c3) + ("triangular",),
                base_assumptions,
                guards,
            )
            patch["annual_churn_rate"] = {
                "min": c_lo,
                "most_likely": c_ml,
                "max": c_hi,
                "distribution": dist,
            }
            trace.append(
                {
                    "target": "annual_churn_rate",
                    "source": "ce_result.annual_retention_rate_*",
                    "transform": "churn=1-retention",
                }
            )

    # ------------------------------------------------------------------
    # 2) BIA -> Feasible uptake/adoption and operating costs
    # ------------------------------------------------------------------
    screen_tri = _scalar_tri_from_payload(bia_result, "screening_participation_pct")

    # Optional algebraic derivation if BIA provides budget envelope + unit cost.
    if screen_tri is None:
        annual_budget = _first_present(
            bia_result,
            ["annual_budget_cap_chf", "annual_affordability_cap_chf", "budget_cap_chf"],
        )
        cost_per_screened = _first_present(
            bia_result,
            ["cost_per_screened_user_chf", "avg_screening_program_cost_chf"],
        )
        adult_population = float(base_assumptions["adult_population"].get("value", 0))

        if annual_budget is not None and cost_per_screened is not None and adult_population > 0:
            feasible_pct = float(annual_budget) / (float(cost_per_screened) * adult_population)
            screen_tri = (feasible_pct, feasible_pct, feasible_pct, "triangular")
            trace.append(
                {
                    "target": "screening_participation_pct",
                    "source": "bia_result.annual_budget_cap_chf + bia_result.cost_per_screened_user_chf",
                    "transform": "budget/(unit_cost*adult_population)",
                }
            )

    if screen_tri is not None:
        lo, ml, hi, dist = _apply_scalar_bounds("screening_participation_pct", screen_tri, base_assumptions, guards)
        patch["screening_participation_pct"] = {
            "min": lo,
            "most_likely": ml,
            "max": hi,
            "distribution": dist,
        }
        if not any(t["target"] == "screening_participation_pct" for t in trace):
            trace.append(
                {
                    "target": "screening_participation_pct",
                    "source": "bia_result.screening_participation_pct_*",
                    "transform": "direct_triangular",
                }
            )

    adoption_values = _to_float_list(
        _first_present(bia_result, ["adoption_ramp", "adoption_ramp_feasible", "feasible_adoption_ramp"])
    )
    if adoption_values is not None:
        if len(adoption_values) != len(YEARS):
            guards.append(
                {
                    "assumption": "adoption_ramp",
                    "rule": "length_mismatch_ignored",
                    "expected_len": len(YEARS),
                    "actual_len": len(adoption_values),
                }
            )
        else:
            bounded = _apply_list_bounds("adoption_ramp", adoption_values, base_assumptions, guards)
            patch["adoption_ramp"] = {
                "min": list(bounded),
                "most_likely": list(bounded),
                "max": list(bounded),
                "distribution": "triangular",
            }
            trace.append(
                {
                    "target": "adoption_ramp",
                    "source": "bia_result.adoption_ramp(_feasible)",
                    "transform": "direct_per_year_fixed_as_triangular",
                }
            )

    for target in ["cac_per_user", "call_center_per_user", "backend_services_per_user"]:
        tri = _scalar_tri_from_payload(bia_result, target)
        if tri is None:
            continue
        lo, ml, hi, dist = _apply_scalar_bounds(target, tri, base_assumptions, guards)
        patch[target] = {"min": lo, "most_likely": ml, "max": hi, "distribution": dist}
        trace.append(
            {
                "target": target,
                "source": f"bia_result.{target}_*",
                "transform": "direct_triangular",
            }
        )

    # Optional policy-mode behavior without introducing new constants:
    # - strict_budget: force adoption_ramp to provided feasible path when present.
    # - value_based: prioritize CE-derived price/churn if both CE and BIA provide same field.
    # - hybrid: accept all non-conflicting mappings (default behavior above).
    # Since all mappings are field-specific and deterministic, this currently affects
    # only conflict resolution in future extension points.

    return {
        "assumption_patch": patch,
        "trace": trace,
        "guards": guards,
        "policy_mode": policy_mode,
    }


def apply_assumption_patch(
    base_assumptions: Dict[str, Dict[str, Any]],
    assumption_patch: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Deep-merge assumption patch into a copy of base assumptions."""
    merged = deepcopy(base_assumptions)
    for name, updates in assumption_patch.items():
        if name not in merged:
            merged[name] = dict(updates)
        else:
            merged[name].update(updates)
    return merged


def run_advanced_simulation_with_ce_bia(
    ce_result: Dict[str, Any],
    bia_result: Dict[str, Any],
    policy_mode: str = "hybrid",
    years: List[int] = YEARS,
    n_paths: int = 5_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    One-call helper for API usage:
      1) derive patch from CE/BIA outputs,
      2) merge patch into ASSUMPTIONS,
      3) run advanced simulation with patched assumptions.
    """
    bridge = derive_assumption_patch(
        ce_result=ce_result,
        bia_result=bia_result,
        base_assumptions=ASSUMPTIONS,
        policy_mode=policy_mode,
    )
    updated_assumptions = apply_assumption_patch(ASSUMPTIONS, bridge["assumption_patch"])
    simulation = run_advanced_simulation(
        assumptions=updated_assumptions,
        years=years,
        n_paths=n_paths,
        seed=seed,
    )

    return {
        "bridge": bridge,
        "simulation": simulation,
    }
