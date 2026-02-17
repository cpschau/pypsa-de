#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 PTES Impact authors
# SPDX-License-Identifier: MIT

"""
PTES savings overview plots.

Generates a figure with 3 rows (Low/Mid/High supply temperature) × 2 columns:
- Left: total German system costs breakdown (bn€) for the NoPTES baseline
- Right: net cost difference (bn€) vs NoPTES for boosting configs

Technologies that change minimally are aggregated as 'other technologies'.
Neighbor countries are shown as a single category on the left.
Star markers show total system savings on the right.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa

import sys

# Add repo root and 'code/pypsa-de' to sys.path so 'scripts._helpers' resolves when run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
CODE_PYPSA_DE = REPO_ROOT / "code" / "pypsa-de"
if str(CODE_PYPSA_DE) not in sys.path:
    sys.path.insert(0, str(CODE_PYPSA_DE))
from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


# ----------------------------- scenario parsing -----------------------------

SUPPLY_TEMPS = ["LowSupplyTemperature", "MidSupplyTemperature", "HighSupplyTemperature"]
COLUMN_CATEGORIES = ["NoPTES", "rhboost", "hpboost_10Cbottom", "noboost"]  # "hpboost",


# ----------------------------- normalization helpers -----------------------------


def _normalize_carrier_name(name: str) -> str:
    """Map raw carrier names to grouped labels.

    - Group 'urban central geothermal heat' and 'urban central geothermal heat pump'
      into 'urban central geothermal heat pump'.
    """
    try:
        n = name.strip()
    except Exception:
        return name
    if n in ("urban central geothermal heat", "urban central geothermal heat pump"):
        return "urban central geothermal heat pump"
    return name


def _normalize_carrier_series(s: pd.Series) -> pd.Series:
    if s is None or len(s) == 0:
        return s
    return s.groupby(s.index.map(_normalize_carrier_name)).sum()


def discover_scenarios(networks: Dict[str, Dict[int, pypsa.Network]]):
    """Map scenarios into (supply_temp, boosting_cfg)."""
    mapping: Dict[Tuple[str, str], str] = {}
    for scen in networks.keys():
        st = next((t for t in SUPPLY_TEMPS if t in scen), None)
        if st is None:
            if "Low" in scen and "Supply" in scen:
                st = "LowSupplyTemperature"
            elif "High" in scen and "Supply" in scen:
                st = "HighSupplyTemperature"
            else:
                st = "MidSupplyTemperature"
        cfg = None
        if "NoPTES" in scen:
            cfg = "NoPTES"
        elif "hpboost_10C" in scen:
            cfg = "hpboost_10Cbottom"
        elif "hpboost" in scen:
            cfg = "hpboost"
        elif "rhboost" in scen:
            cfg = "rhboost"
        elif "noboost" in scen:
            cfg = "noboost"
        if cfg is None:
            continue
        mapping[(st, cfg)] = scen
    return mapping


# ----------------------------- data extraction ------------------------------


def load_networks(run_name: str, scenarios: List[str], planning_horizons: List[int]):
    networks: Dict[str, Dict[int, pypsa.Network]] = {}
    networks_path = os.path.join("results", run_name)
    if not os.path.isdir(networks_path):
        logger.error(f"Run path not found: {networks_path}")
        return {}
    logger.info(f"Searching networks under: {networks_path}")
    for scen in scenarios or []:
        scen_root = os.path.join(networks_path, scen)
        scen_path = os.path.join(scen_root, "networks")
        paths_to_check = []
        if os.path.isdir(scen_path):
            paths_to_check.append(scen_path)
        if os.path.isdir(scen_root):
            paths_to_check.append(scen_root)
        if not paths_to_check:
            # Cross-run fallback: search in any results/* prefix
            try:
                import glob

                alt_scen_roots = glob.glob(os.path.join("results", "*", scen))
                alt_paths = []
                for root in alt_scen_roots:
                    netp = os.path.join(root, "networks")
                    if os.path.isdir(netp):
                        alt_paths.append(netp)
                    if os.path.isdir(root):
                        alt_paths.append(root)
                if alt_paths:
                    logger.warning(
                        f"Scenario path missing under {run_name}; using fallback from other run prefixes for {scen}."
                    )
                    paths_to_check.extend(alt_paths)
                else:
                    logger.warning(
                        f"Scenario path missing and no fallback found: {scen}"
                    )
                    continue
            except Exception:
                logger.warning(f"Scenario path missing: {scen_root}")
                continue
        for year in planning_horizons:
            candidate: str | None = None
            # Prefer files in networks/ then fallback to root
            for base in paths_to_check:
                try:
                    for fname in sorted(os.listdir(base)):
                        if not fname.endswith(".nc"):
                            continue
                        if fname.endswith(f"{year}.nc"):
                            full = os.path.join(base, fname)
                            candidate = full
                            # Prefer base_s_ files if multiple candidates
                            if os.path.basename(candidate).startswith("base_s_"):
                                break
                    if candidate:
                        break
                except FileNotFoundError:
                    continue
            if not candidate:
                # Fallback 1: any .nc for this scenario, pick the newest
                newest: tuple[float, str] | None = None
                for base in paths_to_check:
                    try:
                        for fname in os.listdir(base):
                            if not fname.endswith(".nc"):
                                continue
                            full = os.path.join(base, fname)
                            mtime = os.path.getmtime(full)
                            if newest is None or mtime > newest[0]:
                                newest = (mtime, full)
                    except FileNotFoundError:
                        continue
                if newest is not None:
                    candidate = newest[1]
                    logger.warning(
                        f"Using newest available network for {scen} (fallback): {candidate}"
                    )
                else:
                    # Fallback 2: search across all prefixes under results/*
                    try:
                        import glob

                        pattern = os.path.join(
                            "results", "*", scen, "networks", f"*{year}.nc"
                        )
                        matches = glob.glob(pattern)
                        if not matches:
                            pattern_any = os.path.join(
                                "results", "*", scen, "networks", "*.nc"
                            )
                            matches = glob.glob(pattern_any)
                        if matches:
                            matches.sort(key=lambda p: os.path.getmtime(p))
                            candidate = matches[-1]
                            logger.warning(
                                f"Found network in different run prefix for {scen}: {candidate}"
                            )
                    except Exception:
                        pass
            if not candidate:
                logger.warning(f"No network file for {scen} {year}")
                continue
            try:
                logger.info(f"Loading {scen} {year}: {candidate}")
                n = pypsa.Network(candidate)
                networks.setdefault(scen, {})[year] = n
            except Exception as e:
                logger.error(f"Failed loading {scen} {year} from {candidate}: {e}")
    if not networks:
        logger.error("No networks could be loaded from disk.")
    return networks


def compute_total_costs_by_tech(n: pypsa.Network) -> pd.Series:
    """Return total system costs by carrier (bn€). Robust across components."""
    try:
        capex = n.statistics.capex(
            nice_names=False, groupby=["carrier"]
        )  # works across components
        opex = n.statistics.opex(nice_names=False, groupby=["carrier"])
        series = capex + opex
        # If multi-index, reduce to carrier
        if isinstance(series.index, pd.MultiIndex):
            if "carrier" in series.index.names:
                series = series.groupby(level="carrier").sum()
            else:
                series = series.droplevel(list(range(series.index.nlevels - 1)))
    except Exception:
        # Fallback to helper aggregate if statistics fails
        from scripts._helpers import aggregate_costs as _agg_costs

        costs_df = _agg_costs(n, flatten=True)
        series = costs_df.sum(axis=1)
        series.index = (
            series.index.droplevel(1)
            if isinstance(series.index, pd.MultiIndex)
            else series.index
        )
    # Normalize carrier naming/grouping (e.g., geothermal)
    series = _normalize_carrier_series(series)
    return (series / 1e9).sort_values(key=lambda s: s.abs(), ascending=False)


def _costs_by_bus_carrier(
    n: pypsa.Network, components: List[str], bus_key: str
) -> pd.Series:
    """Helper: costs grouped by (bus_key, carrier) for selected components.

    To be compatible with multiple PyPSA versions, we avoid passing the
    'components=' kwarg (not available in older versions). Instead, we group
    by (bus_key, carrier, component) and then filter the MultiIndex by the
    provided components list, finally summing over the component level.
    """
    try:
        # Try grouping with component level included
        capex = n.statistics.capex(nice_names=False, groupby=[bus_key, "carrier", "component"])  # type: ignore[arg-type]
        opex = n.statistics.opex(nice_names=False, groupby=[bus_key, "carrier", "component"])  # type: ignore[arg-type]
        s = capex.add(opex, fill_value=0)
        if isinstance(s.index, pd.MultiIndex):
            # Normalize component names to match provided list
            comp_level = "component" if "component" in s.index.names else None
            if comp_level is not None:
                comp_names = s.index.get_level_values(comp_level).astype(str)
                # Case-insensitive filter
                want = {c.lower() for c in components}
                mask = comp_names.str.lower().isin(want)
                s = s[mask]
                # Sum over component to get (bus, carrier)
                s = s.groupby(level=[bus_key, "carrier"]).sum()
            else:
                # If no component level, just ensure we only keep desired levels
                names = list(s.index.names)
                levels_to_keep = [names.index(bus_key), names.index("carrier")]
                s = s.groupby(level=levels_to_keep).sum()
        # Normalize carrier naming
        if isinstance(s.index, pd.MultiIndex) and "carrier" in s.index.names:
            df = s.reset_index()
            df["carrier"] = df["carrier"].map(_normalize_carrier_name)
            s = df.groupby([bus_key, "carrier"], as_index=True)[0].sum()
        return s
    except Exception as e:
        logger.warning(f"Cost grouping by {bus_key} for {components} failed: {e}")
        return pd.Series(dtype=float)


def compute_totals_split(n: pypsa.Network) -> Tuple[float, float]:
    """Return (total_system_costs_bn€, germany_system_costs_bn€) using bus-based attribution where possible."""
    # Total cost by carrier (all components)
    total_series = compute_total_costs_by_tech(n)
    total = float(total_series.sum())

    # German-attributed costs from components with a unique bus key
    s_bus = _costs_by_bus_carrier(
        n, ["Generator", "StorageUnit", "Store", "Load"], "bus"
    )
    s_bus0 = _costs_by_bus_carrier(n, ["Link", "Transformer", "Line"], "bus0")
    german_sub = 0.0
    for s in (s_bus, s_bus0):
        if s is not None and not s.empty:
            idx0 = s.index.get_level_values(0).astype(str)
            german_sub += float(s[idx0.str.startswith("DE")].sum())
    return float(total), float(german_sub / 1e9)


def compute_neighbour_costs_by_tech(n: pypsa.Network) -> pd.Series:
    """Neighbour-only costs by carrier (bn€), attributing by bus where available."""
    parts = []
    for comps, bus_key in [
        (["Generator", "StorageUnit", "Store", "Load"], "bus"),
        (["Link", "Transformer", "Line"], "bus0"),
    ]:
        s = _costs_by_bus_carrier(n, comps, bus_key)
        if s is None or s.empty:
            continue
        idx0 = s.index.get_level_values(0).astype(str)
        s_neigh = s[~idx0.str.startswith("DE")]
        if isinstance(s_neigh.index, pd.MultiIndex):
            s_neigh = s_neigh.groupby(level="carrier").sum()
        parts.append(s_neigh)
    if parts:
        series = pd.concat(parts, axis=1).sum(axis=1)
    else:
        # Fallback: compute totals then set neighbours to zero if cannot split
        series = compute_total_costs_by_tech(n) * 0.0
    series = _normalize_carrier_series(series)
    return (series / 1e9).sort_values(key=lambda s: s.abs(), ascending=False)


def compute_dh_price_avg(n: pypsa.Network) -> float:
    """Demand-weighted average marginal price of DE urban central heat [€/MWh]."""
    # Select DE urban central heat buses
    uch_buses = n.buses[
        (n.buses.carrier == "urban central heat") & (n.buses.index.str.startswith("DE"))
    ].index
    if len(uch_buses) == 0:
        return float("nan")
    prices = n.buses_t.marginal_price.reindex(columns=uch_buses, fill_value=0.0)
    # loads per bus
    if n.loads.empty:
        return float("nan")
    # Avoid deprecated groupby with axis=1; transpose, group, and transpose back
    loads_by_bus = n.loads_t.p.T.groupby(n.loads.bus).sum().T
    loads_by_bus = loads_by_bus.reindex(columns=uch_buses, fill_value=0.0).clip(lower=0)
    if (loads_by_bus.sum(axis=1) == 0).all():
        return float("nan")
    # time weighting
    sw = n.snapshot_weightings.generators
    weighted_time = (prices.mul(loads_by_bus).sum(axis=1) * sw).sum()
    demand_time = (loads_by_bus.sum(axis=1) * sw).sum()
    if demand_time == 0:
        return float("nan")
    return float(weighted_time / demand_time)


# ------------------------------ plotting core -------------------------------


def build_figures(
    networks: Dict[str, Dict[int, pypsa.Network]],
    colors: dict,
    year: int,
    out_ptes: str,
    out_neigh: str,
    *,
    agg_threshold: float = 0.01,
):
    scen_map = discover_scenarios(networks)
    # Precompute DH prices and gather y-limits
    dh_prices = {
        scen: compute_dh_price_avg(networks[scen][year])
        for scen in networks
        if year in networks[scen]
    }

    # Compute shared y2 (DH price) limits across all panels
    xcats = COLUMN_CATEGORIES
    y2_vals = []
    for st in SUPPLY_TEMPS:
        for cat in xcats:
            scen = scen_map.get((st, cat))
            if scen in dh_prices:
                val = dh_prices.get(scen)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    y2_vals.append(float(val))
    if y2_vals:
        y2_min, y2_max = float(min(y2_vals)), float(max(y2_vals))
        if np.isclose(y2_min, y2_max):
            y2_max = y2_min + 1e-6
        y2_pad = 0.05 * (y2_max - y2_min)
        common_y2_lim = (y2_min - y2_pad, y2_max + y2_pad)
    else:
        common_y2_lim = (0.0, 1.0)

    # Determine shared y-limits for columns
    left_abs_vals = []
    right_diff_vals = []
    left_neigh_vals = []
    right_neigh_diff_vals = []

    # Collect values for limits
    for st in SUPPLY_TEMPS:
        base = scen_map.get((st, "NoPTES"))
        if base and year in networks[base]:
            abs_series = compute_total_costs_by_tech(networks[base][year])
            left_abs_vals.extend([abs_series.sum(), 0])
            neigh_series_abs = compute_neighbour_costs_by_tech(networks[base][year])
            left_neigh_vals.extend([neigh_series_abs.sum(), 0])
            # diffs vs base
            base_total, base_german = compute_totals_split(networks[base][year])
            for cfg in COLUMN_CATEGORIES[1:]:
                scen = scen_map.get((st, cfg))
                if scen and year in networks[scen]:
                    # system
                    comp_series = compute_total_costs_by_tech(networks[scen][year])
                    diff_series = comp_series.reindex(
                        abs_series.index, fill_value=0
                    ) - abs_series.reindex(abs_series.index, fill_value=0)
                    right_diff_vals.extend([diff_series.sum(), 0])
                    # neighbour
                    comp_neigh = compute_neighbour_costs_by_tech(networks[scen][year])
                    diff_neigh = comp_neigh.reindex(
                        neigh_series_abs.index, fill_value=0
                    ) - neigh_series_abs.reindex(neigh_series_abs.index, fill_value=0)
                    right_neigh_diff_vals.extend([diff_neigh.sum(), 0])

    def compute_ylim(vals):
        if not vals:
            return (-1, 1)
        vmin = min(vals)
        vmax = max(vals)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-3
        pad = 0.15 * (vmax - vmin)
        return (vmin - pad, vmax + pad)

    ylim_left = compute_ylim(left_abs_vals)
    ylim_right = compute_ylim(right_diff_vals)
    ylim_left_neigh = compute_ylim(left_neigh_vals)
    ylim_right_neigh = compute_ylim(right_neigh_diff_vals)

    # Prepare figures with narrower left column (single bar) and slightly smaller size
    fig1, axes1 = plt.subplots(
        len(SUPPLY_TEMPS),
        2,
        figsize=(10, 8),
        sharex=False,
        gridspec_kw={"width_ratios": [1, 3]},
    )
    fig2, axes2 = plt.subplots(
        len(SUPPLY_TEMPS),
        2,
        figsize=(10, 8),
        sharex=False,
        gridspec_kw={"width_ratios": [1, 3]},
    )

    # Common x-categories and ticks
    xcats_right = xcats[1:]  # exclude NoPTES on the right panels
    xticks_right = np.arange(len(xcats_right))
    # Track which labels actually appear to keep legend compact
    used_labels: set[str] = set()

    # Determine which rows actually have data (base NoPTES available for the year)
    rows_with_data = [
        st
        for st in SUPPLY_TEMPS
        if scen_map.get((st, "NoPTES"))
        and year in networks.get(scen_map.get((st, "NoPTES")), {})
    ]
    # Fall back to all SUPPLY_TEMPS to preserve 3-row layout, but annotate missing
    for r, st in enumerate(SUPPLY_TEMPS):
        # System costs row (fig1)
        axL = axes1[r, 0]
        axR = axes1[r, 1]
        # Neighbour costs row (fig2)
        axNL = axes2[r, 0]
        axNR = axes2[r, 1]

        # Baseline
        base = scen_map.get((st, "NoPTES"))
        # If there's no data for this supply temperature, annotate and skip plotting
        if not base or year not in networks.get(base, {}):
            for ax in (axL, axR, axNL, axNR):
                ax.text(
                    0.5,
                    0.5,
                    "No data available",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="0.4",
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_frame_on(False)
                ax2 = ax.twinx()
                ax2.set_yticks([])
                ax2.set_frame_on(False)
            logger.warning(
                f"Missing base scenario for {st} {year}; annotated as missing."
            )
            continue
        # System absolute
        if base and year in networks[base]:
            base_series = compute_total_costs_by_tech(networks[base][year])
        else:
            base_series = pd.Series(dtype=float)
        # Neighbour absolute
        if base and year in networks[base]:
            base_neigh = compute_neighbour_costs_by_tech(networks[base][year])
        else:
            base_neigh = pd.Series(dtype=float)

        # Compute significant German technologies by looking at net differences across boosting configs
        sig_techs: set[str] = set()
        if (
            base
            and year in networks.get(base, {})
            and base_series is not None
            and not base_series.empty
        ):
            diff_df_for_sig = pd.DataFrame(index=base_series.index)
            for cat in xcats_right:
                scen = scen_map.get((st, cat))
                if not scen or year not in networks.get(scen, {}):
                    continue
                comp_series = compute_total_costs_by_tech(networks[scen][year])
                aligned = comp_series.reindex(base_series.index, fill_value=0)
                diff_df_for_sig[cat] = aligned - base_series.reindex(
                    base_series.index, fill_value=0
                )
            if not diff_df_for_sig.empty:
                abs_contrib = diff_df_for_sig.abs().sum(axis=1)
                total_abs = float(abs_contrib.sum())
                if total_abs > 0:
                    sig_techs = set(
                        abs_contrib[(abs_contrib / total_abs) >= agg_threshold].index
                    )

        # Left panels: plot bars only at x=NoPTES index
        def _apply_series_threshold(series: pd.Series, threshold: float) -> pd.Series:
            if series is None or series.empty:
                return series
            total_abs = float(series.abs().sum())
            if total_abs <= 0:
                return series
            keep = (series.abs() / total_abs) >= threshold
            other_val = float(series[~keep].sum())
            s2 = series[keep].copy()
            if not np.isclose(other_val, 0.0):
                s2.loc["Other"] = s2.get("Other", 0.0) + other_val
            return s2

        def plot_abs(
            ax,
            series: pd.Series,
            ylim,
            *,
            include_neighbour_catchall: bool = False,
            neighbour_series: pd.Series | None = None,
            german_sig_only: bool = False,
        ):
            # Optionally add a single 'Neighbour countries' segment
            s_to_plot = series.copy() if series is not None else pd.Series(dtype=float)
            if include_neighbour_catchall:
                if neighbour_series is not None:
                    # Derive German series by subtracting neighbour-only components
                    german_series = s_to_plot.reindex(
                        s_to_plot.index, fill_value=0
                    ) - neighbour_series.reindex(s_to_plot.index, fill_value=0)
                    if german_sig_only and sig_techs:
                        german_shown = german_series.reindex(sorted(sig_techs)).dropna()
                        other_german = float(
                            german_series.drop(
                                index=list(sig_techs), errors="ignore"
                            ).sum()
                        )
                        s_to_plot = german_shown
                        if not np.isclose(other_german, 0.0):
                            s_to_plot.loc["other German costs"] = other_german
                    else:
                        s_to_plot = german_series
                    s_to_plot.loc["Neighbour countries"] = float(neighbour_series.sum())
                else:
                    s_to_plot.loc["Neighbour countries"] = 0.0
            # Threshold aggregation (preserve neighbour catch-all if present)
            if not s_to_plot.empty:
                if "Neighbour countries" in s_to_plot.index:
                    neigh_val = s_to_plot.loc["Neighbour countries"]
                    s_core = (
                        s_to_plot.drop(index=["Neighbour countries"])
                        if len(s_to_plot) > 1
                        else pd.Series(dtype=float)
                    )
                    # Do not collapse 'other German costs' via threshold; threshold the rest
                    if "other German costs" in s_core.index:
                        ogc_val = s_core.loc["other German costs"]
                        s_wo = (
                            s_core.drop(index=["other German costs"])
                            if len(s_core) > 1
                            else pd.Series(dtype=float)
                        )
                        s_wo = _apply_series_threshold(s_wo, agg_threshold)
                        s_core = s_wo
                        s_core.loc["other German costs"] = ogc_val
                    else:
                        s_core = _apply_series_threshold(s_core, agg_threshold)
                    s_to_plot = s_core
                    s_to_plot.loc["Neighbour countries"] = neigh_val
                else:
                    s_to_plot = _apply_series_threshold(s_to_plot, agg_threshold)
            # Sort by absolute value
            s_to_plot = s_to_plot.reindex(
                s_to_plot.abs().sort_values(ascending=False).index
            )
            # Single stacked bar
            if s_to_plot is not None and not s_to_plot.empty:
                bottom = 0.0
                for tech, val in s_to_plot.items():
                    ax.bar(
                        0,
                        val,
                        bottom=bottom,
                        color=colors.get(tech, "black"),
                        edgecolor="none",
                        linewidth=0.0,
                        width=0.8,
                    )
                    used_labels.add(tech)
                    bottom += val
            ax.set_ylim(ylim)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xlim(-0.8, 0.8)
            ax.set_xticks([0])
            ax.set_xticklabels(
                ["NoPTES"]
                if r
                == (
                    len(rows_with_data) - 1 if rows_with_data else len(SUPPLY_TEMPS) - 1
                )
                else []
            )
            # DH price secondary axis
            ax2 = ax.twinx()
            base_scen = scen_map.get((st, "NoPTES"))
            y_val = dh_prices.get(base_scen, np.nan)
            ax2.plot(
                [0], [y_val], color="tab:purple", marker="^", linewidth=0, markersize=6
            )
            ax2.set_ylabel(
                "Avg DH price [€/MWh]" if ax is axR else "", fontsize=11
            )  # only label on right column
            # Apply shared y2 limits across all panels
            ax2.set_ylim(common_y2_lim)
            return ax

        # Left total system costs: only significant German techs + single neighbour segment
        plot_abs(
            axL,
            base_series,
            ylim_left,
            include_neighbour_catchall=True,
            neighbour_series=base_neigh,
            german_sig_only=True,
        )
        plot_abs(axNL, base_neigh, ylim_left_neigh)

        # Right panels: differences vs base for non-NoPTES categories
        def plot_diff(
            ax,
            series_base: pd.Series,
            ylim,
            use_german_marker=True,
            neighbour_only=False,
        ):
            # Build diff DataFrame with rows=tech, columns=xcats; NoPTES zeros
            diff_df = pd.DataFrame(index=series_base.index)
            total_markers = []
            german_markers = []
            for cat in xcats_right:
                scen = scen_map.get((st, cat))
                if not scen or year not in networks[scen]:
                    diff_df[cat] = 0.0
                    total_markers.append(0.0)
                    german_markers.append(0.0)
                    continue
                if neighbour_only:
                    comp_series = compute_neighbour_costs_by_tech(networks[scen][year])
                    base_series_loc = base_neigh
                else:
                    comp_series = compute_total_costs_by_tech(networks[scen][year])
                    base_series_loc = series_base
                aligned = comp_series.reindex(series_base.index, fill_value=0)
                diff = aligned - base_series_loc.reindex(
                    series_base.index, fill_value=0
                )
                diff_df[cat] = diff
                # markers
                if neighbour_only:
                    # neighbour total change only
                    total_markers.append(float(diff.sum()))
                    german_markers.append(0.0)
                else:
                    total_markers.append(float(diff.sum()))
                    # German system savings excludes neighbour countries
                    total_comp, german_comp = compute_totals_split(networks[scen][year])
                    base_total, base_german = (
                        compute_totals_split(networks[base][year])
                        if base and year in networks[base]
                        else (0.0, 0.0)
                    )
                    german_markers.append(float((german_comp - base_german)))
            # Aggregate, sort by contribution, and plot
            if not diff_df.empty:
                diff_df = diff_df.reindex(columns=xcats_right, fill_value=0)
                abs_contrib = diff_df.abs().sum(axis=1)
                total_abs = float(abs_contrib.sum())
                if total_abs > 0:
                    keep_mask = (abs_contrib / total_abs) >= agg_threshold
                    others = diff_df[~keep_mask]
                    diff_df = diff_df[keep_mask]
                    if not others.empty:
                        diff_df.loc["Other"] = others.sum(axis=0)
                order = diff_df.abs().sum(axis=1).sort_values(ascending=False).index
                diff_df = diff_df.reindex(order)
                diff_df.T.plot.bar(
                    stacked=True,
                    ax=ax,
                    color=[colors.get(t, "black") for t in diff_df.index],
                    legend=False,
                    width=0.8,
                    edgecolor="none",
                )
                for p in ax.patches:
                    p.set_linewidth(0.0)
                    p.set_edgecolor("none")
                used_labels.update(diff_df.index.tolist())
            ax.set_ylim(ylim)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_xticks(xticks_right)
            ax.set_xticklabels(
                xcats_right
                if r
                == (
                    len(rows_with_data) - 1 if rows_with_data else len(SUPPLY_TEMPS) - 1
                )
                else []
            )
            # markers
            for i, (tot, ger) in enumerate(zip(total_markers, german_markers)):
                # choose annotation baseline based on sign of negative contributions
                y0 = 0
                # total marker (star)
                ax.scatter(
                    i,
                    tot,
                    s=80,
                    marker="*",
                    facecolor="whitesmoke",
                    edgecolor="black",
                    linewidth=1,
                    zorder=5,
                )
                if use_german_marker and not neighbour_only:
                    # german-only marker (circle)
                    if not np.isclose(ger, tot):
                        ax.scatter(
                            i,
                            ger,
                            s=80,
                            marker="o",
                            facecolor="white",
                            edgecolor="black",
                            linewidth=1,
                            zorder=4,
                        )
            # DH price
            ax2 = ax.twinx()
            y_price = []
            for cat in xcats_right:
                scen = scen_map.get((st, cat))
                y_price.append(dh_prices.get(scen, np.nan))
            ax2.plot(
                xticks_right,
                y_price,
                color="tab:purple",
                marker="^",
                linewidth=2.2,
                markersize=5,
            )
            ax2.set_ylabel("Avg DH price [€/MWh]", fontsize=11)
            # Apply shared y2 limits
            ax2.set_ylim(common_y2_lim)
            return ax

        plot_diff(
            axR, base_series, ylim_right, use_german_marker=True, neighbour_only=False
        )
        plot_diff(
            axNR,
            base_neigh,
            ylim_right_neigh,
            use_german_marker=False,
            neighbour_only=True,
        )

        # Row labels
        row_names = {
            "LowSupplyTemperature": "Low Supply Temperature",
            "MidSupplyTemperature": "Mid Supply Temperature",
            "HighSupplyTemperature": "High Supply Temperature",
        }
        axes1[r, 0].set_ylabel(f"{row_names.get(st, st)}\nCosts [bn€]", fontsize=12)
        axes2[r, 0].set_ylabel(f"{row_names.get(st, st)}\nCosts [bn€]", fontsize=12)

    # Titles and legends
    axes1[0, 0].set_title("Total system costs (NoPTES)", fontsize=13)
    axes1[0, 1].set_title("Net cost difference vs NoPTES", fontsize=13)
    axes2[0, 0].set_title("Neighbour countries costs (NoPTES)", fontsize=13)
    axes2[0, 1].set_title("Neighbour countries net difference vs NoPTES", fontsize=13)

    # X label shared
    fig1.supxlabel("Boosting configuration", fontsize=13)
    fig2.supxlabel("Boosting configuration", fontsize=13)

    # Build legend from actually used labels only
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=colors.get(c, "black"), edgecolor="black", label=c)
        for c in sorted(used_labels)
    ]
    # Add markers reference
    from matplotlib.lines import Line2D

    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                label="Total system savings",
                markerfacecolor="whitesmoke",
                linestyle="",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                label="German system savings",
                markerfacecolor="white",
                linestyle="",
            ),
        ]
    )
    legend_ncol = min(6, max(1, len(legend_handles)))
    fig1.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=legend_ncol,
        frameon=False,
    )
    fig2.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=legend_ncol,
        frameon=False,
    )

    # Tight layout leaving room for legend
    fig1.tight_layout(rect=(0, 0.14, 1, 0.96))
    fig2.tight_layout(rect=(0, 0.14, 1, 0.96))

    os.makedirs(os.path.dirname(out_ptes), exist_ok=True)
    os.makedirs(os.path.dirname(out_neigh), exist_ok=True)
    fig1.savefig(out_ptes, bbox_inches="tight", dpi=300)
    fig2.savefig(out_neigh, bbox_inches="tight", dpi=300)
    logger.info(f"Saved figures:\n - {out_ptes}\n - {out_neigh}")
    plt.close(fig1)
    plt.close(fig2)


def _sanitize_color(c: object) -> str:
    if not isinstance(c, str):
        return "black"
    v = c.strip()
    if not v:
        return "black"
    # Convert 8-digit hex (#RRGGBBAA) to #RRGGBB
    if v.startswith("#") and len(v) == 9:
        v = v[:7]
    return v if mcolors.is_color_like(v) else "black"


def get_colors(networks: Dict[str, Dict[int, pypsa.Network]], override_colors=None):
    if override_colors is None:
        override_colors = {}
    # fetch first available network
    first_net = None
    for scen in networks.values():
        if scen:
            first_net = scen[next(iter(scen))]
            break
    if first_net is None:
        return {}
    colors = first_net.carriers.color
    colors = (
        colors.reindex(colors.index.union(list(override_colors.keys())))
        .fillna("black")
        .to_dict()
    )
    colors.update(override_colors)
    # Sanitize all color values
    colors = {k: _sanitize_color(v) for k, v in colors.items()}
    # Ensure defaults for aggregate categories
    colors.setdefault("Neighbour countries", "#8c8c8c")
    colors.setdefault("Other", "#bdbdbd")
    colors.setdefault("other German costs", "#9e9e9e")
    return colors


def main(snakemake):
    configure_logging(snakemake)
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons
    # Align with other plotting scripts: use the first planning horizon
    year = (
        planning_horizons[0]
        if isinstance(planning_horizons, list)
        else planning_horizons
    )
    override_colors = {}
    try:
        override_colors = snakemake.params.plotting["override_tech_colors"]
    except Exception:
        pass
    agg_threshold = 0.01
    try:
        thr = snakemake.params.plotting.get("aggregate_threshold_pct", 0.01)
        agg_threshold = float(thr)
        if agg_threshold > 1.0:
            agg_threshold = agg_threshold / 100.0
        if agg_threshold < 0:
            agg_threshold = 0.0
    except Exception:
        pass

    networks = load_networks(run_name, scenarios, planning_horizons)
    if not networks:
        logger.error("No networks loaded; aborting plot")
        return
    colors = get_colors(networks, override_colors)
    build_figures(
        networks,
        colors,
        year,
        snakemake.output.ptes_fig,
        snakemake.output.neigh_fig,
        agg_threshold=agg_threshold,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        # Mirror the pattern used in other plotting scripts (e.g., plot_dh_systems):
        # switch to repo root and construct a snakemake object from the Snakefile rule.
        os.chdir(Path(__file__).resolve().parents[2])
        snakemake = mock_snakemake(
            "plot_ptes_savings_and_neighbour_costs",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )
    main(snakemake)
