#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 PTES Impact authors
# SPDX-License-Identifier: MIT

"""
PTES system cost impact overview.

Generates a figure with 3 rows (Low/Mid/High supply temperature) × 2 columns:
- Left: total German system costs breakdown (bn€) for the NoPTES baseline
- Right: net cost difference (bn€) vs NoPTES for boosting configs

Technologies that change minimally are aggregated as 'other technologies'.
EU countries are shown as a single aggregated category on the left.
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
import matplotlib.colors as mcolors
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

# Supply temperatures and PTES configurations
SUPPLY_TEMPS = ["LowSupplyTemperature", "MidSupplyTemperature", "HighSupplyTemperature"]
PTES_CONFIGS = ["hpboost", "rhboost", "hpboost_10Cbottom", "noboost"]


def calc_average_dh_price(n):
    """Calculate average district heating price in EUR MWh$^{-1}$."""
    dh_loads = n.loads_t.p.filter(
        regex=r"DE.*(urban central heat|low-temperature heat for industry)"
    )
    if dh_loads.empty:
        return 0

    # Rename columns to standardize names
    dh_loads.columns = dh_loads.columns.str.replace(
        "low-temperature heat for industry", "urban central heat"
    )
    # Aggregate columns with same name
    dh_loads = dh_loads.T.groupby(level=0).sum().T

    prices = n.buses_t.marginal_price.filter(regex=r"DE0.*urban central heat")
    if prices.empty or dh_loads.sum().sum() == 0:
        return 0

    average_dh_price = dh_loads.mul(prices).sum().sum() / dh_loads.sum().sum()
    return average_dh_price


def get_component_mask(lines_or_links, country, other_countries, bus=0):
    """Create mask for components connecting a country to others."""
    if bus == 0:
        # For links, check both buses
        return lines_or_links.bus0.str.contains(
            country
        ) & lines_or_links.bus1.str.contains("|".join(other_countries))
    elif bus == 1:
        # For links, check the second bus
        return lines_or_links.bus1.str.contains(
            country
        ) & lines_or_links.bus0.str.contains("|".join(other_countries))
    else:
        return (
            lines_or_links.bus0.str.contains(country)
            & lines_or_links.bus1.str.contains("|".join(other_countries))
        ) | (
            lines_or_links.bus0.str.contains("|".join(other_countries))
            & lines_or_links.bus0.str.contains(country)
        )


def calc_ic_capex_correction(n, country, exclude_country=False):
    """Calculate correction value for interconnection CAPEX assuming equal shares of connected countries."""
    other_countries = n.buses.country.unique()
    other_countries = other_countries[
        (other_countries != "") & (other_countries != country)
    ]

    ic_links_bus0_mask = get_component_mask(n.links, country, other_countries, bus=0)
    ic_links_bus0 = n.links.loc[ic_links_bus0_mask]
    capex_ic_links_bus0 = ic_links_bus0.p_nom_opt.mul(ic_links_bus0.capital_cost)

    ic_links_bus1_mask = get_component_mask(n.links, country, other_countries, bus=1)
    ic_links_bus1 = n.links.loc[ic_links_bus1_mask]
    capex_ic_links_bus1 = ic_links_bus1.p_nom_opt.mul(ic_links_bus1.capital_cost)

    ic_links = pd.concat([capex_ic_links_bus0, capex_ic_links_bus1], axis=0)
    if exclude_country:
        ic_links["capex"] = 0.5 * (
            capex_ic_links_bus0.reindex(ic_links.index, fill_value=0)
            - capex_ic_links_bus1.reindex(ic_links.index, fill_value=0)
        )
    else:
        ic_links["capex"] = 0.5 * (
            capex_ic_links_bus1.reindex(ic_links.index, fill_value=0)
            - capex_ic_links_bus0.reindex(ic_links.index, fill_value=0)
        )

    ic_lines_bus0_mask = get_component_mask(n.lines, country, other_countries, bus=0)
    ic_lines_bus0 = n.lines.loc[ic_lines_bus0_mask]
    capex_ic_links_bus0 = ic_lines_bus0.s_nom_opt.mul(ic_lines_bus0.capital_cost)

    ic_lines_bus1_mask = get_component_mask(n.lines, country, other_countries, bus=1)
    ic_lines_bus1 = n.lines.loc[ic_lines_bus1_mask]
    capex_ic_links_bus1 = ic_lines_bus1.s_nom_opt.mul(ic_lines_bus1.capital_cost)

    ic_lines = pd.concat([capex_ic_links_bus0, capex_ic_links_bus1], axis=0)
    if exclude_country:
        ic_lines["capex"] = 0.5 * (
            capex_ic_links_bus0.reindex(ic_lines.index, fill_value=0)
            - capex_ic_links_bus1.reindex(ic_lines.index, fill_value=0)
        )
    else:
        ic_lines["capex"] = 0.5 * (
            capex_ic_links_bus1.reindex(ic_lines.index, fill_value=0)
            - capex_ic_links_bus0.reindex(ic_lines.index, fill_value=0)
        )

    return pd.Series(
        {
            "AC": ic_lines.capex.sum(),
            "DC": ic_links.capex.sum(),
        },
        name="capex",
    )


def calculate_german_fraction(n):
    """Calculate the fraction of each technology that is located in Germany."""
    # Count capacities/components by country
    german_fractions = {}

    # Check generators
    if not n.generators.empty:
        total_gen_capacity = n.generators.groupby("carrier").p_nom_opt.sum()
        german_gen_capacity = (
            n.generators[n.generators.bus.str.startswith("DE")]
            .groupby("carrier")
            .p_nom_opt.sum()
        )
        gen_fractions = german_gen_capacity / total_gen_capacity.reindex(
            german_gen_capacity.index, fill_value=1
        )
        german_fractions.update(gen_fractions.fillna(0).to_dict())

    # Check links
    if not n.links.empty:
        total_link_capacity = n.links.groupby("carrier").p_nom_opt.sum()
        german_link_capacity = (
            n.links[
                n.links.bus0.str.startswith("DE") | n.links.bus1.str.startswith("DE")
            ]
            .groupby("carrier")
            .p_nom_opt.sum()
        )
        link_fractions = german_link_capacity / total_link_capacity.reindex(
            german_link_capacity.index, fill_value=1
        )
        german_fractions.update(link_fractions.fillna(0).to_dict())

    # Check storage units
    if not n.storage_units.empty:
        total_storage_capacity = n.storage_units.groupby("carrier").p_nom_opt.sum()
        german_storage_capacity = (
            n.storage_units[n.storage_units.bus.str.startswith("DE")]
            .groupby("carrier")
            .p_nom_opt.sum()
        )
        storage_fractions = german_storage_capacity / total_storage_capacity.reindex(
            german_storage_capacity.index, fill_value=1
        )
        german_fractions.update(storage_fractions.fillna(0).to_dict())

    # Check stores
    if not n.stores.empty:
        total_store_capacity = n.stores.groupby("carrier").e_nom_opt.sum()
        german_store_capacity = (
            n.stores[n.stores.bus.str.startswith("DE")]
            .groupby("carrier")
            .e_nom_opt.sum()
        )
        store_fractions = german_store_capacity / total_store_capacity.reindex(
            german_store_capacity.index, fill_value=1
        )
        german_fractions.update(store_fractions.fillna(0).to_dict())

    # Default fraction for unknown carriers (use Germany's share in the model)
    default_fraction = 0.15  # Rough estimate based on Germany's economic weight

    return pd.Series(german_fractions).fillna(default_fraction)


def get_system_costs_by_carrier(n, country, exclude_country=False, aggregate_all=False):
    """Calculate total system costs by carrier for all countries modelled, a specified country, or all countries minus a specified country."""
    try:
        # Get costs for all components using PyPSA statistics API
        capex = n.statistics.capex(groupby=["carrier"], nice_names=False)
        opex = n.statistics.opex(groupby=["carrier"], nice_names=False)

        # Extract clean carrier names from MultiIndex if needed
        if isinstance(capex.index, pd.MultiIndex):
            capex = capex.groupby(capex.index.get_level_values(-1)).sum()
        if isinstance(opex.index, pd.MultiIndex):
            opex = opex.groupby(opex.index.get_level_values(-1)).sum()

        # Apply country filtering based on analysis of component locations
        if not aggregate_all:
            if country == "DE":
                # Calculate what fraction of each technology is in Germany
                german_fractions = calculate_german_fraction(n)

                # Align fractions with cost indices
                capex_fractions = german_fractions.reindex(capex.index, fill_value=0.15)
                opex_fractions = german_fractions.reindex(opex.index, fill_value=0.15)

                if exclude_country:
                    # Return costs for all countries except Germany
                    capex = capex * (1 - capex_fractions)
                    opex = opex * (1 - opex_fractions)
                else:
                    # Return costs for Germany only
                    capex = capex * capex_fractions
                    opex = opex * opex_fractions

    except Exception as e:
        logger.warning(f"Statistics API failed: {e}")
        capex = pd.Series(dtype=float)
        opex = pd.Series(dtype=float)

    union_index = capex.index.union(opex.index)
    capex = capex.reindex(union_index).fillna(0)
    opex = opex.reindex(union_index).fillna(0)

    # Create DataFrame
    total = pd.DataFrame({"capex": capex, "opex": opex})

    if not aggregate_all:
        try:
            ic_capex_correction = calc_ic_capex_correction(
                n, country, exclude_country=exclude_country
            )
            for tech in ic_capex_correction.index:
                if tech in total.index:
                    total.loc[tech, "capex"] += ic_capex_correction[tech]
                else:
                    total.loc[tech, "capex"] = ic_capex_correction[tech]
                    total.loc[tech, "opex"] = 0
        except Exception as e:
            logger.warning(f"IC correction failed: {e}")

    # Don't add EU countries here - will be added in main calculation function

    return total


def load_networks(
    run_name: str, scenarios: list, planning_horizons: list
) -> Dict[str, pypsa.Network]:
    """Step 1: Load networks one by one and write them into a dictionary."""
    networks = {}
    networks_path = Path("results") / run_name

    if not networks_path.exists():
        logger.error(f"Run path not found: {networks_path}")
        return {}

    logger.info(f"Loading networks from: {networks_path}")

    # Use first planning horizon
    year = (
        planning_horizons[0]
        if isinstance(planning_horizons, list)
        else planning_horizons
    )

    for scenario in scenarios:
        scenario_path = networks_path / scenario / "networks"
        if not scenario_path.exists():
            # Try fallback in other run prefixes
            fallback_paths = list(Path("results").glob(f"*/{scenario}/networks"))
            if fallback_paths:
                scenario_path = fallback_paths[0]
                logger.warning(f"Using fallback path for {scenario}: {scenario_path}")
            else:
                logger.warning(f"No network path found for scenario: {scenario}")
                continue

        # Look for network file for the specified year
        network_files = list(scenario_path.glob(f"*{year}.nc"))
        if not network_files:
            # Fallback to any .nc file
            network_files = list(scenario_path.glob("*.nc"))

        if network_files:
            # Prefer files starting with "base_s_"
            preferred = [f for f in network_files if f.name.startswith("base_s_")]
            network_file = preferred[0] if preferred else network_files[0]

            try:
                logger.info(f"Loading {scenario}: {network_file}")
                # Add timeout mechanism by loading with limited retries
                network = pypsa.Network(str(network_file))
                networks[scenario] = network
                logger.info(f"Successfully loaded {scenario}")
            except Exception as e:
                logger.error(f"Failed to load {scenario} from {network_file}: {e}")
                # Continue with other networks instead of failing completely
        else:
            logger.warning(f"No network file found for {scenario}")

    return networks


def calculate_german_costs(networks: Dict[str, pypsa.Network]) -> pd.DataFrame:
    """Step 2: Calculate German costs by carrier for each network."""
    cost_data = {}

    for scenario, network in networks.items():
        # Get German costs (excluding EU countries)
        german_costs = get_system_costs_by_carrier(
            network, country="DE", exclude_country=False, aggregate_all=False
        )

        # Calculate EU countries costs separately (don't use aggregate_all=True to avoid double-counting)
        try:
            # Get costs for all non-DE countries
            neighbor_costs_full = get_system_costs_by_carrier(
                network, country="DE", exclude_country=True, aggregate_all=False
            )
            neighbor_total = (
                neighbor_costs_full.sum(axis=1).sum() / 1e9
            )  # Convert to bn€
        except Exception as e:
            logger.warning(
                f"Failed to calculate neighbor costs for {scenario}: {e}. Setting to 0."
            )
            neighbor_total = 0

        # Sum capex and opex for German costs and convert to bn€
        total_costs = german_costs.sum(axis=1) / 1e9

        # Add EU countries as a separate category
        total_costs["EU aggregated"] = neighbor_total

        # Calculate totals for logging
        german_cost = total_costs.drop("EU aggregated").sum()
        total_system_cost = total_costs.sum()

        logger.info(
            f"{scenario}: German={german_cost:.1f} bn€, Neighbors={neighbor_total:.1f} bn€, Total={total_system_cost:.1f} bn€"
        )

        # Expected ranges: German 17-150 bn€, Overall system 773-777 bn€
        if german_cost > 200:
            logger.warning(
                f"German costs seem high ({german_cost:.1f} bn€) - expected 17-150 bn€"
            )
        if total_system_cost < 700 or total_system_cost > 800:
            logger.warning(
                f"Overall system costs seem off ({total_system_cost:.1f} bn€) - expected 773-777 bn€"
            )

        # Apply technology groupings
        costs_df_temp = pd.DataFrame(total_costs).T  # Convert Series to DataFrame row
        grouped_df = apply_technology_groupings(costs_df_temp)
        total_costs = grouped_df.iloc[0]  # Extract back to Series

        cost_data[scenario] = total_costs

    # Create DataFrame with scenarios as rows and technologies as columns
    costs_df = pd.DataFrame(cost_data).T
    costs_df = costs_df.fillna(0)

    logger.info(f"Calculated costs for scenarios: {list(costs_df.index)}")
    logger.info(
        f"Technologies found: {list(costs_df.columns)[:10]}..."
    )  # Show first 10 only

    return costs_df


def calculate_ptes_savings(costs_df: pd.DataFrame) -> pd.DataFrame:
    """Step 3: Calculate savings relative to NoPTES scenarios."""
    savings_data = []

    for supply_temp in SUPPLY_TEMPS:
        # Find NoPTES baseline for this temperature level (exclude _init suffix)
        baseline_scenario = None
        # First try to find scenario without _init suffix
        for scenario in costs_df.index:
            if f"NoPTES_{supply_temp}" in scenario and "_init" not in scenario:
                baseline_scenario = scenario
                break

        if baseline_scenario is None:
            logger.warning(
                f"No NoPTES baseline found for {supply_temp} without _init suffix"
            )
            continue

        logger.info(f"Using baseline scenario: {baseline_scenario} for {supply_temp}")

        baseline_costs = costs_df.loc[baseline_scenario]

        # Calculate differences for each PTES configuration
        for config in PTES_CONFIGS:
            ptes_scenario = None
            for scenario in costs_df.index:
                if (
                    f"{supply_temp}" in scenario
                    and f"_{config}" in scenario
                    and "NoPTES" not in scenario
                ):
                    ptes_scenario = scenario
                    break

            if ptes_scenario is None:
                logger.warning(f"No PTES scenario found for {supply_temp}_{config}")
                continue

            ptes_costs = costs_df.loc[ptes_scenario]
            savings = ptes_costs - baseline_costs

            savings_data.append(
                {"supply_temp": supply_temp, "ptes_config": config, "savings": savings}
            )

    if not savings_data:
        logger.error("No savings data calculated")
        return pd.DataFrame()

    # Create MultiIndex DataFrame
    index_tuples = [(d["supply_temp"], d["ptes_config"]) for d in savings_data]
    index = pd.MultiIndex.from_tuples(
        index_tuples, names=["supply_temp", "ptes_config"]
    )

    savings_df = pd.DataFrame([d["savings"] for d in savings_data], index=index)
    savings_df = savings_df.fillna(0)

    # Apply technology groupings to savings data as well
    if not savings_df.empty:
        for idx in savings_df.index:
            savings_series = savings_df.loc[idx]
            savings_df_temp = pd.DataFrame(savings_series).T  # Convert to DataFrame row
            grouped_df = apply_technology_groupings(savings_df_temp)
            savings_df.loc[idx] = grouped_df.iloc[0]  # Extract back to Series

    return savings_df


def aggregate_technologies_by_significance(
    costs_df: pd.DataFrame, savings_df: pd.DataFrame, threshold: float = 0.05
) -> tuple:
    """Aggregate technologies based on their significance in net differences (savings).

    Technologies that contribute less than threshold (5%) to the sum of absolute values
    for any scenario are aggregated as 'other technologies'.
    """
    logger = logging.getLogger(__name__)

    # Calculate significance based on 5% of sum of absolute values per scenario
    if not savings_df.empty:
        significant_techs = set(["EU aggregated"])  # Always keep explicit

        # For each scenario, check which technologies contribute >5% to absolute sum
        for scenario_idx in savings_df.index:
            scenario_savings = savings_df.loc[scenario_idx]
            abs_sum = scenario_savings.abs().sum()
            significance_threshold = threshold * abs_sum

            for tech in scenario_savings.index:
                if (
                    tech != "EU aggregated"
                    and abs(scenario_savings[tech]) > significance_threshold
                ):
                    significant_techs.add(tech)

        logger.info(
            f"Keeping {len(significant_techs)} significant technologies (>{threshold:.0%} of scenario absolute sums)"
        )
        logger.info(
            f"Significant techs: {sorted(list(significant_techs))[:10]}..."
        )  # Show first 10

    else:
        # Fallback: keep technologies with significant baseline costs
        max_abs_costs = costs_df.abs().max()
        max_cost_value = max_abs_costs.max()
        significance_threshold = threshold * max_cost_value
        significant_techs = set(
            tech
            for tech in max_abs_costs.index
            if max_abs_costs[tech] > significance_threshold
        )
        significant_techs.add("EU aggregated")

    # Aggregate costs dataframe
    costs_agg = costs_df.copy()
    costs_agg["other technologies"] = 0
    techs_to_remove = []

    for tech in costs_df.columns:
        if tech not in significant_techs and tech != "other technologies":
            costs_agg["other technologies"] += costs_agg[tech]
            techs_to_remove.append(tech)

    for tech in techs_to_remove:
        costs_agg = costs_agg.drop(columns=[tech])

    # Aggregate savings dataframe if not empty
    if not savings_df.empty:
        savings_agg = savings_df.copy()
        savings_agg["other technologies"] = 0
        techs_to_remove = []

        for tech in savings_df.columns:
            if tech not in significant_techs and tech != "other technologies":
                if tech in savings_agg.columns:
                    savings_agg["other technologies"] += savings_agg[tech]
                    techs_to_remove.append(tech)

        for tech in techs_to_remove:
            if tech in savings_agg.columns:
                savings_agg = savings_agg.drop(columns=[tech])
    else:
        savings_agg = pd.DataFrame()

    return costs_agg, savings_agg


def apply_technology_groupings(df: pd.DataFrame) -> pd.DataFrame:
    """Group similar technologies into aggregated categories."""
    df_grouped = df.copy()

    # Group AC + DC -> transmission grid
    ac_cols = [col for col in df_grouped.columns if col == "AC"]
    dc_cols = [col for col in df_grouped.columns if col == "DC"]
    if ac_cols or dc_cols:
        transmission_grid_value = 0
        if ac_cols:
            transmission_grid_value += df_grouped[ac_cols].sum(axis=1)
        if dc_cols:
            transmission_grid_value += df_grouped[dc_cols].sum(axis=1)
        df_grouped["transmission grid"] = transmission_grid_value
        df_grouped = df_grouped.drop(columns=ac_cols + dc_cols)

    # Group geothermal technologies -> geothermal heat pumps
    geothermal_cols = [
        col for col in df_grouped.columns if "geothermal" in str(col).lower()
    ]
    if geothermal_cols:
        df_grouped["geothermal heat pumps"] = df_grouped[geothermal_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=geothermal_cols)

    # Group oil primary + gas primary -> fossil primary energy sources
    fossil_cols = [
        col for col in df_grouped.columns if col in ["oil primary", "gas primary"]
    ]
    if fossil_cols:
        df_grouped["fossil primary energy sources"] = df_grouped[fossil_cols].sum(
            axis=1
        )
        df_grouped = df_grouped.drop(columns=fossil_cols)

    return df_grouped


def aggregate_small_technologies(
    df: pd.DataFrame, threshold: float = 0.01
) -> pd.DataFrame:
    """Legacy function for backward compatibility."""
    return df


def get_colors(networks, technologies):
    """Extract colors for technologies from network carriers."""
    colors = {}

    # Get colors from first available network
    first_network = next(iter(networks.values()))

    if hasattr(first_network, "carriers") and hasattr(first_network.carriers, "color"):
        carrier_colors = first_network.carriers.color.dropna().to_dict()

        # Define color sanitization function (from working script)
        def sanitize_color(color):
            if pd.isna(color) or color == "" or color is None:
                return None
            # Handle hex colors
            if isinstance(color, str) and color.startswith("#"):
                return color
            # Handle named colors
            try:
                from matplotlib.colors import is_color_like

                if is_color_like(color):
                    return color
            except:
                pass
            return None

        for tech in technologies:
            if tech in carrier_colors:
                color = sanitize_color(carrier_colors[tech])
                if color:
                    colors[tech] = color

    # Add default colors for special technologies
    default_colors = {
        "EU aggregated": "#cccccc",
        "other technologies": "#999999",
        "transmission grid": "#00b10fff",  # Blue color for power grid
        "geothermal heat pumps": "khaki",  # Keep existing geothermal color
        "fossil primary energy sources": "#8B4513",  # Brown color for fossil fuels
    }

    for tech, color in default_colors.items():
        if tech in technologies and tech not in colors:
            colors[tech] = color

    return colors


def create_plots(
    costs_df, savings_df, colors, output_path, dh_prices=None, threshold=0.05
):
    """Create the PTES system cost impact plots."""
    logger = logging.getLogger(__name__)

    if costs_df.empty:
        logger.warning("Empty costs dataframe, cannot create plots")
        return

    # Extract supply temperatures from scenario names
    temp_candidates = []
    for s in costs_df.index:
        parts = s.split("_")
        for part in parts:
            if "SupplyTemperature" in part:
                temp_candidates.append(part)
    SUPPLY_TEMPS = sorted(list(set(temp_candidates)))
    logger.info(f"Found supply temperatures: {SUPPLY_TEMPS}")

    if not SUPPLY_TEMPS:
        logger.warning("No supply temperatures found in scenario names")
        return

    # Sort technologies by their contribution in PTES scenarios
    tech_order = []
    if not savings_df.empty:
        # Get maximum absolute savings across all scenarios for each tech
        max_savings = savings_df.abs().max(axis=0)
        tech_order = max_savings.sort_values(ascending=False).index.tolist()
    else:
        # Fallback: sort by maximum absolute cost contribution
        max_costs = costs_df.abs().max(axis=0)
        tech_order = max_costs.sort_values(ascending=False).index.tolist()

    # Aggregate technologies based on significance in net differences
    costs_agg, savings_agg = aggregate_technologies_by_significance(
        costs_df, savings_df, threshold
    )

    logger.info(
        f"After aggregation: {len(costs_agg.columns)} cost technologies, {len(savings_agg.columns)} savings technologies"
    )

    # Order technologies by contribution to baseline costs (high shares at bottom, EU countries at top)
    all_techs = set(costs_agg.columns)
    if not savings_agg.empty:
        all_techs.update(savings_agg.columns)

    # Calculate average baseline costs for sorting (exclude EU countries from main sorting)
    baseline_contribution = {}
    for tech in all_techs:
        if tech != "EU aggregated":
            avg_contribution = 0
            count = 0
            for supply_temp in SUPPLY_TEMPS:
                baseline_scenario = None
                for scenario in costs_agg.index:
                    if f"NoPTES_{supply_temp}" in scenario:
                        baseline_scenario = scenario
                        break
                if baseline_scenario and tech in costs_agg.columns:
                    avg_contribution += costs_agg.loc[baseline_scenario, tech]
                    count += 1
            if count > 0:
                baseline_contribution[tech] = avg_contribution / count
            else:
                baseline_contribution[tech] = 0

    # Sort technologies by baseline contribution (descending - largest at bottom of stack)
    sorted_techs = sorted(
        baseline_contribution.keys(),
        key=lambda x: baseline_contribution[x],
        reverse=True,
    )

    # Final order: largest contributions at bottom, EU countries at top
    ordered_techs = sorted_techs
    if "EU aggregated" in all_techs:
        ordered_techs.append("EU aggregated")

    # Add any remaining techs
    for tech in all_techs:
        if tech not in ordered_techs:
            ordered_techs.insert(
                -1 if "EU aggregated" in ordered_techs else len(ordered_techs),
                tech,
            )

    # Create figure with adjusted column ratios for better bar width balance and increased height
    fig, axes = plt.subplots(
        len(SUPPLY_TEMPS), 2, figsize=(9, 7), gridspec_kw={"width_ratios": [0.8, 3.0]}
    )
    if len(SUPPLY_TEMPS) == 1:
        axes = axes.reshape(1, -1)

    # Plot for each supply temperature
    for i, supply_temp in enumerate(SUPPLY_TEMPS):
        ax_left = axes[i, 0]
        ax_right = axes[i, 1]

        # Temperature label positioned much further left to avoid overlap with y-axis
        temp_label = supply_temp.replace("SupplyTemperature", "\nSupply\nTemperature")
        ax_left.text(
            -1.2,
            0.5,
            temp_label,
            transform=ax_left.transAxes,
            rotation=90,
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=10,
            fontweight="bold",
        )

        # Plot baseline costs (left plot) - exclude _init suffix
        baseline_scenario = None
        # First try to find scenario without _init suffix
        for scenario in costs_agg.index:
            if f"NoPTES_{supply_temp}" in scenario and "_init" not in scenario:
                baseline_scenario = scenario
                break
        # Fallback to any NoPTES scenario if none found without _init
        if baseline_scenario is None:
            for scenario in costs_agg.index:
                if f"NoPTES_{supply_temp}" in scenario:
                    baseline_scenario = scenario
                    logger.warning(
                        f"Using _init baseline scenario for plotting: {baseline_scenario}"
                    )
                    break

        if baseline_scenario is None:
            logger.warning(f"No baseline scenario found for {supply_temp}")
            continue

        baseline_costs = costs_agg.loc[baseline_scenario]

        # Calculate German costs for ylim (excluding EU countries)
        german_cost = sum(
            v for k, v in baseline_costs.items() if k != "EU aggregated" and v > 0
        )

        # Create stacked bar for baseline costs with EU countries at top
        bottom = 0
        logger.info(f"Plotting baseline for {supply_temp}: {baseline_scenario}")
        logger.info(
            f"Available technologies in baseline_costs: {list(baseline_costs.index)}"
        )
        logger.info(f"Ordered technologies: {ordered_techs}")

        for tech in ordered_techs:
            if tech in baseline_costs.index and abs(baseline_costs[tech]) > 1e-6:
                value = baseline_costs[tech]
                ax_left.bar(
                    [0],
                    [value],
                    bottom=bottom,
                    color=colors.get(tech, "gray"),
                    label=tech,
                    edgecolor="none",
                    width=0.6,
                )
                bottom += value
                logger.info(f"Added {tech}: {value:.3f} bn€, cumulative: {bottom:.3f}")
            else:
                if tech not in baseline_costs.index:
                    logger.warning(f"Technology {tech} not found in baseline costs")
                elif abs(baseline_costs[tech]) <= 1e-6:
                    logger.info(
                        f"Technology {tech} has negligible cost: {baseline_costs[tech]:.6f}"
                    )

        # Set ylim based on total costs with axis break for EU countries
        total_cost_with_neighbors = german_cost + baseline_costs.get("EU aggregated", 0)
        if baseline_costs.get("EU aggregated", 0) > 0:
            # Adjust upper ylim to 1.3 times German system cost sum (about 200 bn€)
            upper_limit = german_cost * 1.3
            ax_left.set_ylim(0, upper_limit)

            # Add custom y-tick for total system cost and axis break indicators
            # Get current y-ticks
            current_yticks = ax_left.get_yticks()
            current_yticklabels = [
                f"{int(tick)}" for tick in current_yticks if 0 <= tick <= upper_limit
            ]

            # Add total system cost as additional tick at the very top
            total_cost_position = (
                upper_limit  # Position it at the very top of the y-axis
            )
            extended_yticks = list(current_yticks[current_yticks <= upper_limit]) + [
                total_cost_position
            ]
            extended_yticklabels = current_yticklabels + [
                f"{total_cost_with_neighbors:.0f}"
            ]

            ax_left.set_yticks(extended_yticks)
            ax_left.set_yticklabels(extended_yticklabels)

            # Add axis break indicators between German costs and total cost tick
            break_y = german_cost * 1.1  # Position break slightly above German costs
            break_height = upper_limit * 0.05  # Height of break markers

            # Draw zigzag break lines on left y-axis
            ax_left.plot(
                [-0.02, 0.02, -0.02, 0.02],
                [
                    break_y - break_height / 2,
                    break_y,
                    break_y + break_height / 2,
                    break_y + break_height,
                ],
                color="black",
                linewidth=1.5,
                clip_on=False,
                transform=ax_left.get_yaxis_transform(),
            )

        else:
            # If no EU countries, just show German costs with some margin
            ax_left.set_ylim(0, 1.1 * german_cost)

        ax_left.set_xlim(-0.5, 0.5)
        ax_left.set_xticks([])
        ax_left.set_ylabel("System Costs\n[bn EUR a$^{-1}$]", labelpad=5)
        ax_left.grid(True, alpha=0.3)

        # Plot savings (right plot) - use same tech order but sort by net difference contribution
        # The savings_agg index contains tuples like ('HighSupplyTemperature', 'hpboost')
        ptes_scenarios_unsorted = [
            s for s in savings_agg.index if isinstance(s, tuple) and s[0] == supply_temp
        ]

        # Order PTES scenarios as requested: hpboost, rhboost, hpboost_10Cbottom, noboost
        if ptes_scenarios_unsorted and not savings_agg.empty:
            desired_order = ["hpboost", "rhboost", "hpboost_10Cbottom", "noboost"]
            ptes_scenarios = []
            # First add scenarios in the desired order
            for desired in desired_order:
                for scenario in ptes_scenarios_unsorted:
                    if scenario[1] == desired:
                        ptes_scenarios.append(scenario)
            # Add any remaining scenarios not in desired order
            for scenario in ptes_scenarios_unsorted:
                if scenario not in ptes_scenarios:
                    ptes_scenarios.append(scenario)
        else:
            ptes_scenarios = ptes_scenarios_unsorted
        logger.info(f"Found PTES scenarios for {supply_temp}: {ptes_scenarios}")

        if ptes_scenarios and not savings_agg.empty:
            # For net difference plots, sort techs by their contribution magnitude
            # Calculate average absolute contribution across PTES scenarios for this temperature
            temp_tech_contributions = {}
            for tech in ordered_techs:
                if tech != "EU aggregated":
                    avg_contribution = 0
                    count = 0
                    for scenario in ptes_scenarios:
                        if tech in savings_agg.columns:
                            avg_contribution += abs(savings_agg.loc[scenario, tech])
                            count += 1
                    temp_tech_contributions[tech] = avg_contribution / max(count, 1)

            # Sort by contribution (largest first to appear closest to x-axis at bottom of stack)
            sorted_net_techs = sorted(
                [t for t in temp_tech_contributions.keys()],
                key=lambda x: temp_tech_contributions[x],
                reverse=True,
            )
            if "EU aggregated" in ordered_techs:
                sorted_net_techs.append("EU aggregated")

            # Store total savings and bar heights for star markers and annotations
            total_savings_per_scenario = []
            bar_top_positions = []  # Track highest point of bars for annotations

            for j, scenario in enumerate(ptes_scenarios):
                scenario_savings = savings_agg.loc[scenario]

                bottom_pos = 0
                bottom_neg = 0

                # Calculate total net difference for this scenario
                total_net_diff = sum(
                    scenario_savings[tech]
                    for tech in scenario_savings.index
                    if abs(scenario_savings[tech]) > 1e-6
                )
                total_savings_per_scenario.append(total_net_diff)

                for tech in sorted_net_techs:
                    if (
                        tech in scenario_savings.index
                        and abs(scenario_savings[tech]) > 1e-6
                    ):
                        value = scenario_savings[tech]
                        if value > 0:
                            ax_right.bar(
                                [j],
                                [value],
                                bottom=bottom_pos,
                                color=colors.get(tech, "gray"),
                                edgecolor="none",
                                width=0.6,
                            )
                            bottom_pos += value
                        else:
                            ax_right.bar(
                                [j],
                                [value],
                                bottom=bottom_neg,
                                color=colors.get(tech, "gray"),
                                edgecolor="none",
                                width=0.6,
                            )
                            bottom_neg += value

                # Store the highest point for this scenario (for annotation positioning)
                bar_top_positions.append(max(bottom_pos, abs(bottom_neg)))

            # Add black star markers with white border for total net differences
            star_y_positions = total_savings_per_scenario
            ax_right.scatter(
                range(len(ptes_scenarios)),
                star_y_positions,
                marker="*",
                s=200,
                facecolor="black",
                edgecolor="white",
                linewidth=2,
                zorder=10,
                label="Net difference",
            )

            # Add connecting line for star markers (black with white border)
            if len(star_y_positions) > 0:
                # Extend to start at x=-0.5 with y=0 (baseline)
                x_coords = [-0.5] + list(range(len(ptes_scenarios)))
                y_coords = [0] + list(star_y_positions)

                # Draw white outline first
                ax_right.plot(
                    x_coords, y_coords, "-", color="white", linewidth=2, zorder=8
                )
                # Draw main black line
                ax_right.plot(x_coords, y_coords, "k-", linewidth=1, zorder=9)

            # Add annotations for relative changes
            if baseline_scenario:
                baseline_total_cost = costs_agg.loc[baseline_scenario].sum()
                baseline_german_cost = sum(
                    v
                    for k, v in costs_agg.loc[baseline_scenario].items()
                    if k != "EU aggregated"
                )
                logger.info(
                    f"Using baseline {baseline_scenario}: Total={baseline_total_cost:.1f}, German={baseline_german_cost:.1f}"
                )

                for j, (scenario, total_saving, bar_top) in enumerate(
                    zip(ptes_scenarios, star_y_positions, bar_top_positions)
                ):
                    # Calculate percentage relative to total and German costs
                    pct_vs_total = (total_saving / baseline_total_cost) * 100
                    pct_vs_german = (total_saving / baseline_german_cost) * 100

                    # Position all annotations between 3 and 5 on cost savings scale
                    y_position = 4.0  # Fixed position between 3 and 5

                    annotation_text = (
                        f"{pct_vs_total:.1f}% (total)\n{pct_vs_german:.1f}% (DE)"
                    )

                    logger.info(
                        f"Adding annotation for {scenario}: {annotation_text} at y={y_position:.2f}"
                    )

                    ax_right.annotate(
                        annotation_text,
                        (j, y_position),
                        xytext=(0, 0),  # No offset
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        fontsize=8,
                        zorder=20,  # High z-order to appear in front of bars and markers
                    )

                    # Store the annotation position for ylim adjustment
                    if not hasattr(ax_right, "_annotation_positions"):
                        ax_right._annotation_positions = []
                    ax_right._annotation_positions.append(
                        y_position + 0.2
                    )  # Add margin for upper annotations

            ax_right.set_xticks(range(len(ptes_scenarios)))
            # Handle tuple scenario names with improved labels and line breaks
            scenario_labels = []
            for s in ptes_scenarios:
                scenario_name = s[1] if isinstance(s, tuple) else s.split("_")[-1]
                if scenario_name == "hpboost":
                    scenario_labels.append("Booster\nheat pump\nto 35°C")
                elif scenario_name == "rhboost":
                    scenario_labels.append("Resistive\nboosting")
                elif scenario_name == "hpboost_10Cbottom":
                    scenario_labels.append("Booster\nheat pump\nto 10°C")
                elif scenario_name == "noboost":
                    scenario_labels.append("No boosting")
                else:
                    scenario_labels.append(scenario_name)

            # Show xtick labels only for bottom row
            if i == len(SUPPLY_TEMPS) - 1:
                ax_right.set_xticklabels(scenario_labels, fontsize=9)
            else:
                ax_right.set_xticklabels([""] * len(scenario_labels))
            ax_right.axhline(y=0, color="black", linewidth=0.8)
            ax_right.set_ylabel("Cost Savings\n[bn EUR a$^{-1}$]", labelpad=5)
            ax_right.grid(True, alpha=0.3)

            # Remove x-tick labels from NoPTES plots and first row PTES labels
            ax_left.set_xticks([0])
            ax_left.set_xticklabels([""])
            if i == len(SUPPLY_TEMPS) - 1:
                # Add centrally placed xlabel for PTES scenarios only in last row
                ax_right.set_xlabel(
                    "PTES configurations",
                    fontsize=12,
                    fontweight="bold",
                    horizontalalignment="center",
                )

            # Adjust ylim to ensure annotations are visible
            if (
                hasattr(ax_right, "_annotation_positions")
                and ax_right._annotation_positions
            ):
                current_ylim = ax_right.get_ylim()
                max_annotation_y = max(ax_right._annotation_positions)
                if max_annotation_y > current_ylim[1]:
                    ax_right.set_ylim(current_ylim[0], max_annotation_y)

            # Add district heating price on secondary y-axis for right plot
            if dh_prices and ptes_scenarios:
                ax_right_dh = ax_right.twinx()
                baseline_dh_price = None
                # First try to find scenario without _init suffix
                for scenario in dh_prices.keys():
                    if f"NoPTES_{supply_temp}" in scenario and "_init" not in scenario:
                        baseline_dh_price = dh_prices[scenario]
                        break
                # Fallback to any NoPTES scenario
                if baseline_dh_price is None:
                    for scenario in dh_prices.keys():
                        if f"NoPTES_{supply_temp}" in scenario:
                            baseline_dh_price = dh_prices[scenario]
                            break

                if baseline_dh_price is not None and baseline_dh_price > 0:
                    # Plot baseline price and calculate deltas for each PTES scenario
                    dh_deltas = []
                    for scenario in ptes_scenarios:
                        # Convert tuple scenario name back to string format for DH price lookup
                        if isinstance(scenario, tuple):
                            scenario_name = f"{scenario[0]}_MidDH_{scenario[1]}"
                        else:
                            scenario_name = scenario
                        scenario_dh_price = dh_prices.get(
                            scenario_name, baseline_dh_price
                        )
                        delta = (
                            scenario_dh_price - baseline_dh_price
                        )  # Positive = increase, negative = decrease
                        dh_deltas.append(delta)

                    # Plot connected line with red triangle markers starting from x=0
                    if len(dh_deltas) > 0:
                        # Extend to start at x=-0.5 with delta=0
                        x_coords = [-0.5] + list(range(len(ptes_scenarios)))
                        y_coords = [0] + dh_deltas

                        # Draw white outline for entire line first
                        ax_right_dh.plot(
                            x_coords,
                            y_coords,
                            "-",
                            color="white",
                            linewidth=2,
                            zorder=1,
                        )
                        # Draw main red line for entire connection
                        ax_right_dh.plot(
                            x_coords, y_coords, "r-", linewidth=1, zorder=2
                        )
                        # Draw triangle markers only at PTES positions
                        ax_right_dh.plot(
                            range(len(ptes_scenarios)),
                            dh_deltas,
                            "r^",
                            markersize=10,
                            markeredgecolor="white",
                            markeredgewidth=1.5,
                            label="ΔDH price",
                            zorder=2,
                        )
                        ax_right_dh.set_ylabel(
                            "ΔDH Price\n[EUR MWh$^{-1}$]", color="red", labelpad=5
                        )
                        ax_right_dh.tick_params(axis="y", labelcolor="red")
                        # Set DH price limits with reasonable symmetric range around zero
                        ax_right_dh.set_ylim(
                            -6, 6
                        )  # Symmetric range for proper y=0 alignment
                        ax_right_dh.axhline(
                            y=0, color="red", linestyle="--", alpha=0.5, linewidth=1
                        )

                        # Set symmetric y-limits for primary axis (cost savings) to align 0 lines
                        current_ylim = ax_right.get_ylim()
                        max_abs_value = max(abs(current_ylim[0]), abs(current_ylim[1]))
                        ax_right.set_ylim(-max_abs_value, max_abs_value)
        else:
            # If no PTES scenarios, hide the right plot
            ax_right.set_visible(False)

        # Add district heating price on left plot
        if dh_prices:
            ax_left_dh = ax_left.twinx()
            baseline_dh_price = None
            # First try to find scenario without _init suffix
            for scenario in dh_prices.keys():
                if f"NoPTES_{supply_temp}" in scenario and "_init" not in scenario:
                    baseline_dh_price = dh_prices[scenario]
                    break
            # Fallback to any NoPTES scenario
            if baseline_dh_price is None:
                for scenario in dh_prices.keys():
                    if f"NoPTES_{supply_temp}" in scenario:
                        baseline_dh_price = dh_prices[scenario]
                        break

            if baseline_dh_price is not None and baseline_dh_price > 0:
                # Plot smaller circle marker for DH price
                ax_left_dh.plot(
                    [0],
                    [baseline_dh_price],
                    "ro",
                    markersize=8,
                    markeredgecolor="white",
                    markeredgewidth=1.5,
                    label="DH price (baseline)",
                )
                # Add horizontal dashed line at baseline price level
                ax_left_dh.axhline(
                    y=baseline_dh_price,
                    color="red",
                    linestyle="--",
                    alpha=0.6,
                    linewidth=1,
                )
                ax_left_dh.set_ylabel("DH Price\n[EUR MWh$^{-1}$]", color="red")
                ax_left_dh.tick_params(axis="y", labelcolor="red")
                ax_left_dh.set_ylim(0, baseline_dh_price * 1.2)

        # Only show title for first row
        if i == 0:
            ax_right.set_title("Net Cost Difference vs NoPTES", fontweight="bold")

    # Set consistent y-limits across all axes
    if len(SUPPLY_TEMPS) > 1:
        # Get all German costs (excluding EU countries) for consistent scaling
        all_german_costs = []
        for i, supply_temp in enumerate(SUPPLY_TEMPS):
            baseline_scenario = None
            # First try to find scenario without _init suffix
            for scenario in costs_agg.index:
                if f"NoPTES_{supply_temp}" in scenario and "_init" not in scenario:
                    baseline_scenario = scenario
                    break
            # Fallback to any NoPTES scenario
            if baseline_scenario is None:
                for scenario in costs_agg.index:
                    if f"NoPTES_{supply_temp}" in scenario:
                        baseline_scenario = scenario
                        break
            if baseline_scenario:
                baseline_costs = costs_agg.loc[baseline_scenario]
                german_cost = sum(
                    v for k, v in baseline_costs.items() if k != "EU aggregated"
                )
                all_german_costs.append(german_cost)

        if all_german_costs:
            max_german_cost = max(all_german_costs)
            for i in range(len(SUPPLY_TEMPS)):
                axes[i, 0].set_ylim(0, 1.3 * max_german_cost)

    # Create categorized legend
    handles = []
    labels = []

    # German cost components (excluding EU countries and other indicators)
    # Include ALL technologies that appear in any plot, not just first 10
    german_techs = [
        tech
        for tech in ordered_techs
        if tech != "EU aggregated" and tech != "other technologies"
    ]

    if german_techs:
        # Add category label (invisible handle)
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
        labels.append(r"$\mathbf{German\ cost\ components:}$")

        for tech in german_techs:
            handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=colors.get(tech, "gray"), edgecolor="white"
                )
            )
            labels.append(f"  {tech}")

    # Other technologies category
    if "other technologies" in ordered_techs:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
        labels.append(r"$\mathbf{Aggregated:}$")
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors.get("other technologies", "gray"),
                edgecolor="white",
            )
        )
        labels.append("  other technologies")

    # EU aggregated category
    if "EU aggregated" in ordered_techs:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
        labels.append(r"$\mathbf{Other\ countries:}$")
        handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors.get("EU aggregated", "gray"),
                edgecolor="white",
            )
        )
        labels.append("  EU aggregated")

    # Other indicators category
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
    labels.append(r"$\mathbf{Other\ indicators:}$")

    # Add net difference star marker
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="black",
            markeredgecolor="white",
            markeredgewidth=2,
            markersize=12,
            linestyle="None",
        )
    )
    labels.append("  Net difference")

    # Add DH price markers
    if dh_prices:
        # Add baseline DH price marker
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="red", markersize=8, linestyle="None"
            )
        )
        labels.append("  DH price (baseline)")

        # Add DH price delta marker
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="red",
                markersize=8,
                linestyle="-",
                linewidth=2,
            )
        )
        labels.append("  ΔDH price")

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=3,
            frameon=False,
            columnspacing=1.2,
        )

    plt.tight_layout()
    plt.subplots_adjust(
        bottom=0.20, left=0.2, hspace=0.25
    )  # Reduced whitespace: less space for legend (bottom) and between plots (hspace)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to: {output_path}")


def main(snakemake):
    """Main function."""
    configure_logging(snakemake)

    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons

    # Get aggregation threshold
    threshold = 0.01
    try:
        threshold = float(
            snakemake.params.plotting.get("aggregate_threshold_pct", 0.025)
        )
        if threshold > 1.0:
            threshold /= 100.0
    except (AttributeError, KeyError, ValueError):
        pass

    # Get override colors
    override_colors = {}
    try:
        override_colors = snakemake.params.plotting.get("override_tech_colors", {})
    except (AttributeError, KeyError):
        pass

    # Step 1: Load networks
    networks = load_networks(run_name, scenarios, planning_horizons)
    if not networks:
        logger.error("No networks loaded, aborting")
        return

    # Step 2: Calculate German costs
    costs_df = calculate_german_costs(networks)
    if costs_df.empty:
        logger.error("No costs calculated, aborting")
        return

    # Step 3: Calculate PTES savings
    savings_df = calculate_ptes_savings(costs_df)
    if savings_df.empty:
        logger.error("No savings calculated, aborting")
        return

    # Step 4: Calculate district heating prices for all networks
    dh_prices = {}
    for scenario_name, network in networks.items():
        try:
            dh_prices[scenario_name] = calc_average_dh_price(network)
        except Exception as e:
            logger.warning(f"Failed to calculate DH price for {scenario_name}: {e}")
            dh_prices[scenario_name] = 0

    # Get all unique technologies across both datasets
    all_techs = set(costs_df.columns)
    if not savings_df.empty:
        all_techs.update(savings_df.columns)

    # Get colors for all technologies
    colors = get_colors(networks, all_techs)

    # Add override colors
    if override_colors:
        colors.update(override_colors)

    # Steps 5-6: Create plots
    create_plots(
        costs_df, savings_df, colors, snakemake.output.ptes_fig, dh_prices, threshold
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(Path(__file__).resolve().parents[2])
        snakemake = mock_snakemake(
            "plot_ptes_savings_and_neighbour_costs",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )
    main(snakemake)
