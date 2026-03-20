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
import matplotlib.patheffects as path_effects
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
SUPPLY_TEMPS = ["HighSupplyTemperature", "MidSupplyTemperature", "LowSupplyTemperature"]
PTES_CONFIGS = ["freeboost", "freecap", "rhboost", "hpboost_nocooling", "hpboost_10Cbottom"]

# Human-readable labels for PTES configurations
PTES_CONFIG_LABELS = {
    "freeboost": "Free\nboosting*",
    "freecap": "Free\ncapacity*",
    "rhboost": "Resistive\nboosting",
    "hpboost_nocooling": "Heat pump\nno cooling",
    "hpboost_10Cbottom": "Heat pump\nto 10°C",
}


def format_label(label):
    """Format legend labels according to specified naming conventions."""
    if pd.isna(label) or label == "":
        return label

    # Apply string replacements
    replacements = {
        "urban central ": "",
        "CC": "",
        "ptes": "PTES",
        "water tanks": "TTES",
        "water pits": "PTES",
        "river_water": "river water",
    }

    formatted = str(label)
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)

    # Clean up extra spaces that might result from CC removal
    formatted = " ".join(formatted.split())

    # Define acronyms that should remain fully uppercase
    acronyms = {"PV", "PTES", "TTES", "AC", "DC", "EU"}

    # Capitalize first word (but preserve acronyms and existing uppercase patterns)
    if formatted:
        words = formatted.split()
        if words:
            first_word = words[0]
            # Keep acronyms uppercase, otherwise capitalize normally if all lowercase
            if first_word.upper() in acronyms:
                words[0] = first_word.upper()
            elif first_word == first_word.lower():
                words[0] = first_word.capitalize()
            formatted = " ".join(words)

    return formatted


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


def _compute_capex_from_network(n):
    """Compute annualized CAPEX as capital_cost × optimal_capacity for all components.

    Unlike n.statistics.capex() (which only counts new investment for extendable assets),
    this includes the full annualized cost for all assets — existing and new.
    Returns a MultiIndex Series with levels ['component', 'bus', 'carrier'].
    """
    all_capex = []
    component_specs = [
        ("Generator", n.generators, "bus", "p_nom_opt"),
        ("Link", n.links, "bus0", "p_nom_opt"),
        ("StorageUnit", n.storage_units, "bus", "p_nom_opt"),
        ("Store", n.stores, "bus", "e_nom_opt"),
        ("Line", n.lines, "bus0", "s_nom_opt"),
    ]
    for comp_type, df, bus_col, cap_col in component_specs:
        if df.empty or "capital_cost" not in df.columns or cap_col not in df.columns:
            continue
        active = df[(df["capital_cost"] > 0) & (df[cap_col] > 0)]
        if active.empty:
            continue
        capex = active["capital_cost"] * active[cap_col]
        mi = pd.MultiIndex.from_arrays(
            [
                [comp_type] * len(active),
                active[bus_col].values,
                active["carrier"].values,
            ],
            names=["component", "bus", "carrier"],
        )
        all_capex.append(pd.Series(capex.values, index=mi))
    if not all_capex:
        return pd.Series(dtype=float)
    return pd.concat(all_capex).groupby(level=[0, 1, 2]).sum()


def get_system_costs_by_carrier(n, country, exclude_country=False, aggregate_all=False):
    """Calculate total system costs by carrier for all countries modelled, a specified country, or all countries minus a specified country."""
    try:
        # CAPEX: capital_cost × p_nom_opt for ALL components (existing + new investments)
        # n.statistics.capex() only counts expansion of extendable assets, so we compute manually.
        capex = _compute_capex_from_network(n)
        # OPEX: variable + fixed O&M from statistics API
        opex = n.statistics.opex(groupby=["bus", "carrier"], nice_names=False)

        # Apply country filtering based on analysis of component locations
        if not aggregate_all:
            capex_country = capex.loc[
                capex.index.get_level_values(1).str.startswith(country)
            ]
            opex_country = opex.loc[
                opex.index.get_level_values(1).str.startswith(country)
            ]

            capex_others = capex.drop(capex_country.index)
            opex_others = opex.drop(opex_country.index)

            capex_country = capex_country.groupby(level=2).sum()
            opex_country = opex_country.groupby(level=2).sum()

            capex_others = capex_others.groupby(level=2).sum()
            opex_others = opex_others.groupby(level=2).sum()

            ic_capex_correction = calc_ic_capex_correction(n, country)
            # Extend before in-place addition to avoid KeyError in newer pandas
            capex_country = capex_country.reindex(
                capex_country.index.union(ic_capex_correction.index)
            ).fillna(0)
            capex_country.loc[ic_capex_correction.index] += ic_capex_correction
            capex_others = capex_others.reindex(
                capex_others.index.union(ic_capex_correction.index)
            ).fillna(0)
            capex_others.loc[ic_capex_correction.index] -= ic_capex_correction

            if exclude_country:
                union_index = capex_others.index.union(opex_others.index)
                capex_others = capex_others.reindex(union_index).fillna(0)
                opex_others = opex_others.reindex(union_index).fillna(0)
                total_others = capex_others + opex_others
                return total_others
            else:
                union_index = capex_country.index.union(opex_country.index)
                capex_country = capex_country.reindex(union_index).fillna(0)
                opex_country = opex_country.reindex(union_index).fillna(0)
                total_country = capex_country + opex_country
                return total_country
        else:
            capex = capex.groupby(level=2).sum()
            opex = opex.groupby(level=2).sum()
            union_index = capex.index.union(opex.index)
            capex = capex.reindex(union_index).fillna(0)
            opex = opex.reindex(union_index).fillna(0)
            total = capex + opex
            return total

    except Exception as e:
        logger.warning(f"Statistics API failed: {e}")
        capex = pd.Series(dtype=float)
        opex = pd.Series(dtype=float)

    union_index = capex.index.union(opex.index)
    capex = capex.reindex(union_index).fillna(0)
    opex = opex.reindex(union_index).fillna(0)

    # Return a Series (capex + opex per carrier), consistent with the success path
    total = capex + opex

    if not aggregate_all:
        try:
            ic_capex_correction = calc_ic_capex_correction(
                n, country, exclude_country=exclude_country
            )
            total = total.reindex(total.index.union(ic_capex_correction.index)).fillna(
                0
            )
            total.loc[ic_capex_correction.index] += ic_capex_correction
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
        # Ensure Series (older fallback path may return a DataFrame)
        if isinstance(german_costs, pd.DataFrame):
            german_costs = german_costs.sum(axis=1)

        # Calculate EU countries costs separately (don't use aggregate_all=True to avoid double-counting)
        try:
            # Get costs for all non-DE countries
            neighbor_costs_full = get_system_costs_by_carrier(
                network, country="DE", exclude_country=True, aggregate_all=False
            )
            if isinstance(neighbor_costs_full, pd.DataFrame):
                neighbor_costs_full = neighbor_costs_full.sum(axis=1)
            neighbor_total = neighbor_costs_full.sum()  # scalar sum

            # Allow very small numerical differences due to floating point precision
            total_expected = get_system_costs_by_carrier(
                network, country="DE", exclude_country=False, aggregate_all=True
            ).sum()
            total_actual = neighbor_total + german_costs.sum()

            assert abs(total_expected - total_actual) < 1e-2, (
                f"Cost summation mismatch: expected {total_expected:.9f}, "
                f"got {total_actual:.9f}, difference {abs(total_expected - total_actual):.9f}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to calculate neighbor costs for {scenario}: {e}. Setting to 0."
            )
            neighbor_total = 0

        # Sum capex and opex for German costs and convert to bn€
        total_costs = german_costs

        # Add EU countries as a separate category
        total_costs["EU aggregated"] = neighbor_total

        # Calculate totals for logging
        german_cost = total_costs.drop("EU aggregated").sum()
        total_system_cost = total_costs.sum()

        logger.info(
            f"{scenario}: German={german_cost:.1f} €, Neighbors={neighbor_total:.1f} €, Total={total_system_cost:.1f} €"
        )

        # Expected ranges: German 17-150 €, Overall system 773-777 €
        if german_cost > 200:
            logger.warning(
                f"German costs seem high ({german_cost:.1f} €) - expected 17-150 €"
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


def calculate_ptes_savings(costs_df: pd.DataFrame, supply_temps=None) -> pd.DataFrame:
    """Step 3: Calculate savings relative to NoPTES scenarios."""
    savings_data = []
    _supply_temps = supply_temps if supply_temps is not None else SUPPLY_TEMPS

    for supply_temp in _supply_temps:
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
    assert (
        abs(df.sum().sum() - df_grouped.sum().sum()) < 1e-2
    ), "Grouping of transmission technologies changed total sum!"

    # Group solar technologies -> PV (solar + solar-hsat + solar rooftop)
    solar_cols = [
        col
        for col in df_grouped.columns
        if col in ["solar-hsat", "solar", "solar rooftop"]
    ]
    if solar_cols:
        df_grouped["PV"] = df_grouped[solar_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=solar_cols)
    assert (
        abs(df.sum().sum() - df_grouped.sum().sum()) < 1e-2
    ), "Grouping of solar technologies changed total sum!"

    # Group wind technologies -> Wind power
    wind_cols = [
        col
        for col in df_grouped.columns
        if col in ["onwind", "offwind-dc", "offwind-ac"]
    ]
    if wind_cols:
        df_grouped["Wind power"] = df_grouped[wind_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=wind_cols)
    assert (
        abs(df.sum().sum() - df_grouped.sum().sum()) < 1e-2
    ), "Grouping of wind technologies changed total sum!"

    # Group geothermal technologies -> geothermal heat pumps
    geothermal_cols = [
        col for col in df_grouped.columns if "geothermal" in str(col).lower()
    ]
    if geothermal_cols:
        df_grouped["geothermal heat pumps"] = df_grouped[geothermal_cols].sum(axis=1)
        geothermal_cols = [
            col for col in geothermal_cols if col != "geothermal heat pumps"
        ]
        df_grouped = df_grouped.drop(columns=geothermal_cols)
    assert (
        abs(df.sum().sum() - df_grouped.sum().sum()) < 1e-2
    ), "Grouping of geothermal technologies changed total sum!"

    # Group oil primary + gas primary -> fossil primary energy sources
    fossil_cols = [
        col for col in df_grouped.columns if col in ["oil primary", "gas primary"]
    ]
    if fossil_cols:
        df_grouped["fossil primary energy sources"] = df_grouped[fossil_cols].sum(
            axis=1
        )
        fossil_cols = [
            col for col in fossil_cols if col != "fossil primary energy sources"
        ]
        df_grouped = df_grouped.drop(columns=fossil_cols)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of fossil primary energy sources changed total sum!"

    # Group PTES (water pits) variants -> PTES
    ptes_cols = [
        col
        for col in df_grouped.columns
        if "water pits" in str(col).lower() and "ptes" not in str(col).lower()
    ]
    if ptes_cols:
        df_grouped["PTES"] = df_grouped[ptes_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=ptes_cols)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of PTES technologies changed total sum!"

    # Rename PTES heat pump carrier (keep "PTES heat pump" if already present)
    ptes_hp_cols = [
        col for col in df_grouped.columns if "ptes heat pump" in str(col).lower()
    ]
    if ptes_hp_cols:
        df_grouped["PTES heat pump"] = df_grouped[ptes_hp_cols].sum(axis=1)
        cols_to_drop = [c for c in ptes_hp_cols if c != "PTES heat pump"]
        if cols_to_drop:
            df_grouped = df_grouped.drop(columns=cols_to_drop)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of PTES heat pump changed total sum!"

    # Group TTES (water tanks) variants -> TTES
    ttes_cols = [col for col in df_grouped.columns if "water tanks" in str(col).lower()]
    if ttes_cols:
        df_grouped["TTES"] = df_grouped[ttes_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=ttes_cols)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of TTES technologies changed total sum!"

    # Group H2 turbine -> H2 OCGT (also consolidates if H2 OCGT already exists as a carrier)
    h2_ocgt_source_cols = [
        col
        for col in df_grouped.columns
        if col in ["H2 turbine", "H2 OCGT", "H2 retrofit OCGT"]
    ]
    if h2_ocgt_source_cols:
        df_grouped["H2 OCGT"] = df_grouped[h2_ocgt_source_cols].sum(axis=1)
        cols_to_drop = [c for c in h2_ocgt_source_cols if c != "H2 OCGT"]
        if cols_to_drop:
            df_grouped = df_grouped.drop(columns=cols_to_drop)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of H2 OCGT changed total sum!"

    # Group air heat pump variants -> Air heat pump
    air_hp_cols = [
        col
        for col in df_grouped.columns
        if "air heat pump" in str(col).lower() and col != "Air heat pump"
    ]
    if air_hp_cols:
        df_grouped["Air heat pump"] = df_grouped[air_hp_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=air_hp_cols)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of air heat pump changed total sum!"

    # Group resistive heater variants -> Resistive heater
    rh_cols = [
        col
        for col in df_grouped.columns
        if "resistive heater" in str(col).lower() and col != "Resistive heater"
    ]
    if rh_cols:
        df_grouped["Resistive heater"] = df_grouped[rh_cols].sum(axis=1)
        df_grouped = df_grouped.drop(columns=rh_cols)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of resistive heater changed total sum!"

    # Group river water heat pump variants -> River water heat pump (keep destination if already present)
    rwh_cols = [
        col
        for col in df_grouped.columns
        if "river_water heat pump" in str(col).lower()
        or "river water heat pump" in str(col).lower()
    ]
    if rwh_cols:
        df_grouped["River water heat pump"] = df_grouped[rwh_cols].sum(axis=1)
        cols_to_drop = [c for c in rwh_cols if c != "River water heat pump"]
        if cols_to_drop:
            df_grouped = df_grouped.drop(columns=cols_to_drop)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of river water heat pump changed total sum!"

    # Group battery variants -> Battery (includes home battery)
    battery_cols = [
        col
        for col in df_grouped.columns
        if col
        in [
            "Battery",
            "battery",
            "battery charger",
            "battery discharger",
            "battery storage",
            "home battery",
        ]
    ]
    if battery_cols:
        df_grouped["Battery"] = df_grouped[battery_cols].sum(axis=1)
        cols_to_drop = [c for c in battery_cols if c != "Battery"]
        if cols_to_drop:
            df_grouped = df_grouped.drop(columns=cols_to_drop)
    assert (
        abs(df_grouped.sum().sum() - df.sum().sum()) < 1e-2
    ), "Grouping of battery changed total sum!"

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

    # Add default colors for special technologies and grouped carriers
    default_colors = {
        "EU aggregated": "#cccccc",
        "other technologies": "#999999",
        "transmission grid": "#6c9459",  # from config tech_colors: lines / transmission lines
        "geothermal heat pumps": "khaki",  # from override_tech_colors: geothermal heat pump
        "fossil primary energy sources": "#8B4513",
        "PV": "#f9d002",  # solar (from tech_colors)
        "Wind power": "#83D8FF",  # from override_tech_colors: wind power
        # PTES / TTES
        "PTES": "#000F92C1",  # urban central water pits (from override_tech_colors)
        "PTES heat pump": "#3498db",  # urban central ptes heat pump (from carriers)
        "TTES": "#4a5cffc1",  # urban central water tanks (from override_tech_colors)
        # H2
        "H2 OCGT": "#8A2BE2",  # from override_tech_colors
        "H2 CHP": "purple",  # from override_tech_colors
        "H2 Electrolysis": "#ff29d9",  # from tech_colors
        # Heat technologies
        "Air heat pump": "#6cfb6b",  # urban central air heat pump
        "Resistive heater": "#28DBA6",  # from override_tech_colors
        "River water heat pump": "#FF9A03",  # from override_tech_colors (urban central river_water)
        # Storage
        "Battery": "#999999",  # grey (matches reference figure)
    }

    for tech, color in default_colors.items():
        if tech in technologies and tech not in colors:
            colors[tech] = color

    # Grouped technology display colors: always override carrier colors to ensure
    # consistent appearance regardless of what the network carrier DataFrame contains.
    grouped_display_colors = {
        "Wind power": "#83D8FF",        # from override_tech_colors: wind power
        "geothermal heat pumps": "khaki",  # from override_tech_colors: geothermal heat pump
        "Battery": "#999999",           # grey (matches reference figure)
    }
    for tech, color in grouped_display_colors.items():
        if tech in technologies:
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
    # Make everything bn
    costs_df = costs_df / 1e9
    if not savings_df.empty:
        savings_df = savings_df / 1e9
    # Extract supply temperatures from scenario names
    temp_candidates = []
    for s in costs_df.index:
        parts = s.split("_")
        for part in parts:
            if "SupplyTemperature" in part:
                temp_candidates.append(part)
    # Sort supply temperatures from high to low
    temp_order = {
        "HighSupplyTemperature": 0,
        "MidSupplyTemperature": 1,
        "LowSupplyTemperature": 2,
    }
    SUPPLY_TEMPS = sorted(
        list(set(temp_candidates)), key=lambda x: temp_order.get(x, 999)
    )
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

    # Create figure with two side-by-side axes for better use of vertical space
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 8), gridspec_kw={"width_ratios": [0.45, 2.0], "wspace": 0.45}
    )

    # Plot baseline costs for all supply temperatures on left axis
    baseline_scenarios = []
    baseline_costs_list = []
    temp_labels = []

    for i, supply_temp in enumerate(SUPPLY_TEMPS):
        # Temperature label for x-axis
        temp_label = supply_temp.replace("SupplyTemperature", " ST")
        temp_labels.append(temp_label)

        # Find baseline scenario - exclude _init suffix
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

        baseline_scenarios.append(baseline_scenario)
        baseline_costs = costs_agg.loc[baseline_scenario]
        baseline_costs_list.append(baseline_costs)

    # Plot all baseline costs on left axis
    max_german_cost = 0
    for i, (supply_temp, baseline_scenario, baseline_costs) in enumerate(
        zip(SUPPLY_TEMPS, baseline_scenarios, baseline_costs_list)
    ):
        # Calculate German costs for ylim (excluding EU countries)
        german_cost = sum(
            v for k, v in baseline_costs.items() if k != "EU aggregated" and v > 0
        )
        max_german_cost = max(max_german_cost, german_cost)

        # Create stacked bar for baseline costs with EU countries at top
        bottom = 0
        logger.info(f"Plotting baseline for {supply_temp}: {baseline_scenario}")
        logger.info(
            f"Available technologies in baseline_costs: {list(baseline_costs.index)}"
        )
        logger.info(f"Ordered technologies: {ordered_techs}")

        # Calculate maximum total cost for proper scaling
        max_total_cost = max_german_cost + max(
            baseline_costs.get("EU aggregated", 0)
            for baseline_costs in baseline_costs_list
        )

        # Function to transform y-values for broken axis
        def transform_y_for_break(y_val):
            break_start = 160
            break_restart = 750
            compression_range = 50

            if y_val <= break_start:
                return y_val
            elif y_val >= break_restart and max_total_cost > break_restart:
                # Map values above 750 to compressed range starting at 160
                scale_factor = compression_range / max(
                    1, max_total_cost - break_restart
                )
                return break_start + (y_val - break_restart) * scale_factor
            else:
                # Values between 160-750 get mapped to the break point
                return break_start

        for tech in ordered_techs:
            if tech in baseline_costs.index and abs(baseline_costs[tech]) > 1e-6:
                value = baseline_costs[tech]

                # Transform bottom and height for broken axis
                transformed_bottom = transform_y_for_break(bottom)
                transformed_top = transform_y_for_break(bottom + value)
                transformed_height = transformed_top - transformed_bottom

                if transformed_height > 0:  # Only plot if visible after transformation
                    ax_left.bar(
                        [i],
                        [transformed_height],
                        bottom=transformed_bottom,
                        color=colors.get(tech, "gray"),
                        label=(
                            format_label(tech) if i == 0 else ""
                        ),  # Only label first occurrence
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

        # Add total system cost and DE cost annotations
        total_system_cost = sum(
            baseline_costs[tech]
            for tech in baseline_costs.index
            if abs(baseline_costs[tech]) > 1e-6
        )

        # Calculate DE (German) costs only
        de_cost = sum(
            baseline_costs[tech]
            for tech in baseline_costs.index
            if tech != "EU aggregated" and abs(baseline_costs[tech]) > 1e-6
        )

        # Position annotation lower in the plot area to fit in the box
        # Use a position that's about 85% of the way up the plot
        y_position = ax_left.get_ylim()[1] * 0.85

        # Create annotation text with linebreaks like right plot
        annotation_text = f"{total_system_cost:.0f} (total)\n{de_cost:.0f} (DE)"

        # Add annotation
        ax_left.annotate(
            annotation_text,
            (i, y_position),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=8,
            rotation=90,
            fontweight="bold",
            zorder=20,
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )

    # Set ylim and formatting for left axis
    if baseline_costs_list:
        # Check if any baseline has EU countries
        has_eu_countries = any(
            baseline_costs.get("EU aggregated", 0) > 0
            for baseline_costs in baseline_costs_list
        )

        if has_eu_countries:
            # Implement proper axis break at 160 (break starts) and restart at 750
            # Calculate total system cost with neighbors
            total_cost_with_neighbors = max_german_cost + max(
                baseline_costs.get("EU aggregated", 0)
                for baseline_costs in baseline_costs_list
            )

            # Create broken axis: show 0-160, then compressed space for 750-total
            break_start = 160
            break_restart = 750

            # Map the actual total cost to the compressed scale
            # The upper portion (750+) gets compressed into the space from 160 to 210
            compression_range = 50  # Space allocated for the upper range
            if total_cost_with_neighbors > break_restart:
                # Scale the upper portion to fit in the compression range
                scale_factor = compression_range / (
                    total_cost_with_neighbors - break_restart
                )
                upper_limit = break_start + compression_range
            else:
                # If total cost is below break_restart, just show normal scale up to 210
                upper_limit = 210

            ax_left.set_ylim(0, 1.05 * upper_limit)

            # Create custom y-ticks
            # Lower part: 0, 50, 100, 160
            lower_ticks = [0, 50, 100, 160]
            lower_labels = ["0", "50", "100", "160"]

            # Upper part: show the actual total cost
            if total_cost_with_neighbors > break_restart:
                # Position for the total cost in compressed scale
                total_position = (
                    break_start
                    + (total_cost_with_neighbors - break_restart) * scale_factor
                )
                upper_ticks = [total_position]
                upper_labels = [f"{total_cost_with_neighbors:.0f}"]
            else:
                upper_ticks = []
                upper_labels = []

            all_ticks = lower_ticks + upper_ticks
            all_labels = lower_labels + upper_labels

            ax_left.set_yticks(all_ticks)
            ax_left.set_yticklabels(all_labels)

            # Add dashed horizontal line at break restart (750)
            if total_cost_with_neighbors > break_restart:
                restart_position = (
                    break_start + (break_restart - break_restart) * scale_factor
                )  # This equals break_start

            # Add axis break indicators at y=180 (moved from 160)
            break_y = 180
            break_width = 0.015  # Width of break markers
            break_height = 3  # Height of break markers

            # Draw diagonal break lines
            for x_offset in [-break_width, break_width]:
                ax_left.plot(
                    [x_offset - break_width / 2, x_offset + break_width / 2],
                    [break_y - break_height / 2, break_y + break_height / 2],
                    color="black",
                    linewidth=2,
                    clip_on=False,
                    transform=ax_left.get_yaxis_transform(),
                )
        else:
            # If no EU countries, just show German costs with increased margin for better visibility
            ax_left.set_ylim(0, 1.5 * max_german_cost)

    ax_left.set_xlim(-0.5, len(SUPPLY_TEMPS) - 0.5)
    ax_left.set_xticks(range(len(SUPPLY_TEMPS)))
    ax_left.set_xticklabels(
        temp_labels, rotation=90, ha="center", fontsize=13, fontweight="bold"
    )
    ax_left.set_ylabel("System Costs [bn EUR a$^{-1}$]", labelpad=5, fontsize=11)
    ax_left.grid(True, alpha=0.3)

    # Plot all PTES savings on right axis
    # Collect all PTES scenarios across all supply temperatures
    all_ptes_scenarios = []
    bar_positions = []
    temp_group_positions = []
    current_pos = 0

    for temp_idx, supply_temp in enumerate(SUPPLY_TEMPS):
        # Find PTES scenarios for this temperature
        ptes_scenarios_unsorted = [
            s for s in savings_agg.index if isinstance(s, tuple) and s[0] == supply_temp
        ]

        # Order PTES scenarios as in PTES_CONFIGS: freeboost, freecap, rhboost, hpboost_10Cbottom
        if ptes_scenarios_unsorted and not savings_agg.empty:
            desired_order = PTES_CONFIGS
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

        # Store scenarios and positions
        temp_start_pos = current_pos
        for scenario in ptes_scenarios:
            all_ptes_scenarios.append(scenario)
            bar_positions.append(current_pos)
            current_pos += 1

        # Store temperature group center position for secondary x-labels
        temp_group_positions.append((temp_start_pos + current_pos - 1) / 2)

        # Add gap between temperature groups (except after last group)
        if temp_idx < len(SUPPLY_TEMPS) - 1:
            current_pos += 0.5

    # Plot PTES scenarios if we have any
    if all_ptes_scenarios and not savings_agg.empty:
        # Calculate tech contributions for sorting
        temp_tech_contributions = {}
        for tech in ordered_techs:
            if tech != "EU aggregated":
                avg_contribution = 0
                count = 0
                for scenario in all_ptes_scenarios:
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

        # Store total savings for star markers and annotations
        total_savings_per_scenario = []
        bar_top_positions = []

        for j, scenario in enumerate(all_ptes_scenarios):
            scenario_savings = savings_agg.loc[scenario]
            pos = bar_positions[j]

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
                            [pos],
                            [value],
                            bottom=bottom_pos,
                            color=colors.get(tech, "gray"),
                            edgecolor="none",
                            width=0.6,
                        )
                        bottom_pos += value
                    else:
                        ax_right.bar(
                            [pos],
                            [value],
                            bottom=bottom_neg,
                            color=colors.get(tech, "gray"),
                            edgecolor="none",
                            width=0.6,
                        )
                        bottom_neg += value

            # Store the highest point for this scenario
            bar_top_positions.append(max(bottom_pos, abs(bottom_neg)))

        # Add star markers for total net differences
        ax_right.scatter(
            bar_positions,
            total_savings_per_scenario,
            marker="*",
            s=200,
            facecolor="white",
            edgecolor="black",
            linewidth=2,
            zorder=10,
            label=format_label("Net system cost difference"),
        )

        # No connecting line for star markers — discrete scenarios

        # Add annotations for relative changes
        if baseline_scenarios and baseline_costs_list:
            for j, (scenario, total_saving, pos) in enumerate(
                zip(all_ptes_scenarios, total_savings_per_scenario, bar_positions)
            ):
                # Find corresponding baseline for this temperature
                supply_temp = (
                    scenario[0]
                    if isinstance(scenario, tuple)
                    else scenario.split("_")[0]
                )
                baseline_scenario = None
                baseline_costs = None

                for k, temp in enumerate(SUPPLY_TEMPS):
                    if temp == supply_temp and k < len(baseline_scenarios):
                        baseline_scenario = baseline_scenarios[k]
                        baseline_costs = baseline_costs_list[k]
                        break

                if baseline_scenario and baseline_costs is not None:
                    baseline_german_cost = sum(
                        v for k, v in baseline_costs.items() if k != "EU aggregated"
                    )
                    pct_vs_german = (
                        (total_saving / baseline_german_cost) * 100
                        if baseline_german_cost
                        else 0
                    )

                    # Position both annotations in parallel above each bar
                    y_position_base = ax_right.get_ylim()[1] * 0.85

                    # Black annotation: German system cost % change (left position)
                    ax_right.annotate(
                        f"{pct_vs_german:+.1f}%",
                        (pos - 0.15, y_position_base),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="black",
                        rotation=90,
                        zorder=20,
                        path_effects=[
                            path_effects.withStroke(linewidth=2, foreground="white")
                        ],
                    )

                    # Red annotation: ΔDH price % (right position, adjacent to black)
                    if dh_prices:
                        bdp = None
                        for sc in dh_prices:
                            if (
                                f"NoPTES_{supply_temp}" in sc
                                and "_init" not in sc
                            ):
                                bdp = dh_prices[sc]
                                break
                        if bdp is None:
                            for sc in dh_prices:
                                if f"NoPTES_{supply_temp}" in sc:
                                    bdp = dh_prices[sc]
                                    break
                        ptes_sname = (
                            f"{scenario[0]}_MidDH_{scenario[1]}"
                            if isinstance(scenario, tuple)
                            else str(scenario)
                        )
                        pdp = dh_prices.get(ptes_sname, None)
                        if bdp and bdp > 0 and pdp is not None:
                            dh_pct = ((pdp - bdp) / bdp) * 100
                            ax_right.annotate(
                                f"Δ{dh_pct:+.1f}%",
                                (pos + 0.15, y_position_base),
                                xytext=(0, 0),
                                textcoords="offset points",
                                ha="center",
                                va="bottom",
                                fontsize=8,
                                color="red",
                                rotation=90,
                                zorder=20,
                                path_effects=[
                                    path_effects.withStroke(
                                        linewidth=2, foreground="white"
                                    )
                                ],
                            )

        # Add vertical separators between temperature groups
        for temp_idx in range(len(SUPPLY_TEMPS) - 1):
            # Find the boundary between temperature groups
            temp_end_pos = temp_group_positions[temp_idx] + (
                4 / 2
            )  # Assuming 4 scenarios per temp
            ax_right.axvline(
                x=temp_end_pos + 0.25, color="gray", linestyle="--", alpha=0.5
            )

        # Set x-axis for PTES plots
        ax_right.set_xlim(-0.5, max(bar_positions) + 0.5)
        ax_right.set_xticks(bar_positions)

        # Create scenario labels
        scenario_labels = []
        for scenario in all_ptes_scenarios:
            scenario_name = (
                scenario[1] if isinstance(scenario, tuple) else scenario.split("_")[-1]
            )
            scenario_labels.append(PTES_CONFIG_LABELS.get(scenario_name, scenario_name))

        ax_right.set_xticklabels(scenario_labels, fontsize=11, rotation=90, ha="center")

        # Add secondary x-axis for temperature labels
        ax_right_temp = ax_right.twiny()
        ax_right_temp.set_xlim(ax_right.get_xlim())
        ax_right_temp.set_xticks(temp_group_positions)
        ax_right_temp.set_xticklabels(
            [temp.replace("SupplyTemperature", " ST") for temp in SUPPLY_TEMPS],
            fontsize=13,
            fontweight="bold",
        )
        ax_right_temp.tick_params(axis="x", which="major", pad=25)

        ax_right.axhline(y=0, color="black", linewidth=0.8)
        ax_right.set_ylabel("Cost Savings [bn EUR a$^{-1}$]", labelpad=5, fontsize=11)
        ax_right.grid(True, alpha=0.3)
        ax_right.set_xlabel("PTES configurations", fontweight="bold", fontsize=11)

        # Set y-limits — dynamic, fit all stacked bars plus 10% headroom
        if total_savings_per_scenario:
            max_stacked_pos = 0.0
            min_stacked_neg = 0.0
            for idx in savings_agg.index:
                row = savings_agg.loc[idx]
                max_stacked_pos = max(max_stacked_pos, sum(v for v in row.values if v > 0))
                min_stacked_neg = min(min_stacked_neg, sum(v for v in row.values if v < 0))
            y_pad = max(0.10 * (max_stacked_pos - min_stacked_neg), 0.10)
            ax_right.set_ylim(min_stacked_neg - y_pad, max_stacked_pos + y_pad)

    else:
        # If no PTES scenarios, hide the right plot
        ax_right.set_visible(False)

    # Add district heating price information
    if dh_prices:
        # Left axis - baseline DH prices
        ax_left_dh = ax_left.twinx()
        baseline_dh_prices = []

        for i, supply_temp in enumerate(SUPPLY_TEMPS):
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
                baseline_dh_prices.append(baseline_dh_price)
                # Plot circle marker for DH price
                ax_left_dh.plot(
                    [i],
                    [baseline_dh_price],
                    "o",
                    color="white",
                    markersize=6,
                    markeredgecolor="red",
                    markeredgewidth=1.5,
                    label="DH price (baseline)" if i == 0 else "",
                )

        # Add connecting line for baseline DH prices
        if len(baseline_dh_prices) > 1:
            ax_left_dh.plot(
                range(len(baseline_dh_prices)),
                baseline_dh_prices,
                "r-",
                linewidth=1,
                alpha=0.7,
            )

        if baseline_dh_prices:
            ax_left_dh.set_ylabel("DH Price [EUR MWh$^{-1}$]", color="red")
            ax_left_dh.tick_params(axis="y", labelcolor="red")
            ax_left_dh.set_ylim(0, max(baseline_dh_prices) * 1.4)

        # Right axis - DH price deltas
        if all_ptes_scenarios:
            ax_right_dh = ax_right.twinx()
            dh_deltas = []

            for scenario in all_ptes_scenarios:
                # Find corresponding baseline for this temperature
                supply_temp = (
                    scenario[0]
                    if isinstance(scenario, tuple)
                    else scenario.split("_")[0]
                )
                baseline_dh_price = None
                for dh_scenario in dh_prices.keys():
                    if (
                        f"NoPTES_{supply_temp}" in dh_scenario
                        and "_init" not in dh_scenario
                    ):
                        baseline_dh_price = dh_prices[dh_scenario]
                        break
                if baseline_dh_price is None:
                    for dh_scenario in dh_prices.keys():
                        if f"NoPTES_{supply_temp}" in dh_scenario:
                            baseline_dh_price = dh_prices[dh_scenario]
                            break

                # Get scenario DH price
                if isinstance(scenario, tuple):
                    scenario_name = f"{scenario[0]}_MidDH_{scenario[1]}"
                else:
                    scenario_name = scenario
                scenario_dh_price = dh_prices.get(scenario_name, baseline_dh_price)

                delta = (
                    (scenario_dh_price - baseline_dh_price) if baseline_dh_price else 0
                )
                dh_deltas.append(delta)

            # Plot triangle markers for DH price deltas
            if dh_deltas:
                ax_right_dh.plot(
                    bar_positions,
                    dh_deltas,
                    "^",
                    color="white",
                    markersize=8,
                    markeredgecolor="red",
                    markeredgewidth=1.5,
                    label="ΔDH price",
                )
                # No connecting line — markers are discrete (one per scenario)

                ax_right_dh.set_ylabel(
                    "ΔDH Price [EUR MWh$^{-1}$]", color="red", labelpad=5
                )
                ax_right_dh.tick_params(axis="y", labelcolor="red")
                # Align secondary 0 with primary 0: dh_lo/dh_hi = p_lo/p_hi
                p_lo, p_hi = ax_right.get_ylim()
                if p_hi > 0 and p_lo < 0:
                    max_pos = max((d for d in dh_deltas if d > 0), default=0)
                    max_neg_mag = max((-d for d in dh_deltas if d < 0), default=0)
                    dh_hi = max(max_pos, max_neg_mag * p_hi / (-p_lo)) * 1.3
                    dh_hi = max(dh_hi, 0.5)  # minimum range
                    dh_lo = dh_hi * p_lo / p_hi
                else:
                    dh_abs_max = max((abs(d) for d in dh_deltas), default=1.0)
                    dh_hi = dh_abs_max * 1.5
                    dh_lo = -dh_hi
                ax_right_dh.set_ylim(dh_lo, dh_hi)
                ax_right_dh.axhline(
                    y=0, color="red", linestyle="--", alpha=0.5, linewidth=1
                )

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
            labels.append(f"  {format_label(tech)}")

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
        labels.append("  " + format_label("other technologies"))

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
        labels.append("  " + format_label("EU aggregated"))

    # Other indicators category
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
    labels.append(r"$\mathbf{Other\ indicators:}$")

    # Add net difference star marker
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markeredgecolor="black",
            markeredgewidth=2,
            markersize=10,
            markerfacecolor="white",
            linestyle="None",
        )
    )
    labels.append("  " + format_label("Net system cost difference"))

    # Add DH price markers
    if dh_prices:
        # Add baseline DH price marker
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="red", markersize=8, linestyle="None"
            )
        )
        labels.append("  " + format_label("DH price (baseline)"))

        # Add DH price delta marker
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="red",
                markersize=8,
                linestyle="None",
            )
        )
        labels.append("  " + format_label("ΔDH price"))

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=4,
            frameon=False,
            columnspacing=1.5,
            fontsize=11,
            handlelength=1.5,
            handleheight=1.0,
        )

    # Add titles with proper positioning
    ax_left.set_title("NoPTES Baseline Costs", fontweight="bold", pad=20, fontsize=12)
    ax_right.set_title("Net Cost Difference vs NoPTES", fontweight="bold", pad=20, fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35, left=0.07, right=0.95, top=0.90)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to: {output_path}")


def create_sensitivity_plots(
    costs_df,
    savings_df,
    colors,
    output_path,
    dh_prices=None,
    threshold=0.05,
    supply_temps=None,
    temp_labels=None,
):
    """Create PTES sensitivity plots grouped by PTES config (not by supply temperature).

    Left panel: NoPTES baseline costs for each supply temperature group.
    Right panel: 4 groups (one per PTES config), each with N side-by-side bars for
                 supply temperatures, enabling direct comparison across temperature levels.

    Parameters
    ----------
    supply_temps : list of str, optional
        Ordered list of supply-temperature tokens as they appear in scenario names
        (e.g. ["TopTSens_85", "MidSupplyTemperature", "TopTSens_95"]).
        If None, extracted automatically from "SupplyTemperature" substrings.
    temp_labels : list of str, optional
        Human-readable x-tick labels matching supply_temps order
        (e.g. ["85°C", "90°C", "95°C"]).
        If None, derived by stripping "SupplyTemperature" from supply_temps entries.
    """
    logger = logging.getLogger(__name__)

    if costs_df.empty:
        logger.warning("Empty costs dataframe, cannot create plots")
        return

    costs_df = costs_df / 1e9
    if not savings_df.empty:
        savings_df = savings_df / 1e9

    # Extract supply temperatures (use provided list or auto-detect)
    if supply_temps is None:
        temp_candidates = []
        for s in costs_df.index:
            for part in s.split("_"):
                if "SupplyTemperature" in part:
                    temp_candidates.append(part)
        temp_order = {
            "HighSupplyTemperature": 0,
            "MidSupplyTemperature": 1,
            "LowSupplyTemperature": 2,
        }
        supply_temps = sorted(
            set(temp_candidates), key=lambda x: temp_order.get(x, 999)
        )

    if temp_labels is None:
        temp_labels = [st.replace("SupplyTemperature", " ST") for st in supply_temps]

    # Aggregate technologies
    costs_agg, savings_agg = aggregate_technologies_by_significance(
        costs_df, savings_df, threshold
    )

    all_techs = set(costs_agg.columns)
    if not savings_agg.empty:
        all_techs.update(savings_agg.columns)

    # Technology order by baseline contribution
    baseline_contribution = {}
    for tech in all_techs:
        if tech != "EU aggregated":
            vals = []
            for st in supply_temps:
                for sc in costs_agg.index:
                    if (
                        f"NoPTES_{st}" in sc
                        and "_init" not in sc
                        and tech in costs_agg.columns
                    ):
                        vals.append(costs_agg.loc[sc, tech])
                        break
            baseline_contribution[tech] = sum(vals) / max(len(vals), 1)

    sorted_techs = sorted(
        baseline_contribution, key=lambda x: baseline_contribution[x], reverse=True
    )
    ordered_techs = list(sorted_techs)
    if "EU aggregated" in all_techs:
        ordered_techs.append("EU aggregated")
    for tech in all_techs:
        if tech not in ordered_techs:
            ordered_techs.insert(
                -1 if "EU aggregated" in ordered_techs else len(ordered_techs), tech
            )

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 8), gridspec_kw={"width_ratios": [0.5, 2.0], "wspace": 0.45}
    )

    # ---- LEFT PANEL (same as create_plots) ----
    baseline_scenarios = []
    baseline_costs_list = []

    for supply_temp in supply_temps:
        for sc in costs_agg.index:
            if f"NoPTES_{supply_temp}" in sc and "_init" not in sc:
                baseline_scenarios.append(sc)
                baseline_costs_list.append(costs_agg.loc[sc])
                break

    max_german_cost = 0
    for bc in baseline_costs_list:
        german_cost = sum(v for k, v in bc.items() if k != "EU aggregated" and v > 0)
        max_german_cost = max(max_german_cost, german_cost)

    max_total_cost = max_german_cost + max(
        (bc.get("EU aggregated", 0) for bc in baseline_costs_list), default=0
    )

    def transform_y_for_break(
        y_val, break_start=160, break_restart=750, compression_range=50
    ):
        if y_val <= break_start:
            return y_val
        elif y_val >= break_restart and max_total_cost > break_restart:
            scale_factor = compression_range / max(1, max_total_cost - break_restart)
            return break_start + (y_val - break_restart) * scale_factor
        else:
            return break_start

    for i, (supply_temp, baseline_scenario, baseline_costs) in enumerate(
        zip(supply_temps, baseline_scenarios, baseline_costs_list)
    ):
        bottom = 0
        for tech in ordered_techs:
            if tech in baseline_costs.index and abs(baseline_costs[tech]) > 1e-6:
                value = baseline_costs[tech]
                tb = transform_y_for_break(bottom)
                tt = transform_y_for_break(bottom + value)
                th = tt - tb
                if th > 0:
                    ax_left.bar(
                        [i],
                        [th],
                        bottom=tb,
                        color=colors.get(tech, "gray"),
                        label=(format_label(tech) if i == 0 else ""),
                        edgecolor="none",
                        width=0.6,
                    )
                bottom += value

        total_system_cost = sum(
            baseline_costs[t]
            for t in baseline_costs.index
            if abs(baseline_costs[t]) > 1e-6
        )
        de_cost = sum(
            baseline_costs[t]
            for t in baseline_costs.index
            if t != "EU aggregated" and abs(baseline_costs[t]) > 1e-6
        )
        y_pos = ax_left.get_ylim()[1] * 0.85
        ax_left.annotate(
            f"{total_system_cost:.0f} (total)\n{de_cost:.0f} (DE)",
            (i, y_pos),
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=10,
            rotation=90,
            fontweight="bold",
            zorder=20,
            path_effects=[path_effects.withStroke(linewidth=2, foreground="white")],
        )

    has_eu = any(bc.get("EU aggregated", 0) > 0 for bc in baseline_costs_list)
    if has_eu:
        break_start, break_restart, compression_range = 160, 750, 50
        if max_total_cost > break_restart:
            scale_factor = compression_range / (max_total_cost - break_restart)
            upper_limit = break_start + compression_range
        else:
            upper_limit = 210
        ax_left.set_ylim(0, 1.05 * upper_limit)
        lower_ticks = [0, 50, 100, 160]
        lower_labels = ["0", "50", "100", "160"]
        if max_total_cost > break_restart:
            tp = break_start + (max_total_cost - break_restart) * scale_factor
            all_ticks = lower_ticks + [tp]
            all_labels = lower_labels + [f"{max_total_cost:.0f}"]
        else:
            all_ticks, all_labels = lower_ticks, lower_labels
        ax_left.set_yticks(all_ticks)
        ax_left.set_yticklabels(all_labels)
        # Axis break indicators
        break_y = 180
        for x_offset in [-0.015, 0.015]:
            ax_left.plot(
                [x_offset - 0.0075, x_offset + 0.0075],
                [break_y - 1.5, break_y + 1.5],
                color="black",
                linewidth=2,
                clip_on=False,
                transform=ax_left.get_yaxis_transform(),
            )
    else:
        ax_left.set_ylim(0, 1.5 * max_german_cost)

    ax_left.set_xlim(-0.5, len(supply_temps) - 0.5)
    ax_left.set_xticks(range(len(supply_temps)))
    ax_left.set_xticklabels(
        temp_labels, rotation=90, ha="center", fontsize=13, fontweight="bold"
    )
    ax_left.set_ylabel("System Costs [bn EUR a$^{-1}$]", labelpad=5, fontsize=11)
    ax_left.set_title("NoPTES Baseline Costs", fontweight="bold", pad=20, fontsize=12)
    ax_left.grid(True, alpha=0.3)

    # DH prices on left axis
    ax_left_dh = None
    baseline_dh_prices = []
    if dh_prices:
        ax_left_dh = ax_left.twinx()
        for i, supply_temp in enumerate(supply_temps):
            bdp = None
            for sc in dh_prices:
                if f"NoPTES_{supply_temp}" in sc and "_init" not in sc:
                    bdp = dh_prices[sc]
                    break
            if bdp is None:
                for sc in dh_prices:
                    if f"NoPTES_{supply_temp}" in sc:
                        bdp = dh_prices[sc]
                        break
            if bdp is not None and bdp > 0:
                baseline_dh_prices.append(bdp)
                ax_left_dh.plot(
                    [i],
                    [bdp],
                    "o",
                    color="white",
                    markersize=6,
                    markeredgecolor="red",
                    markeredgewidth=1.5,
                    label="DH price (baseline)" if i == 0 else "",
                )
        if len(baseline_dh_prices) > 1:
            ax_left_dh.plot(
                range(len(baseline_dh_prices)),
                baseline_dh_prices,
                "r-",
                linewidth=1,
                alpha=0.7,
            )
        if baseline_dh_prices:
            ax_left_dh.set_ylabel("DH Price [EUR MWh$^{-1}$]", color="red")
            ax_left_dh.tick_params(axis="y", labelcolor="red")
            ax_left_dh.set_ylim(0, max(baseline_dh_prices) * 1.4)

    # ---- RIGHT PANEL: grouped by PTES config ----
    if savings_agg.empty:
        ax_right.set_visible(False)
    else:
        # Build bar positions: 4 groups (configs) × 3 bars (supply temps), with gaps between groups
        bar_positions = []
        config_group_positions = []  # Center of each config group
        current_pos = 0

        # Determine which PTES configs are available in the savings data
        available_configs = []
        for config in PTES_CONFIGS:
            for idx in savings_agg.index:
                if isinstance(idx, tuple) and idx[1] == config:
                    available_configs.append(config)
                    break

        for ci, config in enumerate(available_configs):
            group_start = current_pos
            for st in supply_temps:
                bar_positions.append((config, st, current_pos))
                current_pos += 1
            config_group_positions.append((group_start + current_pos - 1) / 2)
            if ci < len(available_configs) - 1:
                current_pos += 1.0  # gap between groups

        # Sort technologies by savings contribution for right plot
        temp_tech_contributions = {}
        for tech in ordered_techs:
            if tech != "EU aggregated":
                vals = [
                    abs(savings_agg.loc[idx, tech])
                    for idx in savings_agg.index
                    if tech in savings_agg.columns
                ]
                temp_tech_contributions[tech] = sum(vals) / max(len(vals), 1)
        sorted_net_techs = sorted(
            [t for t in temp_tech_contributions],
            key=lambda x: temp_tech_contributions[x],
            reverse=True,
        )
        if "EU aggregated" in ordered_techs:
            sorted_net_techs.append("EU aggregated")

        total_savings_per_bar = []
        all_bar_x = []

        for config, supply_temp, pos in bar_positions:
            all_bar_x.append(pos)
            # Find the savings row for (supply_temp, config)
            idx = (supply_temp, config)
            if idx not in savings_agg.index:
                total_savings_per_bar.append(0)
                continue

            scenario_savings = savings_agg.loc[idx]
            bottom_pos = 0
            bottom_neg = 0

            total_net = sum(
                scenario_savings[t]
                for t in scenario_savings.index
                if abs(scenario_savings[t]) > 1e-6
            )
            total_savings_per_bar.append(total_net)

            for tech in sorted_net_techs:
                if (
                    tech in scenario_savings.index
                    and abs(scenario_savings[tech]) > 1e-6
                ):
                    value = scenario_savings[tech]
                    if value > 0:
                        ax_right.bar(
                            [pos],
                            [value],
                            bottom=bottom_pos,
                            color=colors.get(tech, "gray"),
                            edgecolor="none",
                            width=0.45,
                        )
                        bottom_pos += value
                    else:
                        ax_right.bar(
                            [pos],
                            [value],
                            bottom=bottom_neg,
                            color=colors.get(tech, "gray"),
                            edgecolor="none",
                            width=0.45,
                        )
                        bottom_neg += value

        # Star markers for total savings (discrete — no connecting lines)
        ax_right.scatter(
            all_bar_x,
            total_savings_per_bar,
            marker="*",
            s=200,
            facecolor="white",
            edgecolor="black",
            linewidth=2,
            zorder=10,
            label=format_label("Net system cost difference"),
        )

        # Annotations: % vs baseline
        for (config, supply_temp, pos), total_saving in zip(
            bar_positions, total_savings_per_bar
        ):
            # Find baseline
            bc = None
            for sc, bcosts in zip(baseline_scenarios, baseline_costs_list):
                if f"NoPTES_{supply_temp}" in sc:
                    bc = bcosts
                    break
            if bc is not None:
                baseline_de = sum(v for k, v in bc.items() if k != "EU aggregated")
                pct_de = (total_saving / baseline_de) * 100 if baseline_de else 0

                y_range = ax_right.get_ylim()[1] - ax_right.get_ylim()[0]
                # Place both annotation lines adjacent — shared anchor, offset ±13pt
                y_annot = ax_right.get_ylim()[0] + y_range * 0.82

                # Black annotation: German system cost % change (upper line)
                ax_right.annotate(
                    f"{pct_de:.1f}%",
                    (pos, y_annot),
                    xytext=(0, 13),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    rotation=90,
                    zorder=20,
                    path_effects=[
                        path_effects.withStroke(linewidth=2, foreground="white")
                    ],
                )

                # Red annotation: ΔDH price % (lower line, adjacent to black)
                if dh_prices:
                    bdp = None
                    for sc in dh_prices:
                        if f"NoPTES_{supply_temp}" in sc and "_init" not in sc:
                            bdp = dh_prices[sc]
                            break
                    if bdp is None:
                        for sc in dh_prices:
                            if f"NoPTES_{supply_temp}" in sc:
                                bdp = dh_prices[sc]
                                break
                    pdp = bdp
                    for sc in dh_prices:
                        if (
                            supply_temp in sc
                            and f"_{config}" in sc
                            and "NoPTES" not in sc
                            and "_init" not in sc
                        ):
                            pdp = dh_prices[sc]
                            break
                    if bdp and bdp > 0 and pdp is not None:
                        dh_pct = ((pdp - bdp) / bdp) * 100
                        ax_right.annotate(
                            f"Δ{dh_pct:.1f}%",
                            (pos, y_annot),
                            xytext=(0, -13),
                            textcoords="offset points",
                            ha="center",
                            va="center",
                            fontsize=9,
                            color="red",
                            rotation=90,
                            zorder=20,
                            path_effects=[
                                path_effects.withStroke(
                                    linewidth=2, foreground="white"
                                )
                            ],
                        )

        # Vertical separators between config groups
        for ci in range(len(available_configs) - 1):
            # Find the boundary between groups
            last_pos_of_group = [
                pos for c, _, pos in bar_positions if c == available_configs[ci]
            ][-1]
            first_pos_of_next = [
                pos for c, _, pos in bar_positions if c == available_configs[ci + 1]
            ][0]
            sep_x = (last_pos_of_group + first_pos_of_next) / 2
            ax_right.axvline(x=sep_x, color="gray", linestyle="--", alpha=0.5)

        # Primary x-axis: supply temperature per bar (repeated across groups)
        ax_right.set_xlim(-0.5, max(all_bar_x) + 0.5)
        ax_right.set_xticks(all_bar_x)
        # Repeat supply temp labels for each group
        xtick_labels = []
        for config, supply_temp, pos in bar_positions:
            if supply_temp in supply_temps:
                xtick_labels.append(temp_labels[supply_temps.index(supply_temp)])
            else:
                xtick_labels.append(supply_temp.replace("SupplyTemperature", " ST"))
        ax_right.set_xticklabels(xtick_labels, fontsize=11, rotation=90, ha="center")

        # Secondary x-axis: PTES config names (one label per group)
        ax_right_cfg = ax_right.twiny()
        ax_right_cfg.set_xlim(ax_right.get_xlim())
        ax_right_cfg.set_xticks(config_group_positions)
        ax_right_cfg.set_xticklabels(
            [
                PTES_CONFIG_LABELS.get(c, c).replace("\n", " ")
                for c in available_configs
            ],
            fontsize=13,
            fontweight="bold",
        )
        ax_right_cfg.tick_params(axis="x", which="major", pad=25)

        ax_right.axhline(y=0, color="black", linewidth=0.8)
        ax_right.set_ylabel("Cost Savings [bn EUR a$^{-1}$]", labelpad=5, fontsize=11)
        ax_right.set_xlabel("Supply temperature", fontweight="bold", fontsize=11)
        ax_right.set_title("Net Cost Difference vs NoPTES", fontweight="bold", pad=20, fontsize=12)
        ax_right.grid(True, alpha=0.3)

        # Dynamic ylims — fit all stacked bars plus 15 % headroom
        max_stacked_pos = 0.0
        min_stacked_neg = 0.0
        for idx in savings_agg.index:
            row = savings_agg.loc[idx]
            max_stacked_pos = max(max_stacked_pos, sum(v for v in row.values if v > 0))
            min_stacked_neg = min(min_stacked_neg, sum(v for v in row.values if v < 0))
        y_pad = max(0.15 * (max_stacked_pos - min_stacked_neg), 0.15)
        ax_right.set_ylim(min_stacked_neg - y_pad, max_stacked_pos + y_pad)

        # DH price deltas on right axis
        if dh_prices:
            ax_right_dh = ax_right.twinx()
            dh_deltas = []
            for config, supply_temp, pos in bar_positions:
                bdp = None
                for sc in dh_prices:
                    if f"NoPTES_{supply_temp}" in sc and "_init" not in sc:
                        bdp = dh_prices[sc]
                        break
                # Find PTES scenario DH price (search flexibly to handle different naming conventions)
                ptes_dh = bdp
                for sc in dh_prices:
                    if (
                        supply_temp in sc
                        and f"_{config}" in sc
                        and "NoPTES" not in sc
                        and "_init" not in sc
                    ):
                        ptes_dh = dh_prices[sc]
                        break
                delta = (ptes_dh - bdp) if bdp and ptes_dh else 0
                dh_deltas.append(delta)

            ax_right_dh.plot(
                all_bar_x,
                dh_deltas,
                "^",
                color="white",
                markersize=8,
                markeredgecolor="red",
                markeredgewidth=1.5,
                label="ΔDH price",
            )
            # No connecting line — markers are discrete (one per scenario)
            ax_right_dh.set_ylabel(
                "ΔDH Price [EUR MWh$^{-1}$]", color="red", labelpad=5
            )
            ax_right_dh.tick_params(axis="y", labelcolor="red")
            # Align secondary 0 with primary 0: dh_lo/dh_hi = p_lo/p_hi
            p_lo, p_hi = ax_right.get_ylim()
            if p_hi > 0 and p_lo < 0:
                max_pos = max((d for d in dh_deltas if d > 0), default=0)
                max_neg_mag = max((-d for d in dh_deltas if d < 0), default=0)
                dh_hi = max(max_pos, max_neg_mag * p_hi / (-p_lo)) * 1.3
                dh_hi = max(dh_hi, 0.5)
                dh_lo = dh_hi * p_lo / p_hi
            else:
                dh_abs_max = max((abs(d) for d in dh_deltas), default=1.0)
                dh_hi = dh_abs_max * 1.5
                dh_lo = -dh_hi
            ax_right_dh.set_ylim(dh_lo, dh_hi)
            ax_right_dh.axhline(
                y=0, color="red", linestyle="--", alpha=0.5, linewidth=1
            )

    # ---- LEGEND (same structure as create_plots) ----
    handles, labels = [], []

    german_techs = [
        t for t in ordered_techs if t not in ("EU aggregated", "other technologies")
    ]
    if german_techs:
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
        labels.append(r"$\mathbf{German\ cost\ components:}$")
        for tech in german_techs:
            handles.append(
                plt.Rectangle(
                    (0, 0), 1, 1, facecolor=colors.get(tech, "gray"), edgecolor="white"
                )
            )
            labels.append(f"  {format_label(tech)}")

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
        labels.append("  " + format_label("other technologies"))

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
        labels.append("  " + format_label("EU aggregated"))

    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor="none"))
    labels.append(r"$\mathbf{Other\ indicators:}$")
    handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markeredgecolor="black",
            markeredgewidth=2,
            markersize=10,
            markerfacecolor="white",
            linestyle="None",
        )
    )
    labels.append("  " + format_label("Net system cost difference"))
    if dh_prices:
        handles.append(
            plt.Line2D(
                [0], [0], marker="o", color="red", markersize=8, linestyle="None"
            )
        )
        labels.append("  " + format_label("DH price (baseline)"))
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="^",
                color="red",
                markersize=8,
                linestyle="None",
            )
        )
        labels.append("  " + format_label("ΔDH price"))

    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.04),
            ncol=4,
            frameon=False,
            columnspacing=1.5,
            fontsize=11,
            handlelength=1.5,
            handleheight=1.0,
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35, left=0.07, right=0.95, top=0.90)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved sensitivity plot to: {output_path}")


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
