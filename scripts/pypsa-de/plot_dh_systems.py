#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
This script generates district heating system energy balance comparison plots
for PyPSA networks across different scenarios.
"""

import logging
import os
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pypsa
import re
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "code", "pypsa-de"))
from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def calc_dh_price_range_subnodes(n):
    """
    Calculate demand-weighted district heating prices for each system.

    Parameters:
    -----------
    n : pypsa.Network
        PyPSA network

    Returns:
    --------
    pd.Series
        Demand-weighted district heating prices indexed by system name
    """
    loads = n.loads_t.p.filter(
        regex=r"DE\d.*(urban central|low-temperature) heat"
    ).clip(lower=0)
    # Replace low-temperature heat for industry with urban central heat
    loads.columns = loads.columns.str.replace(
        "low-temperature heat for industry", "urban central heat"
    )
    loads = loads.T.groupby(loads.columns).sum().T
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d.*urban central heat")

    weighted_average_price_system = loads.mul(prices).sum().div(loads.sum())
    # Replace urban central heat with empty string
    weighted_average_price_system.index = (
        weighted_average_price_system.index.str.replace(" urban central heat", "")
    )
    return weighted_average_price_system


def get_colors(networks, override_colors={}):
    """
    Get colors from network and override with custom colors if provided.

    Parameters:
    -----------
    networks : dict
        Dictionary of networks by scenario and year
    override_colors : dict
        Dictionary of color overrides

    Returns:
    --------
    dict
        Color mapping for technologies
    """
    # Extract colors from the first network
    first_network = next(iter(networks.values()))[
        next(iter(networks[next(iter(networks))]))
    ]
    colors = first_network.carriers.color
    extended_index = colors.index.union(override_colors.keys())
    colors = colors.reindex(extended_index).fillna("black").to_dict()
    colors.update(override_colors)

    return colors


def process_networks(run_name, scenarios, planning_horizons):
    """
    Process networks and extract data for analysis.

    Parameters:
    -----------
    run_name : str
        Name of the run
    scenarios : list
        List of scenario names
    planning_horizons : list
        List of planning horizon years

    Returns:
    --------
    dict
        Dictionary of networks organized by scenario and year
    """
    networks = {}
    networks_path = os.path.join("results", run_name)

    logger.info(f"Processing networks for run: {run_name}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Planning horizons: {planning_horizons}")

    for scenario in scenarios:
        scenario_path = os.path.join(networks_path, scenario, "networks")
        if not os.path.exists(scenario_path):
            logger.warning(f"Scenario path {scenario_path} does not exist, skipping")
            continue
        networks[scenario] = {}

        for year in planning_horizons:
            # Find network file for this year
            network_file = None
            for file in os.listdir(scenario_path):
                if file.endswith(f"{year}.nc"):
                    network_file = os.path.join(scenario_path, file)
                    break

            if not network_file:
                logger.warning(
                    f"No network file found for scenario {scenario} and year {year}"
                )
                continue

            # Load network and calculate metrics
            try:
                logger.info(f"Loading network for scenario {scenario} and year {year}")
                n = pypsa.Network(network_file)

                networks[scenario][year] = n
                logger.info(
                    f"Successfully loaded network for scenario {scenario} and year {year}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing network for scenario {scenario} and year {year}: {e}"
                )

    if not networks:
        logger.error("No networks could be loaded")
        return None

    return networks


def plot_energy_balance_comparison(
    network1,
    network2,
    scenarios,
    output_path,
    colors,
    group_chp=False,
    group_heat_pumps=False,
    group_demands=False,
    drop_losses=False,
):
    """
    Plot comparison of energy balance for district heating between two networks.

    Parameters:
    -----------
    network1 : PyPSA Network
        First network (typically No_PTES scenario)
    network2 : PyPSA Network
        Second network (typically Baseline scenario)
    scenarios : list
        List of scenario names [scenario1, scenario2]
    output_path : str
        Path to save the output figure
    colors : dict
        Color mapping for technologies
    group_chp : bool, optional
        If True, group all CHP technologies as "CHP" (default: False)
    group_heat_pumps : bool, optional
        If True, group all heat pump technologies as "Heat Pumps" (default: False)
    group_demands : bool, optional
        If True, group all demand technologies as "District Heating Demand" (default: False)
    drop_losses : bool, optional
        If True, drop loss columns from the plot (default: False)

    Returns:
    --------
    tuple
        (figure, axes) matplotlib objects
    """
    plt.rcParams.update({"font.size": 10})
    title = f"Energy Balance Comparison: {scenarios[0]} vs {scenarios[1]}"

    def prepare_energy_balance_data(
        network,
        group_chp=False,
        group_heat_pumps=False,
        group_demands=False,
        drop_losses=False,
    ):
        """
        Prepare energy balance data for a single network.

        Parameters:
        -----------
        network : pypsa.Network
            PyPSA network
        group_chp : bool
            Whether to group CHP technologies
        group_heat_pumps : bool
            Whether to group heat pump technologies
        group_demands : bool
            Whether to group demand technologies
        drop_losses : bool
            Whether to drop loss columns

        Returns:
        --------
        tuple
            (energy_balance_data, district_heating_prices)
        """
        eb_uch = (
            network.statistics.energy_balance(groupby=["bus", "carrier", "bus_carrier"])
            .xs("urban central heat", level=3)
            .reset_index()
        )
        eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d \d+ .*urban"), :]

        # Strip 'urban central heat' from the bus index
        eb_uch["bus"] = eb_uch["bus"].str.replace(" urban central heat", "")

        eb_uch.drop("component", axis=1, inplace=True)

        # Remove " CC" suffix and aggregate
        eb_uch["carrier"] = eb_uch["carrier"].str.replace(" CC", "", regex=False)
        eb_uch = eb_uch.groupby(["bus", "carrier"], as_index=False).sum()

        # Set index and unstack the last level
        to_plot = eb_uch.set_index(["bus", "carrier"]).unstack(-1)
        to_plot.columns = to_plot.columns.droplevel(0)

        # Remove carriers contributing less than 1% in either system
        total_contribution = to_plot.abs().sum()
        to_plot = to_plot[
            total_contribution[
                total_contribution > 0.0001 * to_plot.abs().sum().sum()
            ].index
        ]

        # Sort columns by total energy
        dh_prices = calc_dh_price_range_subnodes(network)
        to_plot = to_plot.loc[dh_prices.sort_values().index]

        # Calculate relative values
        discharge = to_plot.filter(like=" discharger")
        discharge.columns = discharge.columns.str.replace(
            " discharger", " losses", regex=False
        )

        charge = to_plot.filter(like=" charger")
        charge.columns = charge.columns.str.replace(" charger", " losses", regex=False)

        losses = charge + discharge

        to_plot = pd.concat([to_plot, losses], axis=1)
        ch_to_drop = to_plot.filter(regex=r" charger|losses")
        disch_to_drop = to_plot.filter(regex=r"discharger")
        to_plot_rel_gen = (
            to_plot.clip(lower=0)
            .div(-to_plot.clip(upper=0).drop(ch_to_drop, axis=1).sum(axis=1), axis=0)
            .mul(100)
        )
        to_plot_rel_load = (
            to_plot.clip(upper=0)
            .div(to_plot.clip(upper=0).drop(ch_to_drop, axis=1).sum(axis=1), axis=0)
            .mul(-100)
        )

        to_plot_rel = to_plot_rel_load + to_plot_rel_gen

        # Note: Geothermal heat pumps are now grouped with other heat pumps when group_heat_pumps=True
        if not group_heat_pumps:
            # Only group geothermal separately if not grouping all heat pumps
            geothermal_techs = [
                tech
                for tech in to_plot_rel.columns
                if "urban central geothermal heat pump" in tech
                or "urban central geothermal heat" in tech
            ]
            if len(geothermal_techs) > 0:
                to_plot_rel["geothermal heat pump"] = to_plot_rel[geothermal_techs].sum(
                    axis=1
                )
                to_plot_rel = to_plot_rel.drop(columns=geothermal_techs)

        # Apply additional groupings if requested
        if group_chp:
            chp_techs = [
                tech
                for tech in to_plot_rel.columns
                if "chp" in tech.lower() or "combined heat" in tech.lower()
            ]
            if len(chp_techs) > 0:
                to_plot_rel["CHP"] = to_plot_rel[chp_techs].sum(axis=1)
                to_plot_rel = to_plot_rel.drop(columns=chp_techs)

        if group_heat_pumps:
            heat_pump_techs = [
                tech
                for tech in to_plot_rel.columns
                if "heat pump"
                in tech.lower()  # Include all heat pumps, including geothermal
            ]
            if len(heat_pump_techs) > 0:
                to_plot_rel["Heat Pumps"] = to_plot_rel[heat_pump_techs].sum(axis=1)
                to_plot_rel = to_plot_rel.drop(columns=heat_pump_techs)

        if group_demands:
            demand_techs = [
                tech
                for tech in to_plot_rel.columns
                if "low-temperature heat for industry" in tech.lower()
                or "urban central heat" in tech.lower()
            ]
            if len(demand_techs) > 0:
                to_plot_rel["District Heating Demand"] = to_plot_rel[demand_techs].sum(
                    axis=1
                )
                to_plot_rel = to_plot_rel.drop(columns=demand_techs)

        if drop_losses:
            loss_techs = [
                tech for tech in to_plot_rel.columns if "losses" in tech.lower()
            ]
            if len(loss_techs) > 0:
                to_plot_rel = to_plot_rel.drop(columns=loss_techs)

        return to_plot_rel, dh_prices

    # Prepare data for both networks
    to_plot_rel1, dh_prices1 = prepare_energy_balance_data(
        network1, group_chp, group_heat_pumps, group_demands, drop_losses
    )
    to_plot_rel2, dh_prices2 = prepare_energy_balance_data(
        network2, group_chp, group_heat_pumps, group_demands, drop_losses
    )

    # Calculate price savings (network1 - network2)
    dh_price_savings = dh_prices1 - dh_prices2

    # Sort systems by price savings (highest savings first)
    sorted_systems = dh_price_savings.sort_values(ascending=False).index

    # Reorder both plotting data and price data according to savings
    to_plot_rel1 = to_plot_rel1.loc[sorted_systems]
    to_plot_rel2 = to_plot_rel2.loc[sorted_systems]
    dh_price_savings = dh_price_savings.loc[sorted_systems]

    max_ylim = to_plot_rel2.clip(lower=0).sum(1).max() * 1.05

    # Create subplots with side-by-side layout (further increased height to prevent overlap)
    fig, axes = plt.subplots(1, 2, figsize=(10, 11), sharey=True)

    # Plot for Network 1 (left subplot)
    ax1 = axes[0]

    col_order = [
        # Center left (regular demand)
        "District Heating Demand",  # Grouped demands
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        # Extreme left (demand side, most negative) - Storage charging/losses
        "urban central water tanks",  # TTES (leftmost)
        "urban central water tanks charger",
        "urban central water tanks losses",
        "urban central water pits",  # PTES
        "urban central water pits charger",
        "urban central water pits losses",
        # Center right (regular supply) - ordered from center outward
        "Heat Pumps",  # Grouped heat pumps (including geothermal)
        "urban central electrolysis excess heat pump",
        "geothermal heat pump",
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
        "urban central resistive heater",
        "CHP",  # Grouped CHP
        "urban central gas CHP",
        "urban central solid biomass CHP",
        "urban central lignite CHP",
        "urban central coal CHP",
        "urban central oil CHP",
        "urban central H2 CHP",
        "H2 Electrolysis",
        "waste CHP",
        "urban central gas boiler",
        "Fischer-Tropsch",
        # Extreme right (supply side, most positive) - Storage discharging
        "urban central water tanks discharger",  # TTES discharging
        "urban central water tanks",  # TTES (rightmost)
        "urban central water pits discharger",  # PTES discharging
        "urban central water pits",  # PTES base
    ]
    # Filter to only include columns that exist in the data
    col_order = [c for c in col_order if c in to_plot_rel1.columns]
    # concat col_order with elements from to_plot_rel1 that are not in col_order
    col_order += [c for c in to_plot_rel1.columns if c not in col_order]
    to_plot_rel1 = to_plot_rel1[col_order]  # Align columns

    # Create plots with horizontal stacked bars
    to_plot_rel1.plot.barh(
        stacked=True,
        ax=ax1,
        color=colors,
        legend=False,
        width=0.9,  # Increase bar thickness to reduce white space
    )
    # Create cleaner scenario title
    title1 = scenarios[0].replace(
        "MidSupplyTemperature_MidDH_", "Medium Supply Temperature\nMedium DH\n"
    )
    title1 = title1.replace("hpboost", "heat pump boosting").replace(
        "rhboost", "resistive boosting"
    )
    title1 = title1.replace("noboost", "no boosting").replace(
        "_10Cbottom", " 10°C bottom"
    )
    ax1.set_title(title1, fontsize=10, pad=20, ha="center")
    ax1.set_xlabel("Share of district heating\nconsumption and supply\n[%]", fontsize=9)

    ax1.axvline(x=0, color="black", linestyle="-")
    ax1.set_xlim(-max_ylim, max_ylim)

    # Add secondary x-axis for district heating demand on first subplot
    ax1_demand = ax1.twiny()

    # Calculate district heating demand in TWh for each system using actual loads
    dh_demand = []
    for system in to_plot_rel1.index:
        system_name = system.replace(" urban central heat", "")  # Clean system name

        # Get urban central heat and low-temperature heat for industry loads
        uch_load = (
            network1.loads_t.p.filter(regex=f"{system_name}.*urban central heat")
            .sum()
            .sum()
        )
        industry_load = (
            network1.loads_t.p.filter(
                regex=f"{system_name}.*low-temperature heat for industry"
            )
            .sum()
            .sum()
        )

        # Convert from MWh to TWh and get absolute value
        total_demand_twh = abs(uch_load + industry_load) / 1e6
        dh_demand.append(total_demand_twh)

    dh_demand = pd.Series(dh_demand, index=to_plot_rel1.index)

    # Plot DH demand with squares (white with black border like price savings)
    y_positions1 = range(len(dh_demand))
    ax1_demand.scatter(
        dh_demand.values,
        y_positions1,
        s=40,
        marker="s",  # Square marker
        facecolor="white",
        edgecolor="black",
        linewidth=0.2,
        zorder=20,
        clip_on=False,
    )

    # Add mean DH demand line (dotted, white with black border like price savings)
    # First draw thick black dotted line as border
    ax1_demand.axvline(
        x=dh_demand.mean(),
        color="black",
        linestyle=":",
        linewidth=4,
        alpha=1,
        zorder=5,
    )
    # Then draw thinner white dotted line on top
    ax1_demand.axvline(
        x=dh_demand.mean(),
        color="white",
        linestyle=":",
        linewidth=2,
        alpha=1,
        zorder=6,
    )

    # Set labels for demand axis
    ax1_demand.set_xlabel("DH Demand [TWh]", fontsize=9)
    ax1_demand.tick_params(axis="x", labelsize=9)

    # Set x-limits for demand axis with some padding
    demand_min, demand_max = dh_demand.min(), dh_demand.max()
    demand_range = demand_max - demand_min
    if demand_range > 0:
        padding = demand_range * 0.1
        demand_xlim = (demand_min - padding, demand_max + padding)
    else:
        demand_xlim = (demand_min * 0.95, demand_max * 1.05)
    ax1_demand.set_xlim(demand_xlim)

    # Plot for Network 2 (right subplot)
    ax2 = axes[1]

    # Technology order with PTES/TTES on the very outsides (beyond ±100%)
    # Order: TTES -> PTES -> Demand -> 0 -> Supply -> Fischer-Tropsch -> PTES -> TTES
    col_order = [
        # Center left (regular demand)
        "District Heating Demand",  # Grouped demands
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        # Extreme left (demand side, most negative) - Storage charging/losses
        "urban central water tanks",  # TTES (leftmost)
        "urban central water tanks charger",
        "urban central water tanks losses",
        "urban central water pits",  # PTES
        "urban central water pits charger",
        "urban central water pits losses",
        # Center right (regular supply) - ordered from center outward
        "Heat Pumps",  # Grouped heat pumps (including geothermal)
        "urban central electrolysis excess heat pump",
        "geothermal heat pump",
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
        "urban central resistive heater",
        "CHP",  # Grouped CHP
        "urban central gas CHP",
        "urban central solid biomass CHP",
        "urban central lignite CHP",
        "urban central coal CHP",
        "urban central oil CHP",
        "urban central H2 CHP",
        "H2 Electrolysis",
        "waste CHP",
        "urban central gas boiler",
        "Fischer-Tropsch",
        # Extreme right (supply side, most positive) - Storage discharging
        "urban central water tanks discharger",  # TTES discharging
        "urban central water tanks",  # TTES (rightmost)
        "urban central water pits discharger",  # PTES discharging
        "urban central water pits",  # PTES base
    ]
    # Filter to only include columns that exist in the data
    col_order = [c for c in col_order if c in to_plot_rel2.columns]
    # concat col_order with elements from to_plot_rel2 that are not in col_order
    col_order += [c for c in to_plot_rel2.columns if c not in col_order]
    # Ensure the order of columns matches the first plot
    to_plot_rel2 = to_plot_rel2[col_order]  # Align columns

    to_plot_rel2.plot.barh(
        stacked=True,
        ax=ax2,
        color=colors,
        legend=False,
        width=0.9,  # Increase bar thickness to reduce white space
    )
    # Create cleaner scenario title
    title2 = scenarios[1].replace(
        "MidSupplyTemperature_MidDH_", "Medium Supply Temperature\nMedium DH\n"
    )
    title2 = title2.replace("hpboost", "heat pump boosting").replace(
        "rhboost", "resistive boosting"
    )
    title2 = title2.replace("noboost", "no boosting").replace(
        "_10Cbottom", " 10°C bottom"
    )
    ax2.set_title(title2, fontsize=10, pad=20, ha="center")
    ax2.set_xlabel("Share of district heating\nconsumption and supply\n[%]", fontsize=9)
    ax2.axvline(x=0, color="black", linestyle="-")
    ax2.set_xlim(-max_ylim, max_ylim)

    # Add secondary x-axis for district heating price savings on second subplot only
    ax2_price = ax2.twiny()

    # Plot DH price savings for Network 2 with white circles and black borders
    y_positions2 = range(len(dh_price_savings))
    ax2_price.scatter(
        dh_price_savings.values,
        y_positions2,
        s=40,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=0.2,
        zorder=20,
        clip_on=False,
    )

    # Add mean DH price savings line (more pronounced, white dashed with black border)
    # First draw thick black dashed line as border
    ax2_price.axvline(
        x=dh_price_savings.mean(),
        color="black",
        linestyle="--",
        linewidth=4,
        alpha=1,
        zorder=5,
    )
    # Then draw thinner white dashed line on top
    ax2_price.axvline(
        x=dh_price_savings.mean(),
        color="white",
        linestyle="--",
        linewidth=2,
        alpha=1,
        zorder=6,
    )

    # Set labels and formatting for price savings axis
    ax2_price.set_xlabel("DH Price Savings [€/MWh]", fontsize=9)
    ax2_price.tick_params(axis="x", labelsize=9)

    # Set x-limits for price savings axis with some padding
    price_min, price_max = dh_price_savings.min(), dh_price_savings.max()
    price_range = price_max - price_min
    if price_range > 0:
        padding = price_range * 0.1  # 10% padding
        price_xlim = (price_min - padding, price_max + padding)
    else:
        # If all savings are the same, add some padding around the value
        price_xlim = (price_min * 0.95, price_max * 1.05)

    ax2_price.set_xlim(price_xlim)

    # Decrease fontsize of yticks for both subplots (now that bars are horizontal)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(9)
    for tick in ax2.get_yticklabels():
        tick.set_fontsize(9)

    # Organize legend by categories
    legend_handles = []
    legend_labels = []

    # Define technology categories
    supply_techs = [
        "Heat Pumps",
        "CHP",
        "resistive heater",
        "gas boiler",
        "Fischer-Tropsch",
    ]
    demand_techs = ["District Heating Demand", "heat"]
    storage_techs = ["PTES", "TTES"]

    # Helper function to clean labels
    def clean_label(label):
        label = re.sub(
            "urban central heat$", "heat for residential and services", label
        )
        label = label.replace("urban central ", "")
        label = label.replace("water pits", "PTES")
        label = label.replace("water tanks", "TTES")
        label = label.replace(" charger", "").replace(" discharger", "")
        return label

    # Collect unique technologies with deduplication
    added_techs = set()  # Track added technologies to avoid duplicates

    # Add Supply technologies
    legend_labels.append("Supply Technologies:")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    for carrier, color in colors.items():
        if carrier in to_plot_rel1.columns or carrier in to_plot_rel2.columns:
            clean_name = clean_label(carrier)
            if (
                any(tech.lower() in clean_name.lower() for tech in supply_techs)
                and clean_name not in added_techs
            ):
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean_name)
                added_techs.add(clean_name)

    # Add Demand technologies
    legend_labels.append("Demand:")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    for carrier, color in colors.items():
        if carrier in to_plot_rel1.columns or carrier in to_plot_rel2.columns:
            clean_name = clean_label(carrier)
            if (
                any(tech.lower() in clean_name.lower() for tech in demand_techs)
                and clean_name not in added_techs
            ):
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean_name)
                added_techs.add(clean_name)

    # Add Storage technologies
    legend_labels.append("Storage:")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    for carrier, color in colors.items():
        if carrier in to_plot_rel1.columns or carrier in to_plot_rel2.columns:
            clean_name = clean_label(carrier)
            if (
                any(tech.lower() in clean_name.lower() for tech in storage_techs)
                and clean_name not in added_techs
            ):
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean_name)
                added_techs.add(clean_name)

    # Add markers and indicators
    legend_labels.append("Indicators:")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    # DH Demand marker
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="white",
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=6,
            linestyle="None",
        )
    )
    legend_labels.append("  DH Demand [TWh]")

    # Mean DH demand line
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="white",
            linestyle=":",
            linewidth=3,
            path_effects=[
                matplotlib.patheffects.Stroke(linewidth=4, foreground="black"),
                matplotlib.patheffects.Normal(),
            ],
        )
    )
    legend_labels.append("  Mean DH Demand")

    # DH price savings marker
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=6,
            linestyle="None",
        )
    )
    legend_labels.append("  DH Price Savings [€/MWh]")

    # Mean DH price savings line
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="white",
            linestyle="--",
            linewidth=3,
            path_effects=[
                matplotlib.patheffects.Stroke(linewidth=4, foreground="black"),
                matplotlib.patheffects.Normal(),
            ],
        )
    )
    legend_labels.append("  Mean DH Price Savings")

    fig.legend(
        legend_handles,
        legend_labels,
        title="Technology Categories",
        bbox_to_anchor=(0.5, 1.02),
        loc="lower center",
        frameon=False,
        fontsize=9,
        ncol=2,  # Reduced columns for better organization
    )

    # Replace DE0 at start of yticks with empty string (now y-axis shows regions)
    # Show city names only on the left side for cleaner appearance
    yticks = [label.get_text().replace("DE0 ", "") for label in ax1.get_yticklabels()]
    ax1.set_yticklabels(yticks)

    # Only show y-tick labels on the left subplot
    ax1.tick_params(axis="y", labelleft=True)
    ax2.tick_params(axis="y", labelleft=False, labelright=False)

    # Add light horizontal grid lines for easier comparison (extended beyond borders)
    ax1.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
    ax2.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)

    # Add vertical dashed lines at -100% and +100% (black for prominence)
    ax1.axvline(x=-100, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax1.axvline(x=100, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax2.axvline(x=-100, color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax2.axvline(x=100, color="black", linestyle="--", alpha=0.7, linewidth=1)

    # Set grid to extend beyond plot area
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)

    # The xlabel is now meaningful for horizontal bars, don't remove it

    # Adjust layout and save the plot
    plt.tight_layout()
    # Add extra space at the top for the legend and adjust spacing for much taller figure
    plt.subplots_adjust(top=0.88, wspace=0.2, left=0.18, right=0.95, bottom=0.08)
    fig.savefig(output_path, bbox_inches="tight")

    logger.info(f"Energy balance comparison saved to {output_path}")
    return fig, axes


def main(snakemake):
    """
    Main function to generate district heating system plots from network data.

    Parameters:
    -----------
    snakemake : object
        Snakemake object containing input/output paths and parameters
    """
    # Configure logging
    configure_logging(snakemake)

    # Get parameters from snakemake
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons

    # Get color overrides from config if available
    try:
        override_colors = snakemake.params.plotting["override_tech_colors"]
    except:
        override_colors = {}

    # Get grouping options from config if available (default to True)
    try:
        group_chp = snakemake.params.plotting.get("group_chp", True)
        group_heat_pumps = snakemake.params.plotting.get("group_heat_pumps", True)
        group_demands = snakemake.params.plotting.get("group_demands", True)
        drop_losses = snakemake.params.plotting.get("drop_losses", True)
    except:
        group_chp = True
        group_heat_pumps = True
        group_demands = True
        drop_losses = True

    # Create output directory
    output_path = snakemake.output[0]  # This is a directory
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Generating district heating system plots for run: {run_name}")

    # Process networks
    networks = process_networks(run_name, scenarios, planning_horizons)

    if not networks:
        logger.error("No networks could be loaded")
        return

    # Get color mapping
    colors = get_colors(networks, override_colors)

    # Add default colors for grouped categories
    default_group_colors = {
        "CHP": "#8B4513",  # saddle brown
        "Heat Pumps": "#FF8C00",  # dark orange
        "District Heating Demand": "#CCCCCC",  # lighter grey (same as urban central heat in config)
    }

    for group, color in default_group_colors.items():
        if group not in colors:
            colors[group] = color

    # Generate energy balance comparison plots for each year and scenario pair
    year = planning_horizons[0]  # Use first available year

    # Find scenario pairs (assuming we want to compare scenarios)
    scenario_pairs = []
    for i, scenario_a in enumerate(scenarios):
        for j, scenario_b in enumerate(scenarios):
            if i < j and scenario_a in networks and scenario_b in networks:
                if year in networks[scenario_a] and year in networks[scenario_b]:
                    scenario_pairs.append((scenario_a, scenario_b))

    if not scenario_pairs:
        logger.warning("No valid scenario pairs found for comparison")
        # If no pairs, create individual plots for each scenario (if possible)
        for scenario in scenarios:
            if scenario in networks and year in networks[scenario]:
                # Create a dummy comparison with itself (not very useful, but maintains structure)
                output_file = os.path.join(
                    output_path, f"dh_energy_balance_{scenario}_{year}.pdf"
                )
                plot_energy_balance_comparison(
                    networks[scenario][year],
                    networks[scenario][year],
                    [scenario, scenario],
                    output_file,
                    colors,
                    group_chp=group_chp,
                    group_heat_pumps=group_heat_pumps,
                    group_demands=group_demands,
                    drop_losses=drop_losses,
                )
        return

    # Generate comparison plots for each scenario pair
    for scenario_a, scenario_b in scenario_pairs:
        logger.info(f"Creating energy balance comparison: {scenario_a} vs {scenario_b}")

        output_file = os.path.join(
            output_path, f"dh_energy_balance_{scenario_a}_vs_{scenario_b}_{year}.pdf"
        )

        plot_energy_balance_comparison(
            networks[scenario_a][year],
            networks[scenario_b][year],
            [scenario_a, scenario_b],
            output_file,
            colors,
            group_chp=group_chp,
            group_heat_pumps=group_heat_pumps,
            group_demands=group_demands,
            drop_losses=drop_losses,
        )

    logger.info("District heating system plot generation completed")


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
        snakemake = mock_snakemake(
            "plot_dh_systems",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )
    main(snakemake)
