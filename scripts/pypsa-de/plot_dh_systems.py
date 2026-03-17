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
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


# Define custom handler class for square legend patches
class HandlerSquare(HandlerPatch):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        size = min(width, height)
        center_x = -xdescent + width / 2
        center_y = -ydescent + height / 2
        p = mpatches.Rectangle(
            (center_x - size / 2, center_y - size / 2),
            size,
            size,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            transform=trans,
        )
        return [p]


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


def calc_dh_price_range_subnodes(n, subnodes_only=True):
    """
    Calculate demand-weighted district heating prices for each system.

    Parameters:
    -----------
    n : pypsa.Network
        PyPSA network
    subnodes_only : bool, optional
        If True, only include systems with city names (subnodes) (default: True)

    Returns:
    --------
    pd.Series
        Demand-weighted district heating prices indexed by system name
    """
    if subnodes_only:
        # Only include subnodes with city names
        loads = n.loads_t.p.filter(
            regex=r"DE\d+ \d+ \w+.*(urban central|low-temperature) heat"
        ).clip(lower=0)
        prices = n.buses_t.marginal_price.filter(
            regex=r"DE\d+ \d+ \w+.*urban central heat"
        )
    else:
        # Include all district heating systems
        loads = n.loads_t.p.filter(
            regex=r"DE\d+ \d+.*(urban central|low-temperature) heat"
        ).clip(lower=0)
        prices = n.buses_t.marginal_price.filter(regex=r"DE\d+ \d+.*urban central heat")

    # Replace low-temperature heat for industry with urban central heat
    loads.columns = loads.columns.str.replace(
        "low-temperature heat for industry", "urban central heat"
    )
    loads = loads.T.groupby(loads.columns).sum().T

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
    group_ashp_wshp=False,
    group_demands=False,
    drop_losses=False,
    subnodes_only=True,
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
    subnodes_only : bool, optional
        If True, only show district heating systems with city names (subnodes) (default: True)

    Returns:
    --------
    tuple
        (figure, axes) matplotlib objects
    """
    plt.rcParams.update({"font.size": 12})
    title = f"Energy Balance Comparison: {scenarios[0]} vs {scenarios[1]}"

    def prepare_energy_balance_data(
        network,
        group_chp=False,
        group_heat_pumps=False,
        group_ashp_wshp=False,
        group_demands=False,
        drop_losses=False,
        subnodes_only=True,
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
        subnodes_only : bool
            Whether to only show systems with city names (subnodes)

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
        # Filter for district heating systems
        if subnodes_only:
            # Only show subnodes with city names (pattern: "DE0 1 Berlin urban central heat")
            eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+ \w+.*urban"), :]
        else:
            # Show all district heating systems (including main nodes without city names)
            eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+.*urban"), :]

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
        dh_prices = calc_dh_price_range_subnodes(network, subnodes_only)
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

        # Note: PTES heat pump is kept separate from PTES storage and treated as a supply technology

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
        network1,
        group_chp,
        group_heat_pumps,
        group_ashp_wshp,
        group_demands,
        drop_losses,
        subnodes_only,
    )
    to_plot_rel2, dh_prices2 = prepare_energy_balance_data(
        network2,
        group_chp,
        group_heat_pumps,
        group_ashp_wshp,
        group_demands,
        drop_losses,
        subnodes_only,
    )

    # Calculate price savings (network1 - network2)
    dh_price_savings = dh_prices1 - dh_prices2

    # Calculate district heating demand for sorting
    dh_demand_sort = []
    for system in to_plot_rel1.index:
        system_name = system.replace(" urban central heat", "")  # Clean system name

        # Only calculate demand for subnodes (city systems) to avoid double counting
        # Filter to exact match for the specific subnode system
        if subnodes_only:
            # For subnodes, use exact system name to avoid including mother nodes
            uch_cols = [
                col
                for col in network1.loads_t.p.columns
                if col == f"{system_name} urban central heat"
            ]
            ind_cols = [
                col
                for col in network1.loads_t.p.columns
                if col == f"{system_name} low-temperature heat for industry"
            ]
        else:
            # For all systems, use regex filter as before
            uch_cols = network1.loads_t.p.filter(
                regex=f"{system_name}.*urban central heat"
            ).columns
            ind_cols = network1.loads_t.p.filter(
                regex=f"{system_name}.*low-temperature heat for industry"
            ).columns

        # Calculate weighted loads for the specific columns
        if len(uch_cols) > 0:
            uch_weighted = (
                (
                    network1.loads_t.p[uch_cols].multiply(
                        network1.snapshot_weightings.generators, axis=0
                    )
                )
                .sum()
                .sum()
            )
        else:
            uch_weighted = 0

        if len(ind_cols) > 0:
            industry_weighted = (
                (
                    network1.loads_t.p[ind_cols].multiply(
                        network1.snapshot_weightings.generators, axis=0
                    )
                )
                .sum()
                .sum()
            )
        else:
            industry_weighted = 0

        # Convert to TWh (loads_t.p is in MW, snapshot_weightings gives MWh, so divide by 1e6)
        total_demand_twh = abs(uch_weighted + industry_weighted) / 1e6
        dh_demand_sort.append(total_demand_twh)

    dh_demand_series = pd.Series(dh_demand_sort, index=to_plot_rel1.index)

    # Calculate total DH demand for network2 (for pie chart label)
    dh_demand_sort2 = []
    for system in to_plot_rel2.index:
        system_name = system.replace(" urban central heat", "")
        if subnodes_only:
            uch_cols2 = [
                c
                for c in network2.loads_t.p.columns
                if c == f"{system_name} urban central heat"
            ]
            ind_cols2 = [
                c
                for c in network2.loads_t.p.columns
                if c == f"{system_name} low-temperature heat for industry"
            ]
        else:
            uch_cols2 = network2.loads_t.p.filter(
                regex=f"{system_name}.*urban central heat"
            ).columns
            ind_cols2 = network2.loads_t.p.filter(
                regex=f"{system_name}.*low-temperature heat for industry"
            ).columns
        uch_w2 = (
            network2.loads_t.p[uch_cols2]
            .multiply(network2.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
            if len(uch_cols2) > 0
            else 0
        )
        ind_w2 = (
            network2.loads_t.p[ind_cols2]
            .multiply(network2.snapshot_weightings.generators, axis=0)
            .sum()
            .sum()
            if len(ind_cols2) > 0
            else 0
        )
        dh_demand_sort2.append(abs(uch_w2 + ind_w2) / 1e6)
    dh_demand_series2 = pd.Series(dh_demand_sort2, index=to_plot_rel2.index)

    # Sort systems by demand (highest demand first)
    sorted_systems = dh_demand_series.sort_values(ascending=False).index

    # Reorder both plotting data and price data according to demand
    to_plot_rel1 = to_plot_rel1.loc[sorted_systems]
    to_plot_rel2 = to_plot_rel2.loc[sorted_systems]
    dh_price_savings = dh_price_savings.loc[sorted_systems]
    dh_demand_series = dh_demand_series.loc[sorted_systems]
    dh_demand_series2 = dh_demand_series2.reindex(sorted_systems, fill_value=0)

    # Helper: compute demand-weighted supply mix for pie chart
    def _pie_supply_mix(to_plot_rel, demand_s):
        weights = demand_s / demand_s.sum()
        mix = to_plot_rel.clip(lower=0).multiply(weights, axis=0).sum()
        exclude = [
            "District Heating Demand",
            "heat vent",
            "charger",
            "discharger",
            "losses",
            "low-temperature",
        ]
        mix = mix[
            [c for c in mix.index if not any(p in c for p in exclude) and mix[c] > 0.5]
        ]
        return mix

    max_ylim = to_plot_rel2.clip(lower=0).sum(1).max() * 1.05

    # Create subplots with side-by-side layout and pie charts below
    fig = plt.figure(figsize=(10, 12))

    # Main bar plots (top row) - using grid to leave space for legend on right
    ax_bar1 = plt.subplot2grid((10, 12), (0, 0), rowspan=6, colspan=5)
    ax_bar2 = plt.subplot2grid((10, 12), (0, 5), rowspan=6, colspan=5, sharey=ax_bar1)
    axes = [ax_bar1, ax_bar2]

    # Pie charts (bottom row) - positioned to avoid overlap with upper plots
    ax_pie1 = plt.subplot2grid((10, 12), (7, 0), rowspan=3, colspan=5)
    ax_pie2 = plt.subplot2grid((10, 12), (7, 5), rowspan=3, colspan=5)

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
        # Center right (regular supply) - ordered: geothermal -> electrolysis -> A/WSHP -> resistive -> CHP -> boilers -> storage
        "Heat Pumps",  # Grouped heat pumps (including geothermal)
        "geothermal heat pump",  # First: geothermal
        "urban central electrolysis excess heat pump",  # Second: electrolysis
        "A/WSHP",  # Third: A/WSHP (grouped)
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
        "urban central resistive heater",  # Fourth: Resistive heaters
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
        alpha=0.8,  # Match transparency of lower aggregated charts
    )

    # Create cleaner scenario title with bold formatting and linebreaks
    def format_scenario_title(scenario):
        """Format scenario title with proper linebreaks (no asterisks)."""
        title = scenario

        # Handle supply temperature
        if "HighSupplyTemperature" in title:
            title = title.replace("HighSupplyTemperature_", "High Temperature\n")
        elif "MidSupplyTemperature" in title:
            title = title.replace("MidSupplyTemperature_", "Medium Temperature\n")
        elif "LowSupplyTemperature" in title:
            title = title.replace("LowSupplyTemperature_", "Low Temperature\n")

        # Handle DH level
        if "MidDH_" in title:
            title = title.replace("MidDH_", "")
        elif "HighDH_" in title:
            title = title.replace("HighDH_", "High DH\n")
        elif "LowDH_" in title:
            title = title.replace("LowDH_", "Low DH\n")

        # Handle PTES scenarios
        if "NoPTES" in title:
            title = title.replace("NoPTES", "No PTES")
        elif "hpboost" in title:
            if "35Ctop" in title:
                title = title.replace(
                    "hpboost_35Ctop", "PTES with\nbooster heat pump\nto 35°C"
                )
            elif "10Cbottom" in title:
                title = title.replace(
                    "hpboost_10Cbottom", "PTES with\nbooster heat pump\nto 10°C"
                )
            else:
                title = title.replace("hpboost", "PTES with\nbooster heat pump")
        elif "rhboost" in title:
            title = title.replace("rhboost", "PTES with\nresistive boosting")
        elif "noboost" in title:
            title = title.replace("noboost", "PTES with\nno boosting")

        # Clean up any remaining underscores
        title = title.replace("_", " ")

        return title

    title1 = format_scenario_title(scenarios[0])
    ax1.set_title(title1, fontsize=11, pad=20, ha="center", weight="bold")
    ax1.set_xlabel("Demand and supply [%]", fontsize=12)

    ax1.axvline(x=0, color="black", linestyle="-")
    ax1.set_xlim(-max_ylim, max_ylim)

    # Add secondary x-axis for district heating demand on first subplot
    ax1_demand = ax1.twiny()

    # Calculate district heating demand in TWh for each system using actual loads
    dh_demand = []
    for system in to_plot_rel1.index:
        system_name = system.replace(" urban central heat", "")  # Clean system name

        # Only calculate demand for subnodes (city systems) to avoid double counting
        # Filter to exact match for the specific subnode system
        if subnodes_only:
            # For subnodes, use exact system name to avoid including mother nodes
            uch_cols = [
                col
                for col in network1.loads_t.p.columns
                if col == f"{system_name} urban central heat"
            ]
            ind_cols = [
                col
                for col in network1.loads_t.p.columns
                if col == f"{system_name} low-temperature heat for industry"
            ]
        else:
            # For all systems, use regex filter as before
            uch_cols = network1.loads_t.p.filter(
                regex=f"{system_name}.*urban central heat"
            ).columns
            ind_cols = network1.loads_t.p.filter(
                regex=f"{system_name}.*low-temperature heat for industry"
            ).columns

        # Calculate weighted loads for the specific columns
        if len(uch_cols) > 0:
            uch_weighted = (
                (
                    network1.loads_t.p[uch_cols].multiply(
                        network1.snapshot_weightings.generators, axis=0
                    )
                )
                .sum()
                .sum()
            )
        else:
            uch_weighted = 0

        if len(ind_cols) > 0:
            industry_weighted = (
                (
                    network1.loads_t.p[ind_cols].multiply(
                        network1.snapshot_weightings.generators, axis=0
                    )
                )
                .sum()
                .sum()
            )
        else:
            industry_weighted = 0

        # Convert to TWh (loads_t.p is in MW, snapshot_weightings gives MWh, so divide by 1e6)
        total_demand_twh = abs(uch_weighted + industry_weighted) / 1e6

        dh_demand.append(total_demand_twh)

    dh_demand = pd.Series(dh_demand, index=to_plot_rel1.index)

    # Plot DH demand with squares (black without border)
    y_positions1 = range(len(dh_demand))
    ax1_demand.scatter(
        dh_demand.values,
        y_positions1,
        s=30,
        marker="s",  # Square marker
        facecolor="black",
        edgecolor="black",
        linewidth=0.2,
        zorder=20,
        clip_on=False,
    )

    # Add mean DH demand line (dotted, white with black border like price savings)
    # First draw thick black dotted line as border
    # ax1_demand.axvline(
    #     x=dh_demand.mean(),
    #     color="black",
    #     linestyle=":",
    #     linewidth=4,
    #     alpha=1,
    #     zorder=5,
    # )
    # Then draw thinner white dotted line on top
    # ax1_demand.axvline(
    #     x=dh_demand.mean(),
    #     color="white",
    #     linestyle=":",
    #     linewidth=2,
    #     alpha=1,
    #     zorder=6,
    # )

    # Set labels for demand axis
    ax1_demand.set_xlabel("DH Demand\n[TWh]", fontsize=12)
    ax1_demand.tick_params(axis="x", labelsize=10)

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
        # Center right (regular supply) - ordered: geothermal -> electrolysis -> A/WSHP -> resistive -> CHP -> boilers -> storage
        "Heat Pumps",  # Grouped heat pumps (including geothermal)
        "geothermal heat pump",  # First: geothermal
        "urban central electrolysis excess heat pump",  # Second: electrolysis
        "A/WSHP",  # Third: A/WSHP (grouped)
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
        "urban central resistive heater",  # Fourth: Resistive heaters
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
        alpha=0.8,  # Match transparency of lower aggregated charts
    )
    title2 = format_scenario_title(scenarios[1])
    ax2.set_title(title2, fontsize=11, pad=20, ha="center", weight="bold")
    ax2.set_xlabel("Demand and supply [%]", fontsize=12)
    ax2.axvline(x=0, color="black", linestyle="-")
    ax2.set_xlim(-max_ylim, max_ylim)

    # Add secondary x-axis for district heating price savings on second subplot only
    ax2_price = ax2.twiny()

    # Plot DH price savings for Network 2 with black triangles
    y_positions2 = range(len(dh_price_savings))
    ax2_price.scatter(
        -dh_price_savings.values,
        y_positions2,
        s=30,
        marker="^",
        facecolor="black",
        edgecolor="black",
        linewidth=0.2,
        zorder=20,
        clip_on=False,
    )

    # Add mean DH price savings line (black dashed)
    ax2_price.axvline(
        x=-dh_price_savings.mean(),
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=1,
        zorder=5,
    )
    # Then draw thinner white dashed line on top
    # ax2_price.axvline(
    #     x=-dh_price_savings.mean(),
    #     color="white",
    #     linestyle="--",
    #     linewidth=2,
    #     alpha=1,
    #     zorder=6,
    # )

    # Set labels and formatting for price savings axis
    ax2_price.set_xlabel("ΔDH Price\n[EUR MWh$^{-1}$]", fontsize=12, color="black")
    ax2_price.tick_params(axis="x", labelsize=10, colors="black")

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

    # Update fontsize of yticks for both subplots (now that bars are horizontal)
    for tick in ax1.get_yticklabels():
        tick.set_fontsize(7)
    for tick in ax2.get_yticklabels():
        tick.set_fontsize(7)

    # Organize legend by categories
    legend_handles = []
    legend_labels = []

    # Define technology categories
    supply_techs = [
        "Heat Pumps",
        "geothermal heat pump",
        "electrolysis excess heat pump",
        "air heat pump",
        "river_water heat pump",
        "sea_water heat pump",
        "ptes heat pump",
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
        label = label.replace("A/WSHP", "Air and water sourced heat pumps")

        # Handle PTES capitalization specifically
        if label.lower().startswith("ptes"):
            words = label.split()
            if words:
                words[0] = "PTES"
                label = " ".join(words)

        # Capitalize first letter of first word while preserving abbreviations
        words = label.split()
        if words:
            first_word = words[0]
            # Keep abbreviations in all caps
            if first_word.upper() not in ["CHP", "PTES", "TTES", "H2"]:
                words[0] = first_word.capitalize()
            label = " ".join(words)

        return label

    # Collect unique technologies with deduplication
    added_techs = set()  # Track added technologies to avoid duplicates

    # Add Supply technologies
    legend_labels.append(r"$\bf{Supply\ Technologies:}$")
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
    legend_labels.append(r"$\bf{Demand:}$")
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
    legend_labels.append(r"$\bf{Storage:}$")
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
    legend_labels.append(r"$\bf{Indicators:}$")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    # DH Demand marker
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="black",
            markeredgecolor="black",
            markeredgewidth=0.8,
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
            marker="^",
            color="black",
            markeredgecolor="black",
            markeredgewidth=0.2,
            markersize=6,
            linestyle="None",
        )
    )
    legend_labels.append("  ΔDH Price [EUR MWh$^{-1}$]")

    # Mean DH price savings line
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
        )
    )
    legend_labels.append("  Mean ΔDH Price")

    # Legend to the right of the bar plots
    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.7, 0.72),
        loc="center left",
        frameon=False,
        fontsize=12,
        ncol=1,
        handler_map={mpatches.Patch: HandlerSquare()},
        handlelength=1.0,
        handleheight=1.0,
    )

    # ---- Pie charts ----
    pie_mix1 = _pie_supply_mix(to_plot_rel1, dh_demand_series)
    pie_mix2 = _pie_supply_mix(to_plot_rel2, dh_demand_series2)
    total_twh1 = dh_demand_series.sum()
    total_twh2 = dh_demand_series2.sum()

    for ax_pie, mix, total_twh in [
        (ax_pie1, pie_mix1, total_twh1),
        (ax_pie2, pie_mix2, total_twh2),
    ]:
        pie_colors = [colors.get(c, "gray") for c in mix.index]
        wedges, texts, autotexts = ax_pie.pie(
            mix.values,
            colors=pie_colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=90,
            pctdistance=0.65,
            textprops={"fontsize": 10},
            wedgeprops={"alpha": 0.8},
        )
        for autotext, wedge in zip(autotexts, wedges):
            autotext.set_color("black")
            autotext.set_weight("bold")
            angle = (wedge.theta1 + wedge.theta2) / 2
            if 90 <= angle <= 270:
                rotation = angle + 180
            else:
                rotation = angle
            autotext.set_rotation(rotation)
            autotext.set_horizontalalignment("center")
            autotext.set_verticalalignment("center")
        ax_pie.set_aspect("equal")
        ax_pie.set_title(
            f"Total: {total_twh:.1f} TWh", fontsize=12, pad=5, weight="bold"
        )

    # Replace DE0 at start of yticks with empty string (now y-axis shows regions)
    # Show city names only on the left side for cleaner appearance
    yticks = [label.get_text().replace("DE0 ", "") for label in ax1.get_yticklabels()]
    ax1.set_yticklabels(yticks)

    # Only show y-tick labels on the left subplot and add y-axis label
    ax1.tick_params(axis="y", labelleft=True)
    ax2.tick_params(axis="y", labelleft=False, labelright=False)
    ax1.set_ylabel("District heating system", fontsize=14)

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
    plt.subplots_adjust(
        top=0.95, wspace=0.10, left=0.12, right=0.80, bottom=0.05, hspace=0.15
    )
    fig.savefig(output_path, bbox_inches="tight")

    logger.info(f"Energy balance comparison saved to {output_path}")
    return fig, axes


def plot_energy_balance_triple_comparison(
    network1,
    network2,
    network3,
    scenarios,
    output_path,
    colors,
    group_chp=False,
    group_heat_pumps=False,
    group_ashp_wshp=False,
    group_demands=False,
    drop_losses=False,
    subnodes_only=True,
):
    """
    Plot comparison of energy balance for district heating between three networks.

    Parameters:
    -----------
    network1 : PyPSA Network
        First network (typically No_PTES scenario)
    network2 : PyPSA Network
        Second network (typically PTES with boosting scenario)
    network3 : PyPSA Network
        Third network (typically PTES without boosting scenario)
    scenarios : list
        List of scenario names [scenario1, scenario2, scenario3]
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
    subnodes_only : bool, optional
        If True, only show district heating systems with city names (subnodes) (default: True)

    Returns:
    --------
    tuple
        (figure, axes) matplotlib objects
    """
    plt.rcParams.update({"font.size": 10})

    def prepare_energy_balance_data(
        network,
        group_chp=False,
        group_heat_pumps=False,
        group_ashp_wshp=False,
        group_demands=False,
        drop_losses=False,
        subnodes_only=True,
    ):
        """Prepare energy balance data for a single network (same as in dual comparison)."""
        eb_uch = (
            network.statistics.energy_balance(groupby=["bus", "carrier", "bus_carrier"])
            .xs("urban central heat", level=3)
            .reset_index()
        )
        # Filter for district heating systems
        if subnodes_only:
            # Only show subnodes with city names (pattern: "DE0 1 Berlin urban central heat")
            eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+ \w+.*urban"), :]
        else:
            # Show all district heating systems (including main nodes without city names)
            eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+.*urban"), :]

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
        dh_prices = calc_dh_price_range_subnodes(network, subnodes_only)
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

        # Note: PTES heat pump is kept separate from PTES storage and treated as a supply technology

        # Apply groupings (same logic as dual comparison)
        if not group_heat_pumps:
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
                tech for tech in to_plot_rel.columns if "heat pump" in tech.lower()
            ]
            if len(heat_pump_techs) > 0:
                to_plot_rel["Heat Pumps"] = to_plot_rel[heat_pump_techs].sum(axis=1)
                to_plot_rel = to_plot_rel.drop(columns=heat_pump_techs)
        if group_ashp_wshp:
            ashp_wshp_techs = [
                tech
                for tech in to_plot_rel.columns
                if "urban central air heat pump" in tech
                or "urban central river_water heat pump" in tech
                or "urban central sea_water heat pump" in tech
            ]
            if len(ashp_wshp_techs) > 0:
                to_plot_rel["A/WSHP"] = to_plot_rel[ashp_wshp_techs].sum(axis=1)
                to_plot_rel = to_plot_rel.drop(columns=ashp_wshp_techs)

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

    # Prepare data for all three networks
    to_plot_rel1, dh_prices1 = prepare_energy_balance_data(
        network1,
        group_chp,
        group_heat_pumps,
        group_ashp_wshp,
        group_demands,
        drop_losses,
        subnodes_only,
    )
    to_plot_rel2, dh_prices2 = prepare_energy_balance_data(
        network2,
        group_chp,
        group_heat_pumps,
        group_ashp_wshp,
        group_demands,
        drop_losses,
        subnodes_only,
    )
    to_plot_rel3, dh_prices3 = prepare_energy_balance_data(
        network3,
        group_chp,
        group_heat_pumps,
        group_ashp_wshp,
        group_demands,
        drop_losses,
        subnodes_only,
    )

    # Calculate price savings (network1 - network2, network1 - network3)
    dh_price_savings_2 = dh_prices1 - dh_prices2
    dh_price_savings_3 = dh_prices1 - dh_prices3

    # Calculate district heating demand for sorting
    dh_demand_sort = []
    for system in to_plot_rel1.index:
        system_name = system.replace(" urban central heat", "")  # Clean system name

        # Only calculate demand for subnodes (city systems) to avoid double counting
        # Filter to exact match for the specific subnode system
        if subnodes_only:
            # For subnodes, use exact system name to avoid including mother nodes
            uch_cols = [
                col
                for col in network1.loads_t.p.columns
                if col == f"{system_name} urban central heat"
            ]
            ind_cols = [
                col
                for col in network1.loads_t.p.columns
                if col == f"{system_name} low-temperature heat for industry"
            ]
        else:
            # For all systems, use regex filter as before
            uch_cols = network1.loads_t.p.filter(
                regex=f"{system_name}.*urban central heat"
            ).columns
            ind_cols = network1.loads_t.p.filter(
                regex=f"{system_name}.*low-temperature heat for industry"
            ).columns

        # Calculate weighted loads for the specific columns
        if len(uch_cols) > 0:
            uch_weighted = (
                (
                    network1.loads_t.p[uch_cols].multiply(
                        network1.snapshot_weightings.generators, axis=0
                    )
                )
                .sum()
                .sum()
            )
        else:
            uch_weighted = 0

        if len(ind_cols) > 0:
            industry_weighted = (
                (
                    network1.loads_t.p[ind_cols].multiply(
                        network1.snapshot_weightings.generators, axis=0
                    )
                )
                .sum()
                .sum()
            )
        else:
            industry_weighted = 0

        # Convert to TWh (loads_t.p is in MW, snapshot_weightings gives MWh, so divide by 1e6)
        total_demand_twh = abs(uch_weighted + industry_weighted) / 1e6
        dh_demand_sort.append(total_demand_twh)

    dh_demand_series = pd.Series(dh_demand_sort, index=to_plot_rel1.index)

    # Sort systems by demand (highest demand first)
    sorted_systems = dh_demand_series.sort_values(ascending=False).index

    # Reorder all plotting data according to demand
    to_plot_rel1 = to_plot_rel1.loc[sorted_systems]
    to_plot_rel2 = to_plot_rel2.loc[sorted_systems]
    to_plot_rel3 = to_plot_rel3.loc[sorted_systems]
    dh_price_savings_2 = dh_price_savings_2.loc[sorted_systems]
    dh_price_savings_3 = dh_price_savings_3.loc[sorted_systems]

    max_ylim = (
        max(
            to_plot_rel1.clip(lower=0).sum(1).max(),
            to_plot_rel2.clip(lower=0).sum(1).max(),
            to_plot_rel3.clip(lower=0).sum(1).max(),
        )
        * 1.05
    )

    # Create subplots with three columns and sub-charts below each
    fig = plt.figure(figsize=(8, 12))

    # Main plots (top row) - make them take up most of the space
    axes = []
    for i in range(3):
        ax = plt.subplot2grid(
            (15, 3), (0, i), rowspan=10, sharey=axes[0] if axes else None
        )
        axes.append(ax)

    # Sub-charts (bottom row) for aggregated DH mix with shared y-axis
    sub_axes = []
    for i in range(3):
        sub_ax = plt.subplot2grid(
            (15, 3), (13, i), rowspan=2, sharey=sub_axes[0] if sub_axes else None
        )
        sub_axes.append(sub_ax)

    def format_scenario_title(scenario):
        """Format scenario title with proper linebreaks and correct order for NoPTES."""
        title = scenario

        # Special handling for NoPTES scenarios - reorder components
        if "NoPTES_" in title:
            parts = title.split("_")
            supply_temp = ""
            dh_level = ""

            for part in parts[1:]:  # Skip "NoPTES"
                if "SupplyTemperature" in part:
                    if "High" in part:
                        supply_temp = "High Temperature"
                    elif "Mid" in part:
                        supply_temp = "Medium Temperature"
                    elif "Low" in part:
                        supply_temp = "Low Temperature"
                elif "DH" in part:
                    if "High" in part:
                        dh_level = "High DH"
                    elif "Mid" in part:
                        dh_level = "Medium DH"
                    elif "Low" in part:
                        dh_level = "Low DH"

            # Return in correct order: Temperature -> DH Level -> No PTES
            return f"{supply_temp}\n{dh_level}\nNo PTES"

        # Handle other scenarios (non-NoPTES)
        # Handle supply temperature
        if "HighSupplyTemperature" in title:
            title = title.replace("HighSupplyTemperature_", "High Temperature\n")
        elif "MidSupplyTemperature" in title:
            title = title.replace("MidSupplyTemperature_", "Medium Temperature\n")
        elif "LowSupplyTemperature" in title:
            title = title.replace("LowSupplyTemperature_", "Low Temperature\n")

        # Handle DH level - handle both with and without underscore
        if "MidDH" in title:
            title = title.replace("MidDH_", "").replace("MidDH", "")
        elif "HighDH" in title:
            title = title.replace("HighDH_", "High DH\n").replace("HighDH", "High DH")
        elif "LowDH" in title:
            title = title.replace("LowDH_", "Low DH\n").replace("LowDH", "Low DH")

        # Handle PTES scenarios
        if "hpboost" in title:
            if "35Ctop" in title:
                title = title.replace(
                    "hpboost_35Ctop", "PTES with\nbooster heat pump\nto 35°C"
                )
            elif "10Cbottom" in title:
                title = title.replace(
                    "hpboost_10Cbottom", "PTES with\nbooster heat pump\nto 10°C"
                )
            else:
                title = title.replace("hpboost", "PTES with\nbooster heat pump")
        elif "rhboost" in title:
            title = title.replace("rhboost", "PTES with\nresistive boosting")
        elif "noboost" in title:
            title = title.replace("noboost", "PTES with\nno boosting")

        # Clean up any remaining underscores - convert to linebreaks for proper order
        title = title.replace("_", "\n")

        return title

    # Technology order: geothermal -> electrolysis -> A/WSHP -> Resistive heaters -> CHP -> boilers -> storage
    col_order = [
        "District Heating Demand",
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        "urban central water tanks",
        "urban central water tanks charger",
        "urban central water tanks losses",
        "urban central water pits",
        "urban central water pits charger",
        "urban central water pits losses",
        "Heat Pumps",
        "geothermal heat pump",  # First: geothermal
        "urban central electrolysis excess heat pump",  # Second: electrolysis
        "A/WSHP",  # Third: A/WSHP (grouped)
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
        "urban central resistive heater",  # Fourth: Resistive heaters
        "CHP",  # Fifth: CHP (grouped)
        "urban central gas CHP",
        "urban central solid biomass CHP",
        "urban central lignite CHP",
        "urban central coal CHP",
        "urban central oil CHP",
        "urban central H2 CHP",
        "waste CHP",
        "urban central gas boiler",  # Sixth: boilers
        "H2 Electrolysis",
        "Fischer-Tropsch",
        "urban central water tanks discharger",  # Seventh: storage
        "urban central water pits discharger",
    ]

    # Plot all three scenarios
    plot_data = [to_plot_rel1, to_plot_rel2, to_plot_rel3]
    price_data = [None, dh_price_savings_2, dh_price_savings_3]
    price_axes = []  # Store price axes for standardization

    for i, (ax, data, prices, scenario) in enumerate(
        zip(axes, plot_data, price_data, scenarios)
    ):
        # Filter and order columns
        available_cols = [c for c in col_order if c in data.columns]
        available_cols += [c for c in data.columns if c not in available_cols]
        data = data[available_cols]

        # Create horizontal bar plot
        data.plot.barh(
            stacked=True, ax=ax, color=colors, legend=False, width=0.9, alpha=0.8
        )

        # Format title and labels
        title = format_scenario_title(scenario)
        ax.set_title(title, fontsize=11, pad=20, ha="center", weight="bold")
        ax.set_xlabel(
            "Share of district heating\nconsumption and supply\n[%]", fontsize=12
        )
        ax.axvline(x=0, color="black", linestyle="-")
        ax.set_xlim(-max_ylim, max_ylim)

        # Add grid lines
        ax.grid(True, axis="y", alpha=0.3, linestyle="-", linewidth=0.5)
        ax.axvline(x=-100, color="black", linestyle="--", alpha=0.7, linewidth=1)
        ax.axvline(x=100, color="black", linestyle="--", alpha=0.7, linewidth=1)
        ax.set_axisbelow(True)

        # Add secondary axis for price data or demand
        if i == 0:  # First plot: show DH demand
            ax_secondary = ax.twiny()
            # Calculate DH demand
            dh_demand = []
            for system in data.index:
                system_name = system.replace(" urban central heat", "")
                uch_load = (
                    network1.loads_t.p.filter(
                        regex=f"{system_name}.*urban central heat"
                    )
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
                # Get weighted load values using PyPSA snapshot weightings
                uch_weighted = (
                    (
                        network1.loads_t.p.filter(
                            regex=f"{system_name}.*urban central heat"
                        ).multiply(network1.snapshot_weightings.generators, axis=0)
                    )
                    .sum()
                    .sum()
                )

                industry_weighted = (
                    (
                        network1.loads_t.p.filter(
                            regex=f"{system_name}.*low-temperature heat for industry"
                        ).multiply(network1.snapshot_weightings.generators, axis=0)
                    )
                    .sum()
                    .sum()
                )

                # Convert to TWh (loads_t.p is in MW, snapshot_weightings gives MWh, so divide by 1e6)
                total_demand_twh = abs(uch_weighted + industry_weighted) / 1e6
                dh_demand.append(total_demand_twh)

            dh_demand = pd.Series(dh_demand, index=data.index)
            y_positions = range(len(dh_demand))
            ax_secondary.scatter(
                dh_demand.values,
                y_positions,
                s=30,
                marker="o",
                facecolor="black",
                edgecolor="black",
                linewidth=0.8,
                zorder=20,
                clip_on=False,
            )
            ax_secondary.axvline(
                x=dh_demand.mean(),
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=1,
                zorder=5,
            )
            # ax_secondary.axvline(
            #     x=dh_demand.mean(),
            #     color="white",
            #     linestyle=":",
            #     linewidth=2,
            #     alpha=1,
            #     zorder=6,
            # )
            ax_secondary.set_xlabel("DH Demand\n[TWh]", fontsize=12, color="black")
            ax_secondary.tick_params(axis="x", labelsize=10, colors="black")

            demand_min, demand_max = dh_demand.min(), dh_demand.max()
            demand_range = demand_max - demand_min
            if demand_range > 0:
                padding = demand_range * 0.1
                demand_xlim = (demand_min - padding, demand_max + padding)
            else:
                demand_xlim = (demand_min * 0.95, demand_max * 1.05)
            ax_secondary.set_xlim(demand_xlim)

        elif prices is not None:  # Other plots: show price savings
            ax_secondary = ax.twiny()
            y_positions = range(len(prices))
            ax_secondary.scatter(
                prices.values,
                y_positions,
                s=30,
                marker="^",
                facecolor="black",
                edgecolor="black",
                linewidth=0.8,
                zorder=20,
                clip_on=False,
            )
            ax_secondary.axvline(
                x=prices.mean(),
                color="black",
                linestyle="--",
                linewidth=2,
                alpha=1,
                zorder=5,
            )
            # ax_secondary.axvline(
            #     x=prices.mean(),
            #     color="white",
            #     linestyle="--",
            #     linewidth=2,
            #     alpha=1,
            #     zorder=6,
            # )
            ax_secondary.set_xlabel(
                "ΔDH Price\n[EUR MWh$^{-1}$]", fontsize=12, color="black"
            )
            ax_secondary.tick_params(axis="x", labelsize=10, colors="black")

            # Store price axis for later standardization
            price_axes.append((ax_secondary, prices))

        # Y-axis formatting
        for tick in ax.get_yticklabels():
            tick.set_fontsize(9)

        if i == 0:  # Only show y-labels on leftmost plot
            ax.tick_params(axis="y", labelleft=True)
            ax.set_ylabel("District heating system", fontsize=14, weight="bold")
            # Clean y-tick labels
            yticks = [
                label.get_text().replace("DE0 ", "") for label in ax.get_yticklabels()
            ]
            ax.set_yticklabels(yticks)
        else:
            ax.tick_params(axis="y", labelleft=False, labelright=False)

    # Standardize price axis limits across both price plots with symmetric range to align 0 values
    if "price_axes" in locals() and price_axes:
        # Calculate combined range from both price datasets
        all_price_values = []
        for _, prices in price_axes:
            all_price_values.extend(prices.values)

        combined_min, combined_max = min(all_price_values), max(all_price_values)
        # Use symmetric limits to ensure 0 aligns vertically across all plots
        max_abs_value = max(abs(combined_min), abs(combined_max))
        padding = max_abs_value * 0.1
        symmetric_limit = max_abs_value + padding
        shared_xlim = (-symmetric_limit, symmetric_limit)

        # Apply the same symmetric limits to both price axes
        for ax_secondary, _ in price_axes:
            ax_secondary.set_xlim(shared_xlim)

    # Create aggregated DH mix sub-charts (without storage technologies)
    # For aggregated charts, always include both subnodes and mother nodes
    storage_techs = [
        "urban central water tanks",
        "urban central water tanks charger",
        "urban central water tanks losses",
        "urban central water pits",
        "urban central water pits charger",
        "urban central water pits losses",
        "TTES",
        "PTES",
    ]

    # Helper function to get absolute energy values in TWh for aggregation
    def get_absolute_energy_balance(network, subnodes_only=False):
        """Get absolute energy balance data in TWh for aggregation."""
        eb_uch = (
            network.statistics.energy_balance(groupby=["bus", "carrier", "bus_carrier"])
            .xs("urban central heat", level=3)
            .reset_index()
        )

        # Filter for district heating systems
        if subnodes_only:
            eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+ \w+.*urban"), :]
        else:
            eb_uch = eb_uch.loc[eb_uch.bus.str.contains(r"DE\d+ \d+.*urban"), :]

        # Strip 'urban central heat' from the bus index
        eb_uch["bus"] = eb_uch["bus"].str.replace(" urban central heat", "")
        eb_uch.drop("component", axis=1, inplace=True)

        # Remove " CC" suffix and aggregate
        eb_uch["carrier"] = eb_uch["carrier"].str.replace(" CC", "", regex=False)
        eb_uch = eb_uch.groupby(["bus", "carrier"], as_index=False).sum()

        # Set index and unstack
        to_plot = eb_uch.set_index(["bus", "carrier"]).unstack(-1)
        to_plot.columns = to_plot.columns.droplevel(0)

        # Convert from MWh to TWh (energy balance already accounts for snapshot weightings)
        to_plot_twh = to_plot / 1e6
        return to_plot_twh.clip(lower=0)

    # Get absolute energy data for aggregation (in TWh)
    networks = [network1, network2, network3]
    for i, (network, sub_ax) in enumerate(zip(networks, sub_axes)):
        # Get absolute energy balance in TWh for all systems (both subnodes and mother nodes)
        abs_data = get_absolute_energy_balance(network, subnodes_only=False)

        # Remove storage technologies and group by technology type according to configuration
        grouped_supply = {}

        for col in abs_data.columns:
            if not any(storage_tech in col for storage_tech in storage_techs):
                total_value = abs_data[col].sum()  # Sum across all systems in TWh
                if total_value > 0:
                    # Apply same grouping logic as main plots
                    tech_name = col

                    # Group heat pumps if enabled
                    if group_heat_pumps and "heat pump" in col.lower():
                        tech_name = "Heat Pumps"
                    # Group A/WSHP if enabled (and not already grouped with all heat pumps)
                    elif (
                        group_ashp_wshp
                        and not group_heat_pumps
                        and (
                            "urban central air heat pump" in col.lower()
                            or "urban central river_water heat pump" in col.lower()
                            or "urban central sea_water heat pump" in col.lower()
                        )
                    ):
                        tech_name = "A/WSHP"
                    # Group CHP if enabled
                    elif group_chp and (
                        "chp" in col.lower() or "combined heat" in col.lower()
                    ):
                        tech_name = "CHP"
                    # Group resistive heaters
                    elif "resistive" in col.lower():
                        tech_name = "Resistive Heater"
                    # Clean up other technology names
                    else:
                        tech_name = (
                            col.replace("urban central ", "")
                            .replace("water pits", "PTES")
                            .replace("water tanks", "TTES")
                        )

                    # Add to grouped supply
                    if tech_name not in grouped_supply:
                        grouped_supply[tech_name] = 0
                    grouped_supply[tech_name] += total_value

        # Create stacked bar with proper ordering and colors
        # Define technology order to match main plots: geothermal -> electrolysis -> A/WSHP -> Resistive heaters -> CHP -> boilers -> storage
        # Note: for stacked bars, order from top to bottom (reverse of desired visual order)
        tech_order = [
            # Other supply (top of stack)
            "H2 Electrolysis",
            "Fischer-Tropsch",
            # Boilers
            "gas boiler",
            # CHP technologies (grouped or individual)
            "waste CHP",
            "H2 CHP",
            "oil CHP",
            "coal CHP",
            "lignite CHP",
            "solid biomass CHP",
            "gas CHP",
            "CHP",
            # Resistive heaters
            "resistive heater",
            "Resistive Heater",
            # A/WSHP (grouped or individual)
            "A/WSHP",
            "sea_water heat pump",
            "river_water heat pump",
            "air heat pump",
            "ptes heat pump",
            # Electrolysis excess
            "electrolysis excess heat pump",
            # Heat pumps (grouped)
            "Heat Pumps",
            # Geothermal at the bottom of stack (last, so it appears at bottom visually)
            "geothermal heat pump",
            "geothermal heat",
        ]

        # Use colors from the main color scheme, with fallbacks for grouped categories
        tech_colors_map = colors.copy()

        # Create comprehensive color mapping for cleaned technology names
        for tech, color in colors.items():
            # Map original tech names to cleaned versions
            clean_tech = (
                tech.replace("urban central ", "")
                .replace("water pits", "PTES")
                .replace("water tanks", "TTES")
            )
            tech_colors_map[clean_tech] = color

        # Add specific mappings for grouped and cleaned names
        tech_colors_map.update(
            {
                "Heat Pumps": colors.get("Heat Pumps", "#FF8C00"),  # Orange
                "A/WSHP": colors.get("A/WSHP", "#FFA600"),  # Orange variant
                "Resistive Heater": colors.get(
                    "urban central resistive heater", "#40E0D0"
                ),  # Cyan
                "resistive heater": colors.get(
                    "urban central resistive heater", "#40E0D0"
                ),
                "CHP": colors.get("CHP", "#8B4513"),  # Use same color as main plots
                "gas boiler": colors.get("urban central gas boiler", "#8B0000"),
                "gas CHP": colors.get("urban central gas CHP", "#CD853F"),
                "H2 CHP": colors.get("urban central H2 CHP", "#4169E1"),
                "solid biomass CHP": colors.get(
                    "urban central solid biomass CHP", "#228B22"
                ),
                "waste CHP": colors.get("waste CHP", "#8B4513"),
                "geothermal heat pump": colors.get(
                    "urban central geothermal heat pump", "#B22222"
                ),
                "geothermal heat": colors.get(
                    "urban central geothermal heat", "#B22222"
                ),
                "air heat pump": colors.get("urban central air heat pump", "#FF6347"),
                "sea_water heat pump": colors.get(
                    "urban central sea_water heat pump", "#4682B4"
                ),
                "river_water heat pump": colors.get(
                    "urban central river_water heat pump", "#20B2AA"
                ),
                "ptes heat pump": colors.get("urban central ptes heat pump", "#9932CC"),
                "electrolysis excess heat pump": colors.get(
                    "urban central electrolysis excess heat pump", "#FFD700"
                ),
                "Fischer-Tropsch": colors.get("Fischer-Tropsch", "#A0522D"),
                "H2 Electrolysis": colors.get("H2 Electrolysis", "#4169E1"),
            }
        )

        # Filter to only include technologies that exist and have positive values
        techs = [
            tech
            for tech in tech_order
            if tech in grouped_supply and grouped_supply[tech] > 0
        ]
        # Add any remaining technologies not in the predefined order
        remaining_techs = [
            tech
            for tech in grouped_supply.keys()
            if tech not in techs and grouped_supply[tech] > 0
        ]
        techs.extend(sorted(remaining_techs))

        values = [grouped_supply[tech] for tech in techs]
        tech_colors = [
            tech_colors_map.get(tech, "#808080") for tech in techs
        ]  # Gray fallback

        if techs:  # Only create bars if we have data
            # Reverse the order to flip the stacking (geothermal will now be at top)
            techs_reversed = list(reversed(techs))
            values_reversed = list(reversed(values))
            tech_colors_reversed = list(reversed(tech_colors))

            bottom = 0
            for tech, value, color in zip(
                techs_reversed, values_reversed, tech_colors_reversed
            ):
                sub_ax.bar(0, value, bottom=bottom, color=color, width=1.0, alpha=0.8)
                bottom += value

        # Format sub-chart
        sub_ax.set_xlim(-0.5, 0.5)
        sub_ax.set_xticks([])
        if i == 0:  # Only show y-label on leftmost chart
            sub_ax.set_ylabel("TWh", fontsize=12)
        sub_ax.tick_params(axis="y", labelsize=10)

        # Remove spines except left
        for spine in sub_ax.spines.values():
            spine.set_visible(False)
        sub_ax.spines["left"].set_visible(True)
        sub_ax.grid(True, alpha=0.3, axis="y")

    # Add single centered title for sub-charts (positioned above the bars)
    fig.text(
        0.5,
        0.35,
        "Aggregated DH Supply",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
        transform=fig.transFigure,
    )

    # Create comprehensive legend with all technologies and categorization
    legend_handles = []
    legend_labels = []

    # Define technology categories
    supply_techs = [
        "Heat Pumps",
        "geothermal heat pump",
        "electrolysis excess heat pump",
        "air heat pump",
        "river_water heat pump",
        "sea_water heat pump",
        "ptes heat pump",
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
        label = label.replace("A/WSHP", "Air and water sourced heat pumps")

        # Handle PTES capitalization specifically
        if label.lower().startswith("ptes"):
            words = label.split()
            if words:
                words[0] = "PTES"
                label = " ".join(words)

        # Capitalize first letter of first word while preserving abbreviations
        words = label.split()
        if words:
            first_word = words[0]
            # Keep abbreviations in all caps
            if first_word.upper() not in ["CHP", "PTES", "TTES", "H2"]:
                words[0] = first_word.capitalize()
            label = " ".join(words)

        return label

    # Collect unique technologies with deduplication
    added_techs = set()  # Track added technologies to avoid duplicates

    # Get all columns from all three scenarios
    all_columns = set()
    for data in [to_plot_rel1, to_plot_rel2, to_plot_rel3]:
        all_columns.update(data.columns)

    # Add Supply technologies
    legend_labels.append(r"$\bf{Supply\ Technologies:}$")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    for carrier, color in colors.items():
        if carrier in all_columns:
            clean_name = clean_label(carrier)
            if (
                any(tech.lower() in clean_name.lower() for tech in supply_techs)
                and clean_name not in added_techs
            ):
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean_name)
                added_techs.add(clean_name)

    # Add Demand technologies
    legend_labels.append(r"$\bf{Demand:}$")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    for carrier, color in colors.items():
        if carrier in all_columns:
            clean_name = clean_label(carrier)
            if (
                any(tech.lower() in clean_name.lower() for tech in demand_techs)
                and clean_name not in added_techs
            ):
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean_name)
                added_techs.add(clean_name)

    # Add Storage technologies
    legend_labels.append(r"$\bf{Storage:}$")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    for carrier, color in colors.items():
        if carrier in all_columns:
            clean_name = clean_label(carrier)
            if (
                any(tech.lower() in clean_name.lower() for tech in storage_techs)
                and clean_name not in added_techs
            ):
                legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
                legend_labels.append("  " + clean_name)
                added_techs.add(clean_name)

    # Add markers and indicators
    legend_labels.append(r"$\bf{Indicators:}$")
    legend_handles.append(plt.Rectangle((0, 0), 0, 0, alpha=0))  # Invisible spacer

    # DH Demand marker (black circle)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=6,
            linestyle="None",
        )
    )
    legend_labels.append("  DH Demand [TWh]")

    # Mean DH demand line (black)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
        )
    )
    legend_labels.append("  Mean DH Demand")

    # DH price savings marker
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="^",
            color="black",
            markeredgecolor="black",
            markeredgewidth=0.8,
            markersize=6,
            linestyle="None",
        )
    )
    legend_labels.append("  ΔDH Price [EUR MWh$^{-1}$]")

    # Mean DH price savings line (black)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=2,
        )
    )
    legend_labels.append("  Mean ΔDH Price")

    fig.legend(
        legend_handles,
        legend_labels,
        bbox_to_anchor=(0.5, 0.17),
        loc="upper center",
        frameon=False,
        fontsize=10,
        ncol=3,
    )

    # Adjust layout for narrower plots, sub-charts, and comprehensive legend
    plt.tight_layout()
    plt.subplots_adjust(
        top=0.95, wspace=0.12, left=0.12, right=0.95, bottom=0.23, hspace=0.18
    )
    fig.savefig(output_path, bbox_inches="tight")

    logger.info(f"Triple energy balance comparison saved to {output_path}")
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
        group_ashp_wshp = snakemake.params.plotting.get("group_ashp_wshp", True)
        group_demands = snakemake.params.plotting.get("group_demands", True)
        drop_losses = snakemake.params.plotting.get("drop_losses", True)
        subnodes_only = snakemake.params.plotting.get("subnodes_only", True)
    except:
        group_chp = True
        group_heat_pumps = False
        group_ashp_wshp = True
        group_demands = True
        drop_losses = True
        subnodes_only = True

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
        "A/WSHP": "#FFA600",  # orange
        "District Heating Demand": "#CCCCCC",  # lighter grey (same as urban central heat in config)
    }

    for group, color in default_group_colors.items():
        if group not in colors:
            colors[group] = color

    # Generate energy balance comparison plots for each year
    year = planning_horizons[0]  # Use first available year

    # Get scenario triples from config
    try:
        scenario_triples = snakemake.params.plotting["scenario_triples"]
    except:
        scenario_triples = []

    if scenario_triples:
        # Generate triple comparison plots
        for triple in scenario_triples:
            scenario_a, scenario_b, scenario_c = triple

            # Check if all three scenarios are available
            if (
                scenario_a in networks
                and scenario_b in networks
                and scenario_c in networks
                and year in networks[scenario_a]
                and year in networks[scenario_b]
                and year in networks[scenario_c]
            ):

                logger.info(
                    f"Creating triple energy balance comparison: {scenario_a} vs {scenario_b} vs {scenario_c}"
                )

                output_file = os.path.join(
                    output_path,
                    f"dh_energy_balance_triple_{scenario_a}_vs_{scenario_b}_vs_{scenario_c}_{year}.pdf",
                )

                plot_energy_balance_triple_comparison(
                    networks[scenario_a][year],
                    networks[scenario_b][year],
                    networks[scenario_c][year],
                    [scenario_a, scenario_b, scenario_c],
                    output_file,
                    colors,
                    group_chp=group_chp,
                    group_heat_pumps=group_heat_pumps,
                    group_ashp_wshp=group_ashp_wshp,
                    group_demands=group_demands,
                    drop_losses=drop_losses,
                    subnodes_only=subnodes_only,
                )
            else:
                logger.warning(f"Not all scenarios available for triple: {triple}")

    else:
        # Fall back to dual comparison if no triples defined
        # Find scenario pairs
        scenario_pairs = []
        for i, scenario_a in enumerate(scenarios):
            for j, scenario_b in enumerate(scenarios):
                if i < j and scenario_a in networks and scenario_b in networks:
                    if year in networks[scenario_a] and year in networks[scenario_b]:
                        scenario_pairs.append((scenario_a, scenario_b))

        if not scenario_pairs:
            logger.warning("No valid scenario pairs found for comparison")
            return

        # Generate comparison plots for each scenario pair
        for scenario_a, scenario_b in scenario_pairs:
            logger.info(
                f"Creating energy balance comparison: {scenario_a} vs {scenario_b}"
            )

            output_file = os.path.join(
                output_path,
                f"dh_energy_balance_{scenario_a}_vs_{scenario_b}_{year}.pdf",
            )

            plot_energy_balance_comparison(
                networks[scenario_a][year],
                networks[scenario_b][year],
                [scenario_a, scenario_b],
                output_file,
                colors,
                group_chp=group_chp,
                group_heat_pumps=group_heat_pumps,
                group_ashp_wshp=group_ashp_wshp,
                group_demands=group_demands,
                drop_losses=drop_losses,
                subnodes_only=subnodes_only,
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
