#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
This script processes PyPSA networks and generates plots and metrics
for energy system analysis and comparison across scenarios focusing on
district heating and thermal energy storage.
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
import matplotlib.patheffects as patheffects
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import pypsa
import xarray as xr
import seaborn as sns
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "code", "pypsa-de"))
# Import will be handled conditionally in main()

logger = logging.getLogger(__name__)

# Configure logging for standalone execution
if "snakemake" not in globals():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
else:
    # Import configure_logging only when running via snakemake
    try:
        from scripts._helpers import configure_logging

        configure_logging(logger)
    except ImportError:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def get_supply_temperature(scenario):
    """Extract the supply temperature from the scenario name."""
    if "LowSupplyTemperature" in scenario:
        return "LowTemp"
    elif "MidSupplyTemperature" in scenario:
        return "MedTemp"
    elif "HighSupplyTemperature" in scenario:
        return "HighTemp"
    else:
        raise ValueError(f"Unknown supply temperature in scenario {scenario}")


def get_district_heating_share(scenario):
    """Extract the district heating share from the scenario name."""
    if "LowDH" in scenario:
        return "LowDH"
    elif "MidDH" in scenario:
        return "MedDH"
    elif "HighDH" in scenario:
        return "HighDH"
    else:
        raise ValueError(f"Unknown district heating share in scenario {scenario}")


def get_boosting_technology(scenario):
    """Extract the boosting technology from the scenario name."""
    if "rhboost" in scenario:
        return "rhboost"
    elif "hpboost_10Cbottom" in scenario:
        return "hpboost_10Cbottom"
    elif "hpboost" in scenario:
        return "hpboost"
    elif "noboost" in scenario:
        return "noboost"
    elif "NoPTES" in scenario:
        return None  # Explicitly return None
    else:
        raise ValueError(f"Unknown boosting technology in scenario {scenario}")


def calc_ptes_cycles(n, system, year):
    """Calculate the number of cycles for PTES (water pits) systems."""
    pit_system = n.stores.loc[f"{system} urban central water pits-{year}"]
    if pit_system.e_nom_opt <= 0:
        return None

    discharge = (
        n.links_t.p0[f"{system} urban central water pits discharger-{year}"]
        .mul(n.snapshot_weightings.generators)
        .sum()
    )
    no_cycles = discharge / pit_system.e_nom_opt
    return no_cycles


def get_p2h_chp_share(n, bus_id):
    """Calculate the share of P2H and CHP in total heat generation for a bus."""

    eb = n.statistics.energy_balance(groupby=["bus", "carrier"]).loc[
        (slice(None), bus_id, slice(None))
    ]
    # Drop contributions from chargers and dischargers
    eb = eb.drop(eb.filter(regex="charger").index)
    p2h_gen = eb.filter(regex="heat pump|resistive heater")
    chp_gen = eb.filter(regex="CHP|boiler")
    total_gen = eb.clip(lower=0).sum()

    p2h_share = p2h_gen.sum() / total_gen if total_gen > 0 else 0
    chp_share = chp_gen.sum() / total_gen if total_gen > 0 else 0

    return p2h_share, chp_share


def extract_plotting_data(networks):
    """This function creates a dataframe that extracts for each scenario and planning horizon
    a selection of metrics for all the german district heating systems,that are used as input
    for the plotting functions. These metrics comprise:
    1. Ratio between PTES energy capacity and annual heat demand
    2. Ratio between TTES energy capacity and annual heat demand
    3. The system-specific district heating prices
    4. The annual cycles of the PTES.
    5. The share of P2H district heating (resistive heaters and heat pumps) in the annual heat generation
    6. The share of CHP in the annual heat generation
    """
    plotting_data = pd.DataFrame(
        columns=[
            "scenario",
            "year",
            "ptes",
            "supply_temperature",
            "district_heating_share",
            "boosting_tech",
            "district_heating_system",
            "ptes_to_demand_ratio",
            "ttes_to_demand_ratio",
            "dh_price_EUR_per_MWh",
            "annual_ptes_cycles",
            "p2h_share",
            "chp_share",
        ]
    )
    for scenario, years in networks.items():
        for year, n in years.items():
            logger.info(f"Extracting data for scenario {scenario} and year {year}")

            # Bool if PTES modelled (true if NoPTES not in scenario name)
            ptes_modelled = "NoPTES" not in scenario
            # Supply temperature
            supply_temperature = get_supply_temperature(scenario)
            # determine district heating share
            dh_share = get_district_heating_share(scenario)
            # Determine boosting technology
            boosting_tech = get_boosting_technology(scenario)
            # Identify district heating buses
            dh_buses = n.buses.filter(regex="DE.*urban central heat", axis=0).location
            for bus, location in dh_buses.items():
                # Calculate heat demand directly from loads and links
                # Get all loads connected to this bus
                bus_loads = n.loads[n.loads.bus == bus]
                # Get all DAC links connected to this bus (if any)
                dac_links = n.links[(n.links.bus1 == bus) & (n.links.carrier == "DAC")]

                # Calculate annual heat demand from loads
                annual_heat_demand_loads = 0
                for load_name in bus_loads.index:
                    if load_name in n.loads_t.p_set.columns:
                        load_demand = (
                            n.loads_t.p_set[load_name]
                            * n.snapshot_weightings.generators
                        ).sum()
                        annual_heat_demand_loads += load_demand

                # Calculate annual heat demand from DAC links
                annual_heat_demand_dac = 0
                for link_name in dac_links.index:
                    if link_name in n.links_t.p1.columns:
                        dac_demand = (
                            n.links_t.p1[link_name] * n.snapshot_weightings.generators
                        ).sum()
                        annual_heat_demand_dac += abs(
                            dac_demand
                        )  # Take absolute value since it's a demand

                annual_heat_demand = annual_heat_demand_loads + annual_heat_demand_dac

                if annual_heat_demand <= 0:
                    logger.warning(
                        f"No heat demand found for bus {bus} in scenario {scenario}"
                    )
                    continue

                ptes_capacity = (
                    n.stores.loc[
                        f"{location} urban central water pits-{year}", "e_nom_opt"
                    ]
                    if ptes_modelled
                    else 0
                )
                ttes_capacity = n.stores.loc[
                    f"{location} urban central water tanks-{year}", "e_nom_opt"
                ]
                # Calculate PTES to demand ratio
                ptes_to_demand_ratio = ptes_capacity / annual_heat_demand
                ttes_to_demand_ratio = ttes_capacity / annual_heat_demand
                # Calculate district heating price (use simple average if demand profile not available)
                if bus in n.buses_t.marginal_price.columns:
                    mp = n.buses_t.marginal_price[bus]
                    dh_price = (
                        mp * n.snapshot_weightings.generators
                    ).sum() / n.snapshot_weightings.generators.sum()
                else:
                    logger.warning(f"No marginal price data for bus {bus}")
                    dh_price = 0
                # Calculate annual PTES cycles
                if ptes_modelled:
                    ptes_cycles = calc_ptes_cycles(n, location, year)
                else:
                    ptes_cycles = None
                # Calculate P2H and share in heat generation
                p2h_share, chp_share = get_p2h_chp_share(n, bus)
                logger.debug(
                    f"Bus {bus} in scenario {scenario}: P2H share = {p2h_share:.3f}, CHP share = {chp_share:.3f}"
                )

                # Append data to dataframe
                plotting_data = pd.concat(
                    [
                        plotting_data,
                        pd.DataFrame(
                            {
                                "scenario": [scenario],
                                "year": [year],
                                "ptes": [ptes_modelled],
                                "supply_temperature": [supply_temperature],
                                "district_heating_share": [dh_share],
                                "boosting_tech": [boosting_tech],
                                "district_heating_system": [location],
                                "ptes_to_demand_ratio": [ptes_to_demand_ratio],
                                "ttes_to_demand_ratio": [ttes_to_demand_ratio],
                                "dh_price_EUR_per_MWh": [dh_price],
                                "annual_ptes_cycles": [ptes_cycles],
                                "p2h_share": [p2h_share],
                                "chp_share": [chp_share],
                            }
                        ),
                    ],
                    ignore_index=True,
                )

    return plotting_data


def process_networks(run_name, scenarios, planning_horizons):
    """Process networks and extract data for analysis."""
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
        return None, None, None

    plotting_data = extract_plotting_data(networks)

    return plotting_data


def create_scenario_order(plotting_data):
    """Create ordered scenario list based on boosting tech and temperature."""
    # Filter for MidDH scenarios only
    mid_dh_data = plotting_data[plotting_data["district_heating_share"] == "MedDH"]

    # Debug: Print unique combinations available
    logger.info("Available temperature + boosting combinations in MidDH data:")
    combinations = mid_dh_data[
        ["supply_temperature", "boosting_tech", "scenario"]
    ].drop_duplicates()
    for _, row in combinations.iterrows():
        logger.info(
            f"  Temp: {row['supply_temperature']}, Boost: {row['boosting_tech']}, Scenario: {row['scenario']}"
        )

    # Define order for boosting technologies (NoPTES first)
    boosting_order = [None, "hpboost", "hpboost_10Cbottom", "rhboost", "noboost"]

    # Define order for temperatures
    temp_order = ["HighTemp", "MedTemp", "LowTemp"]

    ordered_scenarios = []
    for temp in temp_order:
        for boost in boosting_order:
            # Find scenarios matching this combination
            if boost is None:
                # For NoPTES scenarios, match boosting_tech being None or NaN
                matching = mid_dh_data[
                    (mid_dh_data["supply_temperature"] == temp)
                    & (
                        (mid_dh_data["boosting_tech"].isna())
                        | (mid_dh_data["boosting_tech"] == None)
                    )
                ]["scenario"].unique()
            else:
                # For other scenarios, match the specific boosting tech
                matching = mid_dh_data[
                    (mid_dh_data["supply_temperature"] == temp)
                    & (mid_dh_data["boosting_tech"] == boost)
                ]["scenario"].unique()

            if len(matching) > 0:
                logger.info(
                    f"Found {len(matching)} scenarios for Temp: {temp}, Boost: {boost}"
                )
                for scenario in matching:
                    logger.info(f"  - {scenario}")

            ordered_scenarios.extend(matching)

    # Remove any empty entries and return unique scenarios
    ordered_scenarios = [s for s in ordered_scenarios if s]
    return ordered_scenarios


def make_violin_plots(plotting_data, run_name):
    """Generate combined violin plots for the extracted metrics."""
    output_dir = os.path.join("results", run_name, "plots")
    os.makedirs(output_dir, exist_ok=True)

    # Debug: Print all available scenarios
    logger.info("Available scenarios in plotting data:")
    for scenario in plotting_data["scenario"].unique():
        logger.info(f"  {scenario}")

    # Debug: Print MidDH scenarios specifically
    mid_dh_scenarios = plotting_data[
        plotting_data["district_heating_share"] == "MedDH"
    ]["scenario"].unique()
    logger.info("Available MidDH scenarios:")
    for scenario in mid_dh_scenarios:
        logger.info(f"  {scenario}")

    # Filter for MidDH scenarios only
    plotting_data_filtered = plotting_data[
        plotting_data["district_heating_share"] == "MedDH"
    ].copy()

    if plotting_data_filtered.empty:
        logger.warning("No MidDH scenarios found for plotting")
        return

    # Get ordered scenarios
    ordered_scenarios = create_scenario_order(plotting_data_filtered)

    # Debug: Print ordered scenarios
    logger.info("Ordered scenarios for plotting:")
    for scenario in ordered_scenarios:
        logger.info(f"  {scenario}")

    # Create consistent x-tick labels for all scenarios
    all_scenario_labels = []
    for scenario in ordered_scenarios:
        scenario_row = plotting_data_filtered[
            plotting_data_filtered["scenario"] == scenario
        ].iloc[0]
        boosting_tech = scenario_row["boosting_tech"]

        # Create clean label based on the actual boosting technology
        if pd.isna(boosting_tech) or boosting_tech is None:
            clean_label = "NoPTES"
        elif boosting_tech == "hpboost_10Cbottom":
            clean_label = "Heat pump to 10°C"
        elif boosting_tech == "hpboost":
            clean_label = "Heat pump"
        elif boosting_tech == "rhboost":
            clean_label = "Resistive Heater"
        elif boosting_tech == "noboost":
            clean_label = "No Boost"
        else:
            clean_label = f"Unknown ({boosting_tech})"

        all_scenario_labels.append(clean_label)

    # Check if we have any NoPTES scenarios
    has_no_ptes = any("NoPTES" in scenario for scenario in ordered_scenarios)

    # Define metrics and their properties - conditionally include PTES metrics
    metrics = [
        ("ttes_to_demand_ratio", "TTES capacity\n/Demand [%]", "#4ECDC4"),
        ("dh_price_EUR_per_MWh", "DH Price\n[EUR MWh⁻¹]", "#45B7D1"),
        ("p2h_share", "P2H Share\n[%]", "#96CEB4"),
        ("chp_share", "CHP Share\n[%]", "#FFEAA7"),
    ]

    # Only add PTES metrics if we have non-NoPTES scenarios
    if not has_no_ptes or any(
        "NoPTES" not in scenario for scenario in ordered_scenarios
    ):
        metrics.insert(
            0, ("ptes_to_demand_ratio", "PTES capacity\n/Demand [%]", "#FF6B6B")
        )
        metrics.append(("annual_ptes_cycles", "PTES Cycles\n[a⁻¹]", "#DDA0DD"))

    # Create figure with subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(16, 16), sharex=True)

    for i, (metric, label, color) in enumerate(metrics):
        ax = axes[i]

        # Prepare data for violin plot
        violin_data = []
        positions = []

        for pos, scenario in enumerate(ordered_scenarios):
            # Skip PTES-specific metrics for NoPTES scenarios
            if "NoPTES" in scenario and metric in [
                "ptes_to_demand_ratio",
                "annual_ptes_cycles",
            ]:
                continue

            scenario_data = plotting_data_filtered[
                plotting_data_filtered["scenario"] == scenario
            ][metric].dropna()

            if len(scenario_data) > 0:
                # Convert to percentages for share metrics and capacity ratios
                if metric in [
                    "p2h_share",
                    "chp_share",
                    "ptes_to_demand_ratio",
                    "ttes_to_demand_ratio",
                ]:
                    violin_data.append(scenario_data.values * 100)
                else:
                    violin_data.append(scenario_data.values)
                positions.append(pos)

        if violin_data:
            # Create violin plot
            parts = ax.violinplot(
                violin_data,
                positions=positions,
                widths=0.7,
                showmeans=True,
                showmedians=True,
                showextrema=True,
            )

            # Customize violin appearance
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor("black")
                pc.set_linewidth(0.5)

            # Customize other elements
            parts["cmeans"].set_color("red")
            parts["cmedians"].set_color("blue")
            parts["cbars"].set_color("black")
            parts["cmins"].set_color("black")
            parts["cmaxes"].set_color("black")

            # Add individual data points
            for pos, data in zip(positions, violin_data):
                # Add small jitter to x-coordinates for visibility
                x_jitter = np.random.normal(pos, 0.02, len(data))
                ax.scatter(x_jitter, data, alpha=0.6, s=20, color="black", zorder=3)

        # Customize subplot
        ax.set_ylabel(label, fontweight="bold", fontsize=18, va="center", labelpad=20)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, len(ordered_scenarios) - 0.5)
        ax.tick_params(axis="y", labelsize=16)

        # Set consistent y-axis limits for percentage metrics
        if metric in ["p2h_share", "chp_share"]:
            ax.set_ylim(0, 100)
        elif metric in ["ptes_to_demand_ratio", "ttes_to_demand_ratio"]:
            ax.set_ylim(0, None)  # Let matplotlib choose upper limit for percentages

        # Add vertical lines between temperature groups (every 5 scenarios)
        for j in range(5, len(ordered_scenarios), 5):
            ax.axvline(x=j - 0.5, color="gray", linestyle="--", alpha=0.7, linewidth=2)

    # Set x-axis labels only on bottom subplot
    axes[-1].set_xticks(range(len(all_scenario_labels)))
    axes[-1].set_xticklabels(all_scenario_labels, rotation=45, ha="right", fontsize=16)
    axes[-1].set_xlabel("Scenarios", fontweight="bold", fontsize=18)
    axes[-1].tick_params(axis="x", labelsize=16)

    # Add temperature group labels at the top
    temp_positions = [2, 7, 12]  # Middle of each group of 5
    temp_labels = ["High Temp", "Medium Temp", "Low Temp"]

    for pos, temp_label in zip(temp_positions, temp_labels):
        if pos < len(ordered_scenarios):
            axes[0].text(
                pos,
                axes[0].get_ylim()[1],
                temp_label,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=20,
            )

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, "district_heating_violin_plots.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()

    logger.info(f"Violin plots saved to {output_file}")


def main():
    """Main function to run the violin plot generation."""
    # Mock snakemake for standalone execution
    if "snakemake" not in globals():
        import sys
        import os

        # Add the scripts directory to Python path
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(script_dir)

        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sysgf_violines",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )

    # Extract parameters
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons

    logger.info(f"Starting violin plot generation for run: {run_name}")

    # Process networks and extract data
    plotting_data = process_networks(run_name, scenarios, planning_horizons)

    if plotting_data is None or plotting_data.empty:
        logger.error("No data extracted from networks")
        return

    logger.info(f"Extracted data for {len(plotting_data)} district heating systems")

    # Generate violin plots
    make_violin_plots(plotting_data, run_name)

    logger.info("Violin plot generation completed")


if __name__ == "__main__":
    main()
