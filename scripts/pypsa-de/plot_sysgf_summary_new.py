#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
Generate summary plots and statistics for system analysis.

This script processes PyPSA networks and generates various plots and metrics
for energy system analysis and comparison across scenarios.
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import yaml

from scripts._helpers import configure_logging, mock_snakemake, read_yaml

logger = logging.getLogger(__name__)


def read_network(network_path):
    """Read a PyPSA network from file."""
    logger.info(f"Reading network from {network_path}")
    return pypsa.Network(network_path)


# Utility functions for calculating metrics
def calc_ptes_cycles(n, mean=True):
    """Calculate the number of cycles for PTES (water pits) systems."""
    pits_de = n.stores.filter(regex="DE0.*water pits", axis=0).query("e_nom_opt > 0")
    if pits_de.empty:
        return 0

    discharge = (
        n.links_t.p0.filter(regex="DE.*pits discharger")
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
    )
    discharge.index = discharge.index.str.replace(" discharger", "")
    no_cycles = discharge.div(pits_de.e_nom_opt)
    if mean:
        return no_cycles.mean()
    else:
        return no_cycles


def calc_thermal_storage_capacity(n, storage_type="pits"):
    """Calculate thermal storage capacity in TWh and GW."""
    regex = f"DE0.*water {storage_type}"
    storage = n.stores.filter(regex=regex, axis=0).query("e_nom_opt > 0")

    if storage.empty:
        return 0, 0

    capacity_twh = storage.e_nom_opt.sum() / 1e6  # TWh

    # Calculate power capacity from chargers/dischargers
    chargers = n.links.filter(regex=f"DE.*{storage_type} charger", axis=0)
    dischargers = n.links.filter(regex=f"DE.*{storage_type} discharger", axis=0)

    capacity_gw = min(
        chargers.p_nom_opt.sum() if not chargers.empty else 0,
        dischargers.p_nom_opt.sum() if not dischargers.empty else 0,
    )

    return capacity_twh, capacity_gw


def calc_h2_store_capacity(n):
    """Calculate hydrogen storage capacity in TWh."""
    h2_stores = n.stores.filter(regex="DE0.*H2 Store", axis=0)
    return h2_stores.e_nom_opt.sum() / 1e6  # TWh


def calc_average_dh_price(n):
    """Calculate average district heating price in EUR/MWh."""
    dh_loads = n.loads_t.p.filter(
        regex="DE.*(urban central heat|low-temperature heat for industry)"
    )
    if dh_loads.empty:
        return 0

    # Rename columns so the substrings are removed
    dh_loads.columns = dh_loads.columns.str.replace(
        "low-temperature heat for industry", "urban central heat"
    )
    # Aggregate columns with same name
    dh_loads = dh_loads.groupby(axis=1, level=0).sum()

    prices = n.buses_t.marginal_price.filter(regex="DE0.*urban central heat")
    if prices.empty or dh_loads.sum().sum() == 0:
        return 0

    average_dh_price = dh_loads.mul(prices).sum().sum() / dh_loads.sum().sum()
    return average_dh_price


def calc_average_elec_price(n):
    """Calculate average electricity price in EUR/MWh."""
    elec_mps = n.buses_t.marginal_price.filter(regex="DE0 \\d+$")
    elec_demand = n.loads_t.p.filter(regex="DE\\d.*(\\d+|electricity|EV)$")

    if elec_mps.empty or elec_demand.empty:
        return 0

    elec_demand.columns = elec_demand.columns.str.split(" ").str[:2].str.join(" ")
    elec_demand = elec_demand.groupby(elec_demand.columns, axis=1).sum()
    elec_costs = (elec_mps * elec_demand).sum().sum()

    if elec_demand.sum().sum() == 0:
        return 0

    return elec_costs / elec_demand.sum().sum()


def calc_peak_prices(n, price_type="electricity"):
    """Calculate peak prices (99th percentile)."""
    if price_type == "electricity":
        loads = n.buses_t.p.filter(regex="DE\\d \\d$").clip(upper=0).mul(-1)
        prices = n.buses_t.marginal_price.filter(regex="DE\\d \\d$")
    else:  # district heating
        loads = n.loads_t.p.filter(
            regex="DE\\d.*(urban central|low-temperature) heat"
        ).clip(lower=0)
        prices = n.buses_t.marginal_price.filter(regex="DE\\d.*urban central heat")

    if loads.empty or prices.empty:
        return 0

    weighted_price = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    peak_price = weighted_price.quantile(0.99)  # 99th percentile
    return peak_price


def calc_curtailment_de(n):
    """Calculate curtailment in TWh."""
    try:
        curtailment = (
            n.statistics.curtailment(
                groupby=n.statistics.groupers.get_bus_and_carrier, nice_names=False
            )
            .xs("Generator", level=0)
            .filter(regex="DE.*wind|solar")
            .div(1e6)
            .sum()
        )
        return curtailment
    except:
        return 0


def calc_heat_venting_de(n):
    """Calculate heat venting in TWh."""
    try:
        heat_venting = (
            n.snapshot_weightings.generators
            @ n.generators_t.p.filter(regex="DE0.*heat vent")
        ).sum() / 1e6
        return heat_venting
    except:
        return 0


def get_system_costs(n, country="DE"):
    """Calculate system costs for a specific country in billion EUR."""
    try:
        s = n.statistics
        capex = s.expanded_capex(
            groupby=n.statistics.groupers.get_bus_and_carrier
        ).filter(regex=country, axis=0)
        opex = s.opex(groupby=n.statistics.groupers.get_bus_and_carrier).filter(
            regex=country, axis=0
        )
        total_costs = (capex.sum().sum() + opex.sum().sum()) / 1e9  # in billion EUR
        return total_costs
    except:
        return 0


def get_total_system_costs(n):
    """Calculate total system costs in billion EUR."""
    try:
        s = n.statistics
        g = s.groupers
        grouper = g.get_country_and_carrier
        system_cost = s.capex(groupby=grouper).add(s.opex(groupby=grouper))
        return system_cost.sum().sum() / 1e9  # in billion EUR
    except:
        return 0


def calc_average_dh_price_t_ordered(n):
    """Calculate time-ordered average district heating price."""
    loads = n.loads_t.p.filter(
        regex="DE\\d.*(urban central|low-temperature) heat"
    ).clip(lower=0)
    prices = n.buses_t.marginal_price.filter(regex="DE\\d.*urban central heat")

    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series()

    weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted_average_price_t


def calc_average_electricity_price_t_ordered(n):
    """Calculate time-ordered average electricity price."""
    loads = n.buses_t.p.filter(regex="DE\\d \\d$").clip(upper=0).mul(-1)
    prices = n.buses_t.marginal_price.filter(regex="DE\\d \\d$")

    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series()

    weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted_average_price_t


# Plotting functions
def plot_price_duration_curve(scenario_data, output_path, price_type="electricity"):
    """Plot price duration curves for multiple scenarios."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario, data in scenario_data.items():
        if f"{price_type}_price_duration" in data:
            prices = data[f"{price_type}_price_duration"]
            prices.plot(ax=ax, label=scenario)

    ax.set_xlabel("Duration Proportion")
    ax.set_ylabel("Price [EUR/MWh]")
    ax.set_title(f"{price_type.capitalize()} Price Duration Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_comparison(
    scenario_data, output_path, metrics=None, reference_scenario=None, normalize=False
):
    """Plot a comparison of key metrics across scenarios."""
    if not metrics:
        metrics = [
            "total_system_costs_bnEUR",
            "PTES_capacity_TWh",
            "TTES_capacity_TWh",
            "dh_price_EUR_per_MWh",
            "electricity_price_EUR_per_MWh",
        ]

    # Create DataFrame for plotting
    data = {
        scenario: {metric: value for metric, value in data.items() if metric in metrics}
        for scenario, data in scenario_data.items()
    }
    df = pd.DataFrame(data).T

    # Normalize to reference scenario if requested
    if normalize and reference_scenario and reference_scenario in df.index:
        ref_values = df.loc[reference_scenario]
        df = df.div(ref_values) - 1  # percentage change

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    df.plot(kind="bar", ax=ax)

    if normalize:
        ax.set_ylabel("Relative Change")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        plt.title("Relative Changes Compared to Reference Scenario")
    else:
        ax.set_ylabel("Value")
        plt.title("Comparison of Key Metrics Across Scenarios")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_analysis(
    scenario_data, output_path, sensitivities, reference_scenario
):
    """Plot sensitivity analysis showing changes compared to reference scenario."""
    # Extract reference data
    if reference_scenario not in scenario_data:
        logger.error(f"Reference scenario {reference_scenario} not found in data")
        return

    ref_data = scenario_data[reference_scenario]

    # Create a figure with subplots for each sensitivity
    fig, axes = plt.subplots(
        nrows=len(sensitivities), figsize=(10, 4 * len(sensitivities))
    )

    if len(sensitivities) == 1:
        axes = [axes]

    # Define metrics to plot
    metrics = [
        "total_system_costs_bnEUR",
        "PTES_capacity_TWh",
        "dh_price_EUR_per_MWh",
        "electricity_price_EUR_per_MWh",
    ]

    for i, sensitivity in enumerate(sensitivities):
        ax = axes[i]

        # Get the low and high scenarios for this sensitivity
        low_scenario = f"Low{sensitivity}"
        high_scenario = f"High{sensitivity}"

        if low_scenario not in scenario_data or high_scenario not in scenario_data:
            continue

        # Calculate percentage change from reference
        low_data = {
            m: (scenario_data[low_scenario].get(m, 0) - ref_data.get(m, 0))
            / ref_data.get(m, 1)
            * 100
            for m in metrics
        }
        high_data = {
            m: (scenario_data[high_scenario].get(m, 0) - ref_data.get(m, 0))
            / ref_data.get(m, 1)
            * 100
            for m in metrics
        }

        # Create DataFrame for plotting
        df = pd.DataFrame({"Low": low_data, "High": high_data})

        # Plot
        df.T.plot(kind="bar", ax=ax)
        ax.set_title(f"{sensitivity} Sensitivity")
        ax.set_ylabel("% Change from Reference")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def process_networks(networks_path, scenario_list, planning_horizons, output_path):
    """Process networks and extract key metrics."""
    scenario_data = {}

    for scenario in scenario_list:
        scenario_path = os.path.join(networks_path, scenario, "networks")
        if not os.path.exists(scenario_path):
            logger.warning(f"Scenario path {scenario_path} does not exist, skipping")
            continue

        for year in planning_horizons:
            # Find network file for this year
            network_file = None
            for file in os.listdir(scenario_path):
                if file.endswith(f"_{year}_final.nc"):
                    network_file = os.path.join(scenario_path, file)
                    break

            if not network_file:
                logger.warning(
                    f"No network file found for scenario {scenario} and year {year}"
                )
                continue

            # Load network and calculate metrics
            try:
                n = read_network(network_file)

                ptes_capacity_twh, ptes_capacity_gw = calc_thermal_storage_capacity(
                    n, "pits"
                )
                ttes_capacity_twh, ttes_capacity_gw = calc_thermal_storage_capacity(
                    n, "tanks"
                )

                # Store metrics
                scenario_data[scenario] = {
                    "year": year,
                    "total_system_costs_bnEUR": get_total_system_costs(n),
                    "system_costs_DE_bnEUR": get_system_costs(n, "DE"),
                    "PTES_capacity_TWh": ptes_capacity_twh,
                    "PTES_capacity_GW": ptes_capacity_gw,
                    "PTES_cycles": calc_ptes_cycles(n),
                    "TTES_capacity_TWh": ttes_capacity_twh,
                    "TTES_capacity_GW": ttes_capacity_gw,
                    "H2_store_TWh": calc_h2_store_capacity(n),
                    "dh_price_EUR_per_MWh": calc_average_dh_price(n),
                    "electricity_price_EUR_per_MWh": calc_average_elec_price(n),
                    "peak_electricity_price_EUR_per_MWh": calc_peak_prices(
                        n, "electricity"
                    ),
                    "peak_dh_price_EUR_per_MWh": calc_peak_prices(n, "dh"),
                    "curtailment_TWh": calc_curtailment_de(n),
                    "heat_venting_TWh": calc_heat_venting_de(n),
                }

                # Calculate price duration curves
                elec_prices = calc_average_electricity_price_t_ordered(n)
                dh_prices = calc_average_dh_price_t_ordered(n)

                if not elec_prices.empty:
                    elec_price_duration = (
                        elec_prices.loc[
                            np.repeat(
                                elec_prices.index, n.snapshot_weightings.generators
                            )
                        ]
                        .sort_values(ascending=False)
                        .reset_index(drop=True)
                    )
                    elec_price_duration.index = elec_price_duration.index / (
                        len(elec_price_duration) - 1
                    )
                    scenario_data[scenario][
                        "electricity_price_duration"
                    ] = elec_price_duration

                if not dh_prices.empty:
                    dh_price_duration = (
                        dh_prices.loc[
                            np.repeat(dh_prices.index, n.snapshot_weightings.generators)
                        ]
                        .sort_values(ascending=False)
                        .reset_index(drop=True)
                    )
                    dh_price_duration.index = dh_price_duration.index / (
                        len(dh_price_duration) - 1
                    )
                    scenario_data[scenario]["dh_price_duration"] = dh_price_duration

                logger.info(
                    f"Successfully processed network for scenario {scenario} and year {year}"
                )
            except Exception as e:
                logger.error(
                    f"Error processing network for scenario {scenario} and year {year}: {e}"
                )

    # Save metrics to CSV
    os.makedirs(output_path, exist_ok=True)
    metrics_df = pd.DataFrame(
        {
            scenario: data
            for scenario, data in scenario_data.items()
            if not isinstance(data.get("electricity_price_duration", None), pd.Series)
        }
    )
    metrics_df.to_csv(os.path.join(output_path, "scenario_metrics.csv"))

    return scenario_data


def generate_summary_pdf(scenario_data, reference_scenario, output_path, run_name):
    """Generate a summary PDF with key findings."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Title and general info
    fig.suptitle(f"System Analysis Summary - Run: {run_name}", fontsize=16)

    # Plot 1: System costs comparison
    ax = axes[0]
    costs_data = {
        k: scenario_data[k]["total_system_costs_bnEUR"] for k in scenario_data
    }
    pd.Series(costs_data).plot(kind="bar", ax=ax)
    ax.set_title("Total System Costs")
    ax.set_ylabel("Billion EUR")
    ax.grid(True, axis="y", alpha=0.3)

    # Plot 2: PTES Capacity comparison
    ax = axes[1]
    ptes_data = {k: scenario_data[k]["PTES_capacity_TWh"] for k in scenario_data}
    pd.Series(ptes_data).plot(kind="bar", ax=ax)
    ax.set_title("PTES Capacity")
    ax.set_ylabel("TWh")
    ax.grid(True, axis="y", alpha=0.3)

    # Plot 3: Price comparison
    ax = axes[2]
    price_data = pd.DataFrame(
        {
            k: [
                scenario_data[k]["electricity_price_EUR_per_MWh"],
                scenario_data[k]["dh_price_EUR_per_MWh"],
            ]
            for k in scenario_data
        },
        index=["Electricity", "District Heating"],
    )
    price_data.plot(kind="bar", ax=ax)
    ax.set_title("Average Energy Prices")
    ax.set_ylabel("EUR/MWh")
    ax.grid(True, axis="y", alpha=0.3)

    # Save figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main(snakemake):
    """Main function to generate plots from network data."""
    # Configure logging
    configure_logging(snakemake)

    # Get parameters from snakemake
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons
    reference_scenario = snakemake.params.reference_scenario
    sensitivities = snakemake.params.sensitivity_runs

    # Create output directory
    output_path = os.path.dirname(snakemake.output.sysgf_summary)
    os.makedirs(output_path, exist_ok=True)

    logger.info(f"Generating system analysis for run: {run_name}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Planning horizons: {planning_horizons}")
    logger.info(f"Reference scenario: {reference_scenario}")

    # Set paths
    networks_path = os.path.join("results", run_name)

    # Process networks and collect data
    scenario_data = process_networks(
        networks_path, scenarios, planning_horizons, output_path
    )

    if not scenario_data:
        logger.error("No data could be collected from networks")
        return

    # Generate plots
    plot_price_duration_curve(
        scenario_data,
        os.path.join(output_path, "electricity_price_duration.pdf"),
        "electricity",
    )

    plot_price_duration_curve(
        scenario_data, os.path.join(output_path, "dh_price_duration.pdf"), "dh"
    )

    # Plot comparison of key metrics
    plot_scenario_comparison(
        scenario_data, os.path.join(output_path, "scenario_comparison.pdf")
    )

    # Plot normalized comparison if reference scenario exists
    if reference_scenario in scenario_data:
        plot_scenario_comparison(
            scenario_data,
            os.path.join(output_path, "scenario_comparison_normalized.pdf"),
            reference_scenario=reference_scenario,
            normalize=True,
        )

    # Plot sensitivity analysis if sensitivities are defined
    if sensitivities and reference_scenario in scenario_data:
        plot_sensitivity_analysis(
            scenario_data,
            os.path.join(output_path, "sensitivity_analysis.pdf"),
            sensitivities,
            reference_scenario,
        )

    # Generate summary PDF
    generate_summary_pdf(
        scenario_data, reference_scenario, snakemake.output.sysgf_summary, run_name
    )

    logger.info("System analysis completed successfully")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sysgf_summary",
            run="20250514_dhsubnodes",
        )
    main(snakemake)
