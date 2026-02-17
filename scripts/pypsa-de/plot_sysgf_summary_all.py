#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
Generate summary plots and statistics across multiple runs for system analysis.

This script processes PyPSA networks from multiple runs and generates 
plots and metrics for energy system analysis and comparison across scenarios.
"""

import logging
import os
from pathlib import Path
import sys

sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import yaml

from scripts._helpers import configure_logging, mock_snakemake
from scripts.pypsa_de.plot_sysgf_summary import (
    get_all_colors, 
    create_summary_df, 
    create_cost_aggregation, 
    create_price_duration_curves,
    plot_price_duration_curves,
    plot_summary_metrics,
    plot_dual_comparison,
    plot_sensitivity_analysis,
    generate_summary_pdf
)

logger = logging.getLogger(__name__)


def process_networks_multi_runs(run_names, scenarios, planning_horizons):
    """
    Process networks from multiple runs and extract data for analysis.
    
    Parameters:
    ----------
    run_names : list of str
        List of run names (directory names under 'results/')
    scenarios : list of str
        List of scenario names to process
    planning_horizons : list of int
        List of planning horizon years
        
    Returns:
    -------
    tuple
        networks, summary_df, costs_agg, price_curves
    """
    networks = {}
    
    logger.info(f"Processing networks for runs: {run_names}")
    logger.info(f"Scenarios: {scenarios}")
    logger.info(f"Planning horizons: {planning_horizons}")

    # Process each run and scenario combination
    for run in run_names:
        networks_path = os.path.join("results", run)
        
        if not os.path.exists(networks_path):
            logger.warning(f"Run path {networks_path} does not exist, skipping")
            continue
            
        for scenario in scenarios:
            scenario_path = os.path.join(networks_path, scenario, "networks")
            if not os.path.exists(scenario_path):
                logger.warning(f"Scenario path {scenario_path} does not exist, skipping")
                continue

            for year in planning_horizons:
                # Find network file for this year
                network_file = None
                for file in os.listdir(scenario_path):
                    if file.endswith(f"{year}_final.nc"):
                        network_file = os.path.join(scenario_path, file)
                        break
                    # Fallback to non-final networks if final ones don't exist
                    elif file.endswith(f"{year}.nc") and network_file is None:
                        network_file = os.path.join(scenario_path, file)

                if not network_file:
                    logger.warning(
                        f"No network file found for run {run}, scenario {scenario} and year {year}"
                    )
                    continue

                # Load network and calculate metrics
                try:
                    # Add run name prefix to scenario for clarity when multiple runs
                    scenario_key = f"{run}_{scenario}" if len(run_names) > 1 else scenario
                    logger.info(f"Loading network for run {run}, scenario {scenario} and year {year}")
                    n = pypsa.Network(network_file)
                    networks[scenario_key] = n
                    logger.info(
                        f"Successfully loaded network for run {run}, scenario {scenario} and year {year}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing network for run {run}, scenario {scenario} and year {year}: {e}"
                    )

    if not networks:
        logger.error("No networks could be loaded")
        return None, None, None, None

    # Create dataframes
    summary_df = create_summary_df(networks)
    costs_agg = create_cost_aggregation(networks)
    price_curves = create_price_duration_curves(networks)

    return networks, summary_df, costs_agg, price_curves


def main(snakemake):
    """Main function to generate plots from network data across multiple runs."""
    # Configure logging
    configure_logging(snakemake)

    # Get parameters from snakemake
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons
    reference_scenario = snakemake.params.reference_scenario
    sensitivity_runs = snakemake.params.sensitivity_runs

    # Get color overrides from config if available
    try:
        override_colors = snakemake.params.plotting["override_tech_colors"]
    except:
        override_colors = {}

    # Create output directory
    output_path = os.path.dirname(snakemake.output.sysgf_summary)
    os.makedirs(output_path, exist_ok=True)

    # Handle run_name as single string or list
    if isinstance(run_name, str):
        run_names = [run_name]
    else:
        run_names = run_name
        
    logger.info(f"Generating system analysis for runs: {run_names}")

    # Process networks and collect data across all runs
    networks, summary_df, costs_agg, price_curves = process_networks_multi_runs(
        run_names, scenarios, planning_horizons
    )

    if not networks:
        logger.error("No networks could be loaded")
        return

    # Get color mapping
    colors = get_all_colors(override_colors)

    # Generate plots
    
    # 1. Plot price duration curves
    plot_price_duration_curves(price_curves, output_path)

    # 2. Plot summary metrics
    plot_summary_metrics(summary_df, output_path)

    # 3. Plot dual comparison if configured
    if (
        "dual_comparison" in snakemake.params.plotting
        and snakemake.params.plotting["dual_comparison"]["enable"]
    ):
        scenario_A = snakemake.params.plotting["dual_comparison"]["scenario_A"]
        scenario_B = snakemake.params.plotting["dual_comparison"]["scenario_B"]
        if scenario_A in costs_agg.index.get_level_values(
            0
        ) and scenario_B in costs_agg.index.get_level_values(0):
            plot_dual_comparison(costs_agg, scenario_A, scenario_B, colors, output_path)

    # 4. Plot sensitivity analysis if configured
    if sensitivity_runs:
        plot_sensitivity_analysis(
            costs_agg, sensitivity_runs, reference_scenario, colors, output_path
        )

    # 5. Generate summary PDF
    generate_summary_pdf(
        summary_df,
        costs_agg,
        reference_scenario,
        snakemake.output.sysgf_summary,
        "_".join(run_names) if len(run_names) > 1 else run_names[0]
    )

    # Save data for further analysis
    summary_df.to_csv(os.path.join(output_path, "summary_metrics.csv"))

    logger.info("System analysis completed successfully")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_sysgf_summary",
            run="Baseline",
        )
    main(snakemake)
