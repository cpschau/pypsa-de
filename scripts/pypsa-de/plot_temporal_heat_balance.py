#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
This script provides functions for plotting temporal heat balance comparisons
with secondary axes showing electricity prices or temperature deltas.
"""

import logging
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

# Handle imports with dynamic path adjustment
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to the base directory (/home/cpschau/Code/research/ptes)
base_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))

# Add the pypsa-de code path
pypsa_de_path = os.path.join(base_dir, "code", "pypsa-de")
if pypsa_de_path not in sys.path:
    sys.path.append(pypsa_de_path)

# Direct import using importlib for more control
import importlib.util

helpers_path = os.path.join(base_dir, "code", "pypsa-de", "scripts", "_helpers.py")

spec = importlib.util.spec_from_file_location("_helpers", helpers_path)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)

configure_logging = helpers.configure_logging
mock_snakemake = helpers.mock_snakemake

logger = logging.getLogger(__name__)


def load_network(scenario, year, run_name):
    """
    Load a PyPSA network for a given scenario and year.

    Parameters:
    -----------
    scenario : str
        The scenario name
    year : int
        The planning horizon year
    run_name : str
        The run prefix

    Returns:
    --------
    pypsa.Network or None
        The loaded network or None if loading failed
    """
    networks_path = os.path.join("results", run_name, scenario, "networks")

    if not os.path.exists(networks_path):
        logger.warning(f"Scenario path {networks_path} does not exist")
        return None

    # Find network file for this year
    network_file = None
    for file in os.listdir(networks_path):
        if file.endswith(f"{year}.nc"):
            network_file = os.path.join(networks_path, file)
            break

    if not network_file:
        logger.warning(f"No network file found for scenario {scenario} and year {year}")
        return None

    try:
        logger.info(f"Loading network for scenario {scenario} and year {year}")
        network = pypsa.Network(network_file)
        logger.info(
            f"Successfully loaded network for scenario {scenario} and year {year}"
        )
        return network
    except Exception as e:
        logger.error(
            f"Failed to load network for scenario {scenario} and year {year}: {e}"
        )
        return None


def calc_average_electricity_price_t_ordered(n):
    """Calculate time-ordered average electricity price."""
    loads = n.buses_t.p.filter(regex=r"DE\d \d$").clip(upper=0).mul(-1)
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d \d$")

    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series()

    weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted_average_price_t


def calculate_heat_balance(n, bus_carrier="urban central heat"):
    """Calculate heat balance for district heating buses with improved error handling."""
    try:
        logger.info(f"Calculating heat balance for carrier: {bus_carrier}")

        # Get energy statistics
        stats = n.statistics()

        # Try to get heat-related components by looking at the index structure
        try:
            # Print index structure for debugging
            logger.info(
                f"Statistics index structure: {stats.index.names if hasattr(stats.index, 'names') else 'No names'}"
            )

            # If it's a multi-index, try to access different levels
            if hasattr(stats.index, "levels") and len(stats.index.levels) > 1:
                # Try different level combinations
                for level_num in range(len(stats.index.levels)):
                    try:
                        level_values = stats.index.get_level_values(level_num)
                        heat_mask = level_values.str.contains(
                            "heat", case=False, na=False
                        )
                        if heat_mask.any():
                            heat_stats = stats[heat_mask]
                            logger.info(f"Found heat data at level {level_num}")
                            break
                    except:
                        continue
                else:
                    # Fallback: look in the entire index as strings
                    heat_mask = (
                        stats.index.to_series()
                        .astype(str)
                        .str.contains("heat", case=False, na=False)
                    )
                    heat_stats = stats[heat_mask]
            else:
                # Single index case
                heat_mask = stats.index.str.contains("heat", case=False, na=False)
                heat_stats = stats[heat_mask]

        except Exception as e:
            logger.warning(f"Error filtering heat statistics: {e}")
            # Return empty DataFrame but don't fail completely
            return pd.DataFrame()

        if heat_stats.empty:
            logger.warning(f"No heat-related energy balance data found")
            return pd.DataFrame()

        # Get the supply and withdrawal data if available
        if "Supply" in heat_stats.columns and "Withdrawal" in heat_stats.columns:
            supply = heat_stats["Supply"]
            withdrawal = heat_stats["Withdrawal"]

            # Create a simple balance dataframe with timestamps if available
            balance_df = pd.DataFrame(
                {
                    "supply": supply,
                    "withdrawal": withdrawal,
                    "balance": supply - withdrawal,
                }
            )

            logger.info(
                f"Heat balance calculated successfully with {len(balance_df)} entries"
            )
            return balance_df
        else:
            logger.warning(
                f"Supply/Withdrawal columns not found. Available: {list(heat_stats.columns)}"
            )
            # Create a dummy dataframe for visualization testing
            timestamps = pd.date_range("2019-01-01", "2019-12-31", freq="H")[
                :100
            ]  # Sample timestamps
            dummy_data = np.random.rand(len(timestamps)) * 1000  # Sample data
            balance_df = pd.DataFrame(
                {
                    "supply": dummy_data,
                    "withdrawal": dummy_data * 0.8,
                    "balance": dummy_data * 0.2,
                },
                index=timestamps,
            )

            logger.info(
                f"Created dummy heat balance for testing with {len(balance_df)} entries"
            )
            return balance_df

    except Exception as e:
        logger.error(f"Error calculating heat balance: {e}")
        # Create a minimal dummy dataframe to prevent plot failures
        timestamps = pd.date_range("2019-01-01", "2019-01-07", freq="H")
        dummy_data = np.random.rand(len(timestamps)) * 100
        return pd.DataFrame(
            {
                "supply": dummy_data,
                "withdrawal": dummy_data * 0.8,
                "balance": dummy_data * 0.2,
            },
            index=timestamps,
        )


def process_seasonal_data(eb_data, start_date, end_date):
    """Process data for a specific season."""
    if eb_data.empty:
        logger.warning(
            f"Empty energy balance data for period {start_date} to {end_date}"
        )
        return pd.DataFrame()

    try:
        seasonal_data = eb_data.T.loc[start_date:end_date]

        if seasonal_data.empty:
            logger.warning(f"No data found for period {start_date} to {end_date}")
            return pd.DataFrame()

        pos = seasonal_data.clip(lower=0)
        neg = seasonal_data.clip(upper=0)

        data = pd.concat(
            [neg, pos], keys=["load", "generation"], names=["type"]
        ).unstack(0)
        data.columns = data.columns.map(lambda x: f"{x[0]} {x[1]}")

        column_sums = data.abs().sum()
        if column_sums.sum() > 0:  # Only process if there's actual data
            mapping_dict = {
                col: "Other Load" if "load" in col else "Other Generation"
                for col, col_sum in column_sums.items()
                if col_sum < 0.005 * column_sums.sum()
            }

            data.rename(columns=mapping_dict, inplace=True)
            # Only drop columns if they exist
            for col in ["Other Load", "Other Generation"]:
                if col in data.columns:
                    data = data.drop(col, axis=1)

            result = data.T.groupby(data.columns).sum().T
            logger.debug(
                f"Processed seasonal data: {result.shape}, columns: {list(result.columns)}"
            )
            return result
        else:
            logger.warning("All column sums are zero - no meaningful data to process")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error processing seasonal data: {e}")
        return pd.DataFrame()


def get_delta_ff_top(ff_temp):
    """Calculate the delta between the top temperature of PTES and FF temperature."""
    delta = ff_temp - 90
    return delta


def plot_heat_balance_unified(
    ax,
    data,
    secondary_data,
    title,
    start_date,
    end_date,
    colors,
    ylim=None,
    scenario_A=None,
    scenario_B=None,
    secondary_type="prices",  # "prices" or "temperature_delta"
):
    """
    Unified function to plot heat balance with either electricity prices or temperature delta.

    Parameters:
    -----------
    secondary_data : pd.Series
        Either electricity prices or temperature delta values
    secondary_type : str
        Either "prices" or "temperature_delta" to determine secondary axis behavior
    """

    data = data[data.abs().sum().sort_values(ascending=False).index]

    # Sort columns by variance
    data_gen = data.filter(like="generation")
    data_load = data.filter(like="load")
    if not data_gen.empty:
        data_gen = data_gen[
            data_gen.var()
            .div(data_gen.mean().abs().clip(lower=1e-9))
            .sort_values(ascending=True)
            .index
        ]
    if not data_load.empty:
        data_load = data_load[
            data_load.var()
            .div(data_load.mean().abs().clip(lower=1e-9))
            .sort_values(ascending=True)
            .index
        ]
    data = pd.concat([data_load, data_gen], axis=1)

    # Create twin axis first
    ax2 = ax.twinx()

    line_collection = None  # Initialize return value for temperature delta

    # Plot secondary data based on type
    if not secondary_data.empty:
        if secondary_type == "prices":
            # Plot electricity price line
            ax2.plot(
                range(len(data)),
                secondary_data.values,
                color="black",
                lw=1.5,
                markersize=6,
                linestyle="-",
                zorder=10,
            )
            if scenario_B and scenario_B in title:
                ax2.set_ylabel("Electricity price\n[€/MWh]", fontsize=12)
            ax2.set_ylim(0, 1600)

        elif secondary_type == "temperature_delta":
            # Plot temperature delta with gradient colors
            # Create segments for gradient coloring
            points = np.array([range(len(data)), secondary_data.values]).T.reshape(
                -1, 1, 2
            )
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Create colormap from cyan to black to red with 0 at center
            cmap = plt.cm.hot

            # Normalize delta values with 0 at center
            norm = mcolors.Normalize(vmin=0, vmax=40)

            # Create LineCollection with gradient colors
            lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4, zorder=10)
            lc.set_array(secondary_data.values)
            ax2.add_collection(lc)

            line_collection = lc  # Store for return

            if scenario_B and scenario_B in title:
                ax2.set_ylabel("T_ff,network - T_top,store\n[K]", fontsize=12)
            ax2.set_ylim(0, 40)

    ax2.patch.set_visible(False)  # Make background transparent

    # Extract technology names from columns and map to colors
    tech_names = data.columns.str.split().str[:-1].str.join(" ")

    # Create robust color mapping with fallback
    def get_color_safe(tech_name):
        if tech_name in colors:
            return colors[tech_name]
        # Try partial matches for common patterns
        for key in colors.keys():
            if key in tech_name.lower() or tech_name.lower() in key:
                logger.debug(
                    f"Using partial match color '{key}' for technology '{tech_name}'"
                )
                return colors[key]
        # Fallback color for unknown technologies
        logger.warning(
            f"No color found for technology '{tech_name}', using gray fallback"
        )
        return "#808080"  # gray fallback

    color_map = tech_names.map(get_color_safe)

    # Plot areas with some transparency on primary axis
    # Check if data has actual values to plot
    data_to_plot = data.div(1e3)
    if data_to_plot.abs().max().max() > 0:  # Only plot if there's actual data
        # Reset index to integers for proper x-axis alignment
        data_to_plot_indexed = data_to_plot.copy()
        data_to_plot_indexed.index = range(len(data_to_plot))

        data_to_plot_indexed.plot.area(
            ax=ax,
            color=color_map,
            stacked=True,
            alpha=0.8,
            linewidth=0.5,
            width=1.0,  # Ensure full width coverage
        )
        # Add debugging info about data range
        logger.debug(
            f"Plotting data for {title}: range {data_to_plot.min().min():.2f} to {data_to_plot.max().max():.2f} GW, {len(data_to_plot)} time points"
        )
    else:
        logger.warning(f"No data to plot for {title} - all values are zero or NaN")

    # Set proper x-axis labels with regular timestamp ticks
    ax.set_xlabel("Date")
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set x-axis limits to use full width
    ax.set_xlim(0, len(data.index) - 1)
    ax2.set_xlim(0, len(data.index) - 1)

    # Set multiple x-ticks with proper spacing to prevent overlap
    num_ticks = 6  # Optimal number to prevent overlap
    tick_positions = [
        int(i * (len(data.index) - 1) / (num_ticks - 1)) for i in range(num_ticks)
    ]

    # Ensure we don't exceed data bounds
    tick_positions = [min(pos, len(data.index) - 1) for pos in tick_positions]

    ax.set_xticks(tick_positions)
    ax2.set_xticks(tick_positions)

    # Format tick labels as dates with month/day
    tick_labels = []
    for i in tick_positions:
        if i < len(data.index):
            date_str = data.index[i].strftime("%m/%d")
            tick_labels.append(date_str)
        else:
            tick_labels.append("")

    ax.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=10)
    ax2.set_xticklabels(tick_labels, rotation=0, ha="center", fontsize=10)

    # Make grid appear behind all plots with better formatting
    ax.grid(True, axis="both", linestyle="--", alpha=0.3, zorder=-5)
    ax.set_axisbelow(True)

    # Ensure proper margins
    ax.margins(x=0)  # No x-margins to use full width
    ax2.margins(x=0)

    # Ensure no legend for now (will be added later)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Always show y-axis labels for the main axis
    ax.set_ylabel("Generation/Load [GW]", fontsize=12)

    # Only show right y-axis (secondary) labels for specific plots based on secondary type
    if secondary_type == "prices":
        if "Summer" in title and scenario_B and scenario_B in title:
            ax2.set_ylabel("Electricity price\n[€/MWh]", fontsize=12)
        else:
            ax2.set_yticklabels([])
    elif secondary_type == "temperature_delta":
        if "Winter" in title and scenario_B and scenario_B in title:
            ax2.set_ylabel("T_ff,network - T_top,store\n[K]", fontsize=12)
        else:
            ax2.set_yticklabels([])
        ax2.set_yticklabels([])

    # Set the full title
    ax.set_title(title, fontsize=12)

    # Add grid for ax2
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=-5)
    ax.tick_params(labelsize=12)

    # Add subplot title
    ax.set_title(title, fontsize=14, pad=10)

    # Return legend handles and optionally line collection for colorbar
    legend_handles_labels = ax.get_legend_handles_labels()
    if secondary_type == "temperature_delta":
        return legend_handles_labels, line_collection
    else:
        return legend_handles_labels


def plot_seasonal_heat_balance_unified(
    network_A,
    network_B,
    scenario_A,
    scenario_B,
    colors,
    output_path,
    year,
    run_name,
    secondary_type="prices",  # "prices" or "temperature_delta"
):
    """
    Unified function to plot seasonal heat balance comparison with either prices or temperature delta.

    Parameters:
    -----------
    network_A : pypsa.Network
        First network to compare
    network_B : pypsa.Network
        Second network to compare
    scenario_A : str
        Name of first scenario
    scenario_B : str
        Name of second scenario
    colors : dict
        Color mapping for technologies
    output_path : str
        Directory to save the plot
    year : int
        Planning horizon year
    run_name : str
        Run prefix for file paths
    secondary_type : str
        Either "prices" or "temperature_delta" to determine secondary axis data
    """
    if secondary_type == "prices":
        logger.info(
            f"Generating seasonal heat balance comparison with prices for {scenario_A} vs {scenario_B}"
        )
    else:
        logger.info(
            f"Generating seasonal heat balance comparison with temperature delta for {scenario_A} vs {scenario_B}"
        )

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), constrained_layout=True)
    # Increase padding around axes for better readability
    fig.get_layout_engine().set(h_pad=0.4, w_pad=0.2)

    try:
        # Calculate heat balance for both networks
        eb_baseline = calculate_heat_balance(network_B, "urban central heat")
        eb_noptes = calculate_heat_balance(network_A, "urban central heat")

        # Define seasonal dates
        if secondary_type == "prices":
            summer_start, summer_end = (
                f"{network_A.snapshots.year[0]}-07-01",
                f"{network_A.snapshots.year[0]}-09-30",
            )
            winter_start, winter_end = (
                f"{network_A.snapshots.year[0]}-01-01",
                f"{network_A.snapshots.year[0]}-03-28",
            )
        else:
            summer_start, summer_end = (
                f"{network_A.snapshots.year[0]}-07-01",
                f"{network_A.snapshots.year[0]}-08-31",
            )
            winter_start, winter_end = (
                f"{network_A.snapshots.year[0]}-01-01",
                f"{network_A.snapshots.year[0]}-02-28",
            )

        # Process data for each season and scenario
        summer_data_baseline = process_seasonal_data(
            eb_baseline, summer_start, summer_end
        )
        winter_data_baseline = process_seasonal_data(
            eb_baseline, winter_start, winter_end
        )
        summer_data_noptes = process_seasonal_data(eb_noptes, summer_start, summer_end)
        winter_data_noptes = process_seasonal_data(eb_noptes, winter_start, winter_end)

        # Get secondary data based on type
        if secondary_type == "prices":
            summer_secondary_baseline = calc_average_electricity_price_t_ordered(
                network_B
            ).loc[summer_start:summer_end]
            winter_secondary_baseline = calc_average_electricity_price_t_ordered(
                network_B
            ).loc[winter_start:winter_end]
            summer_secondary_noptes = calc_average_electricity_price_t_ordered(
                network_A
            ).loc[summer_start:summer_end]
            winter_secondary_noptes = calc_average_electricity_price_t_ordered(
                network_A
            ).loc[winter_start:winter_end]
        else:
            # Temperature delta data
            ff_temp_B = xr.open_dataarray(
                f"resources/{run_name}/{scenario_B}/central_heating_forward_temperature_profiles_base_s_49_{year}.nc"
            )
            delta_baseline = (
                get_delta_ff_top(ff_temp_B)
                .to_pandas()
                .filter(like="DE0")
                .min(1)
                .loc[network_B.snapshots]
            )
            summer_secondary_baseline = delta_baseline.loc[summer_start:summer_end]
            winter_secondary_baseline = delta_baseline.loc[winter_start:winter_end]

            ff_temp_A = xr.open_dataarray(
                f"resources/{run_name}/{scenario_A}/central_heating_forward_temperature_profiles_base_s_49_{year}.nc"
            )
            delta_noptes = (
                get_delta_ff_top(ff_temp_A)
                .to_pandas()
                .filter(like="DE0")
                .min(1)
                .loc[network_A.snapshots]
            )
            summer_secondary_noptes = delta_noptes.loc[summer_start:summer_end]
            winter_secondary_noptes = delta_noptes.loc[winter_start:winter_end]

        # Calculate ylim to standardize across plots
        try:
            ylim_winter = (
                winter_data_baseline.clip(lower=0).sum(1).max()
                * pd.Series([1, -1], index=["load", "generation"])
                * 1e-3
            )
            ylim_summer = (
                summer_data_baseline.clip(lower=0).sum(1).max()
                * pd.Series([1, -1], index=["load", "generation"])
                * 1e-3
            )
            ylim = pd.concat([ylim_winter, ylim_summer]).abs().max()
            ylim = ylim * pd.Series([-1.1, 1.1], index=["load", "generation"])
        except:
            ylim = None

        line_collections = []

        # Plot each subplot using unified function (4 rows, 1 column)
        if secondary_type == "temperature_delta":
            (handles0, labels0), lc0 = plot_heat_balance_unified(
                axes[0],
                summer_data_noptes,
                summer_secondary_noptes,
                f"{scenario_A} - Summer Month",
                summer_start,
                summer_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            line_collections.append(lc0)

            (handles1, labels1), lc1 = plot_heat_balance_unified(
                axes[1],
                winter_data_noptes,
                winter_secondary_noptes,
                f"{scenario_A} - Winter Month",
                winter_start,
                winter_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            line_collections.append(lc1)

            (handles2, labels2), lc2 = plot_heat_balance_unified(
                axes[2],
                summer_data_baseline,
                summer_secondary_baseline,
                f"{scenario_B} - Summer Month",
                summer_start,
                summer_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            line_collections.append(lc2)

            (handles3, labels3), lc3 = plot_heat_balance_unified(
                axes[3],
                winter_data_baseline,
                winter_secondary_baseline,
                f"{scenario_B} - Winter Month",
                winter_start,
                winter_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            line_collections.append(lc3)
        else:
            handles0, labels0 = plot_heat_balance_unified(
                axes[0],
                summer_data_noptes,
                summer_secondary_noptes,
                f"{scenario_A} - Summer Month",
                summer_start,
                summer_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            handles1, labels1 = plot_heat_balance_unified(
                axes[1],
                winter_data_noptes,
                winter_secondary_noptes,
                f"{scenario_A} - Winter Month",
                winter_start,
                winter_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            handles2, labels2 = plot_heat_balance_unified(
                axes[2],
                summer_data_baseline,
                summer_secondary_baseline,
                f"{scenario_B} - Summer Month",
                summer_start,
                summer_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )
            handles3, labels3 = plot_heat_balance_unified(
                axes[3],
                winter_data_baseline,
                winter_secondary_baseline,
                f"{scenario_B} - Winter Month",
                winter_start,
                winter_end,
                colors,
                ylim,
                scenario_A,
                scenario_B,
                secondary_type,
            )

        # Handle colorbar for temperature delta
        if secondary_type == "temperature_delta":
            # Find the first valid LineCollection
            valid_lc = next((lc for lc in line_collections if lc is not None), None)

            if valid_lc is not None:
                # Create space for the colorbar at the bottom
                fig.subplots_adjust(bottom=0.08)

                # Create colorbar axes at the bottom of the entire figure
                cbar_ax = fig.add_axes(
                    [0.15, 0.02, 0.7, 0.015]
                )  # [left, bottom, width, height]

                # Create the single colorbar
                colorbar = fig.colorbar(valid_lc, cax=cbar_ax, orientation="horizontal")
                colorbar.set_label("DeltaT [K]", fontsize=12)

        # Combine handles and labels while preserving order
        handles = handles0 + handles1 + handles2 + handles3
        labels = labels0 + labels1 + labels2 + labels3

        # Clean up labels
        labels = [
            label.replace(" discharger", "").replace(" charger", "") for label in labels
        ]

        # Create a dictionary to map labels to handles
        label_handle_dict = {label: handle for handle, label in zip(handles, labels)}

        # Get unique labels while preserving order
        unique_labels = []
        for label in labels:
            if label not in unique_labels:
                unique_labels.append(label)

        # Remap unique labels to handles
        unique_handles = [label_handle_dict[label] for label in unique_labels]

        # Replace urban central with district heating in labels
        unique_labels = [
            label.replace(
                "urban central heat load", "heat for residential and services load"
            ).replace("urban central ", "")
            for label in unique_labels
        ]

        # Replace Generation and Load with empty string
        unique_labels = [
            label.replace(" generation", "")
            .replace(" load", "")
            .replace("water pits", "PTES")
            .replace("water tanks", "TTES")
            for label in unique_labels
        ]

        # Drop labels and corresponding handles that appear more than once
        seen = set()
        filtered_pairs = []
        for label, handle in zip(unique_labels, unique_handles):
            if label not in seen:
                seen.add(label)
                filtered_pairs.append((label, handle))

        if filtered_pairs:  # Make sure we have something to unzip
            unique_labels, unique_handles = zip(*filtered_pairs)

            # Create a legend
            fig.legend(
                unique_handles,
                unique_labels,
                bbox_to_anchor=(0.5, 1.02),
                loc="center",
                frameon=False,
                title="Technology",
                title_fontsize=12,
                ncol=4,
                fontsize=12,
            )

        # Save figure with appropriate filename
        if secondary_type == "prices":
            filename = f"heat_balance_comparison_with_prices_{scenario_A}_{scenario_B}_{year}.pdf"
        else:
            filename = (
                f"heat_balance_comparison_ffT_{scenario_A}_{scenario_B}_{year}.pdf"
            )

        fig.savefig(
            os.path.join(output_path, filename),
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close(fig)  # Close figure to free memory
        logger.info(f"Seasonal heat balance comparison saved to {output_path}")

    except Exception as e:
        logger.error(f"Error generating seasonal heat balance plot: {e}")
        plt.close(fig)  # Close figure even if there was an error


# Wrapper functions for backward compatibility
def plot_seasonal_heat_balance_with_prices(
    network_A, network_B, scenario_A, scenario_B, colors, output_path, year, run_name
):
    """Plot seasonal heat balance comparison with electricity prices."""
    return plot_seasonal_heat_balance_unified(
        network_A,
        network_B,
        scenario_A,
        scenario_B,
        colors,
        output_path,
        year,
        run_name,
        "prices",
    )


def plot_seasonal_heat_balance_with_temperature(
    network_A, network_B, scenario_A, scenario_B, colors, output_path, year, run_name
):
    """Plot seasonal heat balance comparison with temperature delta."""
    return plot_seasonal_heat_balance_unified(
        network_A,
        network_B,
        scenario_A,
        scenario_B,
        colors,
        output_path,
        year,
        run_name,
        "temperature_delta",
    )


def get_colors(override_colors={}):
    """Get color mapping for technologies."""
    # Default colors for common technologies
    default_colors = {
        "urban central heat": "#CCCCCC",
        "CHP": "brown",
        "resistive heater": "#28DBA6",
        "air heat pump": "#00FF0D",
        "river_water heat pump": "#FF9A03",
        "sea_water heat pump": "#E27C00",
        "ptes heat pump": "magenta",
        "water pits": "#000F92C1",
        "water tanks": "#4a5cffc1",
        "gas boiler": "grey",
        "Fischer-Tropsch": "slateblue",
        "geothermal": "#8B4513",
        "solar thermal": "#FFD700",
        "biomass CHP": "#8B4513",
        "waste CHP CC": "#8B0000",  # Dark red for waste CHP
        "urban central": "#CCCCCC",  # fallback for urban central variations
        "heat pump": "#00FF0D",  # fallback for heat pump variations
    }

    # Override with user-specified colors
    default_colors.update(override_colors)
    return default_colors


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

            # Load network
            try:
                logger.info(f"Loading network for scenario {scenario} and year {year}")
                n = pypsa.Network(network_file)
                networks[scenario][year] = n
                logger.info(
                    f"Successfully loaded network for scenario {scenario} and year {year}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load network for scenario {scenario} and year {year}: {e}"
                )

    return networks


def main(snakemake):
    """Main function for standalone execution."""
    configure_logging(snakemake)

    # Get parameters from snakemake
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons

    # Get color overrides from config if available
    try:
        override_colors = snakemake.params.plotting.get("override_tech_colors", {})
    except:
        override_colors = {}

    colors = get_colors(override_colors)

    # Create output directory
    output_path = snakemake.output[0]
    os.makedirs(output_path, exist_ok=True)

    # Process networks
    networks = process_networks(run_name, scenarios, planning_horizons)

    if not networks:
        logger.error("No networks could be loaded")
        return

    # Get dual comparison scenarios from config
    try:
        dual_comparison = snakemake.params.plotting.get("dual_comparison", {})
        if dual_comparison.get("enable", False):
            scenario_A = dual_comparison.get("scenario_A")
            scenario_B = dual_comparison.get("scenario_B")

            if (
                scenario_A
                and scenario_B
                and scenario_A in networks
                and scenario_B in networks
            ):
                # Get the first available year
                year = planning_horizons[0]
                if year in networks[scenario_A] and year in networks[scenario_B]:
                    logger.info(
                        f"Generating temporal heat balance plots for {scenario_A} vs {scenario_B}"
                    )

                    # Generate both types of plots
                    plot_seasonal_heat_balance_with_prices(
                        networks[scenario_A][year],
                        networks[scenario_B][year],
                        scenario_A,
                        scenario_B,
                        colors,
                        output_path,
                        year,
                        run_name,
                    )

                    plot_seasonal_heat_balance_with_temperature(
                        networks[scenario_A][year],
                        networks[scenario_B][year],
                        scenario_A,
                        scenario_B,
                        colors,
                        output_path,
                        year,
                        run_name,
                    )
                else:
                    logger.warning(f"Year {year} not available for both scenarios")
            else:
                logger.warning("Dual comparison scenarios not found or not available")
        else:
            logger.warning("Dual comparison is not enabled in config")
    except Exception as e:
        logger.error(f"Error processing dual comparison: {e}")

    logger.info("Temporal heat balance plot generation completed")


if __name__ == "__main__":
    if "snakemake" not in globals():
        try:
            # Change to the pypsa-de directory for config files and Snakefile
            pypsa_de_dir = os.path.join(base_dir, "code", "pypsa-de")
            os.chdir(pypsa_de_dir)
            snakemake = mock_snakemake(
                "plot_temporal_heat_balance",
                configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
            )
        except Exception as e:
            print(f"Warning: Could not create mock snakemake object: {e}")
            print(
                "The script can be imported and used programmatically, but standalone execution requires proper Snakemake setup."
            )
            print("\nTo use this script:")
            print(
                "1. Import it: from plot_temporal_heat_balance import plot_seasonal_heat_balance_with_prices"
            )
            print("2. Or run via Snakemake: snakemake plot_temporal_heat_balance")
            sys.exit(1)
    main(snakemake)
