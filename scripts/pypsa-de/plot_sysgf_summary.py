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

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
import numpy as np
import pandas as pd
import pypsa
import seaborn as sns
import yaml
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "code", "pypsa-de"))
from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def calc_ptes_cycles(n, mean=True):
    """Calculate the number of cycles for PTES (water pits) systems."""
    pits_de = n.stores.filter(regex=r"DE0.*water pits", axis=0).query("e_nom_opt > 0")
    if pits_de.empty:
        return 0

    discharge = (
        n.links_t.p0.filter(regex=r"DE.*pits discharger")
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
    )
    discharge.index = discharge.index.str.replace(" discharger", "")
    no_cycles = discharge.div(pits_de.e_nom_opt)
    if mean:
        return no_cycles.mean()
    else:
        return no_cycles


def calc_average_dh_price(n):
    """Calculate average district heating price in EUR/MWh."""
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


def calc_average_elec_price(n):
    """Calculate average electricity price in EUR/MWh."""
    elec_mps = n.buses_t.marginal_price.filter(regex=r"DE0 \d+$")
    elec_demand = n.loads_t.p.filter(regex=r"DE\d.*(\d+|electricity|EV)$")

    if elec_mps.empty or elec_demand.empty:
        return 0

    elec_demand.columns = elec_demand.columns.str.split(" ").str[:2].str.join(" ")
    elec_demand = elec_demand.T.groupby(elec_demand.columns).sum().T
    elec_costs = (elec_mps * elec_demand).sum().sum()

    if elec_demand.sum().sum() == 0:
        return 0

    return elec_costs / elec_demand.sum().sum()


def calc_average_dh_price_t_ordered(n):
    """Calculate time-ordered average district heating price."""
    loads = n.loads_t.p.filter(
        regex=r"DE\d.*(urban central|low-temperature) heat"
    ).clip(lower=0)
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d.*urban central heat")

    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series()

    weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted_average_price_t


def calc_average_electricity_price_t_ordered(n):
    """Calculate time-ordered average electricity price."""
    loads = n.buses_t.p.filter(regex=r"DE\d \d$").clip(upper=0).mul(-1)
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d \d$")

    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series()

    weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted_average_price_t


def calc_curtailment_de(n):
    """Calculate curtailment in TWh."""
    try:
        curtailment = (
            n.statistics.curtailment(groupby=["bus", "carrier"], nice_names=False)
            .xs("Generator", level=0)
            .filter(regex=r"DE.*wind|solar")
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
            @ n.generators_t.p.filter(regex=r"DE0.*heat vent")
        ).sum() / 1e6
        return heat_venting
    except:
        return 0


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


def calc_ic_capex_correction(n, country):
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


def calc_system_costs_country(n, country):
    """Calculate system costs for a country."""
    s = n.statistics
    capex_country = s.capex(groupby=["bus", "carrier"]).filter(regex=country, axis=0)
    opex_country = s.opex(groupby=["bus", "carrier"]).filter(regex=country, axis=0)

    system_costs_country = capex_country.sum().sum() + opex_country.sum().sum()
    ic_costs_correction = calc_ic_capex_correction(n, country).sum()

    return system_costs_country + ic_costs_correction


def calculate_district_heating_costs(n: pypsa.Network) -> float:
    district_heating_carriers = np.array([])
    for c in n.iterate_components():
        if c.name in ["Store", "Link", "Generator"]:
            # Filter rows where any column contains "urban central"
            district_heating_carriers = np.append(
                district_heating_carriers,
                c.df[
                    c.df.apply(
                        lambda x: x.astype(str)
                        .str.contains("urban central", case=False)
                        .any(),
                        axis=1,
                    )
                ].carrier.unique(),
            )

    capex = n.statistics.capex(
        groupby=["bus", "carrier", "bus_carrier"], nice_names=False
    ).filter(like="DE0 ")
    opex = n.statistics.opex(
        groupby=["bus", "carrier", "bus_carrier"], nice_names=False
    ).filter(like="DE0 ")

    capex_dh = capex[
        capex.index.get_level_values("carrier").isin(district_heating_carriers)
    ]
    opex_dh = opex[
        opex.index.get_level_values("carrier").isin(district_heating_carriers)
    ]

    return capex_dh.sum() + opex_dh.sum()


def calc_h2_store_capacity(n):
    """Calculate hydrogen storage capacity in TWh."""
    h2_stores = n.stores.filter(regex=r"DE0.*H2 Store", axis=0)
    return h2_stores.e_nom_opt.sum() / 1e6  # TWh


def calc_costs_per_tech_country(n, country):
    """Calculate costs per technology for a specific country."""
    capex = n.statistics.capex(groupby=["bus", "carrier"], nice_names=False).filter(
        regex=country, axis=0
    )

    opex = n.statistics.opex(groupby=["bus", "carrier"], nice_names=False).filter(
        regex=country, axis=0
    )

    union_index = capex.index.union(opex.index)
    capex = capex.reindex(union_index).fillna(0)
    opex = opex.reindex(union_index).fillna(0)

    # Add Index level for the costs
    total = pd.concat([capex, opex], axis=1, keys=["capex", "opex"])
    total = total.droplevel(["component", "bus"])

    # Sum up the costs for each technology
    total = total.groupby(total.index).sum()

    ic_capex_correction = calc_ic_capex_correction(n, country)
    total.loc[ic_capex_correction.index, "capex"] + ic_capex_correction

    return total


def extract_part(scenario):
    """Extract parameter part from scenario name."""
    for prefix in ["Low", "High", "0.5", "2"]:
        if scenario.startswith(prefix):
            return scenario[len(prefix) :]
    return scenario


def clean_and_aggregate_columns(costs_agg):
    """Clean and aggregate technology columns for better visualization."""
    # Aggregate grid technologies
    costs_agg["power grid"] = costs_agg[
        ["AC", "DC", "electricity distribution grid"]
    ].sum(axis=1)
    costs_agg.drop(["AC", "DC", "electricity distribution grid"], axis=1, inplace=True)

    # Replace residential, services, rural, and urban decentral from the column names
    costs_agg.columns = costs_agg.columns.str.replace("residential ", "")
    costs_agg.columns = costs_agg.columns.str.replace("services ", "")
    costs_agg.columns = costs_agg.columns.str.replace("rural ", "decentral ")
    costs_agg.columns = costs_agg.columns.str.replace("urban decentral ", "decentral ")
    costs_agg = costs_agg.T.groupby(costs_agg.columns).sum().T

    # Aggregate heat pumps
    sum_before = costs_agg.sum().sum()
    costs_agg["decentral heat pump"] = costs_agg.filter(
        regex=r"decentral (ground|air) heat pump"
    ).sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(regex=r"decentral (ground|air) heat pump").columns,
        axis=1,
        inplace=True,
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate H2 pipeline technologies
    sum_before = costs_agg.sum().sum()
    costs_agg["H2 pipeline"] = costs_agg.filter(regex=r"H2 pipeline").sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(regex=r"H2 pipeline").columns.drop(
            "H2 pipeline", errors="ignore"
        ),
        axis=1,
        inplace=True,
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate wind power
    sum_before = costs_agg.sum().sum()
    costs_agg["wind power"] = costs_agg.filter(regex=r"(on|off)wind").sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(regex=r"(on|off)wind").columns, axis=1, inplace=True
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate oil technologies (but not CHP)
    sum_before = costs_agg.sum().sum()
    costs_agg["oil"] = costs_agg.filter(regex=r"(^| )oil(?!.*CHP)").sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(regex=r"(^| )oil(?!.*CHP)").columns.drop(
            "oil", errors="ignore"
        ),
        axis=1,
        inplace=True,
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate solar technologies
    sum_before = costs_agg.sum().sum()
    costs_agg["PV"] = costs_agg.filter(regex=r"solar").sum(axis=1)
    costs_agg.drop(costs_agg.filter(regex=r"solar").columns, axis=1, inplace=True)
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate biomass technologies
    sum_before = costs_agg.sum().sum()
    costs_agg["solid biomass"] = costs_agg.filter(regex=r"solid biomass(?!.*CHP)").sum(
        axis=1
    )
    cols_to_drop = costs_agg.filter(regex=r"solid biomass(?!.*CHP)").columns
    cols_to_drop = cols_to_drop[cols_to_drop != "solid biomass"]
    costs_agg.drop(cols_to_drop, axis=1, inplace=True)
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate biogas
    sum_before = costs_agg.sum().sum()
    costs_agg["biogas"] = costs_agg.filter(regex=r"biogas").sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(regex=r"biogas").columns.drop("biogas", errors="ignore"),
        axis=1,
        inplace=True,
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate battery technologies
    sum_before = costs_agg.sum().sum()
    costs_agg["battery"] = costs_agg.filter(regex=r"battery").sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(regex=r"battery").columns.drop("battery", errors="ignore"),
        axis=1,
        inplace=True,
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    # Aggregate CHP technologies
    sum_before = costs_agg.sum().sum()
    costs_agg["CHP"] = costs_agg.filter(like="CHP").sum(axis=1)
    costs_agg.drop(
        costs_agg.filter(like="CHP").columns.drop("CHP", errors="ignore"),
        axis=1,
        inplace=True,
    )
    assert abs(sum_before - costs_agg.sum().sum()) < 1

    return costs_agg


def create_summary_df(networks):
    """Create summary dataframe with key metrics from networks."""
    df = pd.DataFrame(
        columns=[
            "scenario",
            "year",
            "total_system_costs_bnEUR",
            "total_system_costs_DE_bnEUR",
            "district_heating_costs_DE_bnEUR",
            "PTES_capacity_TWh",
            "PTES_capacity_GW",
            "PTES_no_cycles",
            "TTES_capacity_TWh",
            "TTES_capacity_GW",
            "H2_store_TWh",
            "dh_price_EUR_per_MWh",
            "electricity_price_EUR_per_MWh",
            "peak_electricity_price_EUR_per_MWh",
            "peak_dh_price_EUR_per_MWh",
            "curtailment_TWh",
            "heat_venting_TWh",
            "vRES_capacity_GW",
            "CHP_capacity_GW",
            "resistive_heater_GW",
        ]
    )

    for scenario, years in networks.items():
        networks_scenario = networks[scenario]
        for year, n in networks_scenario.items():
            # Extract metrics from network
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "scenario": scenario,
                            "year": year,
                            "total_system_costs_bnEUR": (
                                n.statistics.capex().sum() + n.statistics.opex().sum()
                            )
                            / 1e9,
                            "total_system_costs_DE_bnEUR": calc_system_costs_country(
                                n, "DE"
                            )
                            / 1e9,
                            "district_heating_costs_DE_bnEUR": (
                                calculate_district_heating_costs(n)
                            )
                            / 1e9,
                            "PTES_capacity_TWh": n.stores.filter(
                                regex=r"DE.*urban central water pits", axis=0
                            )
                            .e_nom_opt.div(1e6)
                            .sum(),
                            "PTES_capacity_GW": n.links.filter(
                                regex=r"DE.*urban central water pits discharger", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "PTES_no_cycles": calc_ptes_cycles(n),
                            "TTES_capacity_TWh": n.stores.filter(
                                regex=r"DE.*urban central water tanks", axis=0
                            )
                            .e_nom_opt.div(1e6)
                            .sum(),
                            "TTES_capacity_GW": n.links.filter(
                                regex=r"DE.*urban central water tanks discharger",
                                axis=0,
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "H2_store_TWh": calc_h2_store_capacity(n),
                            "dh_price_EUR_per_MWh": calc_average_dh_price(n),
                            "electricity_price_EUR_per_MWh": calc_average_elec_price(n),
                            "peak_electricity_price_EUR_per_MWh": n.buses_t.marginal_price.filter(
                                regex=r"DE0 \d+$"
                            )
                            .max()
                            .max(),
                            "peak_dh_price_EUR_per_MWh": n.buses_t.marginal_price.filter(
                                regex=r"DE0.*urban central heat"
                            )
                            .max()
                            .max(),
                            "curtailment_TWh": calc_curtailment_de(n),
                            "heat_venting_TWh": calc_heat_venting_de(n),
                            "vRES_capacity_GW": n.generators.filter(
                                regex=r"DE.*(onwind|offwind|solar-)",
                                axis=0,
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "CHP_capacity_GW": n.links.filter(regex=r"DE.*CHP", axis=0)
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "resistive_heater_GW": n.links.filter(
                                regex=r"DE.*resistive heater", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    # Add parameter extraction for sensitivity analysis
    df["parameter"] = df.scenario.apply(extract_part)
    df["case"] = df.apply(lambda x: x.scenario.replace(x.parameter, ""), axis=1)

    return df


def create_cost_aggregation(networks):
    """Create cost aggregation dataframe by technology."""
    costs_agg = pd.DataFrame()

    for scenario, years in networks.items():
        networks_scenario = networks[scenario]
        for year, n in networks_scenario.items():
            costs_de = calc_costs_per_tech_country(n, "DE").sum(axis=1)
            costs_de["neighbour countries"] = (
                n.statistics.capex().sum() + n.statistics.opex().sum() - costs_de.sum()
            )
            costs_de["scenario"] = scenario
            costs_de["year"] = year
            costs_de = costs_de.to_frame().T.set_index("scenario")
            costs_agg = pd.concat([costs_agg, costs_de])

    # Add parameter extraction for sensitivity analysis
    costs_agg["parameter"] = costs_agg.index.to_series().apply(extract_part)
    costs_agg["case"] = costs_agg.apply(
        lambda x: x.name.replace(x.parameter, ""), axis=1
    )
    costs_agg.set_index(["parameter", "case"], inplace=True)

    # Clean and aggregate technology columns
    costs_agg = clean_and_aggregate_columns(costs_agg)

    costs_agg = costs_agg.set_index(
        [
            costs_agg.index.get_level_values(1) + costs_agg.index.get_level_values(0),
            "year",
        ]
    )

    return costs_agg


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

    # Create dataframes
    summary_df = create_summary_df(networks)
    costs_agg = create_cost_aggregation(networks)

    return networks, summary_df, costs_agg


def get_colors(networks, override_colors={}):
    """Get colors from network and override with custom colors if provided."""
    # Extract colors from the first network
    first_network = next(iter(networks.values()))[
        next(iter(networks[next(iter(networks))]))
    ]
    colors = first_network.carriers.color
    extended_index = colors.index.union(override_colors.keys())
    colors = colors.reindex(extended_index).fillna("black").to_dict()
    colors.update(override_colors)

    return colors


def calc_average_electricity_price_t_ordered(n):
    """Calculate time-ordered average electricity price."""
    loads = n.buses_t.p.filter(regex=r"DE\d \d$").clip(upper=0).mul(-1)
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d \d$")

    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series()

    weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted_average_price_t


def calculate_heat_balance(network, bus_name):
    """Calculate heat balance for a specific bus type."""
    eb = (
        network.statistics.energy_balance(
            groupby=["bus", "carrier", "bus_carrier"], aggregate_time=False
        )
        .xs(bus_name, level="bus_carrier")
        .filter(regex=r"DE\d", axis=0)
    )

    eb.index = eb.index.droplevel(["component", "bus"])
    eb = eb.groupby(eb.index).sum()
    eb.columns = pd.to_datetime(eb.columns)
    eb = eb.div(network.snapshot_weightings.generators)
    return eb


def process_seasonal_data(eb_data, start_date, end_date):
    """Process data for a specific season."""
    seasonal_data = eb_data.T.loc[start_date:end_date]
    pos = seasonal_data.clip(lower=0)
    neg = seasonal_data.clip(upper=0)

    data = pd.concat([neg, pos], keys=["load", "generation"], names=["type"]).unstack(0)
    data.columns = data.columns.map(lambda x: f"{x[0]} {x[1]}")

    column_sums = data.abs().sum()
    mapping_dict = {
        col: "Other Load" if "load" in col else "Other Generation"
        for col, col_sum in column_sums.items()
        if col_sum < 0.01 * column_sums.sum()
    }

    data.rename(columns=mapping_dict, inplace=True)
    # Only drop columns if they exist
    for col in ["Other Load", "Other Generation"]:
        if col in data.columns:
            data = data.drop(col, axis=1)
    return data.T.groupby(data.columns).sum().T


def plot_heat_balance(ax, data, prices, title, start_date, end_date, colors, ylim=None):
    """Plot heat balance for a specific time period."""

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

    # Plot the electricity price line first on secondary axis
    # Using integer indices for plotting the line to match bar positions
    if not prices.empty:
        ax2.plot(
            range(len(data)),
            prices.values,
            color="black",
            lw=1.5,
            markersize=6,
            linestyle="-",
            zorder=10,
        )

    if "chargeboost" in title or "bidiboost" in title:
        ax2.set_ylabel("Electricity price\n[€/MWh]", fontsize=12)
    ax2.patch.set_visible(False)  # Make background transparent

    # Now plot bars with some transparency on primary axis
    data.div(1e3).plot.bar(
        ax=ax,
        color=data.columns.str.split().str[:-1].str.join(" ").map(colors),
        stacked=True,
        width=1,
        alpha=0.9,
        position=0,
    )

    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(ylim)

    # Set x-ticks for both axes
    ax.set_xticks([len(data.index) // 2])
    ax.set_xticklabels([f"{start_date} - {end_date}"], rotation=0, ha="center")

    # Make grid appear behind all plots
    ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=-5)
    ax.set_axisbelow(True)

    # Ensure no legend for now (will be added later)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    ax2.set_ylim(0, 1600)

    # If PTES in title set ax2 yticklabels to ""
    if "NoPTES" in title or "noboost" in title:
        ax2.set_yticklabels([])
    if "chargeboost" in title or "bidiboost" in title:
        ax.set_yticklabels([])

    if "NoPTES" in title or "noboost" in title:
        if "Summer" in title:
            ax.set_ylabel("Summer month:\nGeneration/Load\n[GW]", fontsize=12)
        else:
            ax.set_ylabel("Winter month:\nGeneration/Load\n[GW]", fontsize=12)

    if "Winter" in title:
        title = ""
    else:
        title = title.split(" -")[0]
    ax.set_title(title, fontsize=12)

    # Add grid for ax2
    ax2.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=-5)
    ax.tick_params(labelsize=12)
    return ax.get_legend_handles_labels()


def plot_seasonal_heat_balance(
    network_A, network_B, scenario_A, scenario_B, colors, output_path, year
):
    """Plot seasonal heat balance comparison between two scenarios."""
    logger.info(
        f"Generating seasonal heat balance comparison for {scenario_A} vs {scenario_B}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 6), constrained_layout=True)
    # Increase padding around axes
    fig.get_layout_engine().set(w_pad=0.2)

    try:
        # Calculate heat balance for both networks
        eb_baseline = calculate_heat_balance(network_B, "urban central heat")
        eb_noptes = calculate_heat_balance(network_A, "urban central heat")

        # Define seasonal dates
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
        summer_prices_baseline = calc_average_electricity_price_t_ordered(
            network_B
        ).loc[summer_start:summer_end]
        winter_data_baseline = process_seasonal_data(
            eb_baseline, winter_start, winter_end
        )
        winter_prices_baseline = calc_average_electricity_price_t_ordered(
            network_B
        ).loc[winter_start:winter_end]

        summer_data_noptes = process_seasonal_data(eb_noptes, summer_start, summer_end)
        summer_prices_noptes = calc_average_electricity_price_t_ordered(network_A).loc[
            summer_start:summer_end
        ]
        winter_data_noptes = process_seasonal_data(eb_noptes, winter_start, winter_end)
        winter_prices_noptes = calc_average_electricity_price_t_ordered(network_A).loc[
            winter_start:winter_end
        ]

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

        # Plot each subplot
        handles0, labels0 = plot_heat_balance(
            axes[0, 1],
            summer_data_baseline,
            summer_prices_baseline,
            f"{scenario_B} - Summer Month",
            summer_start,
            summer_end,
            colors,
            ylim,
        )
        handles1, labels1 = plot_heat_balance(
            axes[1, 1],
            winter_data_baseline,
            winter_prices_baseline,
            f"{scenario_B} - Winter Month",
            winter_start,
            winter_end,
            colors,
            ylim,
        )
        handles2, labels2 = plot_heat_balance(
            axes[0, 0],
            summer_data_noptes,
            summer_prices_noptes,
            f"{scenario_A} - Summer Week",
            summer_start,
            summer_end,
            colors,
            ylim,
        )
        handles3, labels3 = plot_heat_balance(
            axes[1, 0],
            winter_data_noptes,
            winter_prices_noptes,
            f"{scenario_A} - Winter Week",
            winter_start,
            winter_end,
            colors,
            ylim,
        )

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
                bbox_to_anchor=(0.5, 1.12),
                loc="center",
                frameon=False,
                title="Technology",
                title_fontsize=12,
                ncol=3,
                fontsize=12,
            )

        # Save figure
        fig.savefig(
            os.path.join(
                output_path,
                f"heat_balance_comparison_{scenario_A}_{scenario_B}_{year}.pdf",
            ),
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close(fig)  # Close figure to free memory
        logger.info(f"Seasonal heat balance comparison saved to {output_path}")

    except Exception as e:
        logger.error(f"Error generating seasonal heat balance plot: {e}")
        plt.close(fig)  # Close figure even if there was an error


def plot_dual_comparison(
    networks, costs_agg, scenario_A, scenario_B, colors, output_path
):
    """Plot comparison between two scenarios (typically with/without PTES)."""
    logger.info(f"Plotting dual comparison between {scenario_A} and {scenario_B}")

    if scenario_A not in costs_agg.index.get_level_values(
        0
    ) or scenario_B not in costs_agg.index.get_level_values(0):
        logger.error(f"Scenarios {scenario_A} or {scenario_B} not found in data")
        return

    # Extract Baseline scenario data and differences by year
    years = costs_agg.index.get_level_values(1).unique()

    for year in years:
        logger.info(f"Generating dual comparison for year {year}")

        fig, ax = plt.subplots(ncols=2, figsize=(12, 6))

        # Extract data for this specific year
        costs_baseline = (
            costs_agg.loc[(scenario_A, year)].squeeze().sort_values(ascending=False)
        )
        costs_baseline = costs_baseline[(costs_baseline != 0)]  # Drop zero-cost entries

        # Extract differences between scenarios for this year
        df_diff = (
            costs_agg.loc[(scenario_B, year)]
            .squeeze()
            .sub(costs_baseline.reindex(costs_agg.columns).fillna(0))
        )
        df_diff = df_diff.loc[(df_diff != 0)]  # Drop zero-difference entries

        # Group small contributions into "other technologies"
        other_indices = df_diff[df_diff.abs() < 0.02 * df_diff.abs().sum()].index
        df_diff["other technologies"] = df_diff[other_indices].sum()
        df_diff.drop(other_indices, inplace=True)

        # Handle other entries in baseline scenario
        other_indices_baseline = costs_baseline.index.intersection(other_indices).union(
            costs_baseline.index.difference(df_diff.index)
        )
        costs_baseline["other technologies"] = costs_baseline[
            other_indices_baseline
        ].sum()
        costs_baseline.drop(other_indices_baseline, inplace=True)

        # Order by magnitude
        costs_baseline = costs_baseline.loc[
            costs_baseline.drop("neighbour countries", errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .index.append(pd.Index(["neighbour countries"]))
        ]

        df_diff = df_diff.loc[
            df_diff.drop("neighbour countries", errors="ignore")
            .abs()
            .sort_values(ascending=False)
            .index.append(pd.Index(["neighbour countries"]))
        ]

        # Plot Baseline scenario
        costs_baseline.to_frame().T.div(1e9).plot.bar(
            stacked=True,
            ax=ax[0],
            color=costs_baseline.index.map(colors).fillna("black"),
            legend=False,
        )
        ax[0].set_ylabel("Costs per Technology [bn€]", fontsize=12)
        ax[0].set_xlabel("")
        ax[0].set_title(f"Total System Costs in {scenario_A} Scenario ({year})")
        ax[0].set_ylim(0, 140)

        # Plot differences
        df_diff.to_frame().T.div(1e9).plot.bar(
            stacked=True,
            ax=ax[1],
            color=df_diff.index.map(colors).fillna("black"),
            legend=False,
        )
        handles, labels = ax[1].get_legend_handles_labels()

        # Add horizontal line at 0
        ax[1].hlines(
            y=0,
            xmin=ax[1].get_xlim()[0],
            xmax=ax[1].get_xlim()[1],
            color="black",
            linewidth=0.5,
            zorder=3,
        )

        # Add markers for the total and German system cost changes
        markers_total = df_diff.sum() / 1e9
        ax[1].hlines(
            y=markers_total,
            xmin=-0.42,
            xmax=0.42,
            color="black",
            linewidth=3,
            zorder=2,
            label="net difference",
            path_effects=[patheffects.withStroke(linewidth=3)],
        )

        markers_de = df_diff.drop("neighbour countries", errors="ignore").sum() / 1e9
        ax[1].hlines(
            y=markers_de,
            xmin=-0.42,
            xmax=0.42,
            color="black",
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label="German system difference",
            path_effects=[patheffects.withStroke(linewidth=3)],
        )

        # Annotate differences
        ref_sum = costs_baseline.sum() / 1e9
        ax[1].annotate(
            (
                f"+{markers_total:.1f}\n(+{markers_total/ref_sum*100:.2f}%)"
                if markers_total > 0
                else f"{markers_total:.1f}\n({markers_total/ref_sum*100:.2f}%)"
            ),
            (0, -0.8),
            textcoords="offset points",
            xytext=(0, -70),
            ha="center",
            va="bottom",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
            bbox=dict(
                boxstyle="round,pad=0", edgecolor="none", facecolor="white", alpha=0
            ),
            fontsize=14,
        )

        ref_sum_de = (
            costs_baseline.drop("neighbour countries", errors="ignore").sum() / 1e9
        )
        ax[1].annotate(
            (
                f"+{markers_de:.1f}\n(+{markers_de/ref_sum_de*100:.2f}%)"
                if markers_de > 0
                else f"{markers_de:.1f}\n({markers_de/ref_sum_de*100:.2f}%)"
            ),
            (0, -0.8),
            textcoords="offset points",
            xytext=(0, 100),
            ha="center",
            va="top",
            path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
            bbox=dict(
                boxstyle="round,pad=0", edgecolor="none", facecolor="white", alpha=0
            ),
            fontsize=14,
        )

        # Add titles and labels
        ax[1].set_ylabel("Cost Difference [bn€]", fontsize=12)
        ax[1].set_title(
            f"Difference in Investments ({scenario_A} - {scenario_B}) ({year})"
        )
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position("right")
        ax[1].set_xlabel("")

        # Increase tick size
        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)

        # Add legend
        from matplotlib.lines import Line2D

        second_legend_lines = [
            Line2D([0], [0], color="black", linewidth=3),
            Line2D([0], [0], color="black", linestyle="dashed", linewidth=3),
        ]
        second_legend_labels = ["Total System Costs", "German System Costs"]

        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(1.05, 0.6),
            loc="center left",
            ncol=1,
            frameon=False,
            title="Technology",
        )

        fig.legend(
            second_legend_lines,
            second_legend_labels,
            bbox_to_anchor=(1.05, 0.15),
            loc="center left",
            ncol=1,
            frameon=False,
            title="Net difference in",
            fontsize=12,
        )

        # Save figure
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_path, f"dual_comparison_{scenario_A}_{scenario_B}_{year}.pdf"
            ),
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close(fig)  # Close figure to free memory

        # Add seasonal heat balance plot if we have the network data
        if scenario_A in networks and scenario_B in networks:
            # Check if networks contain this year
            if year in networks[scenario_A] and year in networks[scenario_B]:
                plot_seasonal_heat_balance(
                    networks[scenario_A][year],
                    networks[scenario_B][year],
                    scenario_A,
                    scenario_B,
                    colors,
                    output_path,
                    year,
                )

    logger.info(f"Dual comparison plots saved to {output_path}")


def plot_sensitivity_analysis(
    costs_agg, sensitivities, scenarios, reference_scenario, colors, output_path
):
    """Plot sensitivity analysis showing changes compared to reference scenario."""
    if reference_scenario not in costs_agg.index.get_level_values(0):
        logger.error(f"Reference scenario {reference_scenario} not found in data")
        return

    logger.info(f"Plotting sensitivity analysis for {sensitivities}")

    # Get available years from the index
    years = costs_agg.index.get_level_values(1).unique()

    # Generate a separate plot for each year
    for year in years:
        logger.info(f"Generating sensitivity analysis plot for year {year}")

        # Initialize figure with 2 columns and enough rows to fit all sensitivities
        num_rows = (len(sensitivities) + 1) // 2
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 5 * num_rows))

        # Flatten the axs array for easier iteration
        axs = axs.flatten() if len(sensitivities) > 1 else [axs]

        # Iterate over sensitivities
        for i, sensitivity in enumerate(sensitivities):
            if i >= len(axs):
                break

            ax = axs[i]

            # Prepare data for both scenarios (Low and High) of the current sensitivity
            scenarios = [f"Low{sensitivity}", f"High{sensitivity}"]
            df_diff_scenarios = []

            for scenario in scenarios:
                if scenario not in costs_agg.index.get_level_values(0):
                    logger.warning(
                        f"Scenario {scenario} not found for sensitivity {sensitivity}"
                    )
                    continue

                # Calculate difference from reference, filtering for the current year
                try:
                    df_diff_scenario = (
                        costs_agg.loc[(scenario, year)]
                        .squeeze()
                        .sub(costs_agg.loc[(reference_scenario, year)].squeeze())
                    )

                    # Group small contributors into "other technologies"
                    other_indices = df_diff_scenario[
                        abs(df_diff_scenario) < 0.02 * abs(df_diff_scenario).sum()
                    ].index
                    df_diff_scenario["other technologies"] = df_diff_scenario[
                        other_indices
                    ].sum()
                    df_diff_scenario.drop(other_indices, inplace=True)

                    # Append to the list for concatenation
                    df_diff_scenarios.append(
                        df_diff_scenario.to_frame(
                            name=scenario.replace(sensitivity, "")
                        ).T
                    )
                except KeyError:
                    logger.warning(
                        f"Data not found for scenario {scenario} and year {year}"
                    )
                    continue

            if not df_diff_scenarios:
                continue

            # Concatenate Low and High data for the current sensitivity
            df_diff_combined = pd.concat(df_diff_scenarios).div(1e9)

            # Sort by magnitude
            df_diff_combined = df_diff_combined[
                df_diff_combined.drop("neighbour countries", axis=1, errors="ignore")
                .abs()
                .sum()
                .sort_values(ascending=False)
                .index.append(
                    pd.Index(["neighbour countries"]).intersection(
                        df_diff_combined.columns
                    )
                )
            ]

            # Plot stacked bar chart
            df_diff_combined.plot.bar(
                stacked=True,
                ax=ax,
                color=df_diff_combined.columns.map(colors).fillna("black"),
                legend=False,
                width=0.8,
            )

            # Add horizontal line at 0
            ax.hlines(
                0, -0.5, len(df_diff_combined) - 0.5, color="black", linewidth=0.5
            )

            # Add markers and annotations for total system cost changes
            markers_total = df_diff_combined.sum(axis=1)

            for x, (idx, y) in enumerate(markers_total.items()):
                ax.hlines(
                    y=y,
                    xmin=x - 0.4,
                    xmax=x + 0.4,
                    color="black",
                    linewidth=3,
                    zorder=3,
                    linestyles="solid",
                )

                # Add absolute value annotation
                ax.text(
                    x,
                    y + 0.1 if y > 0 else y - 0.1,
                    f"+{y:.2f}" if y > 0 else f"{y:.2f}",
                    color="black",
                    ha="center",
                    va="bottom" if y > 0 else "top",
                    fontsize=12,
                    path_effects=[
                        patheffects.withStroke(linewidth=3, foreground="white")
                    ],
                )

                # Add relative change annotation
                try:
                    reference_costs = (
                        costs_agg.loc[(reference_scenario, year)].squeeze().sum() / 1e9
                    )
                    relative_change = (y / reference_costs) * 100
                    ax.text(
                        x,
                        y + 0.3 if y > 0 else y - 0.3,
                        f"({relative_change:+.2f}%)",
                        color="black",
                        ha="center",
                        va="bottom" if y > 0 else "top",
                        fontsize=12,
                        path_effects=[
                            patheffects.withStroke(linewidth=3, foreground="white")
                        ],
                    )
                except:
                    logger.warning(
                        f"Could not calculate relative change for {sensitivity}"
                    )

            # Add titles and labels
            ax.set_title(f"{sensitivity} Parameter Sensitivity")
            ax.set_ylabel("Cost Difference [bn EUR]")
            ax.set_xticklabels(df_diff_combined.index, rotation=0)

            # Set y-axis limits consistently
            y_max = max(2.5, df_diff_combined.abs().max().max() * 1.2)
            ax.set_ylim(-y_max, y_max)

        # Add legend
        handles, labels = zip(*[ax.get_legend_handles_labels() for ax in axs])
        handles = [item for sublist in handles for item in sublist]
        labels = [item for sublist in labels for item in sublist]
        if len(labels) == 0:
            logger.warning(
                f"No data for sensitivity analysis available for year {year}. Check your scenarios and make sure they match the names of the sensitivity runs."
            )
            continue

        # Remove duplicates while preserving order
        unique_labels = []
        unique_handles = []
        for j, label in enumerate(labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handles[j])

        handles, labels = unique_handles, unique_labels

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(labels), 4),
            frameon=False,
        )

        # Remove unused subplots
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        # Adjust layout
        fig.suptitle(f"Sensitivity Analysis - Year {year}", fontsize=16, y=0.98)
        fig.tight_layout(rect=[0, 0.05, 1, 0.95])
        fig.savefig(
            os.path.join(output_path, f"sensitivity_analysis_{year}.pdf"),
            bbox_inches="tight",
            pad_inches=0.1,
        )

        plt.close(fig)  # Close the figure to free memory

    logger.info(f"Sensitivity analysis plots saved to {output_path}")


def plot_price_duration_curves(networks_dict, output_path, figsize=(21, 7)):
    """
    Plot price duration curves for electricity, low voltage electricity, and district heating

    Parameters:
    -----------
    networks_dict : dict
        Dictionary mapping scenario names to PyPSA network objects
    output_path : str
        Path to save the output figure
    figsize : tuple, optional
        Figure size (width, height)
    """

    def calc_average_electricity_price(n):
        loads = n.buses_t.p.filter(regex=r"DE\d \d$").clip(upper=0).mul(-1)
        prices = n.buses_t.marginal_price.filter(regex=r"DE\d \d$")
        weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
        weighted_average_price_t = (
            weighted_average_price_t.loc[
                np.repeat(
                    weighted_average_price_t.index, n.snapshot_weightings.generators
                )
            ]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        weighted_average_price_t.index = weighted_average_price_t.index / (
            len(weighted_average_price_t) - 1
        )
        return weighted_average_price_t

    def calc_average_electricity_lv_price(n):
        loads = n.buses_t.p.filter(regex=r"DE\d \d low voltage").clip(upper=0).mul(-1)
        prices = n.buses_t.marginal_price.filter(regex=r"DE\d \d low voltage")
        weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
        weighted_average_price_t = (
            weighted_average_price_t.loc[
                np.repeat(
                    weighted_average_price_t.index, n.snapshot_weightings.generators
                )
            ]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        weighted_average_price_t.index = weighted_average_price_t.index / (
            len(weighted_average_price_t) - 1
        )
        return weighted_average_price_t

    def calc_average_dh_price(n):
        loads = n.loads_t.p.filter(
            regex=r"DE\d.*(urban central|low-temperature) heat"
        ).clip(lower=0)
        loads.columns = loads.columns.str.replace(
            "low-temperature heat for industry", "urban central heat"
        )
        prices = n.buses_t.marginal_price.filter(regex=r"DE\d.*urban central heat")
        weighted_average_price_t = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
        weighted_average_price_t = (
            weighted_average_price_t.loc[
                np.repeat(
                    weighted_average_price_t.index, n.snapshot_weightings.generators
                )
            ]
            .sort_values(ascending=False)
            .reset_index(drop=True)
        )
        weighted_average_price_t.index = weighted_average_price_t.index / (
            len(weighted_average_price_t) - 1
        )
        return weighted_average_price_t

    fig, ax = plt.subplots(1, 3, figsize=figsize)

    # Plot for each scenario and network
    for scenario, networks_scenario in networks_dict.items():
        for year, network in networks_scenario.items():
            label = f"{scenario}_{year}"

            try:
                # Average Electricity Price
                hv_elec = calc_average_electricity_price(network)
                if not hv_elec.empty:
                    hv_elec.plot(ax=ax[0], label=label, linewidth=0.5)

                # Average Low Voltage Electricity Price
                lv_elec = calc_average_electricity_lv_price(network)
                if not lv_elec.empty:
                    lv_elec.plot(ax=ax[1], label=label, linewidth=0.5)

                # Average District Heating Price
                dh = calc_average_dh_price(network)
                if not dh.empty:
                    dh.plot(ax=ax[2], label=label, linewidth=0.5)
            except Exception as e:
                logger.warning(f"Error plotting price duration curves for {label}: {e}")

    # Set titles and formatting
    ax[0].set_title("Average Electricity Price")
    ax[1].set_title("Average Low Voltage Electricity Price")
    ax[2].set_title("Average District Heating Price")

    for ax_ in ax:
        ax_.set_ylabel("Price [EUR/MWh]")
        ax_.set_xlabel("Duration Proportion")
        ax_.set_xlim(0, 1)
        ax_.legend(title="Scenario")
        ax_.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_path, "price_duration_curves.pdf"),
        bbox_inches="tight",
    )

    logger.info(f"Price duration curves saved to {output_path}")
    return fig, ax


def get_uch_supply(network, drop_stores=True):
    """Extract urban central heat supply from energy balance"""
    # Get energy balance for urban central heat
    eb_uch = network.statistics.energy_balance(
        groupby=["bus", "carrier", "bus_carrier"]
    ).xs("urban central heat", level="bus_carrier")

    # Filter for DE nodes
    eb_uch_de = eb_uch.filter(regex=r"DE\d.*urban", axis=0)

    # Group by carrier and sum across all German regions
    # Focusing only on supply (positive values)
    carriers_supply = eb_uch_de.groupby("carrier").sum()
    carriers_supply = carriers_supply[
        carriers_supply > 0
    ]  # Keep only supply components
    if drop_stores:
        # Drop urban central water pits discharger
        carriers_supply = carriers_supply.drop(
            [
                "urban central water pits discharger",
                "urban central water tanks discharger",
            ],
            errors="ignore",
        )

    return carriers_supply


def plot_uch_supply_comparison(
    networks, output_path="outputs/uch_supply_comparison.png", colors={}
):
    """
    Compare urban central heat supply across different networks

    Parameters:
    -----------
    networks : dict
        Dictionary of PyPSA networks by year
    output_path : str
        Path to save the output figure
    """
    # Get available years and sort them
    years = sorted(networks.keys())

    # Prepare data for all years
    all_supply_data = {}
    all_carriers = set()

    for year in years:
        supply = get_uch_supply(networks[year])
        supply_twh = supply / 1e6
        all_supply_data[year] = supply_twh
        all_carriers.update(supply_twh.index)

    # Create a DataFrame with all carriers and years
    df = pd.DataFrame(index=sorted(all_carriers), columns=years)

    # Fill the DataFrame with values
    for year in years:
        for carrier in df.index:
            if carrier in all_supply_data[year].index:
                df.loc[carrier, year] = all_supply_data[year][carrier]
            else:
                df.loc[carrier, year] = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot stacked bars

    df.T.plot(kind="bar", stacked=True, ax=ax, color=colors)

    # Add labels and title
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Energy [TWh]", fontsize=12)
    ax.set_title("Urban Central Heat Supply Comparison", fontsize=14)

    # Add legend with custom colors
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in df.index]
    ax.legend(
        handles,
        df.index,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        frameon=True,
        fontsize=10,
    )

    # Add total values on top of each bar
    for i, year in enumerate(years):
        total = df[year].sum()
        ax.text(i, total, f"{total:.1f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")

    logger.info(f"Urban central heat supply comparison saved to {output_path}")
    return fig, ax


def plot_uch_supply(networks_dict, output_path, colors):
    """
    Plot urban central heat supply from all scenarios in one consolidated figure
    with one row/subplot per scenario

    Parameters:
    -----------
    networks_dict : dict
        Dictionary mapping scenario names to networks by year
    output_path : str
        Path to save the output figure
    colors : dict
        Dictionary of colors for each technology
    """
    scenarios = list(networks_dict.keys())
    n_scenarios = len(scenarios)

    if n_scenarios == 0:
        logger.warning("No scenarios to plot for urban central heat supply")
        return

    # Create figure with one subplot per scenario
    fig, axes = plt.subplots(n_scenarios, 1, figsize=(12, 4 * n_scenarios), sharex=True)
    if n_scenarios == 1:
        axes = [axes]  # Convert to list if only one subplot

    # Process each scenario
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        networks_scenario = networks_dict[scenario]

        # Get available years and sort them
        years = sorted(networks_scenario.keys())

        # Prepare data for all years in this scenario
        all_supply_data = {}
        all_carriers = set()

        for year in years:
            supply = get_uch_supply(networks_scenario[year])
            supply_twh = supply / 1e6
            all_supply_data[year] = supply_twh
            all_carriers.update(supply_twh.index)

        # Create a DataFrame with all carriers and years for this scenario
        df = pd.DataFrame(index=sorted(all_carriers), columns=years)

        # Fill the DataFrame with values
        for year in years:
            for carrier in df.index:
                if carrier in all_supply_data[year].index:
                    df.loc[carrier, year] = all_supply_data[year][carrier]
                else:
                    df.loc[carrier, year] = 0

        # Plot stacked bars for this scenario
        df.T.plot(kind="bar", stacked=True, ax=ax, color=colors)

        # Add labels and title for this subplot
        ax.set_title(f"Scenario: {scenario}", fontsize=14)
        ax.set_ylabel("Energy [TWh]", fontsize=12)

        # Only set xlabel for the bottom subplot
        if i == n_scenarios - 1:
            ax.set_xlabel("Year", fontsize=12)

        # Add total values on top of each bar
        for j, year in enumerate(years):
            total = df[year].sum()
            ax.text(j, total, f"{total:.1f}", ha="center", va="bottom")

        # Remove individual subplot legends
        ax.get_legend().remove() if ax.get_legend() else None

    # Add a single consolidated legend for all subplots
    all_carriers = set()
    for scenario in scenarios:
        for year in networks_dict[scenario]:
            carriers = get_uch_supply(networks_dict[scenario][year]).index
            all_carriers.update(carriers)

    all_carriers = sorted(all_carriers)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[carrier]) for carrier in all_carriers
    ]

    fig.legend(
        legend_handles,
        all_carriers,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, len(all_carriers)),
        frameon=True,
        fontsize=10,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for the legend at the top
    plt.savefig(output_path, bbox_inches="tight")

    logger.info(
        f"Consolidated urban central heat supply comparison saved to {output_path}"
    )
    return fig, axes


def plot_summary_metrics(summary_df, output_path):
    """Plot key summary metrics across scenarios."""
    # Select key metrics to plot
    key_metrics = [
        "total_system_costs_bnEUR",
        "total_system_costs_DE_bnEUR",
        "electricity_price_EUR_per_MWh",
        "dh_price_EUR_per_MWh",
        "PTES_capacity_TWh",
        "PTES_no_cycles",
        "TTES_capacity_TWh",
        "CHP_capacity_GW",
        "curtailment_TWh",
        "H2_store_TWh",
        "vRES_capacity_GW",
        "resistive_heater_GW",
    ]

    # Create a pivot table for plotting
    plot_data = summary_df.groupby(["scenario", "year"])[key_metrics].sum()

    # Sort scenarios by total system costs, and within scenarios by year
    plot_data = plot_data.sort_values(
        by=["year", "total_system_costs_bnEUR"], ascending=[True, True]
    )

    # Plot the data
    fig, axes = plt.subplots(4, 3, figsize=(10, 15))
    axes = axes.flatten()

    for i, metric in enumerate(key_metrics):
        ax = axes[i]
        # Create a grouped bar plot - grouped by scenario with one bar per year
        years = plot_data.index.get_level_values(1).unique()
        scenarios = plot_data.index.get_level_values(0).unique()

        # Set up colors for different years
        year_colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

        # Create x positions for the bars
        x = np.arange(len(scenarios))
        width = 0.8 / len(years)  # Width of each bar

        # Plot bars for each year within each scenario group
        for i, year in enumerate(years):
            year_data = [
                (
                    plot_data.loc[(scenario, year), metric]
                    if (scenario, year) in plot_data.index
                    else 0
                )
                for scenario in scenarios
            ]
            bars = ax.bar(
                x + (i - len(years) / 2 + 0.5) * width,
                year_data,
                width,
                label=f"{year}",
                color=year_colors[i],
            )

            # Add value annotations
            for j, value in enumerate(year_data):
                ax.text(
                    x[j] + (i - len(years) / 2 + 0.5) * width,
                    value + 0.01 * value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        # Set ylim to accommodate annotations
        ax.set_ylim(0, plot_data[metric].max() * 1.25)
        # Set x-axis labels to scenario names
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=45, ha="right")

        ax.set_title(metric.replace("_", " "))
        ax.set_ylabel(metric.split("_")[-1])
        ax.grid(True, alpha=0.3)

    # DO one legend for all axes
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        title="Year",
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(output_path, "summary_metrics.png"),
        bbox_inches="tight",
        bbox_extra_artists=[leg],
    )


def calc_dh_price_range_subnodes(n):

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


def get_ptes_pot_to_demand_ratio(n, system):
    """Calculate ratio of PTES potential to district heating demand."""
    dh_demand = (
        n.loads_t.p.filter(regex=rf"{system} (urban central|low-temperature) heat")
        .sum(1)
        .mul(n.snapshot_weightings.generators)
        .sum()
    )
    ptes_pot = n.stores.filter(regex=rf"{system}.*water pits", axis=0).e_nom_max.sum()
    return ptes_pot / dh_demand


def plot_energy_balance_comparison(network1, network2, scenarios, output_path, colors):
    """
    Plot comparison of energy balance for district heating between two networks.

    Parameters:
    -----------
    network1 : PyPSA Network
        First network (typically No_PTES scenario)
    network2 : PyPSA Network
        Second network (typically Baseline scenario)
    title : str
        Title for the plot
    output_path : str
        Path to save the output figure
    """
    plt.rcParams.update({"font.size": 10})
    title = f"Energy Balance Comparison: {scenarios[0]} vs {scenarios[1]}"

    def prepare_energy_balance_data(network):
        eb_uch = (
            network.statistics.energy_balance(groupby=["bus", "carrier", "bus_carrier"])
            .xs("urban central heat", level=3)
            .reset_index()
        )
        eb_uch = eb_uch.loc[eb_uch.bus.str.contains("DE\d \d .*urban"), :]

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

        return to_plot_rel_load + to_plot_rel_gen, dh_prices

    # Prepare data for both networks
    to_plot_rel1, dh_prices1 = prepare_energy_balance_data(network1)
    to_plot_rel2, dh_prices2 = prepare_energy_balance_data(network2)
    to_plot_rel1 = to_plot_rel1.loc[to_plot_rel2.index]

    # Create subplots
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Plot for Network 1
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    col_order = [
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        "urban central geothermal heat pump",
        "urban central geothermal heat direct utilisation",
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central resistive heater",
        "H2 Electrolysis",
        "urban central solid biomass CHP",
        "urban central gas CHP",
        "urban central oil CHP",
        "urban central coal CHP",
        "waste CHP",
        "urban central H2 CHP",
        "urban central gas boiler",
        "urban central water tanks discharger",
        "urban central water tanks charger",
        "urban central water tanks losses",
    ]

    # Filter to only include columns that exist in the data
    col_order = [c for c in col_order if c in to_plot_rel1.columns]
    # concat col_order with elements from to_plot_rel1 that are not in col_order
    col_order += [c for c in to_plot_rel1.columns if c not in col_order]
    to_plot_rel1 = to_plot_rel1[col_order]  # Align columns

    to_plot_rel1.plot.bar(
        stacked=True,
        ax=ax1,
        color=colors,
        legend=False,
        width=0.8,
    )
    ax1.set_title(scenarios[0])
    ax1.set_ylabel(
        "Share of district heating\nconsumption and supply\n[%]", fontsize=10
    )
    ax1_twin.set_ylabel(
        "Ratio between PTES potential\nand District Heating Demand",
        color="black",
        fontsize=10,
    )
    ax1_twin.set_yscale("log")
    ax1.axhline(y=0, color="black", linestyle="-")

    # Add horizontal lines for ratio of PTES potential to district heating load in Baseline scenario
    for system in dh_prices1.index:
        ptes_pot_to_demand_ratio = get_ptes_pot_to_demand_ratio(network2, system)
        idx = to_plot_rel1.index.get_loc(system)
        ax1_twin.scatter(idx, ptes_pot_to_demand_ratio, color="black", marker="x", s=10)

    # Plot for Network 2
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    col_order = [
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        "urban central geothermal heat pump",
        "urban central geothermal heat direct utilisation",
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central resistive heater",
        "H2 Electrolysis",
        "urban central solid biomass CHP",
        "urban central gas CHP",
        "urban central coal CHP",
        "waste CHP",
        "urban central H2 CHP",
        "urban central gas boiler",
        "urban central water tanks discharger",
        "urban central water pits discharger",
        "urban central water tanks charger",
        "urban central water tanks losses",
        "urban central water pits charger",
        "urban central water pits losses",
    ]

    # Filter to only include columns that exist in the data
    col_order = [c for c in col_order if c in to_plot_rel2.columns]
    # concat col_order with elements from to_plot_rel2 that are not in col_order
    col_order += [c for c in to_plot_rel2.columns if c not in col_order]
    # Ensure the order of columns matches the first plot
    to_plot_rel2 = to_plot_rel2[col_order]  # Align columns

    to_plot_rel2.plot.bar(
        stacked=True,
        ax=ax2,
        color=colors,
        legend=False,
        width=0.8,
    )
    ax2.set_title(scenarios[1])
    ax2.set_ylabel(
        "Share of district heating\nconsumption and supply\n[%]", fontsize=10
    )
    ax2_twin.set_ylabel("District Heating Price\n[€/MWh]", color="black", fontsize=10)
    ax2.axhline(y=0, color="black", linestyle="-")

    # Decrease fontsize of ax2 xticks

    for tick in ax2.get_xticklabels():
        tick.set_fontsize(10)

    # Add horizontal lines for DH prices
    for system, price in dh_prices1.items():
        idx = to_plot_rel2.index.get_loc(system)
        ax2_twin.axhline(
            y=price,
            xmin=idx / len(to_plot_rel2),
            xmax=(idx + 1) / len(to_plot_rel2),
            color="black",
            linestyle="-",
            markeredgecolor="white",
        )

    for system, price in dh_prices2.items():
        idx = to_plot_rel2.index.get_loc(system)
        ax2_twin.scatter(
            idx, price, color="black", marker="o", s=10, edgecolors="white"
        )

    # Shared legend
    handles, labels = [], []
    for carrier, color in colors.items():
        if carrier in to_plot_rel1.columns or carrier in to_plot_rel2.columns:
            handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            labels.append(carrier)

    # Replace urban central with district heating in labels
    import re

    labels = [
        re.sub(
            "urban central heat$",
            "urban central heat for residential and services",
            label,
        )
        for label in labels
    ]
    # Replace urban central with empty string in labels
    labels = [label.replace("urban central ", "") for label in labels]
    # Replace water pits with PTES
    labels = [label.replace("water pits", "PTES") for label in labels]
    # Replace water tanks with TTES
    labels = [label.replace("water tanks", "TTES") for label in labels]
    # Replace charge and discharge with empty string
    labels = [
        label.replace(" charger", "").replace(" discharger", "") for label in labels
    ]

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handles[i])

    fig.legend(
        unique_handles,
        unique_labels,
        title="Technology",
        bbox_to_anchor=(1.17, 0.65),
        loc="center",
        frameon=False,
        fontsize=10,
    )

    # Second shared legend for PTES potential to demand ratio and DH prices
    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color="black", marker="x", linestyle="None"))
    labels.append("PTES potential to demand ratio")
    handles.append(plt.Line2D([0], [0], color="black", linestyle="-"))
    labels.append("Average marginal price\nof district heat in No_PTES")
    handles.append(plt.Line2D([0], [0], color="black", linestyle="None", marker="o"))
    labels.append("Average marginal price\nof district heat in Baseline")

    fig.legend(
        handles,
        labels,
        title="",
        bbox_to_anchor=(1.17, 0.2),
        loc="center",
        frameon=False,
        fontsize=10,
    )

    # Replace DE0 at start of xticks with empty string
    xticks = [label.get_text().replace("DE0 ", "") for label in ax2.get_xticklabels()]
    ax2.set_xticklabels(xticks)

    # Remove xlabel
    ax2.set_xlabel("")

    # Adjust layout and save the plot
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    logger.info(f"Energy balance comparison saved to {output_path}")
    return fig, axes


def plot_ptes_price_impact_scatter(
    networks, scenario_tuples, output_path, figsize=(7, 5)
):
    """
    Plot scatter plots showing the relationship between PTES expansion ratio
    and district heating price decrease for multiple scenario comparisons.

    Parameters:
    -----------
    networks : dict
        Dictionary mapping scenario names to networks by year
    scenario_tuples : list of tuples
        List of (reference_scenario, comparison_scenario) tuples
        where reference_scenario is without PTES and comparison_scenario is with PTES
    output_path : str
        Path to save the output figure
    figsize : tuple, optional
        Figure size (width, height)
    """

    def get_ptes_expansion_ratio_and_price_decrease(
        networks, ref_scenario, comp_scenario
    ):
        """Calculate PTES expansion ratio and price decrease for each district heating system."""

        # Check if required scenarios exist
        if ref_scenario not in networks or comp_scenario not in networks:
            logger.warning(
                f"Required scenarios '{ref_scenario}' and '{comp_scenario}' not found in networks"
            )
            return pd.DataFrame()

        # Get the year (assuming 2045 or the first available year)
        year = list(networks[ref_scenario].keys())[0]

        if year not in networks[ref_scenario] or year not in networks[comp_scenario]:
            logger.warning(f"Year {year} not found in both scenarios")
            return pd.DataFrame()

        # Get district heating prices for both scenarios
        no_ptes_prices = calc_dh_price_range_subnodes(networks[ref_scenario][year])
        baseline_prices = calc_dh_price_range_subnodes(networks[comp_scenario][year])

        # Calculate price decrease
        price_decrease = no_ptes_prices - baseline_prices

        # Get only systems present in both scenarios
        common_systems = price_decrease.index

        # Get PTES expansion (e_nom_opt) for each system
        ptes_expansion = pd.Series(index=common_systems, dtype=float)

        # Get total district heating demand for each system
        dh_demand = pd.Series(index=common_systems, dtype=float)

        # Calculate PTES expansion for each system
        for system in common_systems:
            # Get PTES storage in the system
            ptes_stores = networks[comp_scenario][year].stores.filter(
                regex=rf"{system}.*water pits", axis=0
            )

            # Sum the optimal energy capacity (e_nom_opt)
            if not ptes_stores.empty:
                ptes_expansion[system] = ptes_stores.e_nom_opt.sum()
            else:
                ptes_expansion[system] = 0

            # Calculate total district heating demand
            dh_demand[system] = (
                networks[comp_scenario][year]
                .loads_t.p.filter(
                    regex=rf"{system} (urban central|low-temperature) heat"
                )
                .sum(1)
                .mul(networks[comp_scenario][year].snapshot_weightings.generators)
                .sum()
            )

        # Calculate ratio of PTES expansion to district heating demand
        ptes_ratio = ptes_expansion / dh_demand

        # Create a DataFrame with both metrics
        result = pd.DataFrame(
            {
                "price_decrease": price_decrease,
                "ptes_ratio": ptes_ratio,
                "dh_demand": dh_demand,
                "ptes_expansion": ptes_expansion,
            }
        )

        # Drop any rows with NaN values
        result.dropna(inplace=True)

        return result

    # Filter scenario tuples to only include those with available data
    available_tuples = []
    for ref_scenario, comp_scenario in scenario_tuples:
        if ref_scenario in networks and comp_scenario in networks:
            available_tuples.append((ref_scenario, comp_scenario))

    if not available_tuples:
        logger.warning(
            "No valid scenario tuples found for PTES price impact scatter plots"
        )
        return None, None

    # Create subplots for each scenario comparison
    n_comparisons = len(available_tuples)
    fig, axs = plt.subplots(
        1, n_comparisons, figsize=(figsize[0] * n_comparisons, figsize[1])
    )

    # If only one comparison, convert axs to a list for consistent indexing
    if n_comparisons == 1:
        axs = [axs]

    # Process each scenario pair
    for i, (ref_scenario, comp_scenario) in enumerate(available_tuples):
        ax = axs[i]

        # Get data for the plot
        systems_data = get_ptes_expansion_ratio_and_price_decrease(
            networks, ref_scenario, comp_scenario
        )

        if systems_data.empty:
            logger.warning(
                f"No data available for PTES price impact scatter plot: {ref_scenario} vs {comp_scenario}"
            )
            ax.set_visible(False)
            continue

        # Size points by district heating demand (normalized for better visibility)
        sizes = systems_data.dh_demand / systems_data.dh_demand.max() * 200

        # Create scatter plot with sized points
        sc = ax.scatter(
            systems_data.ptes_ratio,
            systems_data.price_decrease,
            s=sizes,
            alpha=0.6,
            c="blue",
            edgecolor="black",
        )

        # Add trendline
        if len(systems_data) > 1:  # Need at least 2 points for trendline
            z = np.polyfit(systems_data.ptes_ratio, systems_data.price_decrease, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                systems_data.ptes_ratio.min(), systems_data.ptes_ratio.max(), 100
            )
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)

        # Annotate systems (simplified for multiple plots)
        try:
            from adjustText import adjust_text

            texts = []
            # Annotate only top 10 systems by demand to avoid overcrowding
            to_annotate = (
                systems_data.sort_values(by="dh_demand", ascending=False).head(10).index
            )

            for system, row in systems_data.loc[to_annotate].iterrows():
                # Extract just the city name without DE prefix for cleaner labels
                label = system.split(" ")[-1] if " " in system else system
                texts.append(
                    ax.text(row.ptes_ratio, row.price_decrease, label, fontsize=8)
                )

            # Adjust text positions to avoid overlapping
            adjust_text(
                texts, arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5)
            )

        except ImportError:
            # Fallback if adjustText is not available
            for system, row in systems_data.head(
                5
            ).iterrows():  # Limit to avoid overcrowding
                # Extract just the city name without DE prefix for cleaner labels
                label = system.split(" ")[-1] if " " in system else system
                ax.annotate(
                    label,
                    xy=(row.ptes_ratio, row.price_decrease),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7
                    ),
                )

        # Add legend for point sizes (only on the first subplot)
        if i == 0:
            kw = dict(
                prop="sizes",
                num=3,
                fmt="{x:.1f} TWh/a",
                func=lambda s: s / 200 * systems_data.dh_demand.max() / 1e6,
            )
            legend1 = ax.legend(
                *sc.legend_elements(**kw),
                title="District Heating Demand",
                loc="upper left",
            )
            ax.add_artist(legend1)

        # Labels and title
        if i == 0:
            ax.set_ylabel("District Heating Cost\nSavings (€/MWh)", fontsize=12)
        ax.set_xlabel("PTES Capacity to\nDistrict Heating Demand Ratio", fontsize=12)

        # Use the comparison scenario name as title (the one with PTES)
        ax.set_title(f"{comp_scenario}", fontsize=14)

        # Grid
        ax.grid(True, linestyle="--", alpha=0.7)

    # Add overall title
    fig.suptitle(
        "Impact of PTES Investments on District Heating Prices", fontsize=16, y=1.02
    )

    # Adjust layout
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_path, "ptes_price_decrease_impact_scatter.pdf")
    fig.savefig(output_file, dpi=300, bbox_inches="tight")

    logger.info(f"PTES price impact scatter plots saved to {output_file}")

    return fig, axs


def plot_ptes_socs(
    networks, output_path="outputs/ptes_soc_ranges.png", figsize=(10, 6)
):
    """
    Plot the state of charge (SoC) ranges for PTES across different scenarios.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)

    for scenario, networks_scenario in networks.items():
        for year, networks_year in networks_scenario.items():
            soc = networks_year.stores_t.e.filter(regex="DE.*water pits").sum(1)
            if soc is not None:
                plt.plot(soc.index, soc, alpha=0.5, label=scenario)

    plt.title("PTES State of Charge Ranges")
    plt.xlabel("Time")
    plt.ylabel("State of Charge")
    plt.grid()
    plt.savefig(output_path)
    plt.close()


def plot_ptes_price_impact_scatter(
    networks, scenario_tuples, output_path, figsize=(7, 5)
):
    """
    Plot scatter plots showing the relationship between PTES expansion ratio
    and district heating price decrease for multiple scenario comparisons.
    Creates separate plots for each scenario comparison with unified y-axis limits.

    Parameters:
    -----------
    networks : dict
        Dictionary mapping scenario names to networks by year
    scenario_tuples : list of tuples
        List of (reference_scenario, comparison_scenario) tuples
        where reference_scenario is without PTES and comparison_scenario is with PTES
    output_path : str
        Path to save the output figures
    figsize : tuple, optional
        Figure size (width, height)
    """

    def get_ptes_expansion_ratio_and_price_decrease(
        networks, ref_scenario, comp_scenario
    ):
        """Calculate PTES expansion ratio and price decrease for each district heating system."""

        # Check if required scenarios exist
        if ref_scenario not in networks or comp_scenario not in networks:
            logger.warning(
                f"Required scenarios '{ref_scenario}' and '{comp_scenario}' not found in networks"
            )
            return pd.DataFrame()

        # Get the year (assuming 2045 or the first available year)
        year = list(networks[ref_scenario].keys())[0]

        if year not in networks[ref_scenario] or year not in networks[comp_scenario]:
            logger.warning(f"Year {year} not found in both scenarios")
            return pd.DataFrame()

        # Get district heating prices for both scenarios
        no_ptes_prices = calc_dh_price_range_subnodes(networks[ref_scenario][year])
        baseline_prices = calc_dh_price_range_subnodes(networks[comp_scenario][year])

        # Calculate price decrease
        price_decrease = no_ptes_prices - baseline_prices

        # Get only systems present in both scenarios
        common_systems = price_decrease.index

        # Get PTES expansion (e_nom_opt) for each system
        ptes_expansion = pd.Series(index=common_systems, dtype=float)

        # Get total district heating demand for each system
        dh_demand = pd.Series(index=common_systems, dtype=float)

        # Calculate PTES expansion for each system
        for system in common_systems:
            # Get PTES storage in the system
            ptes_stores = networks[comp_scenario][year].stores.filter(
                regex=rf"{system}.*water pits", axis=0
            )

            # Sum the optimal energy capacity (e_nom_opt)
            if not ptes_stores.empty:
                ptes_expansion[system] = ptes_stores.e_nom_opt.sum()
            else:
                ptes_expansion[system] = 0

            # Calculate total district heating demand
            dh_demand[system] = (
                networks[comp_scenario][year]
                .loads_t.p.filter(
                    regex=rf"{system} (urban central|low-temperature) heat"
                )
                .sum(1)
                .mul(networks[comp_scenario][year].snapshot_weightings.generators)
                .sum()
            )

        # Calculate ratio of PTES expansion to district heating demand
        ptes_ratio = ptes_expansion / dh_demand

        # Create a DataFrame with both metrics
        result = pd.DataFrame(
            {
                "price_decrease": price_decrease,
                "ptes_ratio": ptes_ratio,
                "dh_demand": dh_demand,
                "ptes_expansion": ptes_expansion,
            }
        )

        # Drop any rows with NaN values
        result.dropna(inplace=True)

        return result

    # Filter scenario tuples to only include those with available data
    available_tuples = []
    for ref_scenario, comp_scenario in scenario_tuples:
        if ref_scenario in networks and comp_scenario in networks:
            available_tuples.append((ref_scenario, comp_scenario))

    if not available_tuples:
        logger.warning(
            "No valid scenario tuples found for PTES price impact scatter plots"
        )
        return None

    # First pass: collect all data to determine unified axis limits
    all_data = []
    for ref_scenario, comp_scenario in available_tuples:
        systems_data = get_ptes_expansion_ratio_and_price_decrease(
            networks, ref_scenario, comp_scenario
        )
        if not systems_data.empty:
            all_data.append(systems_data)

    if not all_data:
        logger.warning("No valid data found for any scenario tuples")
        return None

    # Combine all data to determine unified limits
    combined_data = pd.concat(all_data, ignore_index=True)

    # Calculate unified axis limits with some padding
    y_min = combined_data["price_decrease"].min()
    y_max = combined_data["price_decrease"].max()
    y_padding = (y_max - y_min) * 0.1  # 10% padding
    unified_ylim = (y_min - y_padding, y_max + y_padding)

    # x_min = combined_data['ptes_ratio'].min()
    # x_max = combined_data['ptes_ratio'].max()
    # x_padding = (x_max - x_min) * 0.1  # 10% padding
    # unified_xlim = (max(0, x_min - x_padding), x_max + x_padding)  # Ensure x_min >= 0

    logger.info(f"Unified y-axis limits: {unified_ylim}")
    # print(f"Unified x-axis limits: {unified_xlim}")

    # Create separate plots for each scenario comparison
    figures = []

    for ref_scenario, comp_scenario in available_tuples:
        logger.info(
            f"Creating PTES price impact scatter plot for: {ref_scenario} vs {comp_scenario}"
        )

        # Create individual figure for this comparison
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Get data for the plot
        systems_data = get_ptes_expansion_ratio_and_price_decrease(
            networks, ref_scenario, comp_scenario
        )

        if systems_data.empty:
            logger.warning(
                f"No data available for PTES price impact scatter plot: {ref_scenario} vs {comp_scenario}"
            )
            plt.close(fig)
            continue

        # Size points by district heating demand (normalized for better visibility)
        sizes = systems_data.dh_demand / systems_data.dh_demand.max() * 200

        # Create scatter plot with sized points
        sc = ax.scatter(
            systems_data.ptes_ratio,
            systems_data.price_decrease,
            s=sizes,
            alpha=0.6,
            c="blue",
            edgecolor="black",
        )

        # Add trendline
        if len(systems_data) > 1:  # Need at least 2 points for trendline
            z = np.polyfit(systems_data.ptes_ratio, systems_data.price_decrease, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                systems_data.ptes_ratio.min(), systems_data.ptes_ratio.max(), 100
            )
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8)

        # Annotate systems
        try:
            from adjustText import adjust_text

            texts = []
            # Annotate top systems by demand for better readability
            to_annotate = (
                systems_data.sort_values(by="dh_demand", ascending=False).head(15).index
            )

            for system, row in systems_data.loc[to_annotate].iterrows():
                # Extract just the city name without DE prefix for cleaner labels
                label = system.split(" ")[-1] if " " in system else system
                texts.append(
                    ax.text(row.ptes_ratio, row.price_decrease, label, fontsize=10)
                )

            # Adjust text positions to avoid overlapping
            adjust_text(
                texts, arrowprops=dict(arrowstyle="->", color="gray", alpha=0.5)
            )

        except ImportError:
            # Fallback if adjustText is not available
            for system, row in systems_data.head(10).iterrows():
                # Extract just the city name without DE prefix for cleaner labels
                label = system.split(" ")[-1] if " " in system else system
                ax.annotate(
                    label,
                    xy=(row.ptes_ratio, row.price_decrease),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=10,
                    bbox=dict(
                        boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7
                    ),
                )

        # Add legend for point sizes
        kw = dict(
            prop="sizes",
            num=3,
            fmt="{x:.1f} TWh/a",
            func=lambda s: s / 200 * systems_data.dh_demand.max() / 1e6,
        )
        legend1 = ax.legend(
            *sc.legend_elements(**kw), title="District Heating Demand", loc="upper left"
        )
        ax.add_artist(legend1)

        # Labels and title
        ax.set_ylabel("District Heating Cost\nSavings (€/MWh)", fontsize=12)
        ax.set_xlabel("PTES Capacity to\nDistrict Heating Demand Ratio", fontsize=12)

        # Use the comparison scenario name as title (the one with PTES)
        ax.set_title(
            f"Impact of PTES Investments on District Heating Prices\n{comp_scenario}",
            fontsize=14,
        )

        # Set unified axis limits
        # ax.set_ylim(unified_ylim)
        # ax.set_xlim(unified_xlim)

        # Grid
        ax.grid(True, linestyle="--", alpha=0.7)

        # Adjust layout
        plt.tight_layout()

        # Save individual figure
        output_file = os.path.join(
            output_path, f"ptes_price_impact_scatter_{comp_scenario}.pdf"
        )
        fig.savefig(output_file, dpi=300, bbox_inches="tight")

        logger.info(f"PTES price impact scatter plot saved: {output_file}")

        figures.append(fig)

    return figures


def plot_ptes_savings_comparison(
    scenario_tuples, costs_agg, colors, year, output_path, figsize=(8, 4)
):
    """
    Plot cost savings comparison between scenarios with and without PTES.

    Parameters:
    -----------
    scenario_tuples : list of tuples
        List of (reference_scenario, comparison_scenario) tuples
        where reference_scenario is without PTES and comparison_scenario is with PTES
    costs_agg : DataFrame
        Cost aggregation dataframe with technology breakdown
    colors : dict
        Dictionary of colors for each technology
    year : int
        Year to plot
    output_path : str
        Path to save the output figure
    figsize : tuple, optional
        Figure size (width, height)
    """
    logger.info(f"Generating PTES savings comparison for year {year}")

    # Filter the costs data for the specified year
    costs_year = costs_agg.xs(year, level="year")

    # Create figure with one subplot for each scenario pair
    n_comparisons = len(scenario_tuples)
    fig, axs = plt.subplots(1, n_comparisons, figsize=figsize, sharey=True)

    # If only one comparison, convert axs to a list for consistent indexing
    if n_comparisons == 1:
        axs = [axs]

    # Keep track of all displayed technologies across all comparisons for the legend
    all_displayed_techs = set()

    # Process each scenario pair
    for i, (ref_scenario, comp_scenario) in enumerate(scenario_tuples):
        ax = axs[i]

        # Check if both scenarios exist in the data
        if (
            ref_scenario not in costs_year.index
            or comp_scenario not in costs_year.index
        ):
            logger.warning(
                f"Scenarios {ref_scenario} or {comp_scenario} not found in data for year {year}"
            )
            # Clear the unused axis
            ax.set_visible(False)
            continue

        logger.info(f"  Processing comparison: {ref_scenario} vs {comp_scenario}")

        # Calculate cost differences: comp_scenario - ref_scenario
        # Positive values mean cost increases in the comparison scenario with PTES
        df_diff = costs_year.loc[comp_scenario].sub(costs_year.loc[ref_scenario])

        # Filter out technologies with zero contribution
        df_diff = df_diff[df_diff != 0]

        # Group small contributors into "other technologies"
        small_indices = df_diff.index[df_diff.abs() < 0.01 * df_diff.abs().sum()]
        if len(small_indices) > 0:
            df_diff["other technologies"] = df_diff[small_indices].sum()
            df_diff = df_diff.drop(small_indices)

        # Sort by magnitude (absolute value), but keep "neighbour countries" at the end
        if "neighbour countries" in df_diff.index:
            neighbour_value = df_diff["neighbour countries"]
            df_diff = df_diff.drop("neighbour countries")

            # Sort by absolute value
            df_diff = df_diff.reindex(df_diff.abs().sort_values(ascending=False).index)

            # Add neighbour countries back at the end
            df_diff["neighbour countries"] = neighbour_value
        else:
            # Sort by absolute value
            df_diff = df_diff.reindex(df_diff.abs().sort_values(ascending=False).index)

        # Convert to billion EUR
        df_diff_bn = df_diff.div(1e9)

        # Add the displayed technologies from this comparison to our set
        all_displayed_techs.update(df_diff_bn.index)

        # Plot stacked bar
        df_diff_bn.to_frame().T.plot.bar(
            stacked=True,
            ax=ax,
            color=df_diff_bn.index.map(colors).fillna("black"),
            legend=False,
            width=0.8,
        )

        # Add horizontal line at 0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, zorder=1)

        # Add markers for total system cost savings and German system cost savings
        total_savings = df_diff_bn.sum()
        germany_savings = df_diff_bn.drop("neighbour countries", errors="ignore").sum()

        # Add total system savings marker
        ax.hlines(
            y=total_savings,
            xmin=-0.42,
            xmax=0.42,
            color="black",
            linewidth=3,
            zorder=10,
            path_effects=[patheffects.withStroke(linewidth=3)],
            label="Total system savings",
        )

        # Add German system savings marker (if different)
        if "neighbour countries" in df_diff_bn:
            ax.hlines(
                y=germany_savings,
                xmin=-0.42,
                xmax=0.42,
                color="black",
                linewidth=1.5,
                linestyle="--",
                zorder=10,
                path_effects=[patheffects.withStroke(linewidth=3)],
                label="German system savings",
            )

        # Compute y-axis limits
        y_lims = ax.get_ylim()
        padding = 0.05 * (y_lims[1] - y_lims[0])

        y_top = y_lims[1] - padding  # always near the top of plot
        y_bottom = y_lims[0] + padding  # always near the bottom of plot

        # Calculate percentage against reference total
        ref_total = costs_year.loc[ref_scenario].sum() / 1e9
        pct_change_total = (total_savings / ref_total) * 100

        # Add annotation for total system savings
        sign_total = "+" if total_savings > 0 else ""
        # ax.annotate(
        #     f"Total:\n{sign_total}{total_savings:.1f} bn€\n({sign_total}{pct_change_total:.1f}%)",
        #     xy=(0, total_savings),
        #     xytext=(0, 1.3),
        #     textcoords="data",
        #     ha="center",
        #     va="top",
        #     fontweight="bold",
        #     path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
        #     bbox=dict(boxstyle="round,pad=0", edgecolor="none", facecolor="white", alpha=0),
        #     fontsize=9
        # )

        # Add annotation for German system savings (if different)
        if "neighbour countries" in df_diff_bn:
            ref_germany_total = (
                costs_year.loc[ref_scenario].drop("neighbour countries").sum() / 1e9
            )
            pct_change_germany = (germany_savings / ref_germany_total) * 100

            sign_germany = "+" if germany_savings > 0 else ""
            # ax.annotate(
            #     f"Germany:\n{sign_germany}{germany_savings:.1f} bn€\n({sign_germany}{pct_change_germany:.1f}%)",
            #     xy=(0, germany_savings),
            #     xytext=(0, -1.7),
            #     textcoords="data",
            #     ha="center",
            #     va="bottom",
            #     fontweight="bold",
            #     path_effects=[patheffects.withStroke(linewidth=3, foreground="white")],
            #     bbox=dict(boxstyle="round,pad=0", edgecolor="none", facecolor="white", alpha=0),
            #     fontsize=9
            # )

        # Set the comparison scenario name as x-tick label instead of title
        ax.set_xticks([0])
        ax.set_xticklabels([comp_scenario], fontsize=16)
        ax.tick_params(axis="x", which="both", bottom=True, labelbottom=True)

        # Remove the title setting since we're using x-tick labels instead
        # title = comp_scenario

        # Set axis labels and title with larger font sizes
        # ax.set_title(title, fontsize=18)
        if i == 0:
            ax.set_ylabel("Cost Difference [bn€]", fontsize=16)

        # Increase tick label sizes
        ax.tick_params(axis="y", labelsize=14)

    # Add legend for the markers - position more centered
    from matplotlib.lines import Line2D

    line_handles = [
        Line2D([0], [0], color="black", linewidth=3),
        Line2D([0], [0], color="black", linewidth=3, linestyle="--"),
    ]
    line_labels = ["Total system", "German system"]

    fig.legend(
        line_handles,
        line_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),  # More centered position
        frameon=True,
        title="Net difference in",
        fontsize=14,
        ncol=2,  # Horizontal layout
    )

    # Create a separate legend for technologies - ONLY for actually displayed technologies
    # Sort displayed technologies by their total importance across all comparisons
    tech_importance = {}
    for tech in all_displayed_techs:
        importance = 0
        for ref_scenario, comp_scenario in scenario_tuples:
            if ref_scenario in costs_year.index and comp_scenario in costs_year.index:
                diff = costs_year.loc[comp_scenario].sub(costs_year.loc[ref_scenario])
                if tech in diff:
                    importance += abs(diff[tech])
        tech_importance[tech] = importance

    sorted_displayed_techs = sorted(
        all_displayed_techs, key=lambda x: tech_importance.get(x, 0), reverse=True
    )

    # Create legend handles and labels for displayed technologies only
    tech_handles = []
    tech_labels = []
    for tech in sorted_displayed_techs:
        # Get color, using black as fallback for missing or empty colors
        tech_color = colors.get(tech, "black")
        if not tech_color or tech_color == "":  # Handle empty strings
            tech_color = "black"

        tech_handles.append(plt.Rectangle((0, 0), 1, 1, color=tech_color))
        tech_labels.append(tech)

    # Add legend for technologies on the right side
    fig.legend(
        tech_handles,
        tech_labels,
        title="Technology",
        bbox_to_anchor=(0.9, 0.5),
        loc="center left",
        ncol=1,
        frameon=True,
        fontsize=12,
    )

    # Remove the overall title to avoid overlap
    # fig.suptitle(f"Cost Impact of PTES Across Different Scenarios - {year}", fontsize=20, y=0.98)

    # Adjust layout with more space for right legend
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, right=0.75)  # Reduced top margin since no suptitle

    # Save figure
    output_file = os.path.join(output_path, f"ptes_savings_comparison_{year}.pdf")
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.3)

    logger.info(f"PTES savings comparison plot saved to {output_file}")

    return fig, axs


def plot_system_costs(costs_agg, scenarios, year, output_path, colors):
    """Plot system costs for different scenarios for the passed year as stacked bar plot.
    One bar per scenario, with different colors for each technology.
    There should be one row for the total system costs and one row for the German system costs,
    meaning without the column 'neighbour countries'.
    """
    costs_agg_year = costs_agg.xs(year, level="year")

    # Filter to only include scenarios that exist in the data
    available_scenarios = [s for s in scenarios if s in costs_agg_year.index]

    if not available_scenarios:
        logger.warning(f"No data found for any scenarios in year {year}")
        return

    # Create DataFrame with scenarios as rows and technologies as columns
    plot_data = costs_agg_year.loc[available_scenarios].copy()

    # Group small technologies into "other technologies"
    other_indices = plot_data.loc[
        :, (plot_data.max() < 0.001 * plot_data.sum(1).max())
    ].columns
    plot_data["other technologies"] = plot_data[other_indices].sum(1)
    plot_data.drop(other_indices, axis=1, inplace=True)

    # Sort technologies by their total contribution across all scenarios (descending order)
    # This puts larger contributions at the bottom of the stack
    tech_totals = plot_data.sum(axis=0).sort_values(ascending=False)
    scenario_order = plot_data.sum(axis=1).sort_values(ascending=False).index
    plot_data = plot_data.loc[scenario_order, tech_totals.index]

    # Create figure with 2 subplots (total system costs and German system costs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    # Plot 1: Total system costs - simplified to Germany vs neighbour countries
    plot_data_total = plot_data.div(1e9)  # Convert to billion EUR

    # Create simplified data for total costs plot: aggregate all non-neighbour countries as "Germany"
    plot_data_simplified = pd.DataFrame(index=plot_data_total.index)
    if "neighbour countries" in plot_data_total.columns:
        plot_data_simplified["Germany"] = plot_data_total.drop(
            "neighbour countries", axis=1
        ).sum(axis=1)
        plot_data_simplified["neighbour countries"] = plot_data_total[
            "neighbour countries"
        ]
    else:
        plot_data_simplified["Germany"] = plot_data_total.sum(axis=1)

    # Define colors for simplified plot
    simplified_colors = {"Germany": "#1f77b4", "neighbour countries": "#D3D3D3"}

    plot_data_simplified.plot.bar(
        stacked=True,
        ax=ax1,
        color=plot_data_simplified.columns.map(simplified_colors).fillna("#1f77b4"),
        width=0.8,
        legend=False,  # Disable individual legend
    )
    ax1.set_title(f"Total System Costs in {year}")
    ax1.set_ylabel("Billion EUR per year")
    ax1.set_xlabel("")

    # Plot 2: German system costs (excluding neighbour countries) with full technological detail
    plot_data_de = plot_data.drop("neighbour countries", axis=1, errors="ignore").div(
        1e9
    )
    # Sort the German data with the same order as total data (excluding neighbour countries if not present)
    if "neighbour countries" in tech_totals.index:
        de_order = [
            col
            for col in tech_totals.index
            if col != "neighbour countries" and col in plot_data_de.columns
        ]
    else:
        de_order = [col for col in tech_totals.index if col in plot_data_de.columns]
    plot_data_de = plot_data_de[de_order]

    plot_data_de.plot.bar(
        stacked=True,
        ax=ax2,
        color=plot_data_de.columns.map(colors).fillna("black"),
        width=0.8,
        legend=False,  # Disable individual legend
    )
    ax2.set_title(f"German System Costs in {year}")
    ax2.set_ylabel("Billion EUR per year")
    ax2.set_xlabel("Scenario")

    # Add value labels on top of each bar for both plots
    for ax, data in zip([ax1, ax2], [plot_data_simplified, plot_data_de]):
        for i, scenario in enumerate(data.index):
            total = data.loc[scenario].sum()
            ax.text(
                i,
                total + total * 0.01,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Set y-axis limit to accommodate annotations
        max_value = data.sum(axis=1).max()
        ax.set_ylim(0, max_value * 1.2)  # Add 20% padding above the highest bar

    # Add individual value annotations for Germany and neighbour countries on first plot
    for i, scenario in enumerate(plot_data_simplified.index):
        germany_value = plot_data_simplified.loc[scenario, "Germany"]

        # Annotate Germany value (at the center of the Germany bar)
        ax1.text(
            i,
            germany_value / 2,
            f"{germany_value:.1f}",
            ha="center",
            va="center",
            fontweight="bold",
            color="white",
            fontsize=10,
        )

        # Annotate neighbour countries value if it exists
        if "neighbour countries" in plot_data_simplified.columns:
            neighbour_value = plot_data_simplified.loc[scenario, "neighbour countries"]
            if neighbour_value > 0:  # Only annotate if there's a value
                # Position at center of neighbour countries bar
                ax1.text(
                    i,
                    germany_value + neighbour_value / 2,
                    f"{neighbour_value:.1f}",
                    ha="center",
                    va="center",
                    fontweight="bold",
                    color="black",
                    fontsize=10,
                )

    # Create legend for regional breakdown (above the plots)
    simplified_handles = []
    simplified_labels = []
    for region in plot_data_simplified.columns:
        simplified_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=simplified_colors[region])
        )
        simplified_labels.append(region)

    # Add legend for regional breakdown above the plots
    fig.legend(
        simplified_handles,
        simplified_labels,
        title="Region",
        bbox_to_anchor=(1.1, 1.1),
        loc="upper center",
        ncol=len(simplified_labels),
        frameon=True,
    )

    # Create legend for the detailed German costs (below the plots)
    all_technologies = list(plot_data_de.columns)

    # Create legend handles and labels for detailed technologies
    legend_handles = []
    legend_labels = []
    for tech in all_technologies:
        if tech in colors:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=colors[tech]))
            legend_labels.append(tech)

    # Add unified legend for technologies to the right side of the plots
    fig.legend(
        legend_handles,
        legend_labels,
        title="Technology",
        bbox_to_anchor=(1.05, 0.5),
        loc="center left",
        ncol=2,
        frameon=True,
        # alpha=1
    )

    # Adjust layout to accommodate legends
    plt.subplots_adjust(top=0.85, bottom=0.25)
    plt.savefig(
        os.path.join(output_path, f"system_costs_comparison_{year}.pdf"),
        bbox_inches="tight",
    )

    logger.info(f"System costs comparison plot saved for year {year}")
    return fig, (ax1, ax2)


def main(snakemake):
    """Main function to generate plots from network data."""
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

    logger.info(f"Generating system analysis for run: {run_name}")

    # Process networks and collect data
    networks, summary_df, costs_agg = process_networks(
        run_name, scenarios, planning_horizons
    )

    if not networks:
        logger.error("No networks could be loaded")
        return

    # Get color mapping
    colors = get_colors(networks, override_colors)

    # Generate plots

    # 1. Plot total system costs
    plot_system_costs(
        costs_agg,
        scenarios,
        2045,
        output_path,
        colors,
    )

    # 1. Plot price duration curves with new implementation
    plot_price_duration_curves(networks, output_path)

    # 2. Plot urban central heat supply comparison for each scenario - consolidated in one figure
    plot_uch_supply(networks, os.path.join(output_path, "uch_supply.pdf"), colors)

    # 3. Plot summary metrics
    plot_summary_metrics(summary_df, output_path)

    # 4. Plot PTES SOCs ranges for networks
    plot_ptes_socs(networks, output_path + "/soc_comparison.png")

    # Plot PTES savings comparison
    # Define scenario tuples for PTES comparison
    scenario_tuples = snakemake.params.plotting["scenario_tuples"]

    # Ensure all required scenarios are available in the data
    available_tuples = []
    for ref, comp in scenario_tuples:
        if ref in costs_agg.index.get_level_values(
            0
        ) and comp in costs_agg.index.get_level_values(0):
            available_tuples.append((ref, comp))

    if available_tuples:
        plot_ptes_price_impact_scatter(networks, available_tuples, output_path)

    # 6. Plot PTES savings comparison
    if available_tuples:
        for year in planning_horizons:
            plot_ptes_savings_comparison(
                available_tuples, costs_agg, colors, year, output_path, figsize=(8, 8)
            )

    # 5. Plot dual comparison if configured
    if (
        "dual_comparison" in snakemake.params.plotting
        and snakemake.params.plotting["dual_comparison"]["enable"]
    ):
        scenario_A = snakemake.params.plotting["dual_comparison"]["scenario_A"]
        scenario_B = snakemake.params.plotting["dual_comparison"]["scenario_B"]
        if scenario_A in costs_agg.index.get_level_values(
            0
        ) and scenario_B in costs_agg.index.get_level_values(0):
            plot_dual_comparison(
                networks, costs_agg, scenario_A, scenario_B, colors, output_path
            )

            # Also plot energy balance comparison when dual comparison is enabled
            if scenario_A in networks and scenario_B in networks:
                # First check that we have valid network data for both scenarios
                network_A = networks[scenario_A]
                network_B = networks[scenario_B]

                if network_A and network_B:
                    # Get the first (and typically only) year for each network
                    network_A_year = list(network_A.values())[0]
                    network_B_year = list(network_B.values())[0]

                    # Plot the energy balance comparison
                    plot_energy_balance_comparison(
                        network_A_year,
                        network_B_year,
                        f"Energy Balance Comparison: {scenario_A} vs {scenario_B}",
                        os.path.join(
                            output_path,
                            f"energy_balance_comparison_{scenario_A}_{scenario_B}.pdf",
                        ),
                        colors,
                    )

    # 5. Plot sensitivity analysis if configured
    if sensitivity_runs:
        plot_sensitivity_analysis(
            costs_agg,
            sensitivity_runs,
            scenarios,
            reference_scenario,
            colors,
            output_path,
        )

    # Save data for further analysis
    summary_df.to_csv(snakemake.output.sysgf_summary)

    logger.info("System analysis completed successfully")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
        snakemake = mock_snakemake(
            "plot_sysgf_summary",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )
    main(snakemake)
