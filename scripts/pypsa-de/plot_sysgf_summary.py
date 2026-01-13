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
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "code", "pypsa-de"))
from scripts._helpers import configure_logging, mock_snakemake

# Import temporal heat balance functions
# import importlib.util

# spec = importlib.util.spec_from_file_location(
#     "plot_temporal_heat_balance",
#     os.path.join(
#         os.getcwd(),
#         "code",
#         "pypsa-de",
#         "scripts",
#         "pypsa-de",
#         "plot_temporal_heat_balance.py",
#     ),
# )
# plot_temporal_module = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(plot_temporal_module)

# plot_seasonal_heat_balance_unified = (
#     plot_temporal_module.plot_seasonal_heat_balance_unified
# )
# plot_seasonal_heat_balance_with_prices = (
#     plot_temporal_module.plot_seasonal_heat_balance_with_prices
# )
# plot_seasonal_heat_balance_with_temperature = (
#     plot_temporal_module.plot_seasonal_heat_balance_with_temperature
# )

logger = logging.getLogger(__name__)


def calc_ptes_cycles(n, mean=True):
    """Calculate the number of cycles for PTES (water pits) systems."""
    pits_de = n.stores.filter(regex=r"DE0.*water pits", axis=0).query("e_nom_opt > 0")
    if pits_de.empty:
        return 0

    pits_de_e_max_pu = n.stores_t.e_max_pu.reindex(
        pits_de.index, axis=1, fill_value=1
    ).mean()

    discharge = (
        n.links_t.p0.filter(regex=r"DE.*pits discharger")
        .mul(n.snapshot_weightings.generators, axis=0)
        .sum()
    )
    discharge.index = discharge.index.str.replace(" discharger", "")
    no_cycles = discharge.div(pits_de.e_nom_opt * pits_de_e_max_pu)
    if mean:
        return no_cycles.mean()
    else:
        return no_cycles


def get_pit_capacities_de(n, mode="effective"):
    pits_de = n.stores.filter(regex=r"DE0.*water pits", axis=0).query("e_nom_opt > 0")
    pits_de_e_max_pu = n.stores_t.e_max_pu.reindex(
        pits_de.index, axis=1, fill_value=1
    ).mean()
    if mode == "effective":
        caps = pits_de.e_nom_opt * pits_de_e_max_pu
    elif mode == "volume":
        caps = pits_de.e_nom_opt / 4500 * 70000
    elif mode == "capex":
        caps = pits_de.e_nom_opt * pits_de.capital_cost
    else:
        caps = pits_de.e_nom_opt
    return caps


def calc_average_dh_price_t_ordered(n, aggregate_time=False):
    """Calculate time-ordered average district heating price."""
    dh_price_t = n.buses_t.marginal_price.filter(regex=r"DE0 \d+.*urban central heat")
    dh_wd_t_ordered = n.statistics.withdrawal(
        groupby=["bus", "carrier", "bus_carrier"],
        aggregate_time=False,
        bus_carrier="urban central heat",
    ).filter(like="DE0", axis=0)
    # Drop chargers
    to_drop = dh_wd_t_ordered.filter(like="charger", axis=0).index
    dh_wd_t_ordered = dh_wd_t_ordered.drop(to_drop, axis=0)
    # Drop chargers
    dh_wd_t_ordered = dh_wd_t_ordered.groupby(["bus"]).sum()

    weighted_dh_price_t = (
        dh_price_t.mul(dh_wd_t_ordered.T)
        .sum(1)
        .div(dh_wd_t_ordered.sum())
        .sort_values()
    )
    if aggregate_time:
        weighted_dh_price = (
            weighted_dh_price_t.mul(dh_wd_t_ordered.sum()).sum()
            / dh_wd_t_ordered.sum().sum()
        )
        return weighted_dh_price
    else:
        return weighted_dh_price_t


def calc_average_elec_price_t_ordered(n, aggregate_time=False):
    """Calculate time-ordered average electricity price."""
    elec_price_t = n.buses_t.marginal_price.filter(regex=r"DE0 \d+$")
    elec_wd_t_ordered = n.statistics.withdrawal(
        groupby=["bus", "carrier", "bus_carrier"],
        aggregate_time=False,
        bus_carrier="AC",
    ).filter(like="DE0", axis=0)
    # Drop carrier=='DC' and component=='Line' entries
    to_drop = elec_wd_t_ordered.index[
        (elec_wd_t_ordered.index.get_level_values("carrier") == "DC")
        | (elec_wd_t_ordered.index.get_level_values("component") == "Line")
    ]
    elec_wd_t_ordered = elec_wd_t_ordered.drop(to_drop, axis=0)
    elec_wd_t_ordered = elec_wd_t_ordered.groupby(["bus"]).sum()
    weighted_elec_price_t = (
        elec_price_t.mul(elec_wd_t_ordered.T)
        .sum(1)
        .div(elec_wd_t_ordered.sum())
        .sort_values()
    )
    if aggregate_time:
        weighted_elec_price = (
            weighted_elec_price_t.mul(elec_wd_t_ordered.sum()).sum()
            / elec_wd_t_ordered.sum().sum()
        )
        return weighted_elec_price
    else:
        return weighted_elec_price_t


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


def calc_vres_gen_de(n):
    try:
        gen = (
            n.statistics.energy_balance(groupby=["bus", "carrier"], nice_names=False)
            .xs("Generator", level=0)
            .filter(regex=r"DE.*wind|solar")
            .div(1e6)
            .sum()
        )
        return gen
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


def get_delta_ff_top(ff_temp):
    """Calculate the delta between the top temperature of PTES and FF temperature."""
    delta = ff_temp - 90
    return delta


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


# def calculate_district_heating_costs(n: pypsa.Network) -> float:
#     district_heating_carriers = np.array([])
#     for c in n.iterate_components():
#         if c.name in ["Store", "Link", "Generator"]:
#             # Filter rows where any column contains "urban central"
#             district_heating_carriers = np.append(
#                 district_heating_carriers,
#                 c.df[
#                     c.df.apply(
#                         lambda x: x.astype(str)
#                         .str.contains("urban central", case=False)
#                         .any(),
#                         axis=1,
#                     )
#                 ].carrier.unique(),
#             )

#     capex = n.statistics.capex(
#         groupby=["bus", "carrier", "bus_carrier"], nice_names=False
#     ).filter(like="DE0 ")
#     opex = n.statistics.opex(
#         groupby=["bus", "carrier", "bus_carrier"], nice_names=False
#     ).filter(like="DE0 ")

#     capex_dh = capex[
#         capex.index.get_level_values("carrier").isin(district_heating_carriers)
#     ]
#     opex_dh = opex[
#         opex.index.get_level_values("carrier").isin(district_heating_carriers)
#     ]

#     return capex_dh.sum() + opex_dh.sum()


def calculate_district_heating_costs(n: pypsa.Network) -> float:
    dh_mp = n.buses_t.marginal_price.filter(regex=r"DE0.*urban central heat")
    dh_loads = n.loads_t.p.filter(regex=r"DE\d.*(urban central|low-temperature) heat")
    dac_load = n.links_t.p1.filter(regex=r"DE0.*urban central DAC")
    all_loads = pd.concat([dh_loads, dac_load], axis=1).fillna(0)
    # Replace all the words following central with " heat" in the column names
    all_loads.columns = all_loads.columns.str.replace(
        r"central.*", "central heat", regex=True
    ).str.replace(r"low-temperature.*", "urban central heat", regex=True)
    # Aggregate columns with same name
    all_loads = all_loads.T.groupby(level=0).sum().T

    # Get snapshot weightings
    snapshot_weightings = n.snapshot_weightings.generators

    # Calculate dh consumer costs
    dh_consumer_costs = snapshot_weightings @ (dh_mp * all_loads)

    return dh_consumer_costs.sum()


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
            "PTES_capacity_TWh_scaled",
            "PTES_capacity_m3",
            "PTES_capacity_GW",
            "PTES_no_cycles",
            "PTES_investment_bn€",
            "TTES_capacity_TWh",
            "TTES_capacity_GW",
            "H2_store_TWh",
            "co2_price_EU_EUR_per_ton",
            "co2_price_DE_EUR_per_ton",
            "dh_price_EUR_per_MWh",
            "electricity_price_EUR_per_MWh",
            "peak_electricity_price_EUR_per_MWh",
            "peak_dh_price_EUR_per_MWh",
            "curtailment_TWh",
            "vres_gen_TWh",
            "relative_curtailment",
            "heat_venting_TWh",
            "solar capacity_GW",
            "onwind_capacity_GW",
            "hp_capacity_GW",
            "booster_hp_capacity",
            "electrolysis_cf",
            "RSHP_cf",
            "GSHP_cf",
            "booster_hp_cf",
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
                            "co2_price_EU_EUR_per_ton": -n.global_constraints.loc[
                                "CO2Limit", "mu"
                            ],
                            "co2_price_DE_EUR_per_ton": -n.global_constraints.loc[
                                "co2_limit-DE", "mu"
                            ],
                            "dh_price_EUR_per_MWh": calc_average_dh_price_t_ordered(
                                n, aggregate_time=True
                            ),
                            "electricity_price_EUR_per_MWh": calc_average_elec_price_t_ordered(
                                n, aggregate_time=True
                            ),
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
                            "vres_gen_TWh": calc_vres_gen_de(n),
                            "relative_curtailment": (
                                calc_curtailment_de(n)
                                / (calc_vres_gen_de(n) + calc_curtailment_de(n))
                                if (calc_vres_gen_de(n) + calc_curtailment_de(n)) > 0
                                else 0
                            ),
                            "heat_venting_TWh": calc_heat_venting_de(n),
                            "vRES_capacity_GW": n.generators.filter(
                                regex=r"DE.*(onwind|offwind|solar-)",
                                axis=0,
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "solar_capacity_GW": n.generators.filter(
                                regex=r"DE.*solar", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "wind_capacity_GW": n.generators.filter(
                                regex=r"DE.*(onwind|offwind)", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "CHP_capacity_GW": n.links.filter(regex=r"DE.*CHP", axis=0)
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "H2_CHP_capacity_GW": n.links.filter(
                                regex=r"DE.*H2 CHP", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "resistive_heater_GW": n.links.filter(
                                regex=r"DE.*resistive heater", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "PTES_capacity_TWh_scaled": get_pit_capacities_de(
                                n, mode="effective"
                            ).sum(),
                            "PTES_capacity_m3": get_pit_capacities_de(
                                n, mode="volume"
                            ).sum(),
                            "PTES_investment_bn€": get_pit_capacities_de(
                                n, mode="effective"
                            ).sum(),
                            "booster_hp_GW": (
                                n.links.filter(regex=r"DE.*ptes heat pump", axis=0)
                                .p_nom_opt.div(1e3)
                                .sum()
                                if "hpboost" in scenario
                                else 0
                            ),
                            "booster_hp_cf": (
                                n.statistics.capacity_factor(
                                    groupby=["country", "carrier"]
                                )
                                .xs("DE", level="country")
                                .xs("urban central ptes heat pump", level="carrier")
                                .mean()
                                if "hpboost" in scenario
                                else 0
                            ),
                            "onwind_capacity_GW": n.generators.filter(
                                regex=r"DE.*onwind", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "solar_capacity_GW": n.generators.filter(
                                regex=r"DE.*solar", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "hp_capacity_GW": n.links.filter(
                                regex="DE.*heat pump", axis=0
                            )
                            .p_nom_opt.mul(1 / n.links_t.efficiency.max())
                            .dropna()
                            .div(1e3)
                            .sum(),
                            "electrolysis_cf": n.statistics.capacity_factor(
                                groupby=["country", "carrier"]
                            )
                            .xs("DE", level="country")
                            .xs("H2 Electrolysis", level="carrier")
                            .mean(),
                            "electrolysis_capacity_GW": n.links.filter(
                                regex="DE.*H2 Electrolysis", axis=0
                            )
                            .p_nom_opt.div(1e3)
                            .sum(),
                            "RSHP_cf": n.statistics.capacity_factor(
                                groupby=["country", "carrier"]
                            )
                            .xs("DE", level="country")
                            .xs("urban central river_water heat pump", level="carrier")
                            .mean(),
                            "GSHP_cf": n.statistics.capacity_factor(
                                groupby=["country", "carrier"]
                            )
                            .xs("DE", level="country")
                            .xs("urban central geothermal heat pump", level="carrier")
                            .mean(),
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
        if col_sum < 0.005 * column_sums.sum()
    }

    data.rename(columns=mapping_dict, inplace=True)
    # Only drop columns if they exist
    for col in ["Other Load", "Other Generation"]:
        if col in data.columns:
            data = data.drop(col, axis=1)
    return data.T.groupby(data.columns).sum().T


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

    # Plot bars with some transparency on primary axis
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

    # Handle y-tick labels based on scenarios
    if scenario_A and scenario_A in title:
        ax2.set_yticklabels([])
    if scenario_B and scenario_B in title:
        ax.set_yticklabels([])

    if scenario_A and scenario_A in title:
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

    # Return legend handles and optionally line collection for colorbar
    legend_handles_labels = ax.get_legend_handles_labels()
    if secondary_type == "temperature_delta":
        return legend_handles_labels, line_collection
    else:
        return legend_handles_labels


def plot_dual_comparison(
    networks,
    costs_agg,
    scenario_A,
    scenario_B,
    colors,
    output_path,
    energy_balances_path=None,
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
        other_indices = df_diff[df_diff.abs() < 0.01 * df_diff.abs().sum()].index
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
            costs_baseline.abs().sort_values(ascending=False).index
        ]

        df_diff = df_diff.loc[df_diff.abs().sort_values(ascending=False).index]

        # Plot Baseline scenario
        costs_baseline.to_frame().T.div(1e9).plot.bar(
            stacked=True,
            ax=ax[0],
            color=costs_baseline.index.map(colors).fillna("black"),
            legend=False,
        )
        ax[0].set_ylabel("Costs per Technology [bn€]", fontsize=12)
        ax[0].set_xlabel("")
        ax[0].set_title(f"{scenario_A} Scenario ({year})", fontsize=12)
        # ax[0].set_ylim(0, 140)

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

        # Add titles and labels
        ax[1].set_ylabel("Cost Difference [bn€]", fontsize=12)
        ax[1].set_title(
            f"Cost Difference ({scenario_A} - {scenario_B}) ({year})", fontsize=12
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
                balance_output_path = (
                    energy_balances_path if energy_balances_path else output_path
                )
                plot_seasonal_heat_balance_with_temperature(
                    networks[scenario_A][year],
                    networks[scenario_B][year],
                    scenario_A,
                    scenario_B,
                    colors,
                    balance_output_path,
                    year,
                    snakemake.params.run,  # Add run_name parameter
                )

    logger.info(f"Dual comparison plots saved to {output_path}")


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

    # Collect all price data to calculate 99.5 percentile
    all_hv_prices = []
    all_lv_prices = []
    all_dh_prices = []

    # Plot for each scenario and network
    for scenario, networks_scenario in networks_dict.items():
        # Drop init, CAPEX, freeboost, and freehp scenarios
        if (
            "capex" in scenario.lower()
            or "freeboost" in scenario.lower()
            or "freehp" in scenario.lower()
            or "init" in scenario.lower()
        ):
            continue
        for year, network in networks_scenario.items():
            label = f"{scenario}_{year}"

            try:
                # Average Electricity Price
                hv_elec = calc_average_electricity_price(network)
                if not hv_elec.empty:
                    hv_elec.plot(ax=ax[0], label=label, linewidth=0.5)
                    all_hv_prices.extend(hv_elec.values)

                # Average Low Voltage Electricity Price
                lv_elec = calc_average_electricity_lv_price(network)
                if not lv_elec.empty:
                    lv_elec.plot(ax=ax[1], label=label, linewidth=0.5)
                    all_lv_prices.extend(lv_elec.values)

                # Average District Heating Price
                dh = calc_average_dh_price(network)
                if not dh.empty:
                    dh.plot(ax=ax[2], label=label, linewidth=0.5)
                    all_dh_prices.extend(dh.values)
            except Exception as e:
                logger.warning(f"Error plotting price duration curves for {label}: {e}")

    # Calculate 99.5 percentile for each price type
    hv_ylim = np.percentile(all_hv_prices, 99.5) if all_hv_prices else None
    lv_ylim = np.percentile(all_lv_prices, 99.5) if all_lv_prices else None
    dh_ylim = np.percentile(all_dh_prices, 99.5) if all_dh_prices else None

    # Set titles and formatting
    ax[0].set_title("Average Electricity Price")
    ax[1].set_title("Average Low Voltage Electricity Price")
    ax[2].set_title("Average District Heating Price")

    # Set y-axis limits based on 99.5 percentile
    # logger.info(f"HV Y-Limit: {hv_ylim}")
    if hv_ylim is not None and hv_ylim != np.nan and hv_ylim != np.inf:
        ax[0].set_ylim(0, hv_ylim)
    if lv_ylim is not None:
        ax[1].set_ylim(0, lv_ylim)
    if dh_ylim is not None:
        ax[2].set_ylim(0, dh_ylim)

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
        year_colors = plt.cm.coolwarm(np.linspace(0, 1, len(years)))

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

        # Group geothermal technologies
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

        return to_plot_rel, dh_prices

    # Prepare data for both networks
    to_plot_rel1, dh_prices1 = prepare_energy_balance_data(network1)
    to_plot_rel2, dh_prices2 = prepare_energy_balance_data(network2)

    # Calculate price savings (network1 - network2)
    dh_price_savings = dh_prices1 - dh_prices2

    # Sort systems by price savings (highest savings first)
    sorted_systems = dh_price_savings.sort_values(ascending=False).index

    # Reorder both plotting data and price data according to savings
    to_plot_rel1 = to_plot_rel1.loc[sorted_systems]
    to_plot_rel2 = to_plot_rel2.loc[sorted_systems]
    dh_price_savings = dh_price_savings.loc[sorted_systems]

    max_ylim = to_plot_rel2.clip(lower=0).sum(1).max() * 1.05

    # Create subplots with original dimensions
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Plot for Network 1
    ax1 = axes[0]

    col_order = [
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        "urban central electrolysis excess heat pump",
        "geothermal heat pump",
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
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
    ]  # Filter to only include columns that exist in the data
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

    ax1.axhline(y=0, color="black", linestyle="-")
    ax1.set_ylim(-max_ylim, max_ylim)

    # Plot for Network 2
    ax2 = axes[1]

    col_order = [
        "low-temperature heat for industry",
        "urban central heat",
        "urban central heat vent",
        "urban central electrolysis excess heat pump",
        "geothermal heat pump",
        "urban central river_water heat pump",
        "urban central sea_water heat pump",
        "urban central air heat pump",
        "urban central ptes heat pump",
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
        "urban central water pits discharger",
        "urban central water tanks charger",
        "urban central water tanks losses",
        "urban central water pits charger",
        "urban central water pits losses",
    ]  # Filter to only include columns that exist in the data
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
    ax2.axhline(y=0, color="black", linestyle="-")
    ax2.set_ylim(-max_ylim, max_ylim)

    # Add secondary y-axis for district heating price savings on second subplot only
    ax2_price = ax2.twinx()

    # Plot DH price savings for Network 2 with white circles and black borders
    x_positions2 = range(len(dh_price_savings))
    ax2_price.scatter(
        x_positions2,
        dh_price_savings.values,
        s=40,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=0.2,
        zorder=20,
        clip_on=False,
    )

    # Add mean DH price savings line
    ax2_price.axhline(
        y=dh_price_savings.mean(),
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=1,
        zorder=5,
    )

    # Set labels and formatting for price savings axis
    ax2_price.set_ylabel("DH Price Savings [€/MWh]", fontsize=10)
    ax2_price.tick_params(axis="y", labelsize=10)

    # Set y-limits for price savings axis with some padding
    price_min, price_max = dh_price_savings.min(), dh_price_savings.max()
    price_range = price_max - price_min
    if price_range > 0:
        padding = price_range * 0.1  # 10% padding
        price_ylim = (price_min - padding, price_max + padding)
    else:
        # If all savings are the same, add some padding around the value
        price_ylim = (price_min * 0.95, price_max * 1.05)

    ax2_price.set_ylim(price_ylim)

    # Decrease fontsize of ax2 xticks

    for tick in ax2.get_xticklabels():
        tick.set_fontsize(10)

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

    # Add DH price savings indicators to legend
    from matplotlib.lines import Line2D

    # Add DH price savings marker to legend
    unique_handles.append(
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
    unique_labels.append("Demand-weighted\nDH price savings")

    # Add mean DH price savings line to legend
    unique_handles.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
    )
    unique_labels.append("Mean DH price savings")

    fig.legend(
        unique_handles,
        unique_labels,
        title="Technology",
        bbox_to_anchor=(1.17, 0.5),
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
    import matplotlib.cm as cm
    import numpy as np

    plt.figure(figsize=figsize)

    # Collect all scenario-year combinations first to determine color mapping
    scenario_year_combinations = []
    cycle_counts = []

    for scenario, networks_scenario in networks.items():
        # Skip CAPEX and freeboost and freecap scenarios
        if (
            "CAPEX" in scenario
            or "freeboost" in scenario
            or "freecap" in scenario
            or "init" in scenario
        ):
            continue
        for year, networks_year in networks_scenario.items():
            soc = (
                networks_year.stores_t.e.filter(regex="DE.*water pits").div(1e6).sum(1)
            )
            if soc is not None and not soc.empty:
                # Calculate number of cycles (approximate)
                # A cycle is defined as the total energy discharged divided by storage capacity
                ptes_stores = networks_year.stores.filter(
                    regex="DE.*water pits", axis=0
                )
                total_capacity = ptes_stores.e_nom_opt.sum() / 1e6  # TWh

                # Calculate total energy discharged (positive values from stores_t.p)
                ptes_discharge = (
                    networks_year.stores_t.p.filter(regex="DE.*water pits")
                    .clip(lower=0)
                    .sum(1)
                )
                total_discharged = (
                    ptes_discharge * networks_year.snapshot_weightings.generators
                ).sum() / 1e6  # TWh

                # Number of cycles = total discharged / capacity
                cycles = total_discharged / total_capacity if total_capacity > 0 else 0

                scenario_year_combinations.append((scenario, year, soc))
                cycle_counts.append(cycles)

    # Sort scenarios by number of cycles (lowest cycles first for darkest blue)
    n_combinations = len(scenario_year_combinations)
    if n_combinations > 0:
        # Create list of tuples with (cycle_count, scenario, year, soc) for sorting
        sorted_combinations = sorted(
            zip(cycle_counts, scenario_year_combinations),
            key=lambda x: x[0],  # Sort by cycle count (ascending)
        )

        # Extract sorted data
        sorted_cycle_counts = [x[0] for x in sorted_combinations]
        sorted_scenario_year_combinations = [x[1] for x in sorted_combinations]

        # Create Blues colormap with darkest blue for lowest cycles
        # Reverse the color mapping so lowest cycles get darkest blue (1.0) and highest get lightest (0.3)
        colors = cm.Blues(
            np.linspace(1.0, 0.3, n_combinations)
        )  # Start from 1.0 (darkest) to 0.3 (lightest)

        years = list(set(year for _, year, _ in sorted_scenario_year_combinations))

        # Plot each SOC curve with assigned color
        for i, (scenario, year, soc) in enumerate(sorted_scenario_year_combinations):
            label = f"{scenario}_{year}" if len(years) > 1 else scenario
            plt.plot(
                soc.index,
                soc,
                alpha=0.7,
                color=colors[i],
                label=label,
                linewidth=1.5,
            )

        # Add cycle annotations with smart positioning to avoid overlaps
        # Find good positions for annotations (spread them vertically)
        y_positions = []
        x_positions = []

        for i, (scenario, year, soc) in enumerate(sorted_scenario_year_combinations):
            # Use the maximum SOC value as the y position
            max_soc_idx = soc.idxmax()
            max_soc_val = soc.max()

            # Store positions for overlap checking
            y_positions.append(max_soc_val)
            x_positions.append(max_soc_idx)

        # Adjust y positions to avoid overlaps
        adjusted_y_positions = []
        for i, (y_pos, x_pos) in enumerate(zip(y_positions, x_positions)):
            adjusted_y = y_pos

            # Check for overlaps with previously placed annotations
            for j, prev_y in enumerate(adjusted_y_positions):
                if (
                    abs(adjusted_y - prev_y)
                    < (max(y_positions) - min(y_positions)) * 0.05
                ):  # 5% of range
                    # Move annotation up or down to avoid overlap
                    if i % 2 == 0:
                        adjusted_y = (
                            prev_y + (max(y_positions) - min(y_positions)) * 0.08
                        )
                    else:
                        adjusted_y = (
                            prev_y - (max(y_positions) - min(y_positions)) * 0.08
                        )

            adjusted_y_positions.append(adjusted_y)

        # Add the cycle annotations
        for i, (scenario, year, soc) in enumerate(sorted_scenario_year_combinations):
            cycles = sorted_cycle_counts[i]
            x_pos = x_positions[i]
            y_pos = adjusted_y_positions[i]

            # Create annotation text
            cycles_text = f"{cycles:.0f} cycles"

            # Add annotation with arrow pointing to the curve
            plt.annotate(
                cycles_text,
                xy=(x_pos, y_positions[i]),  # Point to actual curve
                xytext=(x_pos, y_pos),  # Position of text (adjusted)
                arrowprops=dict(arrowstyle="->", color=colors[i], alpha=0.7, lw=1),
                fontsize=9,
                fontweight="bold",
                color=colors[i],
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=colors[i],
                    alpha=0.8,
                ),
            )

    plt.title("PTES State of Charge Ranges")
    plt.xlabel("Time")
    plt.ylabel("State of Charge [TWh]")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
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
    scenario_tuples,
    costs_agg,
    colors,
    year,
    output_path,
    figsize=(8, 4),
    output_suffix="",
):
    """
    Plot cost savings comparison between scenarios with and without PTES.
    All subplots share the same y-axis limits for better comparability.

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
    output_suffix : str, optional
        Suffix to add to the output filename (e.g., "HighSupplyTemperature", "LowSupplyTemperature")
    """
    logger.info(f"Generating PTES savings comparison for year {year}")

    # Filter the costs data for the specified year
    costs_year = costs_agg.xs(year, level="year")

    # Create single figure for all scenarios
    n_comparisons = len(scenario_tuples)
    # Adjust figure width based on number of comparisons
    adjusted_figsize = (max(6, n_comparisons * 1.2), figsize[1])
    fig, ax = plt.subplots(1, 1, figsize=adjusted_figsize)

    # Keep track of all displayed technologies across all comparisons for the legend
    all_displayed_techs = set()

    # First pass: collect all cost differences to determine global y-axis limits
    all_cumulative_values = []
    scenario_data = {}
    baseline_costs = {}  # Store baseline costs for percentage calculations

    for ref_scenario, comp_scenario in scenario_tuples:
        # Check if both scenarios exist in the data
        if (
            ref_scenario not in costs_year.index
            or comp_scenario not in costs_year.index
        ):
            continue

        # Get baseline costs for percentage calculation
        ref_costs = costs_year.loc[ref_scenario]
        baseline_total = ref_costs.sum()
        baseline_german = ref_costs.drop("neighbour countries", errors="ignore").sum()
        baseline_costs[(ref_scenario, comp_scenario)] = {
            "total": baseline_total,
            "german": baseline_german,
        }

        # Calculate cost differences: comp_scenario - ref_scenario
        df_diff = costs_year.loc[comp_scenario].sub(costs_year.loc[ref_scenario])
        df_diff = df_diff[df_diff != 0]

        # Group geothermal technologies
        geothermal_techs = [
            tech
            for tech in df_diff.index
            if "urban central geothermal heat pump" in tech
            or "urban central geothermal heat" in tech
        ]
        if len(geothermal_techs) > 0:
            df_diff["geothermal heat pump"] = df_diff[geothermal_techs].sum()
            df_diff = df_diff.drop(geothermal_techs)

        # Group small contributors into "other technologies"
        small_indices = df_diff.index[df_diff.abs() < 0.02 * df_diff.abs().sum()]
        if len(small_indices) > 0:
            df_diff["other technologies"] = df_diff[small_indices].sum()
            df_diff = df_diff.drop(small_indices)

        # Convert to billion EUR for plotting
        df_diff_bn = df_diff.div(1e9)

        # Sort by magnitude for proper stacking calculation
        if "neighbour countries" in df_diff_bn.index:
            neighbour_value = df_diff_bn["neighbour countries"]
            df_diff_bn_sorted = df_diff_bn.drop("neighbour countries")
            df_diff_bn_sorted = df_diff_bn_sorted.reindex(
                df_diff_bn_sorted.abs().sort_values(ascending=False).index
            )
            df_diff_bn_sorted["neighbour countries"] = neighbour_value
        else:
            df_diff_bn_sorted = df_diff_bn.reindex(
                df_diff_bn.abs().sort_values(ascending=False).index
            )

        # Store for later plotting (in billion EUR for the bars)
        scenario_data[(ref_scenario, comp_scenario)] = df_diff_bn_sorted

        # Calculate cumulative sums for stacked bar limits
        # For stacked bars, we need to consider positive and negative contributions separately
        positive_values = df_diff_bn_sorted[df_diff_bn_sorted > 0]
        negative_values = df_diff_bn_sorted[df_diff_bn_sorted < 0]

        # Calculate cumulative positive and negative sums
        max_positive_cumsum = positive_values.sum() if not positive_values.empty else 0
        min_negative_cumsum = negative_values.sum() if not negative_values.empty else 0

        # Also include the total sum (which represents the net difference markers)
        total_sum = df_diff_bn_sorted.sum()

        # Collect all extreme values
        all_cumulative_values.extend(
            [max_positive_cumsum, min_negative_cumsum, total_sum, 0]
        )

    # Calculate global y-axis limits with some padding
    if all_cumulative_values:
        global_min = min(all_cumulative_values)
        global_max = max(all_cumulative_values)
        y_range = global_max - global_min
        padding = y_range * 0.15  # 15% padding for better visibility
        global_ylim = (global_min - padding, global_max + padding)
        logger.info(
            f"Global y-axis limits for cost comparison: [{global_ylim[0]:.2f}, {global_ylim[1]:.2f}] bn€"
        )
    else:
        global_ylim = None

    # Sort scenario tuples by their total system savings (descending order)
    scenario_savings = []
    for ref_scenario, comp_scenario in scenario_tuples:
        if (
            ref_scenario in costs_year.index
            and comp_scenario in costs_year.index
            and (ref_scenario, comp_scenario) in scenario_data
        ):
            total_savings = scenario_data[(ref_scenario, comp_scenario)].sum()
            scenario_savings.append((total_savings, ref_scenario, comp_scenario))

    # Sort by total savings (descending - largest savings first)
    scenario_savings.sort(key=lambda x: x[0], reverse=True)
    sorted_scenario_tuples = [(ref, comp) for _, ref, comp in scenario_savings]

    logger.info(f"Scenarios ordered by total system savings:")
    for savings, ref, comp in scenario_savings:
        logger.info(f"  {comp}: {savings:.2f} bn€")

    # Process each scenario pair for plotting
    plot_data_list = []
    x_labels = []

    for ref_scenario, comp_scenario in sorted_scenario_tuples:

        # Check if both scenarios exist in the data
        if (
            ref_scenario not in costs_year.index
            or comp_scenario not in costs_year.index
        ):
            logger.warning(
                f"Scenarios {ref_scenario} or {comp_scenario} not found in data for year {year}"
            )
            continue

        logger.info(f"  Processing comparison: {ref_scenario} vs {comp_scenario}")

        # Get pre-calculated and sorted data
        if (ref_scenario, comp_scenario) not in scenario_data:
            logger.warning(f"No data available for {ref_scenario} vs {comp_scenario}")
            continue

        df_diff_bn = scenario_data[(ref_scenario, comp_scenario)]

        # Add the displayed technologies from this comparison to our set
        all_displayed_techs.update(df_diff_bn.index)

        # Store data for plotting
        plot_data_list.append(df_diff_bn)
        x_labels.append(comp_scenario)

    # Create combined DataFrame for plotting
    if not plot_data_list:
        logger.warning("No valid scenario comparisons found for plotting")
        return None, None

    # Combine all data into one DataFrame with scenarios as columns
    combined_plot_data = pd.DataFrame(index=sorted(all_displayed_techs))
    for i, (data, label) in enumerate(zip(plot_data_list, x_labels)):
        combined_plot_data[label] = data.reindex(combined_plot_data.index, fill_value=0)

    # Transpose so scenarios are rows and technologies are columns
    combined_plot_data = combined_plot_data.T

    # Reorder technology (column) order by total absolute contribution (descending)
    if not combined_plot_data.empty:
        abs_order = (
            combined_plot_data.abs().sum(axis=0).sort_values(ascending=False).index
        )
        combined_plot_data = combined_plot_data[abs_order]

    # Calculate savings markers from the combined plot data (AFTER sorting and combining)
    total_savings_list = []
    germany_savings_list = []

    for i, scenario in enumerate(combined_plot_data.index):
        # Find the corresponding reference scenario for this comparison
        ref_scenario, comp_scenario = None, None
        for ref_scen, comp_scen in sorted_scenario_tuples:
            if comp_scen == scenario:
                ref_scenario, comp_scenario = ref_scen, comp_scen
                break

        if ref_scenario is None:
            # Fallback: total savings is the sum of all technologies for this scenario (in bn€)
            total_savings = combined_plot_data.loc[scenario].sum()
            germany_savings = (
                combined_plot_data.loc[scenario]
                .drop("neighbour countries", errors="ignore")
                .sum()
            )
            # Convert to percentage for annotations (assuming baseline of 100 bn€ as fallback)
            total_savings_pct = (total_savings / 100) * 100  # Fallback conversion
            germany_savings_pct = (germany_savings / 100) * 100
        else:
            # Total savings is the sum of all technologies for this scenario (in bn€)
            total_savings = combined_plot_data.loc[scenario].sum()

            # German savings excludes neighbour countries if present (in bn€)
            germany_data = combined_plot_data.loc[scenario].drop(
                "neighbour countries", errors="ignore"
            )
            germany_savings = germany_data.sum()

            # Convert to percentage for annotations using stored baseline costs
            baseline_info = baseline_costs[(ref_scenario, comp_scenario)]
            total_savings_pct = (
                total_savings / baseline_info["total"] * 1e9
            ) * 100  # Convert back from bn€ to €, then to %
            germany_savings_pct = (
                germany_savings / baseline_info["german"] * 1e9
            ) * 100  # Use total baseline for consistency

        total_savings_list.append(total_savings)
        germany_savings_list.append(germany_savings)

        # Store percentage values for annotations
        if i == 0:  # Initialize lists on first iteration
            total_savings_pct_list = []
            germany_savings_pct_list = []
        total_savings_pct_list.append(total_savings_pct)
        germany_savings_pct_list.append(germany_savings_pct)

    # Plot stacked bar chart
    combined_plot_data.plot.bar(
        stacked=True,
        ax=ax,
        color=combined_plot_data.columns.map(colors).fillna("black"),
        legend=False,
        width=0.8,
    )

    # Add horizontal line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, zorder=1)

    # Add markers for total system cost savings and German system cost savings
    for i, (
        total_savings,
        germany_savings,
        total_savings_pct,
        germany_savings_pct,
    ) in enumerate(
        zip(
            total_savings_list,
            germany_savings_list,
            total_savings_pct_list,
            germany_savings_pct_list,
        )
    ):
        # Calculate gross savings (sum of all negative entries) for this scenario
        scenario_data_row = combined_plot_data.iloc[i]
        gross_savings = scenario_data_row[scenario_data_row < 0].sum()

        # Determine annotation position based on gross savings direction
        if gross_savings < 0:
            # Net savings - place annotations below the bar
            annotation_y = gross_savings
            y_offset = -15
            va = "top"
        else:
            # Net costs - place annotations above the bar (at 0 line)
            annotation_y = 0
            y_offset = 15
            va = "bottom"

        # Add total system savings marker (white star with black border)
        ax.scatter(
            i,
            total_savings,
            s=100,
            marker="*",
            facecolor="whitesmoke",
            edgecolor="black",
            linewidth=1,
            zorder=12,
        )

        # Add annotation for total system savings with star symbol
        ax.annotate(
            f"★ {total_savings_pct:.2f} %",
            xy=(i, annotation_y),
            xytext=(0, y_offset),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va=va,
            zorder=15,
        )

        # Add German system savings marker (white circle with black border) if different
        if germany_savings != total_savings:
            ax.scatter(
                i,
                germany_savings,
                s=100,
                marker="o",
                facecolor="white",
                edgecolor="black",
                linewidth=1,
                zorder=10,
            )

            # Add annotation for German system savings with circle symbol
            # Position slightly offset from total system annotation
            ax.annotate(
                f"○ {germany_savings_pct:.2f} %",
                xy=(i, annotation_y),
                xytext=(0, y_offset + (-10 if va == "top" else 10)),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va=va,
                zorder=15,
            )

    # Set labels and formatting
    ax.set_xlabel("Boosting configuration", fontsize=16)
    ax.set_ylabel("Cost Difference [bn€]", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    # Process x-tick labels: split at underscore and remove first part
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    processed_labels = []
    for label in current_labels:
        if "_" in label:
            # Split at underscore and take everything after the first part
            parts = label.split("_")
            processed_label = "_".join(parts[1:])
        else:
            processed_label = label
        processed_labels.append(processed_label)

    # Set the processed labels
    ax.set_xticklabels(processed_labels)

    # Rotate x-axis labels for better readability and ensure proper alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add plot title if output_suffix is provided
    if output_suffix:
        ax.set_title(output_suffix, fontsize=18, pad=20)

    # Set global y-axis limits for uniform comparison
    if global_ylim is not None:
        ax.set_ylim(global_ylim)

    # Add legend for the markers - position more compactly with proper marker symbols
    from matplotlib.lines import Line2D

    line_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=10,
            linestyle="None",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            markeredgecolor="black",
            markeredgewidth=1,
            markersize=10,
            linestyle="None",
        ),
    ]
    line_labels = ["Total system", "German system"]

    # Create a separate legend for technologies - ONLY for actually displayed technologies
    # Sort displayed technologies by their total importance across all comparisons
    # Use the same absolute ordering for legend importance
    tech_importance = combined_plot_data.abs().sum().to_dict()
    sorted_displayed_techs = list(combined_plot_data.columns)

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

    # Add legend for the markers on the right side above technologies
    fig.legend(
        line_handles,
        line_labels,
        title="Net difference for",
        bbox_to_anchor=(0.8, 0.9),
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )

    # Add legend for technologies on the right side, below markers
    fig.legend(
        tech_handles,
        tech_labels,
        title="Technology",
        bbox_to_anchor=(0.8, 0.5),
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )

    # Adjust layout with reduced space for more compact figure
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # More space for legends on the right

    # Save figure
    if output_suffix:
        output_file = os.path.join(
            output_path, f"ptes_savings_comparison_{year}_{output_suffix}.pdf"
        )
    else:
        output_file = os.path.join(output_path, f"ptes_savings_comparison_{year}.pdf")
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.1)  # Reduced padding

    logger.info(f"PTES savings comparison plot saved to {output_file}")

    return fig, ax


def plot_neighbour_countries_cost_comparison(
    networks,
    scenario_tuples,
    colors,
    year,
    output_path,
    figsize=(8, 4),
    output_suffix="",
):
    """
    Plot cost differences by technology in neighbour countries between scenarios.

    Parameters:
    -----------
    networks : dict
        Dictionary mapping scenario names to networks by year
    scenario_tuples : list of tuples
        List of (reference_scenario, comparison_scenario) tuples
    colors : dict
        Dictionary of colors for each technology
    year : int
        Year to plot
    output_path : str
        Path to save the output figure
    figsize : tuple, optional
        Figure size (width, height)
    output_suffix : str, optional
        Suffix to add to the output filename
    """
    logger.info(f"Generating neighbour countries cost comparison for year {year}")

    def get_neighbour_costs_by_technology(network):
        """Extract neighbour countries costs by technology from a network."""
        capex = network.statistics.capex(
            nice_names=False, groupby=["bus", "carrier", "bus_carrier"]
        )
        capex = capex.drop(capex.filter(like="DE").index)

        opex = network.statistics.opex(
            nice_names=False, groupby=["bus", "carrier", "bus_carrier"]
        )
        opex = opex.drop(opex.filter(like="DE").index)

        # Combine capex and opex
        costs = pd.concat([capex, opex], axis=1).sum(axis=1)

        # Group by carrier (technology) by dropping bus and component levels
        costs_by_tech = costs.droplevel([0, 1, 3]).groupby(level=0).sum()

        return costs_by_tech

    # Create single figure for all scenarios
    n_comparisons = len(scenario_tuples)
    adjusted_figsize = (max(6, n_comparisons * 1.2), figsize[1])
    fig, ax = plt.subplots(1, 1, figsize=adjusted_figsize)

    # Keep track of all displayed technologies across all comparisons
    all_displayed_techs = set()

    # First pass: collect all cost differences to determine global y-axis limits
    all_cumulative_values = []
    scenario_data = {}
    baseline_costs = {}

    for ref_scenario, comp_scenario in scenario_tuples:
        # Check if both scenarios exist in the networks
        if (
            ref_scenario not in networks
            or comp_scenario not in networks
            or year not in networks[ref_scenario]
            or year not in networks[comp_scenario]
        ):
            logger.warning(
                f"Missing data for {ref_scenario} or {comp_scenario} in year {year}"
            )
            continue

        # Get neighbour costs for both scenarios
        ref_costs = get_neighbour_costs_by_technology(networks[ref_scenario][year])
        comp_costs = get_neighbour_costs_by_technology(networks[comp_scenario][year])

        # Store baseline costs for percentage calculations
        baseline_total = ref_costs.sum()
        baseline_costs[(ref_scenario, comp_scenario)] = {"total": baseline_total}

        # Calculate cost differences: comp_scenario - ref_scenario
        # Align indices to include all technologies from both scenarios
        all_techs = ref_costs.index.union(comp_costs.index)
        ref_costs_aligned = ref_costs.reindex(all_techs, fill_value=0)
        comp_costs_aligned = comp_costs.reindex(all_techs, fill_value=0)

        df_diff = comp_costs_aligned.sub(ref_costs_aligned)
        df_diff = df_diff[df_diff != 0]  # Remove zero differences

        # Group small contributors into "other technologies"
        small_indices = df_diff.index[df_diff.abs() < 0.02 * df_diff.abs().sum()]
        if len(small_indices) > 0:
            df_diff["other technologies"] = df_diff[small_indices].sum()
            df_diff = df_diff.drop(small_indices)

        # Convert to billion EUR for plotting
        df_diff_bn = df_diff.div(1e9)

        # Sort by magnitude for proper stacking calculation
        df_diff_bn_sorted = df_diff_bn.reindex(
            df_diff_bn.abs().sort_values(ascending=False).index
        )

        # Store for later plotting
        scenario_data[(ref_scenario, comp_scenario)] = df_diff_bn_sorted

        # Calculate cumulative sums for stacked bar limits
        positive_values = df_diff_bn_sorted[df_diff_bn_sorted > 0]
        negative_values = df_diff_bn_sorted[df_diff_bn_sorted < 0]

        max_positive_cumsum = positive_values.sum() if not positive_values.empty else 0
        min_negative_cumsum = negative_values.sum() if not negative_values.empty else 0
        total_sum = df_diff_bn_sorted.sum()

        all_cumulative_values.extend(
            [max_positive_cumsum, min_negative_cumsum, total_sum, 0]
        )

    # Calculate global y-axis limits with padding
    if all_cumulative_values:
        global_min = min(all_cumulative_values)
        global_max = max(all_cumulative_values)
        y_range = global_max - global_min
        padding = y_range * 0.15
        global_ylim = (global_min - padding, global_max + padding)
        logger.info(
            f"Global y-axis limits for neighbour countries comparison: [{global_ylim[0]:.2f}, {global_ylim[1]:.2f}] bn€"
        )
    else:
        global_ylim = None

    # Sort scenario tuples by their total system cost changes
    scenario_savings = []
    for ref_scenario, comp_scenario in scenario_tuples:
        if (ref_scenario, comp_scenario) in scenario_data:
            total_change = scenario_data[(ref_scenario, comp_scenario)].sum()
            scenario_savings.append((total_change, ref_scenario, comp_scenario))

    # Sort by total cost change (descending - largest savings first)
    scenario_savings.sort(key=lambda x: x[0], reverse=True)
    sorted_scenario_tuples = [(ref, comp) for _, ref, comp in scenario_savings]

    logger.info(f"Neighbour countries scenarios ordered by total cost change:")
    for change, ref, comp in scenario_savings:
        logger.info(f"  {comp}: {change:.2f} bn€")

    # Process each scenario pair for plotting
    plot_data_list = []
    x_labels = []

    for ref_scenario, comp_scenario in sorted_scenario_tuples:
        if (ref_scenario, comp_scenario) not in scenario_data:
            continue

        df_diff_bn = scenario_data[(ref_scenario, comp_scenario)]
        all_displayed_techs.update(df_diff_bn.index)
        plot_data_list.append(df_diff_bn)
        x_labels.append(comp_scenario)

    # Create combined DataFrame for plotting
    if not plot_data_list:
        logger.warning(
            "No valid scenario comparisons found for neighbour countries plotting"
        )
        return None, None

    combined_plot_data = pd.DataFrame(index=sorted(all_displayed_techs))
    for i, (data, label) in enumerate(zip(plot_data_list, x_labels)):
        combined_plot_data[label] = data.reindex(combined_plot_data.index, fill_value=0)

    # Transpose so scenarios are rows and technologies are columns
    combined_plot_data = combined_plot_data.T

    # Calculate cost change markers
    total_changes_list = []
    total_changes_pct_list = []

    for i, scenario in enumerate(combined_plot_data.index):
        # Find the corresponding reference scenario
        ref_scenario, comp_scenario = None, None
        for ref_scen, comp_scen in sorted_scenario_tuples:
            if comp_scen == scenario:
                ref_scenario, comp_scenario = ref_scen, comp_scen
                break

        total_change = combined_plot_data.loc[scenario].sum()
        total_changes_list.append(total_change)

        if ref_scenario and (ref_scenario, comp_scenario) in baseline_costs:
            baseline_info = baseline_costs[(ref_scenario, comp_scenario)]
            total_change_pct = (total_change / baseline_info["total"] * 1e9) * 100
        else:
            total_change_pct = 0  # Fallback

        total_changes_pct_list.append(total_change_pct)

    # Plot stacked bar chart
    combined_plot_data.plot.bar(
        stacked=True,
        ax=ax,
        color=combined_plot_data.columns.map(colors).fillna("black"),
        legend=False,
        width=0.8,
    )

    # Add horizontal line at 0
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, zorder=1)

    # Add markers for total cost changes
    for i, (total_change, total_change_pct) in enumerate(
        zip(total_changes_list, total_changes_pct_list)
    ):
        # Determine annotation position
        scenario_data_row = combined_plot_data.iloc[i]
        gross_change = (
            scenario_data_row[scenario_data_row < 0].sum()
            if (scenario_data_row < 0).any()
            else 0
        )

        if gross_change < 0:
            annotation_y = gross_change
            y_offset = -15
            va = "top"
        else:
            annotation_y = 0
            y_offset = 15
            va = "bottom"

        # Add total cost change marker
        ax.scatter(
            i,
            total_change,
            s=100,
            marker="^",
            facecolor="lightblue",
            edgecolor="black",
            linewidth=1,
            zorder=12,
        )

        # Add annotation
        ax.annotate(
            f"▲ {total_change_pct:.2f} %",
            xy=(i, annotation_y),
            xytext=(0, y_offset),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va=va,
            zorder=15,
        )

    # Set labels and formatting
    ax.set_xlabel("Boosting configuration", fontsize=16)
    ax.set_ylabel("Cost Difference in Neighbour Countries [bn€]", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

    # Process x-tick labels: split at underscore and remove first part
    current_labels = [label.get_text() for label in ax.get_xticklabels()]
    processed_labels = []
    for label in current_labels:
        if "_" in label:
            parts = label.split("_")
            processed_label = "_".join(parts[1:])
        else:
            processed_label = label
        processed_labels.append(processed_label)

    ax.set_xticklabels(processed_labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add plot title
    title = f"Neighbour Countries Cost Impact"
    if output_suffix:
        title += f" - {output_suffix}"
    ax.set_title(title, fontsize=18, pad=20)

    # Set global y-axis limits
    if global_ylim is not None:
        ax.set_ylim(global_ylim)

    # Add legends
    from matplotlib.lines import Line2D

    # Marker legend
    marker_handle = Line2D(
        [0],
        [0],
        marker="^",
        color="lightblue",
        markeredgecolor="black",
        markeredgewidth=1,
        markersize=10,
        linestyle="None",
    )

    # Technology legend
    sorted_displayed_techs = sorted(
        all_displayed_techs, key=lambda x: sum(abs(combined_plot_data[x])), reverse=True
    )

    tech_handles = []
    tech_labels = []
    for tech in sorted_displayed_techs:
        tech_color = colors.get(tech, "black")
        if not tech_color or tech_color == "":
            tech_color = "black"
        tech_handles.append(plt.Rectangle((0, 0), 1, 1, color=tech_color))
        tech_labels.append(tech)

    # Add legends
    fig.legend(
        [marker_handle],
        ["Total cost change"],
        title="Net difference",
        bbox_to_anchor=(0.8, 0.9),
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )

    fig.legend(
        tech_handles,
        tech_labels,
        title="Technology",
        bbox_to_anchor=(0.8, 0.5),
        loc="center left",
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    # Save figure
    if output_suffix:
        output_file = os.path.join(
            output_path,
            f"neighbour_countries_cost_comparison_{year}_{output_suffix}.pdf",
        )
    else:
        output_file = os.path.join(
            output_path, f"neighbour_countries_cost_comparison_{year}.pdf"
        )

    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.1)
    logger.info(f"Neighbour countries cost comparison plot saved to {output_file}")

    return fig, ax


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
        :, (plot_data.max() < 0.01 * plot_data.sum(1).max())
    ].columns
    plot_data["other technologies"] = plot_data[other_indices].sum(1)
    plot_data.drop(other_indices, axis=1, inplace=True)

    # Sort technologies by their total contribution across all scenarios (descending order)
    # This puts larger contributions at the bottom of the stack
    tech_totals = plot_data.sum(axis=0).sort_values(ascending=False)
    scenario_order = plot_data.sum(axis=1).sort_values(ascending=False).index
    plot_data = plot_data.loc[scenario_order, tech_totals.index]

    # Scale figure size based on number of scenarios
    n_scenarios = len(available_scenarios)
    base_width = 6
    width = max(base_width, n_scenarios * 0.4)  # Minimum 6, scale with scenarios

    # Create figure with 2 subplots (total system costs and German system costs)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width, 8), sharex=True)

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
    # Scale annotation font size based on number of scenarios
    annotation_fontsize = max(
        6, 10 - n_scenarios * 0.3
    )  # Minimum 6, scale down with more scenarios

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
                fontsize=annotation_fontsize,
                rotation=90,  # Rotate annotations vertically
            )

        # Set y-axis limit to accommodate annotations
        max_value = data.sum(axis=1).max()
        ax.set_ylim(0, max_value * 1.4)  # Increased padding for vertical annotations

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
            fontsize=annotation_fontsize,
            rotation=90,  # Rotate annotations vertically
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
                    fontsize=annotation_fontsize,
                    rotation=90,  # Rotate annotations vertically
                )

    # Create legend for regional breakdown (above the plots)
    simplified_handles = []
    simplified_labels = []
    for region in plot_data_simplified.columns:
        simplified_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=simplified_colors[region])
        )
        simplified_labels.append(region)

    # Add legend for regional breakdown below the plots
    fig.legend(
        simplified_handles,
        simplified_labels,
        title="Region",
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        ncol=len(simplified_labels),
        frameon=False,
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

    # Add unified legend for technologies below the plots
    # Calculate number of columns based on number of technologies and available width
    n_tech_cols = min(
        4, max(2, len(legend_labels) // 7)
    )  # 2-4 columns depending on tech count
    fig.legend(
        legend_handles,
        legend_labels,
        title="Technology",
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=n_tech_cols,
        frameon=False,
        fontsize=10,
    )

    # Adjust layout to accommodate legends below
    plt.subplots_adjust(top=0.95, bottom=0.25)
    plt.savefig(
        os.path.join(output_path, f"system_costs_comparison_{year}.pdf"),
        bbox_inches="tight",
    )

    logger.info(f"System costs comparison plot saved for year {year}")
    return fig, (ax1, ax2)


def plot_storage_psd(networks_dict, year, output_path):
    """
    Plot power spectrum density (PSD) analysis for different storage technologies across scenarios.

    Parameters:
    -----------
    networks_dict : dict
        Dictionary mapping scenario names to year-network dictionaries
    year : int
        Year to analyze
    output_path : str
        Path to save the output figure
    """
    logger.info(f"Generating storage power spectrum analysis for year {year}")

    # Define total hours per year
    total_hours_per_year = 8760

    # Define frequency bands
    bands = {
        "Intersemestral": (1, 2),
        "Intrasemestral": (2, 17),
        "Synoptical": (17, 51),
        "Intraweekly": (51, 364),
        "Daily": (364, 367),
        "Intradaily": (367, np.inf),
    }

    # Storage configurations
    storage_colors = {
        "PTES": "Blues_r",
        "TTES": "Blues_r",
        "H2 Storage": "Blues_r",
        "Battery": "Blues_r",
    }

    # Get networks for the specified year
    year_networks = {}
    for scenario, networks_scenario in networks_dict.items():
        if year in networks_scenario:
            year_networks[scenario] = networks_scenario[year]

    if not year_networks:
        logger.warning(f"No networks found for year {year}")
        return

    # Compute intersemestral share for sorting scenarios
    intersemestral_shares = {}
    for scenario, n in year_networks.items():
        try:
            storage_ts = n.stores_t.p[
                n.stores.filter(regex="DE.*urban central water pits", axis=0).index
            ]
            if storage_ts.empty:
                intersemestral_shares[scenario] = 0
                continue

            storage_ts = storage_ts.resample("H").ffill().mean(axis=1).values

            # Compute FFT and Power Spectrum
            n_samples = len(storage_ts)
            fft_result = np.fft.fft(storage_ts)
            freqs = np.fft.fftfreq(n_samples, d=1 / total_hours_per_year)
            power_spectrum = np.abs(fft_result) ** 2

            # Filter positive frequencies and normalize power spectrum
            positive_indices = freqs > 0
            freqs = freqs[positive_indices]
            power_spectrum = power_spectrum[positive_indices]
            power_spectrum /= power_spectrum.sum()

            # Calculate intersemestral share
            low, high = bands["Intersemestral"]
            band_indices = (freqs >= low) & (freqs < high)
            intersemestral_shares[scenario] = power_spectrum[band_indices].sum()
        except Exception as e:
            logger.warning(f"Error computing intersemestral share for {scenario}: {e}")
            intersemestral_shares[scenario] = 0

    # Sort scenarios by intersemestral share
    sorted_scenarios = sorted(
        intersemestral_shares.keys(),
        key=lambda x: intersemestral_shares[x],
        reverse=True,
    )

    if not sorted_scenarios:
        logger.warning(f"No valid scenarios found for storage PSD analysis")
        return

    # Initialize the plot with increased spacing between subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Increase padding between subplots
    axes = axes.flatten()

    # Loop over storage technologies and create subplots
    for ax, (storage_name, cmap_name) in zip(axes, storage_colors.items()):
        scenario_data = []  # Store normalized band power data for all scenarios
        capacity_data = []  # Store capacity data for secondary y-axis

        for scenario in sorted_scenarios:
            n = year_networks[scenario]

            try:
                # Select storage time series and capacity for the current scenario
                if storage_name == "PTES":
                    storage_ts = n.stores_t.p[
                        n.stores.filter(regex="DE.*urban central water pits", axis=0)
                        .query("e_nom_opt > 0")
                        .index
                    ]
                    # Get total capacity for this storage type
                    storage_capacity = (
                        n.stores.filter(
                            regex="DE.*urban central water pits", axis=0
                        ).e_nom_opt.sum()
                        / 1e6
                    )  # Convert to TWh
                elif storage_name == "TTES":
                    storage_ts = n.stores_t.p[
                        n.stores.filter(regex="DE.*urban central water tanks", axis=0)
                        .query("e_nom_opt > 0")
                        .index
                    ]
                    storage_capacity = (
                        n.stores.filter(
                            regex="DE.*urban central water tanks", axis=0
                        ).e_nom_opt.sum()
                        / 1e6
                    )  # Convert to TWh
                elif storage_name == "H2 Storage":
                    storage_ts = n.stores_t.p[
                        n.stores.filter(regex="DE.*H2 Store", axis=0)
                        .query("e_nom_opt > 0")
                        .index
                    ]
                    storage_capacity = (
                        n.stores.filter(regex="DE.*H2 Store", axis=0).e_nom_opt.sum()
                        / 1e6
                    )  # Convert to TWh
                elif storage_name == "Battery":
                    storage_ts = n.stores_t.p[
                        n.stores.filter(regex="DE.*battery", axis=0)
                        .query("e_nom_opt > 0")
                        .index
                    ]
                    storage_capacity = (
                        n.stores.filter(regex="DE.*battery", axis=0).e_nom_opt.sum()
                        / 1e6
                    )  # Convert to TWh
                else:
                    continue

                capacity_data.append(storage_capacity)

                if storage_ts.empty:
                    # Add zeros for scenarios without this storage type
                    scenario_data.append([0] * len(bands))
                    continue

                storage_ts = storage_ts.resample("H").ffill().sum(axis=1).values

                # Compute FFT and Power Spectrum
                n_samples = len(storage_ts)
                fft_result = np.fft.fft(storage_ts)
                freqs = np.fft.fftfreq(
                    n_samples, d=1 / total_hours_per_year
                )  # Convert to cycles per year
                power_spectrum = np.abs(fft_result) ** 2

                # Filter positive frequencies and normalize power spectrum
                positive_indices = freqs > 0
                freqs = freqs[positive_indices]
                power_spectrum = power_spectrum[positive_indices]
                power_spectrum /= power_spectrum.sum()

                # Categorize power into bands and normalize to a sum of 1
                band_values = []
                for band_name, (low, high) in bands.items():
                    band_indices = (freqs >= low) & (freqs < high)
                    band_values.append(power_spectrum[band_indices].sum())
                scenario_data.append(band_values)

            except Exception as e:
                logger.warning(
                    f"Error processing {storage_name} for scenario {scenario}: {e}"
                )
                # Add zeros for failed scenarios
                scenario_data.append([0] * len(bands))
                capacity_data.append(0)

        # Stack the band values for each scenario and plot
        scenario_data = np.array(scenario_data)
        bottom_stack = np.zeros(len(sorted_scenarios))
        cmap = cm.get_cmap(cmap_name, len(bands))
        colors = [mcolors.rgb2hex(cmap(i)) for i in range(len(bands))]

        for i, (band_name, color) in enumerate(zip(bands.keys(), colors)):
            ax.bar(
                sorted_scenarios,
                scenario_data[:, i],
                bottom=bottom_stack,
                color=color,
                width=1.0,
                label=band_name,
            )
            bottom_stack += scenario_data[:, i]

        # Create secondary y-axis for capacity
        ax2 = ax.twinx()

        # Plot capacity as line with markers
        x_positions = range(len(sorted_scenarios))
        if capacity_data and max(capacity_data) > 0:
            ax2.plot(
                x_positions,
                capacity_data,
                "ro-",
                linewidth=2,
                markersize=6,
                color="red",
                alpha=0.8,
                # label=f"{storage_name} Capacity",
            )
            ax2.set_ylabel(f"{storage_name} Capacity [TWh]", fontsize=10, color="red")
            ax2.tick_params(axis="y", labelcolor="red", labelsize=9)

            # Add capacity annotations
            for i, capacity in enumerate(capacity_data):
                if capacity > 0:  # Only annotate non-zero capacities
                    ax2.annotate(
                        f"{capacity:.1f}",
                        xy=(i, capacity),
                        xytext=(0, 10),  # 10 points above the marker
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        color="red",
                        fontweight="bold",
                    )

            # Set y-limits for capacity axis to give some breathing room
            max_cap = max(capacity_data)
            if max_cap > 0:
                ax2.set_ylim(0, max_cap * 1.2)

        # Format the subplot
        ax.set_title(f"{storage_name}", fontsize=12)
        ax.set_ylabel("Normalized\nPower Spectrum", fontsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)

    # X-axis labels (shared across subplots)
    for ax in axes:
        ax.set_xticks(range(len(sorted_scenarios)))
        ax.set_xticklabels(sorted_scenarios, rotation=45, ha="right")
    axes[-1].set_xlabel("Scenario", fontsize=12)

    # Create a single legend for the entire figure
    # Use the last subplot's data to create legend handles and labels
    legend_handles = []
    legend_labels = list(bands.keys())
    cmap = cm.get_cmap("Blues_r", len(bands))
    colors_legend = [mcolors.rgb2hex(cmap(i)) for i in range(len(bands))]

    for i, (band_name, color) in enumerate(zip(legend_labels, colors_legend)):
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))

    # Add the legend to the figure - positioned below plots centrally with 3 columns
    fig.legend(
        legend_handles,
        legend_labels,
        title="Frequency Band",
        bbox_to_anchor=(0.5, -0.03),
        loc="upper center",
        fontsize=12,
        title_fontsize=12,
        frameon=False,
        ncol=3,
    )

    # Adjust layout to accommodate the legend and increased padding
    fig.tight_layout()

    # Save and show the plot
    output_file = os.path.join(output_path, f"storage_power_spectrum_{year}.pdf")
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    logger.info(f"Storage power spectrum analysis saved to {output_file}")


def compute_psd(storage_ts, total_hours_per_year):
    """Compute power spectrum density for storage time series."""
    # Compute FFT and Power Spectrum
    n = len(storage_ts)
    fft_result = np.fft.fft(storage_ts)
    freqs = np.fft.fftfreq(n, d=1 / total_hours_per_year)  # Convert to cycles per year
    power_spectrum = np.abs(fft_result) ** 2

    # Filter positive frequencies and normalize power spectrum
    positive_indices = freqs > 0
    freqs = freqs[positive_indices]
    power_spectrum = power_spectrum[positive_indices]
    power_spectrum /= power_spectrum.sum()

    return pd.Series(power_spectrum, index=freqs)


def plot_storage_psd_stacked(networks_dict, year, output_path):
    """
    Plot normalized PSD stacked by frequency for PTES and TTES side by side with capacity info.

    Parameters:
    -----------
    networks_dict : dict
        Dictionary mapping scenario names to year-network dictionaries
    year : int
        Year to analyze
    output_path : str
        Path to save the output figure
    """
    logger.info(f"Generating stacked storage PSD analysis for year {year}")

    # Define total hours per year
    total_hours_per_year = 8760

    # Get networks for the specified year
    year_networks = {}
    for scenario, networks_scenario in networks_dict.items():
        if year in networks_scenario:
            year_networks[scenario] = networks_scenario[year]

    if not year_networks:
        logger.warning(f"No networks found for year {year}")
        return

    # Create figure with space for horizontal colorbar positioned lower
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        4, 2, height_ratios=[1, 0.1, 0.05, 0.1], hspace=0.4, wspace=0.4
    )
    ax1 = fig.add_subplot(gs[0, 0])  # Top left
    ax2 = fig.add_subplot(gs[0, 1])  # Top right
    cbar_ax = fig.add_subplot(gs[2, :])  # Horizontal colorbar positioned lower

    # First pass: collect all data to determine common scenario order
    all_ptes_data = pd.DataFrame()
    all_ttes_data = pd.DataFrame()
    all_caps_data = {}

    storage_configs = [
        ("PTES", "DE0.*water pits", all_ptes_data),
        ("TTES", "DE0.*water tanks", all_ttes_data),
    ]

    # Collect data for both storage types
    for storage_name, regex_pattern, df_storage in storage_configs:
        caps_data = {}

        for scenario, n in year_networks.items():
            try:
                # Get storage components
                storage_components = n.stores.filter(regex=regex_pattern, axis=0).query(
                    "e_nom_opt > 0"
                )

                if storage_components.empty:
                    continue

                # Get storage time series
                storage_ts = n.stores_t.p[storage_components.index]
                storage_ts = storage_ts.resample("H").ffill().sum(axis=1)

                # Compute PSD
                storage_psd = compute_psd(
                    storage_ts, total_hours_per_year=total_hours_per_year
                )

                # Calculate the capacity for this scenario (convert to TWh)
                caps = storage_components.e_nom_opt.sum() / 1e6
                caps_data[scenario] = caps

                # Round frequencies to integers for grouping and add scenario name
                storage_psd.index = storage_psd.index.round().astype(int)
                storage_psd.name = scenario

                # Append to DataFrame
                if storage_name == "PTES":
                    all_ptes_data = pd.concat([all_ptes_data, storage_psd], axis=1)
                else:
                    all_ttes_data = pd.concat([all_ttes_data, storage_psd], axis=1)

            except Exception as e:
                logger.warning(
                    f"Error processing {storage_name} for scenario {scenario}: {e}"
                )
                continue

        all_caps_data[storage_name] = caps_data

    # Determine common scenario order based on PTES intersemestral share (frequency = 1)
    # or TTES if PTES is not available, or capacity as final fallback
    common_scenario_order = None

    if not all_ptes_data.empty and 1 in all_ptes_data.index:
        common_scenario_order = all_ptes_data.loc[1].sort_values(ascending=False).index
    elif not all_ttes_data.empty and 1 in all_ttes_data.index:
        common_scenario_order = all_ttes_data.loc[1].sort_values(ascending=False).index
    elif all_caps_data.get("PTES"):
        common_scenario_order = (
            pd.Series(all_caps_data["PTES"]).sort_values(ascending=False).index
        )
    elif all_caps_data.get("TTES"):
        common_scenario_order = (
            pd.Series(all_caps_data["TTES"]).sort_values(ascending=False).index
        )
    else:
        common_scenario_order = list(year_networks.keys())

    # Process PTES and TTES with common ordering
    storage_types = [
        ("PTES", "DE0.*water pits", "PTES (Water Pits)", ax1, all_ptes_data),
        ("TTES", "DE0.*water tanks", "TTES (Water Tanks)", ax2, all_ttes_data),
    ]

    # Determine common frequency range for colorbar
    all_freqs = set()
    if not all_ptes_data.empty:
        all_freqs.update(all_ptes_data.index)
    if not all_ttes_data.empty:
        all_freqs.update(all_ttes_data.index)

    if all_freqs:
        norm = mcolors.LogNorm(
            vmin=max(1, min(all_freqs)),
            vmax=min(max(all_freqs), 365 * n.snapshot_weightings.generators.min()),
        )
    else:
        norm = mcolors.LogNorm(vmin=1, vmax=365)

    cmap = cm.Blues_r

    for storage_name, regex_pattern, title, ax, df in storage_types:
        if df.empty:
            logger.warning(f"No valid data found for {storage_name}")
            ax.text(
                0.5,
                0.5,
                f"No {storage_name} data available",
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(title, fontsize=14)
            continue

        # Apply common scenario order (only include scenarios that exist in this dataset)
        available_scenarios = [s for s in common_scenario_order if s in df.columns]
        if available_scenarios:
            df = df[available_scenarios]

        caps_data = all_caps_data.get(storage_name, {})
        cap_data = [caps_data.get(i, 0) for i in df.columns]

        # Normalize each column so that the total bar height is 1
        df = df.div(df.sum(axis=0), axis=1)

        # Iterate through frequencies and plot each frequency as a stacked contribution
        x_positions = np.arange(len(df.columns))
        width = 1  # Bar width
        bottom_stack = np.zeros(len(df.columns))

        for freq in df.index:
            heights = df.loc[freq]
            ax.bar(
                x_positions,
                heights,
                width=width,
                color=cmap(norm(freq)),
                bottom=bottom_stack,
                edgecolor="none",
            )
            bottom_stack += heights  # Update the bottom stack for the next segment

        # Add a secondary y-axis for the capacity
        if cap_data and max(cap_data) > 0:
            ax_twin = ax.twinx()
            ax_twin.scatter(
                x_positions,
                cap_data,
                color="red",
                marker="o",
                s=100,
                zorder=10,
            )
            ax_twin.set_ylabel(
                f"{storage_name} capacity [TWh]", fontsize=12, color="red"
            )
            ax_twin.tick_params(axis="y", labelcolor="red")
            ax_twin.set_ylim(0, max(cap_data) * 1.1)  # Add some padding for clarity

        # Format the plot
        ax.set_xticks(x_positions)
        ax.set_xticklabels(df.columns, rotation=45, ha="right")
        ax.set_title(f"{title}", fontsize=14)
        ax.set_ylabel("Normalized Density", fontsize=12)
        ax.set_xlabel("Scenario", fontsize=12)
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_ylim(0, 1)
        ax.set_xlim(-0.5, len(df.columns) - 0.5)

    # Add horizontal colorbar between the plots
    if all_freqs:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Required for colorbar
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", pad=0.1)
        cbar.set_label("Frequency (cycles per year)", fontsize=12)

        # Set custom ticks and labels for frequencies
        freq_labels = [1, 17, 52, 365]
        available_freqs = [
            f for f in freq_labels if f <= max(all_freqs) and f >= min(all_freqs)
        ]
        if available_freqs:
            cbar.set_ticks(available_freqs)
            cbar.set_ticklabels(
                [
                    (
                        f"Yearly\n(1)"
                        if f == 1
                        else (
                            f"Synoptical\n(17)"
                            if f == 17
                            else (
                                f"Weekly\n(52)"
                                if f == 52
                                else f"Daily\n(365)" if f == 365 else str(f)
                            )
                        )
                    )
                    for f in available_freqs
                ]
            )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(
        output_path, f"storage_psd_stacked_comparison_{year}.png"
    )
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)

    logger.info(f"Stacked storage PSD comparison saved to {output_file}")


def resample_to_snapshots(
    n: pypsa.Network, series: pd.Series, func: str = "mean"
) -> pd.Series:
    """
    Resample a series to match the network's snapshots.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network object containing the snapshots.
    series : pd.Series
        The series to be resampled.
    func : str
        The mode of resampling, e.g., 'mean', 'sum', etc.

    Returns
    -------
    pd.Series
        The resampled series with the same index as the network's snapshots.
    """
    sns = n.snapshots
    sw = n.snapshot_weightings.generators

    # Append last snapshot of year for bin assignment using concat instead of deprecated append
    last_snapshot = sns[-1] + pd.Timedelta(hours=sw[sns[-1]])
    sns_extended = pd.Index(sns.tolist() + [last_snapshot])

    # Create bins: each interval is between snapshot_weightings.index[i] and [i+1]
    bins = pd.IntervalIndex.from_breaks(sns_extended, closed="left")

    # Assign each p_max_source timestamp to a bin
    bin_labels = pd.cut(series.index, bins)

    # Group by bin and apply the specified mode using agg with dictionary
    binned_values = series.groupby(bin_labels, observed=True).agg(func)

    # Reindex to match the network's snapshots
    binned_values = binned_values.reindex(sns, fill_value=0)

    return binned_values


def get_ptes_discharge(network):
    """Get PTES discharge energy weighted by snapshot weightings."""
    return network.snapshot_weightings.generators.mul(
        network.stores_t.p.clip(lower=0).filter(regex="DE.*water pit").T
    ).T


def get_ptes_charge(network):
    """Get PTES charge energy weighted by snapshot weightings."""
    return network.snapshot_weightings.generators.mul(
        network.stores_t.p.clip(upper=0).filter(regex="DE.*water pit").T
    ).T


def plot_ptes_energy_quantiles(
    network,
    boosting_ratio_data,
    output_path,
    scenario_name="scenario",
    figsize=(6, 6),
    xlim=None,
):
    """
    Plot charged and discharged energy at different electricity price quantiles
    including boosting energy visualization.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object
    boosting_ratio_data : xr.DataArray or pd.DataFrame
        The boosting ratio data for PTES dischargers
    output_path : str
        Path to save the output plot
    scenario_name : str
        Name of the scenario for plot title
    figsize : tuple
        Figure size for the plot
    xlim : tuple, optional
        Fixed x-axis limits as (left_lim, right_lim) for uniform comparison across plots
    """
    logger.info(f"Generating PTES energy quantiles plot for {scenario_name}")

    # Get PTES operation data
    ptes_charge = get_ptes_charge(network).rename(
        columns=lambda x: x.replace(" urban central water pits-2045", "")
    )
    ptes_discharge = get_ptes_discharge(network).rename(
        columns=lambda x: x.replace(" urban central water pits-2045", "")
    )

    if ptes_charge.empty or ptes_discharge.empty:
        logger.warning(f"No PTES data found for {scenario_name}")
        return

    # Handle boosting ratio data - convert to pandas if needed
    if hasattr(boosting_ratio_data, "to_pandas"):
        discharger_boosting_ratio = boosting_ratio_data.to_pandas().filter(like="DE")
    else:
        discharger_boosting_ratio = boosting_ratio_data.filter(like="DE")

    # Match columns between discharge data and boosting ratio data
    common_columns = ptes_discharge.columns.intersection(
        discharger_boosting_ratio.columns
    )
    if common_columns.empty:
        logger.warning(
            f"No matching columns between discharge data and boosting ratio data for {scenario_name}"
        )
        return

    # Filter to common columns
    ptes_discharge_filtered = ptes_discharge[common_columns]
    discharger_boosting_ratio_filtered = discharger_boosting_ratio[common_columns]

    # Resample boosting ratios and calculate alpha (inverse)
    try:
        # Handle the case where boosting ratio data might be a DataFrame with multiple columns
        if isinstance(discharger_boosting_ratio_filtered, pd.DataFrame):
            alpha = pd.DataFrame(index=network.snapshots, columns=common_columns)
            for col in common_columns:
                if col in discharger_boosting_ratio_filtered.columns:
                    resampled_series = resample_to_snapshots(
                        network, discharger_boosting_ratio_filtered[col]
                    )
                    alpha[col] = resampled_series.pow(-1).replace(np.inf, 0)
        else:
            # Single series case
            alpha = (
                resample_to_snapshots(network, discharger_boosting_ratio_filtered)
                .pow(-1)
                .replace(np.inf, 0)
            )
            alpha = pd.DataFrame(alpha).reindex(columns=common_columns, fill_value=0)
    except Exception as e:
        logger.warning(f"Error processing boosting ratio data for {scenario_name}: {e}")
        # Create a default alpha matrix with all ones (no boosting)
        alpha = pd.DataFrame(1.0, index=network.snapshots, columns=common_columns)

    # Create electricity prices dataframe
    electricity_prices = pd.DataFrame(index=network.snapshots, columns=common_columns)
    for col in common_columns:
        bus_name = " ".join(col.split()[:2])
        if bus_name in network.buses_t.marginal_price.columns:
            electricity_prices[col] = network.buses_t.marginal_price[bus_name]

    # Filter charge data to common columns as well
    ptes_charge_filtered = (
        ptes_charge[common_columns] if len(common_columns) > 0 else ptes_charge
    )

    # Create combined dataframes
    charge_combined = pd.concat(
        [ptes_charge_filtered.unstack(0), electricity_prices.unstack(0)], axis=1
    )
    charge_combined.columns = ["charge", "electricity_price"]
    charge_combined = charge_combined.dropna()

    discharge_combined = pd.concat(
        [
            ptes_discharge_filtered.unstack(0),
            electricity_prices.unstack(0),
            alpha.unstack(-1),
        ],
        axis=1,
    )
    discharge_combined.columns = ["discharge", "electricity_price", "alpha"]
    discharge_combined = discharge_combined.dropna()

    # --- reshape to long format ---
    def to_long_signed_charge(df):
        out = pd.DataFrame(
            {
                "system": df.index.get_level_values(0),
                "ts": df.index.get_level_values(1),
                "price": pd.to_numeric(df["electricity_price"], errors="coerce"),
                "e": pd.to_numeric(df["charge"], errors="coerce"),
            }
        ).dropna(subset=["price", "e"])
        out["e"] = out["e"].clip(upper=0)  # keep charge negative
        return out

    def to_long_signed_discharge(df):
        out = pd.DataFrame(
            {
                "system": df.index.get_level_values(0),
                "ts": df.index.get_level_values(1),
                "price": pd.to_numeric(df["electricity_price"], errors="coerce"),
                "e": pd.to_numeric(df["discharge"], errors="coerce"),
                "alpha": pd.to_numeric(df["alpha"], errors="coerce"),
            }
        ).dropna(subset=["price", "e", "alpha"])
        out["e"] = out["e"].clip(lower=0)  # keep discharge positive
        return out

    charge_long = to_long_signed_charge(charge_combined)
    discharge_long = to_long_signed_discharge(discharge_combined)

    if charge_long.empty or discharge_long.empty:
        logger.warning(f"No valid charge/discharge data for {scenario_name}")
        return

    # --- 20 shared price-quantiles ---
    n_q = 20
    all_prices = pd.concat([charge_long["price"], discharge_long["price"]]).to_numpy()
    edges = np.unique(
        np.quantile(all_prices, np.linspace(0, 1, n_q + 1))
    )  # handle flats
    n_bins = len(edges) - 1

    def add_q(df, edges):
        q = pd.cut(
            df["price"], bins=edges, labels=False, include_lowest=True, right=True
        )
        return df.assign(q=(q + 1).astype("Int64")).dropna(subset=["q"])

    charge_q = add_q(charge_long, edges)
    discharge_q = add_q(discharge_long, edges)

    # --- aggregate per quantile ---
    qs = pd.RangeIndex(1, n_bins + 1)

    # charge totals (negative)
    charge_tot = (
        charge_q.groupby("q", observed=True)["e"].sum().reindex(qs, fill_value=0.0)
    )

    # discharge totals and boosting energy
    g = discharge_q.groupby("q", observed=True)
    dis_tot = g["e"].sum().reindex(qs, fill_value=0.0)

    # Calculate boosting energy: sum of (alpha * discharge) for each quantile
    boosting_energy = g.apply(
        lambda d: (d["alpha"] * d["e"]).sum(), include_groups=False
    ).reindex(qs, fill_value=0.0)

    # --- labels: rounded € price ranges ---
    er = np.round(edges).astype(int)
    y_labels = [f"€{er[i]}–€{er[i+1]}" for i in range(n_bins)]

    # --- convert to TWh ---
    charge_twh = charge_tot / 1e6
    dis_twh = dis_tot / 1e6
    boosting_twh = boosting_energy / 1e6

    # Calculate total energy amounts for legend labels
    total_charge = abs(charge_twh.sum())  # absolute value since charge is negative
    total_discharge = dis_twh.sum()
    total_boosting = boosting_twh.sum()

    # --- create the plot ---
    fig, ax = plt.subplots(figsize=figsize)

    # charge bars (negative side) - blue
    bars_charge = ax.barh(
        qs, charge_twh.values, label=f"Charge ({total_charge:.1f} TWh)", color="blue"
    )

    # discharge bars (positive side) - red
    bars_discharge = ax.barh(
        qs, dis_twh.values, label=f"Discharge ({total_discharge:.1f} TWh)", color="red"
    )

    # boosting energy stacked on top of discharge - teal
    bars_boosting = ax.barh(
        qs,
        boosting_twh.values,
        left=dis_twh.values,
        label=f"Boosting energy ({total_boosting:.1f} TWh)",
        color="teal",
    )

    # zero line
    ax.axvline(0, color="black", linewidth=1.2)

    # axes & ticks with consistent font sizing
    ax.set_xlabel("Aggregate energy [TWh]", fontsize=12)
    ax.set_ylabel("Electricity price range per quantile", fontsize=12)
    ax.set_yticks(qs)
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    # x-limits with more space to ensure 0.5 TWh tick is included
    if xlim is not None:
        # Use provided global limits for uniform comparison
        left_lim, right_lim = xlim
    else:
        # Use hardcoded default limits if no xlim parameter is provided
        left_lim, right_lim = -20, 10

    ax.grid(True, axis="x", linestyle=":", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(f"PTES Energy by Price Quantile - {scenario_name}", fontsize=14)

    plt.tight_layout()

    # Apply xlim after tight_layout to prevent it from being overridden
    ax.set_xlim(left_lim, right_lim)

    # Save the plot
    output_file = os.path.join(
        output_path, f"ptes_energy_quantiles_{scenario_name}.pdf"
    )
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

    logger.info(f"PTES energy quantiles plot saved to {output_file}")


def prepare_energy_data(network, exclude_water_pits=True):
    """Prepare energy balance data for district heating analysis."""
    eb_t = network.statistics.energy_balance(
        groupby=pypsa.statistics.groupers["bus", "carrier", "bus_carrier"],
        nice_names=False,
        aggregate_time=False,
    )
    uch_de_t = eb_t.xs("urban central heat", level="bus_carrier").filter(
        regex="DE", axis=0
    )

    uch_de_t = uch_de_t.droplevel(["component", "bus"]).groupby("carrier").sum()

    return uch_de_t


def process_generation_and_load(uch_de_t, network):
    """Process generation and load data into electricity price quartiles (previously ventiles)."""
    uch_de_t_gen = (
        uch_de_t.clip(lower=0)
        .T.resample("3h")
        .mean()
        .T.mul(network.snapshot_weightings.generators)
    )
    uch_de_t_gen = uch_de_t_gen.loc[uch_de_t_gen[uch_de_t_gen.sum(axis=1) > 0].index]

    uch_de_t_gen_prices = (
        calc_average_electricity_price_t_ordered(network).resample("3h").mean()
    )

    uch_de_t_load = (
        uch_de_t.clip(upper=0)
        .T.resample("3h")
        .mean()
        .T.mul(network.snapshot_weightings.generators)
    )
    uch_de_t_load = uch_de_t_load.loc[
        uch_de_t_load[uch_de_t_load.sum(axis=1) < 0].index
    ]

    percentiles = [
        0,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1,
    ]
    price_bins, bin_edges = pd.qcut(
        uch_de_t_gen_prices, q=percentiles, labels=None, retbins=True
    )

    def format_bin_edge(value):
        if value < 1:
            return f"{value:.4f}"
        if value < 10:
            return f"{value:.2f}"
        elif value < 100:
            return f"{value:.1f}"
        elif value < 1000:
            return f"{value:.0f} "
        else:
            return f"{value:.0f}"

    # Create labels based on actual bin edges (which may be fewer than percentiles due to duplicates='drop')
    bin_labels_with_edges = [
        f"< {format_bin_edge(bin_edges[i+1])}" for i in range(len(bin_edges) - 1)
    ]

    uch_de_t_gen_prices = pd.qcut(
        uch_de_t_gen_prices,
        q=percentiles,
        labels=bin_labels_with_edges,
    )

    uch_de_t_gen.rename(columns=uch_de_t_gen_prices, inplace=True)
    uch_de_t_gen = uch_de_t_gen.T.groupby(uch_de_t_gen.columns).sum().T
    uch_de_t_gen = uch_de_t_gen[bin_labels_with_edges]

    uch_de_t_load.rename(columns=uch_de_t_gen_prices, inplace=True)
    uch_de_t_load = uch_de_t_load.T.groupby(uch_de_t_load.columns).sum().T
    uch_de_t_load = uch_de_t_load[bin_labels_with_edges]

    return uch_de_t_gen, uch_de_t_load, bin_labels_with_edges


def get_boosting_energy(
    n: pypsa.Network,
    boosting_technology: str = "resistive heater",
    boosting_ratio_fn: str = None,
):
    """Calculate boosting energy for PTES discharge per snapshot (Germany).

    Parameters
    ----------
    n : pypsa.Network
        Network with snapshots and time series.
    boosting_technology : str
        Either "resistive heater" or "urban central ptes heat pump".
    boosting_ratio_fn : str | None
        Path to an xarray DataArray/Dataset containing boosting ratios per node and time
        (required for "resistive heater").

    Returns
    -------
    pd.Series
        Boosting energy per snapshot (same units as the weighted discharge/power integration in the network,
        typically MWh per snapshot when multiplied by snapshot weights).
    """
    # Branch 1: Resistive heater uses a boosting ratio times PTES discharge per node
    if boosting_technology.lower() == "resistive heater":
        if boosting_ratio_fn is None:
            logger.error("boosting_ratio_fn is required for resistive heater boosting")
            return pd.Series(0.0, index=n.snapshots)

        # Lazy import to avoid global dependency if unused
        try:
            import xarray as xr
        except Exception as e:
            logger.error(f"xarray is required to read boosting ratio file: {e}")
            return pd.Series(0.0, index=n.snapshots)

        # Load boosting ratio from file as a DataArray
        try:
            try:
                br_da = xr.open_dataarray(boosting_ratio_fn)
            except Exception:
                ds = xr.open_dataset(boosting_ratio_fn)
                # Pick the first data variable if dataset
                if len(ds.data_vars) == 0:
                    raise ValueError("Dataset has no data variables")
                first_var = list(ds.data_vars)[0]
                br_da = ds[first_var]
        except Exception as e:
            logger.error(f"Failed to load boosting ratio from {boosting_ratio_fn}: {e}")
            return pd.Series(0.0, index=n.snapshots)

        # Identify a time-like dimension (fallback to first dim)
        time_dim = None
        for d in br_da.dims:
            if "time" in d.lower() or "snapshot" in d.lower() or "date" in d.lower():
                time_dim = d
                break
        if time_dim is None:
            # Fallback to the first dim
            time_dim = br_da.dims[0]

        # Choose a node dimension if present
        node_dim = None
        for d in br_da.dims:
            if d != time_dim:
                node_dim = d
                break

        # Convert to a pandas DataFrame: index=time, columns=node
        try:
            s = br_da.to_series().dropna()
            # Ensure the time dimension is the index level
            if time_dim in s.index.names and node_dim in s.index.names:
                br_df = s.unstack(node_dim)
            elif time_dim in s.index.names:
                # Single series over time
                br_df = s.rename("ratio").to_frame()
            else:
                # Put the first level as time
                br_df = s.unstack(0)
            # Ensure datetime index if possible
            try:
                br_df.index = pd.to_datetime(br_df.index)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to convert boosting ratio to pandas: {e}")
            return pd.Series(0.0, index=n.snapshots)

        # PTES discharge (positive), weighted per snapshot, DE nodes only
        ptes_discharge = get_ptes_discharge(n)
        ptes_discharge.columns = ptes_discharge.columns.str.split(" urban").str[0]
        # Align boosting ratio columns to discharge columns
        common_cols = ptes_discharge.columns.intersection(br_df.columns)
        if common_cols.empty:
            logger.warning(
                "No overlapping nodes between boosting ratio and PTES discharge; returning zeros"
            )
            return pd.Series(0.0, index=n.snapshots)

        # Resample boosting ratio to network snapshots per column
        alpha_df = pd.DataFrame(index=n.snapshots, columns=common_cols, dtype=float)
        for col in common_cols:
            try:
                series = br_df[col].dropna()
                series = series[~series.index.duplicated(keep="last")]
                alpha_df[col] = resample_to_snapshots(n, series, func="mean")
            except Exception as e:
                logger.warning(f"Failed resampling boosting ratio for {col}: {e}")
                alpha_df[col] = 1.0  # fallback to no boosting

        # Multiply element-wise by PTES discharge and sum over nodes to get per-snapshot boosting energy
        boosting_energy_df = (
            ptes_discharge[common_cols]
            .div(alpha_df[common_cols])
            .replace(np.inf, 0.0)
            .fillna(0.0)
        )
        boosting_ts = boosting_energy_df.sum(axis=1)
        # Ensure index matches snapshots
        boosting_ts = boosting_ts.reindex(n.snapshots, fill_value=0.0)
        return boosting_ts

    # Branch 2: Heat pump booster — sum p0 over all relevant nodes (weighted)
    if boosting_technology.lower() in [
        "heat pump",
        "urban central ptes heat pump",
        "ptes heat pump",
    ]:
        try:
            hp_cols = n.links_t.p0.filter(
                regex=r"DE.*urban central ptes heat pump"
            ).columns
        except Exception:
            # If naming differs slightly, try a broader pattern
            hp_cols = n.links_t.p0.filter(regex=r"DE.*ptes.*heat pump").columns

        if len(hp_cols) == 0:
            logger.warning("No PTES heat pump (p0) time series found; returning zeros")
            return pd.Series(0.0, index=n.snapshots)

        hp_p0 = n.links_t.p0[hp_cols]
        # Weight to convert to per-snapshot energy consistent with other calculations
        weighted = n.snapshot_weightings.generators.mul(hp_p0.T).T
        boosting_ts = weighted.sum(axis=1)
        boosting_ts = boosting_ts.reindex(n.snapshots, fill_value=0.0)
        return boosting_ts

    logger.error(f"Unknown boosting technology: {boosting_technology}")
    return pd.Series(0.0, index=n.snapshots)


def plot_energy_balance_combined(
    uch_de_t_gen_dict,
    uch_de_t_load_dict,
    output_file,
    scenario_names,
    colors,
    networks=None,
    boosting_ratio_files=None,
    dh_supply_temperatures=None,
):
    """Plot energy balance comparison across scenarios and price quartiles.

    Notes
    -----
    The function now derives bin labels directly from the provided DataFrames' column
    ordering (already aggregated to quartiles upstream). The former argument
    `bin_labels_with_edges` (ventile-based) has been removed to avoid mismatches.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    handles, labels = None, None

    # Track if we already added a temperature colorbar; store mappable for later
    temp_colorbar_added = False
    sm_for_colorbar = None

    for ax, (scenario, uch_de_t_gen) in zip(axes, uch_de_t_gen_dict.items()):
        uch_de_t_load = uch_de_t_load_dict[scenario]

        # Optionally compute boosting energy per price bin for this scenario
        boosting_by_label_twh = None
        try:
            if networks is not None and scenario in networks:
                n_obj = networks[scenario]
                # networks can be a dict of years -> network, pick the first if needed
                if isinstance(n_obj, dict):
                    # Pick the first available network
                    n = next(iter(n_obj.values()))
                else:
                    n = n_obj

                # Decide boosting technology from scenario name
                scen_lower = str(scenario).lower()
                if "hpboost" in scen_lower:
                    boosting_tech = "heat pump"
                elif "rhboost" in scen_lower:
                    boosting_tech = "resistive heater"
                else:
                    boosting_tech = None

                boosting_ts = None
                if boosting_tech is not None:
                    # Resolve file for resistive heater if provided as dict or string
                    ratio_fn = None
                    if boosting_tech == "resistive heater":
                        if isinstance(boosting_ratio_files, dict):
                            ratio_fn = boosting_ratio_files.get(scenario)
                        elif isinstance(boosting_ratio_files, str):
                            ratio_fn = boosting_ratio_files
                    # Compute boosting time series per snapshot
                    boosting_ts = get_boosting_energy(
                        n,
                        boosting_technology=boosting_tech,
                        boosting_ratio_fn=ratio_fn,
                    )

                # Build price series and bin into same quantiles (3h resolution like input processing)
                prices_3h = (
                    calc_average_electricity_price_t_ordered(n).resample("3h").mean()
                )
                boosting_3h = (
                    boosting_ts.resample("3h").sum()
                    if boosting_ts is not None
                    else None
                )

                # Use same percentiles and labeling approach as process_generation_and_load
                percentiles = [0, 0.25, 0.5, 0.75, 1]
                # First qcut to get bin edges (labels=None)
                _price_bins_tmp, bin_edges = pd.qcut(
                    prices_3h, q=percentiles, labels=None, retbins=True
                )

                def _format_bin_edge(value):
                    if value < 1:
                        return f"{value:.4f}"
                    if value < 10:
                        return f"{value:.2f}"
                    elif value < 100:
                        return f"{value:.1f}"
                    elif value < 1000:
                        return f"{value:.0f} "
                    else:
                        return f"{value:.0f}"

                # Build labels from edges exactly as in process_generation_and_load
                _bin_labels_with_edges = [
                    f"< {_format_bin_edge(bin_edges[i+1])}"
                    for i in range(len(bin_edges) - 1)
                ]
                # Second qcut with labels assigned
                price_bins_labeled = pd.qcut(
                    prices_3h, q=percentiles, labels=_bin_labels_with_edges
                )
                # Aggregate boosting energy by those labeled bins
                boosting_by_label = (
                    boosting_3h.groupby(price_bins_labeled, observed=True).sum()
                    if boosting_3h is not None
                    else None
                )
                # Align to current bin index order (to_plot index after transpose)
                if boosting_by_label is not None:
                    boosting_by_label = boosting_by_label.reindex(
                        index=uch_de_t_gen.columns, fill_value=0.0
                    )
                    boosting_by_label_twh = boosting_by_label / 1e6
        except Exception as e:
            logger.warning(
                f"Failed to compute boosting energy for scenario {scenario}: {e}"
            )

        # Use absolute energy per quartile in TWh (positive supply, negative loads)
        to_plot_gen = (uch_de_t_gen / 1e6).T
        to_plot_load = (uch_de_t_load / 1e6).T  # keep negative values for loads
        to_plot = pd.concat([to_plot_gen, to_plot_load], axis=1)

        # (Directly creating quartile bins now; no post-hoc collapse needed.)
        try:
            original_bin_labels = list(uch_de_t_gen.columns)
            if len(original_bin_labels) == 20:
                # Build mapping index -> quartile label
                quartile_map = {}
                for i, lbl in enumerate(original_bin_labels):
                    q_index = i // 5  # 0..3
                    quartile_label = ["Q1", "Q2", "Q3", "Q4"][q_index]
                    quartile_map.setdefault(quartile_label, []).append(lbl)

                def _aggregate_quartiles(df: pd.DataFrame) -> pd.DataFrame:
                    out_rows = []
                    for qlbl in ["Q1", "Q2", "Q3", "Q4"]:
                        cols = quartile_map.get(qlbl, [])
                        if not cols:
                            # create zeros if missing
                            out_rows.append(pd.Series(0.0, index=df.index, name=qlbl))
                        else:
                            out_rows.append(df[cols].sum(axis=1).rename(qlbl))
                    return pd.DataFrame(out_rows).set_index(
                        pd.Index(["Q1", "Q2", "Q3", "Q4"], name=df.index.name)
                    )

                # Apply to positive and negative frames separately (already concatenated)
                # Since current shape is (20 x techs) after T transpose, we aggregate along rows (index)
                to_plot = _aggregate_quartiles(to_plot.T).T  # back to (4 x techs)

                # Aggregate boosting energy per quartile if available
                if boosting_by_label_twh is not None:
                    boosting_q = []
                    for qlbl in ["Q1", "Q2", "Q3", "Q4"]:
                        cols = quartile_map.get(qlbl, [])
                        boosting_q.append(
                            (boosting_by_label_twh.reindex(cols, fill_value=0.0)).sum()
                        )
                    boosting_by_label_twh = pd.Series(
                        boosting_q, index=["Q1", "Q2", "Q3", "Q4"]
                    )

                # Aggregate temperatures if they were already computed (later block will check shape)
                # We'll store mapping for later use (re-coloring happens later after temps_per_bin_aligned creation)
            else:
                logger.debug(
                    f"Expected 20 ventile bins before quartile aggregation, found {len(original_bin_labels)}; skipping quartile collapse."
                )
        except Exception as _quart_e:
            logger.debug(f"Quartile aggregation failed: {_quart_e}")

        # 1. Drop heat vents from the technologies
        heat_vent_cols = [col for col in to_plot.columns if "heat vent" in col.lower()]
        to_plot.drop(heat_vent_cols, axis=1, inplace=True, errors="ignore")

        # 2. Aggregate CHPs (H2, gas, and waste) as one technology
        chp_cols = [
            col
            for col in to_plot.columns
            if any(
                chp_type in col.lower()
                for chp_type in ["h2 chp", "gas chp", "waste chp"]
            )
        ]
        if chp_cols:
            to_plot["CHP"] = to_plot[chp_cols].sum(axis=1)
            to_plot.drop(chp_cols, axis=1, inplace=True)

        # 3. Aggregate heat pumps as one technology (excluding PTES heat pump)
        heat_pump_cols = [
            col
            for col in to_plot.columns
            if any(
                hp_type in col.lower()
                for hp_type in [
                    "air heat pump",
                    "geothermal heat pump",
                    "river_water heat pump",
                    "sea_water heat pump",
                    "electrolysis excess heat pump",
                ]
            )
            and "ptes" not in col.lower()
        ]
        if heat_pump_cols:
            to_plot["Heat pumps"] = to_plot[heat_pump_cols].sum(axis=1)
            to_plot.drop(heat_pump_cols, axis=1, inplace=True)

        # 4. Aggregate heat demand categories
        heat_demand_cols = []
        # Find all columns that contain demand-related keywords
        for col in to_plot.columns:
            col_lower = col.lower()
            if any(
                keyword in col_lower
                for keyword in [
                    "low-temperature heat for industry",
                    "urban central heat",
                ]
            ):
                heat_demand_cols.append(col)

        if heat_demand_cols:
            to_plot["Heat demand"] = to_plot[heat_demand_cols].sum(axis=1)
            to_plot.drop(heat_demand_cols, axis=1, inplace=True)

        # 5. No rescaling for absolute values (keep TWh units)

        # Group irrelevant columns into two buckets without mixing signs for area plots
        other_techs = to_plot.T.where(to_plot.abs().sum() < 5).dropna().index
        if len(other_techs) > 0:
            # Sum positive and negative parts separately to avoid cancellation
            pos_sum = to_plot[other_techs].clip(lower=0).sum(axis=1)
            neg_sum = to_plot[other_techs].clip(upper=0).sum(axis=1)
            # Split into positive-only supply and negative-only loads
            to_plot["other supply technologies"] = pos_sum
            to_plot["other loads"] = neg_sum
            # Drop the individual small columns
            to_plot.drop(other_techs, inplace=True, axis=1)
        # Ensure the two 'other' columns always exist (zero if empty)
        if "other supply technologies" not in to_plot.columns:
            to_plot["other supply technologies"] = 0.0
        if "other loads" not in to_plot.columns:
            to_plot["other loads"] = 0.0
        # Drop zero-only 'other' columns to keep legend clean
        for _col in ["other supply technologies", "other loads"]:
            if to_plot[_col].abs().sum() == 0:
                to_plot.drop(columns=[_col], inplace=True)

        to_plot = to_plot[to_plot.abs().sum().sort_values(ascending=False).index]

        # Map colors from the provided color scheme (with sensible defaults for new buckets)
        try:
            colors_local = colors.copy()
        except Exception:
            colors_local = dict(colors)
        colors_local.setdefault("other supply technologies", "#A9A9A9")  # darkgray
        colors_local.setdefault("other loads", "#696969")  # dimgray

        # Determine positive (supply) and negative (demand) columns
        pos_cols = [c for c in to_plot.columns if (to_plot[c] > 0).any()]
        neg_cols = [c for c in to_plot.columns if (to_plot[c] < 0).any()]

        # Identify storage columns (PTES/TTES)
        def is_storage(col: str) -> bool:
            cl = col.lower()
            return ("water pits" in cl) or ("water tanks" in cl)

        # For supply: PTES/TTES should be on top (drawn last)
        pos_storage = [c for c in pos_cols if is_storage(c)]
        pos_others = [c for c in pos_cols if c not in pos_storage]
        pos_order = pos_others + pos_storage

        # For demand: PTES/TTES should be at the very bottom (farthest below zero)
        # For negative stacks, the last drawn series ends up bottom-most -> draw storage last
        neg_storage = [c for c in neg_cols if is_storage(c)]
        neg_others = [c for c in neg_cols if c not in neg_storage]
        neg_order = neg_others + neg_storage

        # Build color lists aligned to the custom orders
        pos_colors = [colors_local.get(c, "black") for c in pos_order]
        neg_colors = [colors_local.get(c, "black") for c in neg_order]

        # Build separated DataFrames for plotting
        pos_df = to_plot[pos_order].clip(lower=0)
        neg_df = to_plot[neg_order].clip(upper=0)

        # Plot positive and negative stacks separately to control ordering
        if not pos_df.empty:
            pos_df.plot.area(
                ax=ax,
                stacked=True,
                color=pos_colors,
                linewidth=0.5,
                alpha=0.8,
            )
        if not neg_df.empty:
            neg_df.plot.area(
                ax=ax,
                stacked=True,
                color=neg_colors,
                linewidth=0.5,
                alpha=0.8,
            )
        # Remove extra horizontal padding added by area plot
        ax.margins(x=0)

        # Ensure all x-ticks/labels for price quantiles are shown consistently
        from matplotlib.ticker import FixedLocator

        bin_labels = list(to_plot.index)  # quartile labels inferred from qcut edges
        x_positions = np.arange(len(bin_labels))
        ax.xaxis.set_major_locator(FixedLocator(x_positions))
        ax.set_xticks(x_positions)
        ax.set_xticklabels(bin_labels, rotation=90, ha="right")

        # Light vertical grid at each price quartile to visualize distribution
        ax.set_axisbelow(True)
        ax.grid(
            True, axis="x", linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8
        )

        # Map scatter markers to numeric positions to align with fixed ticks
        pos_map = {lbl: i for i, lbl in enumerate(bin_labels)}

        ax.axhline(0, color="black", linewidth=0.5)
        ax2 = ax.twinx()
        # If boosting energy data is available, plot cumulative boosting energy; otherwise, fall back to previous markers
        if boosting_by_label_twh is not None:
            # Align boosting values to current bin label order and build cumulative sum
            boosting_per_bin = [
                boosting_by_label_twh.get(lbl, 0.0) for lbl in bin_labels
            ]
            boosting_vals = np.cumsum(boosting_per_bin)

            # --- Optional temperature-based coloring ---
            temp_colors = None
            cmap = plt.get_cmap("coolwarm")
            norm = None
            temps_per_bin_aligned = None
            try:
                # Auto-load temperature file if not provided for this scenario
                if (dh_supply_temperatures is None) or (
                    scenario not in (dh_supply_temperatures or {})
                ):
                    try:
                        # Attempt to access global snakemake (same pattern as seasonal plot)
                        run_name_auto = snakemake.params.run  # type: ignore  # noqa: F821
                        temp_path = (
                            f"resources/{run_name_auto}/{scenario}/"
                            "central_heating_forward_temperature_profiles_base_s_49_2045.nc"
                        )
                        if os.path.exists(temp_path):
                            ff_temp_auto = xr.open_dataarray(temp_path)
                            # Convert to pandas, filter DE0 buses, take min across columns (as in seasonal plot)
                            temp_series_auto = (
                                get_delta_ff_top(ff_temp_auto)
                                .to_pandas()
                                .filter(like="DE0")
                                .mean(1)
                            )
                            # Initialize dict if needed
                            if dh_supply_temperatures is None:
                                dh_supply_temperatures = {}
                            dh_supply_temperatures[scenario] = temp_series_auto
                        else:
                            logger.debug(
                                f"Temperature file not found for scenario {scenario}: {temp_path}"
                            )
                    except Exception as auto_e:
                        logger.debug(
                            f"Automatic temperature loading failed for {scenario}: {auto_e}"
                        )

                if (
                    dh_supply_temperatures is not None
                    and scenario in dh_supply_temperatures
                ):
                    temp_series = dh_supply_temperatures[scenario]
                    # Ensure datetime index & align
                    if not isinstance(temp_series.index, pd.DatetimeIndex):
                        temp_series.index = pd.to_datetime(temp_series.index)
                    # 3h resample like price bins
                    temps_3h = temp_series.resample("3h").mean()
                    # Use same binning as prices (price_bins_labeled exists only inside try above, so recompute locally)
                    # Reconstruct quartile percentiles (aligned with earlier price binning)
                    percentiles = [0, 0.25, 0.5, 0.75, 1]
                    # Need prices again for consistent labeled bins
                    if networks is not None and scenario in networks:
                        n_obj_temp = networks[scenario]
                        n_temp = (
                            next(iter(n_obj_temp.values()))
                            if isinstance(n_obj_temp, dict)
                            else n_obj_temp
                        )
                        prices_3h_temp = (
                            calc_average_electricity_price_t_ordered(n_temp)
                            .resample("3h")
                            .mean()
                        )
                        _tmp_bins, _edges = pd.qcut(
                            prices_3h_temp, q=percentiles, labels=None, retbins=True
                        )

                        def _fmt_edge(v):
                            if v < 10:
                                return f"{v:.2f}"
                            if v < 1000:
                                return f"{v:.0f} "
                            return f"{v:.0f}"

                        _labels = [
                            f"< {_fmt_edge(_edges[i+1])}"
                            for i in range(len(_edges) - 1)
                        ]
                        price_bins_for_t = pd.qcut(
                            prices_3h_temp, q=percentiles, labels=_labels
                        )
                        temps_grouped = temps_3h.groupby(
                            price_bins_for_t, observed=True
                        ).mean()
                        temps_per_bin_aligned = temps_grouped.reindex(
                            uch_de_t_gen.columns
                        )
                        if temps_per_bin_aligned.notna().any():
                            temps_arr = (
                                temps_per_bin_aligned.fillna(method="ffill")
                                .fillna(method="bfill")
                                .values
                            )
                            t_min = np.nanmin(temps_arr)
                            t_max = np.nanmax(temps_arr)
                            if np.isclose(t_min, t_max):
                                t_max = t_min + 1e-6
                            norm = plt.Normalize(vmin=t_min, vmax=t_max)
                            temp_colors = [cmap(norm(t)) for t in temps_arr]
            except Exception as _e:
                logger.debug(f"Temperature coloring skipped for {scenario}: {_e}")

            if temp_colors is None:
                # Fallback: single-color line & white markers (original behavior adapted)
                ax2.plot(
                    x_positions,
                    boosting_vals,
                    marker="o",
                    markersize=6,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    color="black",
                    linewidth=1.0,
                )
            else:
                # Plot colored line segment by segment for gradient effect
                for i in range(1, len(x_positions)):
                    ax2.plot(
                        [x_positions[i - 1], x_positions[i]],
                        [boosting_vals[i - 1], boosting_vals[i]],
                        color=temp_colors[i],
                        linewidth=1.2,
                    )
                # Colored markers
                ax2.scatter(
                    x_positions,
                    boosting_vals,
                    c=temp_colors,
                    s=36,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
                # Register colorbar mappable once (draw later horizontally)
                if not temp_colorbar_added and norm is not None:
                    sm_for_colorbar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm_for_colorbar.set_array([])
                    temp_colorbar_added = True
        else:
            # Fallback: scatter cumulative generation without storage (legacy)
            storage_discharge_indices = uch_de_t_gen.filter(
                regex="discharge", axis=0
            ).index
            markers_to_plot = (
                uch_de_t_gen.drop(storage_discharge_indices).sum().div(1e6).cumsum()
            )
            marker_x, marker_y = [], []
            for lbl, val in markers_to_plot.items():
                if lbl in pos_map:
                    marker_x.append(pos_map[lbl])
                    marker_y.append(val)
            ax2.scatter(marker_x, marker_y, color="white", s=50, edgecolor="black")

        # Set labels and titles based on scenario position
        if ax == axes[0]:
            ax.set_ylabel("District heating energy [TWh]", fontsize=14)
            ax2.set_yticks([])
        else:
            ax2.set_ylabel(
                (
                    "Cumulative boosting energy [TWh]"
                    if boosting_by_label_twh is not None
                    else "Cumulative heat generation\nwithout storage [TWh]"
                ),
                fontsize=14,
            )

        # Use the provided scenario name or clean up the key
        scenario_title = scenario_names.get(scenario, scenario)
        ax.set_title(scenario_title, fontsize=16)
        ax.set_xlabel("Electricity price quartiles [€/MWh]", fontsize=14)
        ax.set_xlim(-0.5, len(bin_labels) - 0.5)
        # Dynamic symmetric y-limits based on stacked totals
        pos_tot = to_plot.clip(lower=0).sum(axis=1).max() if not to_plot.empty else 0
        neg_tot = to_plot.clip(upper=0).sum(axis=1).min() if not to_plot.empty else 0
        y_max = max(pos_tot, abs(neg_tot)) if (pos_tot or neg_tot) else 1
        ax.set_ylim(-1.1 * y_max, 1.1 * y_max)
        # Dynamic secondary axis based on boosting markers (or fallback)
        if boosting_by_label_twh is not None:
            max_val = max(1e-9, np.nanmax(boosting_vals) if len(boosting_vals) else 0.0)
            ax2.set_ylim(0, 1.1 * max_val if max_val > 0 else 1)
        else:
            if marker_y:
                ax2.set_ylim(0, 1.1 * max(marker_y))
            else:
                ax2.set_ylim(0, 1)

        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()
        else:
            handles += ax.get_legend_handles_labels()[0]
            labels += ax.get_legend_handles_labels()[1]

        # Turn ax legend off
        if ax.get_legend():
            ax.get_legend().remove()
        # Increase xticklabel fontsize
        ax.tick_params(labelsize=13)
        ax2.tick_params(labelsize=13)

    # Clean up labels for legend
    import re

    labels = [
        re.sub(
            "urban central heat$",
            "urban central heat for residential and services",
            label,
        )
        for label in labels
    ]
    labels = [label.replace("urban central ", "") for label in labels]
    labels = [label.replace("water pits", "PTES") for label in labels]
    labels = [label.replace("water tanks", "TTES") for label in labels]
    labels = [
        label.replace(" charger", "").replace(" discharger", "") for label in labels
    ]

    # Handle new aggregated technology names
    labels = [label.replace("CHP", "Combined heat and power") for label in labels]
    labels = [
        label.replace("Heat pumps", "Heat pumps (without booster)") for label in labels
    ]
    labels = [label.replace("Heat demand", "Heat demand") for label in labels]

    # More aggressive cleanup for heat demand variations
    labels = [
        (
            "Heat demand"
            if any(
                demand_term in label.lower()
                for demand_term in [
                    "heat for residential and services",
                    "low-temperature heat for industry",
                    "residential and services",
                ]
            )
            else label
        )
        for label in labels
    ]

    # Remove duplicates
    unique_labels = list(dict.fromkeys(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]

    # Reserve space at bottom for legend and (optional) horizontal colorbar
    if temp_colorbar_added:
        fig.subplots_adjust(bottom=0.3)
        legend_y = 0.27
    else:
        fig.subplots_adjust(bottom=0.18)
        legend_y = 0.1

    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=4,
        frameon=False,
        title="Technology",
        fontsize=10,
    )

    # Draw horizontal colorbar under legend if temperature data used
    if temp_colorbar_added and sm_for_colorbar is not None:
        cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.025])  # left, bottom, width, height
        cbar = fig.colorbar(sm_for_colorbar, cax=cbar_ax, orientation="horizontal")
        cbar.set_label("Avg DH supply temperature [°C]", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.2)
    logger.info(
        f"Energy balance with price ventiles comparison plot saved to {output_file}"
    )
    plt.close(fig)

    logger.info(f"Energy balance comparison plot saved to {output_file}")


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

    # Create output directory and subdirectories for organization
    output_path = os.path.dirname(snakemake.output.sysgf_summary)
    os.makedirs(output_path, exist_ok=True)

    # Create subdirectories for different plot categories
    subdirs = {
        "costs": os.path.join(output_path, "costs"),
        "energy_balances": os.path.join(output_path, "energy_balances"),
        "storage_operation": os.path.join(output_path, "storage_operation"),
        "prices": os.path.join(output_path, "prices"),
        "supply": os.path.join(output_path, "supply"),
        "summary": os.path.join(output_path, "summary"),
    }

    for subdir in subdirs.values():
        os.makedirs(subdir, exist_ok=True)

    logger.info(f"Generating system analysis for run: {run_name}")

    # Process networks and collect data
    networks, summary_df, costs_agg = process_networks(
        run_name, scenarios, planning_horizons
    )

    # Save summary DataFrame
    summary_df.to_csv(os.path.join(subdirs["summary"], "summary.csv"), index=False)

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
        subdirs["costs"],
        colors,
    )

    # 1. Plot price duration curves with new implementation
    plot_price_duration_curves(networks, subdirs["prices"])

    # 2. Plot urban central heat supply comparison for each scenario - consolidated in one figure
    plot_uch_supply(networks, os.path.join(subdirs["supply"], "uch_supply.pdf"), colors)

    # 3. Plot summary metrics
    plot_summary_metrics(summary_df, subdirs["summary"])

    # 4. Plot PTES SOCs ranges for networks
    plot_ptes_socs(
        networks, os.path.join(subdirs["storage_operation"], "soc_comparison.png")
    )

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
        plot_ptes_price_impact_scatter(networks, available_tuples, subdirs["prices"])

    # 6. Plot PTES savings comparison grouped by supply temperature scenarios
    if available_tuples:
        # Split available tuples by supply temperature scenarios
        high_temp_tuples = []
        mid_temp_tuples = []
        low_temp_tuples = []

        for ref_scenario, comp_scenario in available_tuples:
            # Check if either scenario contains "HighSupplyTemperature" or "LowSupplyTemperature"
            if (
                "HighSupplyTemperature" in ref_scenario
                or "HighSupplyTemperature" in comp_scenario
            ):
                high_temp_tuples.append((ref_scenario, comp_scenario))
            elif (
                "MidSupplyTemperature" in ref_scenario and "MidDH" in ref_scenario
            ) or ("MidSupplyTemperature" in comp_scenario and "MidDH" in comp_scenario):
                mid_temp_tuples.append((ref_scenario, comp_scenario))
            elif (
                "LowSupplyTemperature" in ref_scenario
                or "LowSupplyTemperature" in comp_scenario
            ):
                low_temp_tuples.append((ref_scenario, comp_scenario))
            else:
                # If neither scenario contains the temperature keywords, add to both groups
                # (this handles edge cases where scenario naming might be different)
                logger.warning(
                    f"Scenario tuple {ref_scenario} vs {comp_scenario} doesn't contain temperature keywords"
                )

        # Plot PTES savings comparison for HighSupplyTemperature scenarios
        if high_temp_tuples:
            logger.info(
                f"Plotting PTES savings comparison for HighSupplyTemperature scenarios: {high_temp_tuples}"
            )
            for year in planning_horizons:
                plot_ptes_savings_comparison(
                    high_temp_tuples,
                    costs_agg,
                    colors,
                    year,
                    subdirs["costs"],
                    figsize=(6, 8),
                    output_suffix="HighSupplyTemperature",
                )
                # Call the neighbour countries cost comparison function
                plot_neighbour_countries_cost_comparison(
                    networks=networks,  # Your networks dictionary
                    scenario_tuples=high_temp_tuples,
                    colors=colors,  # Same colors dictionary you use for other plots
                    year=year,  # Or whatever year you're analyzing
                    output_path=subdirs["costs"],
                    output_suffix="HighSupplyTemperature",  # Optional, same as for regular function
                )
        else:
            logger.info("No HighSupplyTemperature scenario tuples found")

        if mid_temp_tuples:
            logger.info(
                f"Plotting PTES savings comparison for MidSupplyTemperature scenarios: {mid_temp_tuples}"
            )
            for year in planning_horizons:
                plot_ptes_savings_comparison(
                    mid_temp_tuples,
                    costs_agg,
                    colors,
                    year,
                    subdirs["costs"],
                    figsize=(6, 8),
                    output_suffix="MidSupplyTemperature",
                )
                # Call the neighbour countries cost comparison function
                plot_neighbour_countries_cost_comparison(
                    networks=networks,  # Your networks dictionary
                    scenario_tuples=mid_temp_tuples,
                    colors=colors,  # Same colors dictionary you use for other plots
                    year=year,  # Or whatever year you're analyzing
                    output_path=subdirs["costs"],
                    output_suffix="MidSupplyTemperature",  # Optional, same as for regular function
                )
        else:
            logger.info("No MidSupplyTemperature scenario tuples found")

        # Plot PTES savings comparison for LowSupplyTemperature scenarios
        if low_temp_tuples:
            logger.info(
                f"Plotting PTES savings comparison for LowSupplyTemperature scenarios: {low_temp_tuples}"
            )
            for year in planning_horizons:
                plot_ptes_savings_comparison(
                    low_temp_tuples,
                    costs_agg,
                    colors,
                    year,
                    subdirs["costs"],
                    figsize=(6, 8),
                    output_suffix="LowSupplyTemperature",
                )
                # Call the neighbour countries cost comparison function
                plot_neighbour_countries_cost_comparison(
                    networks=networks,  # Your networks dictionary
                    scenario_tuples=low_temp_tuples,
                    colors=colors,  # Same colors dictionary you use for other plots
                    year=year,  # Or whatever year you're analyzing
                    output_path=subdirs["costs"],
                    output_suffix="LowSupplyTemperature",  # Optional, same as for regular function
                )
        else:
            logger.info("No LowSupplyTemperature scenario tuples found")

    # 5. Plot dual comparison for all available tuples
    if available_tuples:
        for scenario_A, scenario_B in available_tuples:
            plot_dual_comparison(
                networks,
                costs_agg,
                scenario_A,
                scenario_B,
                colors,
                subdirs["costs"],
                subdirs["energy_balances"],
            )

            # Also plot energy balance comparison for each tuple
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
                        [scenario_A, scenario_B],
                        os.path.join(
                            subdirs["energy_balances"],
                            f"energy_balance_comparison_{scenario_A}_{scenario_B}.pdf",
                        ),
                        colors,
                    )

                    # Plot seasonal heat balance comparison with deltaT visualization
                    plot_seasonal_heat_balance_with_temperature(
                        network_A_year,
                        network_B_year,
                        scenario_A,
                        scenario_B,
                        colors,
                        subdirs["energy_balances"],
                        list(network_A_year.snapshots.year)[0],
                        run_name,
                    )

                    # Plot seasonal heat balance comparison with electricity prices
                    plot_seasonal_heat_balance_with_prices(
                        network_A_year,
                        network_B_year,
                        scenario_A,
                        scenario_B,
                        colors,
                        subdirs["energy_balances"],
                        list(network_A_year.snapshots.year)[0],
                        run_name,
                    )

                    # Plot energy balance combined comparison across price ventiles
                    try:
                        uch_de_t_A = prepare_energy_data(network_A_year)
                        uch_de_t_B = prepare_energy_data(network_B_year)

                        uch_de_t_gen_A, uch_de_t_load_A, bin_labels_A = (
                            process_generation_and_load(uch_de_t_A, network_A_year)
                        )
                        uch_de_t_gen_B, uch_de_t_load_B, bin_labels_B = (
                            process_generation_and_load(uch_de_t_B, network_B_year)
                        )

                        # Defensive alignment: ensure both have identical quartile labels.
                        # If a scenario produced fewer (e.g., due to duplicate edges), pad with synthetic labels.
                        def _pad_bins(df_gen, df_load, labels):
                            # Expect 4 quartiles; pad if shorter.
                            expected = 4
                            current = len(labels)
                            if current == expected:
                                return df_gen, df_load, labels
                            # Create padded labels Q1..Q4
                            full_labels = [f"Q{i}" for i in range(1, expected + 1)]
                            # Map existing labels to their order; fill missing with zeros
                            new_gen = pd.DataFrame(index=full_labels)
                            new_load = pd.DataFrame(index=full_labels)
                            for lbl in full_labels:
                                if lbl in df_gen.columns:
                                    new_gen[lbl] = df_gen[lbl]
                                else:
                                    new_gen[lbl] = 0.0
                                if lbl in df_load.columns:
                                    new_load[lbl] = df_load[lbl]
                                else:
                                    new_load[lbl] = 0.0
                            return new_gen.T, new_load.T, full_labels

                        uch_de_t_gen_A, uch_de_t_load_A, bin_labels_A = _pad_bins(
                            uch_de_t_gen_A, uch_de_t_load_A, bin_labels_A
                        )
                        uch_de_t_gen_B, uch_de_t_load_B, bin_labels_B = _pad_bins(
                            uch_de_t_gen_B, uch_de_t_load_B, bin_labels_B
                        )
                        # Final consistency check
                        if bin_labels_A != bin_labels_B:
                            logger.warning(
                                f"Quartile label mismatch {bin_labels_A} vs {bin_labels_B}; using first set."
                            )

                        # Create dictionaries for the plotting function
                        uch_de_t_gen_dict = {
                            scenario_A: uch_de_t_gen_A,
                            scenario_B: uch_de_t_gen_B,
                        }
                        uch_de_t_load_dict = {
                            scenario_A: uch_de_t_load_A,
                            scenario_B: uch_de_t_load_B,
                        }

                        # Create clean scenario names for display
                        scenario_names = {
                            scenario_A: scenario_A.replace("_", " "),
                            scenario_B: scenario_B.replace("_", " "),
                        }

                        # Resolve boosting ratio files for each scenario-year (if available)
                        boosting_ratio_files = {}
                        try:
                            # Determine years used for A and B from the selected network objects

                            for scen in [scenario_A, scenario_B]:
                                resources_path = os.path.join(
                                    "resources", run_name, scen
                                )
                                if os.path.exists(resources_path):
                                    try:
                                        candidates = [
                                            f
                                            for f in os.listdir(resources_path)
                                            if f.startswith(
                                                "ptes_discharger_temperature_boosting_ratio_profiles"
                                            )
                                        ]
                                        if candidates:
                                            boosting_ratio_files[scen] = os.path.join(
                                                resources_path, candidates[0]
                                            )
                                    except Exception as e:
                                        logger.warning(
                                            f"Failed to scan boosting ratio files in {resources_path}: {e}"
                                        )
                                else:
                                    logger.debug(
                                        f"Resources path {resources_path} not found for scenario {scen}"
                                    )
                        except Exception as e:
                            logger.debug(
                                f"Could not resolve boosting ratio files for ventiles plot: {e}"
                            )

                        output_fname = os.path.join(
                            subdirs["energy_balances"],
                            f"uch_balance_price_quartiles_{scenario_A}_{scenario_B}.pdf",
                        )
                        try:
                            plot_energy_balance_combined(
                                uch_de_t_gen_dict,
                                uch_de_t_load_dict,
                                output_fname,
                                scenario_names,
                                colors,
                                networks=networks,
                                boosting_ratio_files=(
                                    boosting_ratio_files
                                    if boosting_ratio_files
                                    else None
                                ),
                            )
                            logger.info(
                                f"Energy balance with price quartiles comparison plot saved to {output_fname}"
                            )
                        except Exception as inner_e:
                            logger.warning(
                                "Detailed quartile comparison failure: "
                                f"{inner_e} | binsA={bin_labels_A} binsB={bin_labels_B}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to generate price quartiles comparison for {scenario_A} vs {scenario_B}: {e}"
                        )

    # 7. Plot storage power spectrum analysis for sensitivity runs
    if (
        "sensitivities" in snakemake.params.plotting
        and snakemake.params.plotting["sensitivities"]["enable"]
    ):
        sensitivity_reference = snakemake.params.plotting["sensitivities"]["reference"]
        sensitivity_runs = snakemake.params.plotting["sensitivities"]["runs"]

        # Create filtered networks dict with only reference and sensitivity scenarios
        sensitivity_scenarios = [sensitivity_reference] + sensitivity_runs
        filtered_networks = {
            scenario: networks[scenario]
            for scenario in sensitivity_scenarios
            if scenario in networks
        }

        if filtered_networks:
            for year in planning_horizons:
                plot_storage_psd(filtered_networks, year, subdirs["storage_operation"])
                plot_storage_psd_stacked(
                    filtered_networks, year, subdirs["storage_operation"]
                )
                plot_ptes_socs(
                    filtered_networks,
                    output_path=os.path.join(
                        subdirs["storage_operation"],
                        f"soc_comparison_sensitivity_{year}.png",
                    ),
                )
        else:
            logger.warning(
                "No sensitivity scenarios found in networks for storage PSD analysis"
            )
    else:
        logger.info("Sensitivity analysis disabled, skipping storage PSD plot")

    # 8. Plot PTES energy quantiles for each scenario with uniform x-axis limits
    # Use simplified global xlim: -20 to +10 TWh
    global_ptes_xlim = (-20, 10)

    for scenario in scenarios:
        if scenario in networks:
            scenario_networks = networks[scenario]
            for year, network in scenario_networks.items():
                # Try to find the corresponding boosting ratio file dynamically
                resources_path = os.path.join("resources", run_name, scenario)

                if os.path.exists(resources_path):
                    # Look for the boosting ratio file
                    for file in os.listdir(resources_path):
                        if file.startswith(
                            "ptes_discharger_temperature_boosting_ratio_profiles"
                        ) and file.endswith(f"{year}.nc"):
                            boosting_ratio_file = os.path.join(resources_path, file)

                            try:
                                logger.info(
                                    f"Loading boosting ratio data from {boosting_ratio_file}"
                                )
                                boosting_ratio_data = xr.open_dataarray(
                                    boosting_ratio_file
                                )
                                if "noboost" in scenario.lower():
                                    boosting_ratio_data = boosting_ratio_data.where(
                                        boosting_ratio_data <= 0, 0
                                    )

                                # Generate the plot with uniform x-axis limits
                                plot_ptes_energy_quantiles(
                                    network,
                                    boosting_ratio_data,
                                    subdirs["storage_operation"],
                                    scenario_name=f"{scenario}_{year}",
                                    figsize=(8, 8),
                                    xlim=global_ptes_xlim,
                                )
                                break  # Found and processed the file
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load boosting ratio data for {scenario} {year}: {e}"
                                )
                else:
                    logger.warning(
                        f"Resources path {resources_path} does not exist for scenario {scenario}"
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
