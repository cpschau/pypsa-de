#!/usr/bin/env python3
# SPDX-FileCopyrightText: : 2024 PyPSA-DE authors
#
# SPDX-License-Identifier: MIT

"""
District heating energy balance over electricity price terciles (3x5 grid).
- Rows: supply temperatures (Low/Mid/High)
- Columns: boosting configurations ordered by PTES contribution
- X axis: electricity price terciles (T1..T3)
- Primary Y: DH energy [TWh], shared limits across panels
- Secondary Y: cumulative boosting energy [TWh], shared limits across panels
- Colored markers for boosting energy using ΔT to 90°C with a shared colorbar below legend
"""

import logging
import os
import sys
from pathlib import Path

import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pypsa
import xarray as xr

# Ensure we can import helpers regardless of current working directory
REPO_ROOT = Path(__file__).resolve().parents[2]  # .../code/pypsa-de
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from scripts._helpers import configure_logging, mock_snakemake

logger = logging.getLogger(__name__)


def calc_average_electricity_price_t_ordered(n: pypsa.Network) -> pd.Series:
    loads = n.buses_t.p.filter(regex=r"DE\d \d$").clip(upper=0).mul(-1)
    prices = n.buses_t.marginal_price.filter(regex=r"DE\d \d$")
    if loads.empty or prices.empty or loads.sum(axis=1).isnull().any():
        return pd.Series(dtype=float)
    weighted = loads.mul(prices).sum(axis=1).div(loads.sum(axis=1))
    return weighted


def get_delta_ff_top(ff_temp: xr.DataArray) -> xr.DataArray:
    # Δ to 90°C
    return ff_temp - 90


def resample_to_snapshots(
    n: pypsa.Network, series: pd.Series, func: str = "mean"
) -> pd.Series:
    sns = n.snapshots
    sw = n.snapshot_weightings.generators
    last_snapshot = sns[-1] + pd.Timedelta(hours=sw[sns[-1]])
    sns_extended = pd.Index(sns.tolist() + [last_snapshot])
    bins = pd.IntervalIndex.from_breaks(sns_extended, closed="left")
    bin_labels = pd.cut(series.index, bins)
    binned_values = series.groupby(bin_labels, observed=True).agg(func)
    return binned_values.reindex(sns, fill_value=0)


def get_ptes_discharge(n: pypsa.Network) -> pd.DataFrame:
    return n.snapshot_weightings.generators.mul(
        n.stores_t.p.clip(lower=0).filter(regex="DE.*water pit").T
    ).T


def get_boosting_energy(
    n: pypsa.Network,
    boosting_technology: str = "resistive heater",
    boosting_ratio_fn: str | None = None,
) -> pd.Series:
    """Boosting energy (per snapshot) for a scenario.
    - resistive heater: needs boosting_ratio_fn (xarray per node/time) and divides discharge by alpha
    - heat pump: sum links_t.p0 for PTES HP and weight by snapshot
    Returns series indexed by snapshots (MWh-equivalent per snapshot, consistent with weighting usage).
    """
    tech = boosting_technology.lower()
    if tech == "resistive heater":
        if boosting_ratio_fn is None:
            return pd.Series(0.0, index=n.snapshots)
        try:
            try:
                br_da = xr.open_dataarray(boosting_ratio_fn)
            except Exception:
                ds = xr.open_dataset(boosting_ratio_fn)
                first_var = list(ds.data_vars)[0]
                br_da = ds[first_var]
        except Exception:
            return pd.Series(0.0, index=n.snapshots)

        # Convert to DataFrame time x node
        s = br_da.to_series().dropna()
        if not isinstance(s.index, pd.MultiIndex):
            br_df = s.rename("ratio").to_frame()
        else:
            # assume first level is time
            time_level = 0
            if any("time" in str(nm).lower() for nm in s.index.names):
                time_level = s.index.names.index(
                    next(nm for nm in s.index.names if "time" in str(nm).lower())
                )
            br_df = s.unstack(
                level=[lvl for lvl in range(s.index.nlevels) if lvl != time_level][0]
            )
        try:
            br_df.index = pd.to_datetime(br_df.index)
        except Exception:
            pass

        discharge = get_ptes_discharge(n)
        discharge.columns = discharge.columns.str.split(" urban").str[0]
        common_cols = discharge.columns.intersection(br_df.columns)
        if common_cols.empty:
            return pd.Series(0.0, index=n.snapshots)

        alpha_df = pd.DataFrame(index=n.snapshots, columns=common_cols, dtype=float)
        for col in common_cols:
            series = br_df[col].dropna()
            series = series[~series.index.duplicated(keep="last")]
            alpha_df[col] = resample_to_snapshots(n, series, func="mean")
        boosting_df = (
            discharge[common_cols]
            .div(alpha_df[common_cols])
            .replace(np.inf, 0.0)
            .fillna(0.0)
        )
        return boosting_df.sum(axis=1).reindex(n.snapshots, fill_value=0.0)

    if tech in {"heat pump", "urban central ptes heat pump", "ptes heat pump"}:
        hp_cols = n.links_t.p0.filter(regex=r"DE.*ptes.*heat pump").columns
        if len(hp_cols) == 0:
            return pd.Series(0.0, index=n.snapshots)
        weighted = n.snapshot_weightings.generators.mul(-n.links_t.p0[hp_cols].T).T
        return weighted.sum(axis=1).reindex(n.snapshots, fill_value=0.0)

    return pd.Series(0.0, index=n.snapshots)


def prepare_energy_data(n: pypsa.Network) -> pd.DataFrame:
    """Return a DataFrame with index=carrier and columns=snapshots for DE urban central heat.

    We rely on energy_balance to provide a time-indexed DataFrame with a MultiIndex over columns
    that includes (at least) 'carrier' and, when available, 'bus' and/or 'bus_carrier'.
    """
    # Compute energy balance with robust grouping fallbacks

    eb_t = n.statistics.energy_balance(
        groupby=["bus", "carrier", "bus_carrier"],
        nice_names=False,
        aggregate_time=False,
    )

    eb_t_uch = eb_t.xs("urban central heat", level="bus_carrier")
    eb_t_uch_de = eb_t_uch.loc[
        eb_t_uch.index.get_level_values("bus").str.startswith("DE")
    ]

    # Drop component level if it exists
    if "component" in eb_t_uch_de.index.names:
        eb_t_uch_de.index = eb_t_uch_de.index.droplevel("component")

    # Return dataframe with snapshot columns and multiindex (bus, carrier)
    return eb_t_uch_de


def process_generation_and_load(uch_de_t: pd.DataFrame, n: pypsa.Network):
    """Aggregate generation (+) and load (-) into price terciles (T1..T3)."""
    prices_3h = calc_average_electricity_price_t_ordered(n).resample("3h").mean()
    percentiles = [0, 0.33333, 0.66667, 1]
    # Build labeled terciles aligned to prices_3h index
    price_bins = pd.qcut(prices_3h, q=percentiles, labels=None, duplicates="drop")
    # Labels T1..Tk based on number of categories
    k = price_bins.cat.categories.size if hasattr(price_bins, "cat") else 0
    q_labels = [f"T{i+1}" for i in range(k)]
    if k > 0:
        price_bins = price_bins.cat.rename_categories(q_labels)

    # Weight per-snapshot by snapshot weightings, then aggregate to 3h sums
    sw = n.snapshot_weightings.generators
    # Make index of uch_de_t datetime-like if possible
    try:
        uch_de_t.columns = pd.to_datetime(uch_de_t.columns)
    except Exception:
        logger.warning(
            f"Failed to convert energy balance index: {type(uch_de_t.index[0])} is not convertible to datetime, at position 0"
        )
    gen_t = uch_de_t.T.clip(lower=0).mul(sw, axis=0).resample("3h").sum()
    load_t = uch_de_t.T.clip(upper=0).mul(sw, axis=0).resample("3h").sum()

    # Group rows (time) by price bin series; then transpose back
    if k > 0:
        try:
            # Group using a positional Series to avoid index-name join issues
            gen_g = gen_t.groupby(
                pd.Series(price_bins.values, index=gen_t.index), observed=True
            ).sum(min_count=1)
        except Exception as e:
            logger.exception(
                "Grouping generation by price bins failed: index/gen_t.index alignment issue?"
            )
            raise
        try:
            load_g = load_t.groupby(
                pd.Series(price_bins.values, index=load_t.index), observed=True
            ).sum(min_count=1)
        except Exception as e:
            logger.exception(
                "Grouping load by price bins failed: index/load_t.index alignment issue?"
            )
            raise
        # Keep only positive in gen and negative in load after grouping
        gen = gen_g.T
        gen = gen[gen.sum(axis=1) > 0]
        load = load_g.T
        load = load[load.sum(axis=1) < 0]
        # Ensure column order Q1..Qk
        gen = gen.reindex(columns=q_labels, fill_value=0)
        load = load.reindex(columns=q_labels, fill_value=0)
        return gen, load, q_labels
    else:
        # Fallback: no distinct bins; return empty columns to keep pipeline alive
        empty = []
        gen = gen_t.T.iloc[:, :0]
        load = load_t.T.iloc[:, :0]
        return gen, load, empty


def plot_single_energy_balance(
    uch_de_t_gen: pd.DataFrame,
    uch_de_t_load: pd.DataFrame,
    ax: plt.Axes,
    colors: dict,
    scenario: str,
    networks=None,
    boosting_ratio_files=None,
    dh_supply_temperatures=None,
    global_y_max=None,
    global_y_min=None,
    global_b_max=None,
    show_bottom_xticklabels: bool = False,
    temp_norm=None,
    marker_size: int = 84,
    show_left_y_ticks: bool = True,
    show_right_y2_ticks: bool = False,
    set_right_ylabel: bool = False,
    plot_type: str = "all",  # "all", "storage", or "supply_demand"
):
    """Plot one panel and return legend handles/labels plus (norm, cmap) if available."""
    boosting_by_label_twh = None
    try:
        if networks is not None and scenario in networks:
            n_obj = networks[scenario]
            n = next(iter(n_obj.values())) if isinstance(n_obj, dict) else n_obj
            scen_lower = scenario.lower()
            if "hpboost" in scen_lower:
                boosting_tech = "heat pump"
            elif "rhboost" in scen_lower:
                boosting_tech = "resistive heater"
            else:
                boosting_tech = None
            if boosting_tech is not None:
                ratio_fn = None
                if boosting_tech == "resistive heater":
                    if isinstance(boosting_ratio_files, dict):
                        ratio_fn = boosting_ratio_files.get(scenario)
                    elif isinstance(boosting_ratio_files, str):
                        ratio_fn = boosting_ratio_files
                boosting_ts = get_boosting_energy(
                    n, boosting_technology=boosting_tech, boosting_ratio_fn=ratio_fn
                )
                prices_3h = (
                    calc_average_electricity_price_t_ordered(n).resample("3h").mean()
                )
                percentiles = [0, 0.33333, 0.66667, 1]
                bins = pd.qcut(prices_3h, q=percentiles, labels=None, duplicates="drop")
                k = bins.cat.categories.size if hasattr(bins, "cat") else 0
                q_labels = [f"T{i+1}" for i in range(k)]
                if k > 0:
                    bins = bins.cat.rename_categories(q_labels)
                boosting_3h = boosting_ts.resample("3h").sum()
                if k > 0:
                    boosting_by_label = (
                        boosting_3h.groupby(
                            pd.Series(bins.values, index=boosting_3h.index),
                            observed=True,
                        )
                        .sum()
                        .reindex(q_labels, fill_value=0)
                    )
                    boosting_by_label_twh = boosting_by_label / 1e6
    except Exception as e:
        logger.debug(f"Boosting computation failed for {scenario}: {e}")

    # Assemble to_plot (price_bins, technologies)
    to_plot = pd.concat([(uch_de_t_gen / 1e6).T, (uch_de_t_load / 1e6).T], axis=1)
    to_plot.index.name = None  # avoid pandas adding index name as title
    # Drop first column level and group by remaining column level carrier
    to_plot.columns = to_plot.columns.get_level_values("carrier")
    to_plot = to_plot.T.groupby(to_plot.columns).sum().T
    # Drop vents
    to_drop = [c for c in to_plot.columns if "heat vent" in c.lower()]
    to_plot.drop(columns=to_drop, inplace=True, errors="ignore")

    # Aggregate CHPs
    chp_cols = [
        c
        for c in to_plot.columns
        if any(t in c.lower() for t in ["h2 chp", "gas chp", "waste chp"])
    ]
    if chp_cols:
        to_plot["CHP"] = to_plot[chp_cols].sum(axis=1)
        to_plot.drop(columns=chp_cols, inplace=True)

    # Aggregate HT Heat Pumps (geothermal and electrolysis excess)
    ht_hp_cols = [
        c
        for c in to_plot.columns
        if any(
            t in c.lower()
            for t in [
                "geothermal heat pump",
                "electrolysis excess heat pump",
            ]
        )
        and "ptes" not in c.lower()
    ]
    if ht_hp_cols:
        to_plot["HT heat pumps"] = to_plot[ht_hp_cols].sum(axis=1)
        to_plot.drop(columns=ht_hp_cols, inplace=True)

    # Aggregate LT Heat Pumps (air, river_water, sea_water)
    lt_hp_cols = [
        c
        for c in to_plot.columns
        if any(
            t in c.lower()
            for t in [
                "air heat pump",
                "river_water heat pump",
                "sea_water heat pump",
            ]
        )
        and "ptes" not in c.lower()
    ]
    if lt_hp_cols:
        to_plot["LT heat pumps"] = to_plot[lt_hp_cols].sum(axis=1)
        to_plot.drop(columns=lt_hp_cols, inplace=True)

    # Aggregate PTES Heat Pumps (booster)
    ptes_hp_cols = [
        c for c in to_plot.columns if "ptes" in c.lower() and "heat pump" in c.lower()
    ]
    if ptes_hp_cols:
        to_plot["Booster heat pump"] = to_plot[ptes_hp_cols].sum(axis=1)
        to_plot.drop(columns=ptes_hp_cols, inplace=True)

    # Aggregate heat demand buckets (including other loads)
    demand_cols = [
        c
        for c in to_plot.columns
        if any(
            k in c.lower()
            for k in ["low-temperature heat for industry", "urban central heat"]
        )
    ]
    # Also include other loads in heat demand
    other_load_cols = [c for c in to_plot.columns if "other loads" in c.lower()]
    demand_cols.extend(other_load_cols)

    if demand_cols:
        to_plot["Heat demand"] = to_plot[demand_cols].sum(axis=1)
        to_plot.drop(columns=demand_cols, inplace=True)

    # Filter technologies based on plot_type
    if plot_type == "storage":
        # Keep only storage technologies
        storage_cols = [
            c
            for c in to_plot.columns
            if any(
                s in c.lower() for s in ["water pits", "water tanks", "ptes", "ttes"]
            )
        ]
        to_plot = to_plot[storage_cols]
    elif plot_type == "supply_demand":
        # Remove storage technologies
        storage_cols = [
            c
            for c in to_plot.columns
            if any(
                s in c.lower() for s in ["water pits", "water tanks", "ptes", "ttes"]
            )
        ]
        to_plot = to_plot.drop(columns=storage_cols, errors="ignore")

    # Group tiny techs (only if not storage-only plot)
    if plot_type != "storage":
        small = to_plot.columns[(to_plot.abs().sum() < 5)]
        if len(small) > 0:
            to_plot["other supply technologies"] = (
                to_plot[small].clip(lower=0).sum(axis=1)
            )
            # Don't create separate "other loads" since they're now part of heat demand
            remaining_other_loads = to_plot[small].clip(upper=0).sum(axis=1)
            if not np.isclose(remaining_other_loads.abs().sum(), 0):
                if "Heat demand" in to_plot.columns:
                    to_plot["Heat demand"] = (
                        to_plot["Heat demand"] + remaining_other_loads
                    )
                else:
                    to_plot["Heat demand"] = remaining_other_loads
            to_plot.drop(columns=small, inplace=True)
        # Clean up empty other supply technologies
        if "other supply technologies" in to_plot and np.isclose(
            to_plot["other supply technologies"].abs().sum(), 0
        ):
            to_plot.drop(columns=["other supply technologies"], inplace=True)

    # Order and colors
    to_plot = to_plot[to_plot.abs().sum().sort_values(ascending=False).index]
    colors_local = dict(colors)
    colors_local.setdefault("other supply technologies", "#A9A9A9")
    colors_local.setdefault("other loads", "#696969")
    # Set colors for new heat pump categories
    colors_local.setdefault(
        "HT heat pumps",
        colors_local.get(
            "geothermal heat pump",
            colors_local.get("urban central geothermal heat pump", "#8B4513"),
        ),
    )
    colors_local.setdefault("LT heat pumps", "#FFA600")  # Yellow color
    colors_local.setdefault(
        "Booster heat pump",
        colors_local.get(
            "urban central ptes heat pump",
            colors_local.get("ptes heat pump", "#FF6347"),
        ),
    )
    pos_cols = [c for c in to_plot.columns if (to_plot[c] > 0).any()]
    neg_cols = [c for c in to_plot.columns if (to_plot[c] < 0).any()]

    def is_storage(c: str) -> bool:
        cl = c.lower()
        return ("water pits" in cl) or ("water tanks" in cl)

    pos_order = [c for c in pos_cols if not is_storage(c)] + [
        c for c in pos_cols if is_storage(c)
    ]
    neg_order = [c for c in neg_cols if not is_storage(c)] + [
        c for c in neg_cols if is_storage(c)
    ]
    pos_df = to_plot[pos_order].clip(lower=0)
    neg_df = to_plot[neg_order].clip(upper=0)
    if not pos_df.empty:
        pos_df.plot.bar(
            ax=ax,
            stacked=True,
            color=[colors_local.get(c, "black") for c in pos_order],
            linewidth=0.5,
            alpha=0.8,
            width=1,
        )
    if not neg_df.empty:
        neg_df.plot.bar(
            ax=ax,
            stacked=True,
            color=[colors_local.get(c, "black") for c in neg_order],
            linewidth=0.5,
            alpha=0.8,
            width=1,
        )
    ax.margins(x=0)

    # X ticks as Q1..Qk; show only on bottom row
    from matplotlib.ticker import FixedLocator

    # Use column order of uch_de_t_gen (quartile labels Q1..Qk)
    bin_labels = list(uch_de_t_gen.columns)
    x_positions = np.arange(len(bin_labels))
    ax.xaxis.set_major_locator(FixedLocator(x_positions))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels if show_bottom_xticklabels else [])
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#CCCCCC", alpha=0.8)
    ax.set_axisbelow(True)
    ax.axhline(0, color="black", linewidth=0.6)

    # Secondary axis: boosting markers
    ax2 = ax.twinx()
    norm = None
    cmap = None
    if boosting_by_label_twh is not None:
        boosting_vals = np.cumsum(
            boosting_by_label_twh.reindex(bin_labels, fill_value=0.0).values
        )
        # Color by temperature
        temp_colors = None
        try:
            if (dh_supply_temperatures is None) or (
                scenario not in (dh_supply_temperatures or {})
            ):
                try:
                    run_name_auto = snakemake.params.run  # type: ignore  # noqa: F821
                    temp_path = (
                        f"resources/{run_name_auto}/{scenario}/"
                        "central_heating_forward_temperature_profiles_base_s_49_2045.nc"
                    )
                    if os.path.exists(temp_path):
                        ff_temp = xr.open_dataarray(temp_path)
                        temp_series = (
                            get_delta_ff_top(ff_temp)
                            .to_pandas()
                            .filter(like="DE0")
                            .mean(1)
                        )
                        if dh_supply_temperatures is None:
                            dh_supply_temperatures = {}
                        dh_supply_temperatures[scenario] = temp_series
                except Exception:
                    pass
            if (
                dh_supply_temperatures is not None
                and scenario in dh_supply_temperatures
            ):
                ts = dh_supply_temperatures[scenario]
                if not isinstance(ts.index, pd.DatetimeIndex):
                    ts.index = pd.to_datetime(ts.index)
                temps_3h = ts.resample("3h").mean()
                # reuse price bins to label temperatures into the same categories as gen/load columns
                if networks is not None and scenario in networks:
                    n_obj_temp = networks[scenario]
                    n_temp = (
                        next(iter(n_obj_temp.values()))
                        if isinstance(n_obj_temp, dict)
                        else n_obj_temp
                    )
                    prices_3h = (
                        calc_average_electricity_price_t_ordered(n_temp)
                        .resample("3h")
                        .mean()
                    )
                    percentiles = [0, 0.33333, 0.66667, 1]
                    t_bins = pd.qcut(
                        prices_3h, q=percentiles, labels=None, duplicates="drop"
                    )
                    # align to same labels order as x-axis
                    if hasattr(t_bins, "cat"):
                        cats = t_bins.cat.categories
                        # Map categories to provided bin_labels if same length; otherwise keep as-is
                        if len(cats) == len(bin_labels):
                            t_bins = t_bins.cat.rename_categories(bin_labels)
                    temps_grouped = (
                        temps_3h.groupby(
                            pd.Series(t_bins.values, index=temps_3h.index),
                            observed=True,
                        )
                        .mean()
                        .reindex(bin_labels)
                    )
                    arr = (
                        temps_grouped.fillna(method="ffill")
                        .fillna(method="bfill")
                        .values
                    )
                    tmin = float(np.nanmin(arr))
                    tmax = float(np.nanmax(arr))
                    if temp_norm is not None:
                        tmin, tmax = temp_norm
                    if np.isclose(tmin, tmax):
                        tmax = tmin + 1e-6
                    cmap = plt.get_cmap("Wistia")
                    norm = plt.Normalize(vmin=tmin, vmax=tmax)
                    temp_colors = [cmap(norm(t)) for t in arr]
        except Exception as e:
            logger.debug(f"Temperature coloring failed for {scenario}: {e}")
            temp_colors = None

        if temp_colors is None:
            # Fallback: use a simple gradient based on quartile position
            if len(x_positions) > 0:
                cmap = plt.get_cmap("Wistia")
                temp_colors = [
                    cmap(i / max(1, len(x_positions) - 1))
                    for i in range(len(x_positions))
                ]
            else:
                temp_colors = ["orange"] * len(x_positions)

        # Always use colored markers
        for i in range(1, len(x_positions)):
            ax2.plot(
                [x_positions[i - 1], x_positions[i]],
                [boosting_vals[i - 1], boosting_vals[i]],
                color=temp_colors[i] if i < len(temp_colors) else temp_colors[-1],
                linewidth=1.4,
            )
        ax2.scatter(
            x_positions,
            boosting_vals,
            c=temp_colors[: len(x_positions)],
            s=marker_size,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    # Use unified global y-limits across all axes
    if global_y_max is not None and global_y_min is not None:
        # Use global limits with small margin (10%)
        y_margin_pos = 0.1 * abs(global_y_max) if global_y_max > 0 else 0.1
        y_margin_neg = 0.1 * abs(global_y_min) if global_y_min < 0 else 0.1
        ax.set_ylim(
            global_y_min - y_margin_neg,
            global_y_max + y_margin_pos,
        )
    elif not to_plot.empty:
        # Fallback to local limits if global not available
        pos_tot = to_plot.clip(lower=0).sum(axis=1).max()
        neg_tot = to_plot.clip(upper=0).sum(axis=1).min()
        y_max = max(pos_tot, abs(neg_tot)) if (pos_tot or neg_tot) else 1
        ax.set_ylim(
            1.1 * neg_tot if neg_tot < 0 else -0.1 * y_max,
            1.1 * pos_tot if pos_tot > 0 else 0.1 * y_max,
        )
    else:
        ax.set_ylim(-1, 1)

    # Use consistent global_b_max for all secondary y-axes
    if global_b_max is not None and global_b_max > 0:
        ax2.set_ylim(0, 1.05 * global_b_max)
    else:
        # Fallback for scenarios without boosting data or when global_b_max is 0
        if boosting_by_label_twh is not None and not boosting_by_label_twh.empty:
            local_max = float(
                np.nanmax(
                    np.cumsum(
                        boosting_by_label_twh.reindex(
                            to_plot.index if hasattr(to_plot, "index") else [],
                            fill_value=0.0,
                        ).values
                    )
                )
            )
            ax2.set_ylim(0, 1.05 * (local_max if local_max > 0 else 1.0))
        else:
            ax2.set_ylim(0, 1)

    ax.set_xlim(-0.5, len(bin_labels) - 0.5)

    if ax.get_legend():
        ax.get_legend().remove()
    # Tick labels visibility controls
    ax.tick_params(labelsize=12, labelleft=show_left_y_ticks)
    ax2.tick_params(labelsize=12, labelright=show_right_y2_ticks)
    if not show_left_y_ticks:
        ax.set_yticklabels([])
    if not show_right_y2_ticks:
        ax2.set_yticklabels([])
    # Cumulative boosting energy label is now handled globally, not per-axis
    # Keep this section empty but maintain the if structure for compatibility
    if set_right_ylabel:
        pass

    handles, labels = ax.get_legend_handles_labels()
    return (
        handles,
        labels,
        (norm, plt.get_cmap("Wistia") if norm is not None else None),
    )


def plot_dh_grid_over_elec_prices(
    networks,
    output_file: str,
    colors: dict,
    run_name: str,
    boosting_ratio_files=None,
    dh_supply_temperatures=None,
):
    """Generate both storage and supply/demand plots."""
    # Generate storage plot
    storage_output = output_file.replace(".pdf", "_storage.pdf")
    plot_dh_grid_single_type(
        networks,
        storage_output,
        colors,
        run_name,
        boosting_ratio_files,
        dh_supply_temperatures,
        "storage",
    )

    # Generate supply/demand plot
    supply_demand_output = output_file.replace(".pdf", "_supply_demand.pdf")
    plot_dh_grid_single_type(
        networks,
        supply_demand_output,
        colors,
        run_name,
        boosting_ratio_files,
        dh_supply_temperatures,
        "supply_demand",
    )


def plot_dh_grid_single_type(
    networks,
    output_file: str,
    colors: dict,
    run_name: str,
    boosting_ratio_files=None,
    dh_supply_temperatures=None,
    plot_type: str = "all",
):
    # Grouping and column order
    supply_temp_groups = [
        "HighSupplyTemperature",
        "MidSupplyTemperature",
        "LowSupplyTemperature",
    ]
    boosting_configs_match = [
        "hpboost_10Cbottom",
        "hpboost",
        "rhboost",
        "NoPTES",
        "noboost",
    ]
    boosting_configs_display = [
        "NoPTES",
        "hpboost",
        "hpboost_10Cbottom",
        "rhboost",
        "noboost",
    ]

    # Map scenarios to (row,col) categories
    scenario_mapping: dict[tuple[str, str], str] = {}
    for scenario in networks.keys():
        supply_temp = None
        for t in supply_temp_groups:
            if t in scenario:
                supply_temp = t
                break
        if supply_temp is None:
            if "Low" in scenario and "Supply" in scenario:
                supply_temp = "LowSupplyTemperature"
            elif "High" in scenario and "Supply" in scenario:
                supply_temp = "HighSupplyTemperature"
            else:
                supply_temp = "MidSupplyTemperature"
        boosting_config = None
        for c in boosting_configs_match:
            if c in scenario:
                boosting_config = c
                break
        if boosting_config is None:
            if "NoPTES" in scenario:
                boosting_config = "NoPTES"
            elif "hpboost_10C" in scenario:
                boosting_config = "hpboost_10Cbottom"
            elif "hpboost" in scenario:
                boosting_config = "hpboost"
            elif "rhboost" in scenario:
                boosting_config = "rhboost"
            elif "noboost" in scenario:
                boosting_config = "noboost"
            else:
                continue
        scenario_mapping[(supply_temp, boosting_config)] = scenario

    # Determine which columns actually have any scenario mapped (to avoid empty columns)
    available_columns = [
        cfg
        for cfg in boosting_configs_display
        if any((t, cfg) in scenario_mapping for t in supply_temp_groups)
    ]
    if not available_columns:
        logger.error("No scenarios mapped to any column; aborting plot")
        return

    # First pass: global limits and temperature range per plot type
    global_y_max_storage = 0.0
    global_y_min_storage = 0.0
    global_y_max_supply_demand = 0.0
    global_y_min_supply_demand = 0.0
    global_b_max = 0.0
    global_t_min = None
    global_t_max = None
    all_quartile_temps = []  # Collect all tercile temperature averages

    for (supply_temp, boosting_config), scenario in scenario_mapping.items():
        n_obj = networks[scenario]
        n = next(iter(n_obj.values())) if isinstance(n_obj, dict) else n_obj
        try:
            uch = prepare_energy_data(n)
            gen, load, q_labels = process_generation_and_load(uch, n)

            # Filter data based on plot type for more accurate limits
            if plot_type == "storage":
                # For storage plots, only consider storage technologies (PTES, TTES)
                storage_keywords = ["ptes", "ttes", "water pits", "water tanks"]

                # Handle MultiIndex by checking the last level (carrier names)
                if not gen.empty:
                    if isinstance(gen.index, pd.MultiIndex):
                        # Get the carrier level (assuming it's the last level)
                        carrier_level = gen.index.get_level_values(-1)
                        storage_mask_gen = carrier_level.str.contains(
                            "|".join(storage_keywords), case=False, na=False
                        )
                        gen_filtered = gen[storage_mask_gen]
                    else:
                        gen_filtered = gen[
                            gen.index.str.contains(
                                "|".join(storage_keywords), case=False, na=False
                            )
                        ]
                else:
                    gen_filtered = gen

                if not load.empty:
                    if isinstance(load.index, pd.MultiIndex):
                        # Get the carrier level (assuming it's the last level)
                        carrier_level = load.index.get_level_values(-1)
                        storage_mask_load = carrier_level.str.contains(
                            "|".join(storage_keywords), case=False, na=False
                        )
                        load_filtered = load[storage_mask_load]
                    else:
                        load_filtered = load[
                            load.index.str.contains(
                                "|".join(storage_keywords), case=False, na=False
                            )
                        ]
                else:
                    load_filtered = load

                pos_tot = (
                    (gen_filtered / 1e6).T.sum(axis=1).max()
                    if not gen_filtered.empty
                    else 0.0
                )
                neg_tot = (
                    (load_filtered / 1e6).T.sum(axis=1).min()
                    if not load_filtered.empty
                    else 0.0
                )
            else:
                # For supply/demand plots, use all technologies
                pos_tot = (gen / 1e6).T.sum(axis=1).max() if not gen.empty else 0.0
                neg_tot = (load / 1e6).T.sum(axis=1).min() if not load.empty else 0.0

            # Robust numeric handling (avoid Python 'or' on potential array-like)
            pt = float(pos_tot) if pd.notnull(pos_tot) else 0.0
            nt = float(neg_tot) if pd.notnull(neg_tot) else 0.0

            # Update limits based on plot type
            if plot_type == "storage":
                global_y_max_storage = max(global_y_max_storage, pt)
                global_y_min_storage = min(global_y_min_storage, nt)
            elif plot_type == "supply_demand":
                global_y_max_supply_demand = max(global_y_max_supply_demand, pt)
                global_y_min_supply_demand = min(global_y_min_supply_demand, nt)

            scen_lower = scenario.lower()
            boosting_tech = (
                "heat pump"
                if "hpboost" in scen_lower
                else ("resistive heater" if "rhboost" in scen_lower else None)
            )
            if boosting_tech is not None:
                ratio_fn = None
                if boosting_tech == "resistive heater":
                    if isinstance(boosting_ratio_files, dict):
                        ratio_fn = boosting_ratio_files.get(scenario)
                    elif isinstance(boosting_ratio_files, str):
                        ratio_fn = boosting_ratio_files
                boosting_ts = get_boosting_energy(
                    n, boosting_technology=boosting_tech, boosting_ratio_fn=ratio_fn
                )
                prices_3h = (
                    calc_average_electricity_price_t_ordered(n).resample("3h").mean()
                )
                percentiles = [0, 0.33333, 0.66667, 1]
                bins = pd.qcut(prices_3h, q=percentiles, labels=None, duplicates="drop")
                k = bins.cat.categories.size if hasattr(bins, "cat") else 0
                qls = [f"T{i+1}" for i in range(k)]
                if k > 0:
                    bins = bins.cat.rename_categories(qls)
                b3h = boosting_ts.resample("3h").sum()
                b_by_label = (
                    (
                        b3h.groupby(
                            pd.Series(bins.values, index=b3h.index), observed=True
                        ).sum()
                        / 1e6
                    ).reindex(qls, fill_value=0.0)
                    if k > 0
                    else pd.Series([], dtype=float)
                )
                cum = np.cumsum(b_by_label.values)
                if len(cum):
                    global_b_max = max(global_b_max, float(np.nanmax(cum)))

            # Temperature quartile averages calculation
            try:
                temp_path = (
                    f"resources/{run_name}/{scenario}/"
                    "central_heating_forward_temperature_profiles_base_s_49_2045.nc"
                )
                if os.path.exists(temp_path):
                    ff_temp = xr.open_dataarray(temp_path)
                    ts = (
                        get_delta_ff_top(ff_temp).to_pandas().filter(like="DE0").mean(1)
                    )
                    if not isinstance(ts.index, pd.DatetimeIndex):
                        ts.index = pd.to_datetime(ts.index)
                    temps_3h = ts.resample("3h").mean()

                    # Calculate tercile averages for this scenario
                    prices_3h = (
                        calc_average_electricity_price_t_ordered(n)
                        .resample("3h")
                        .mean()
                    )
                    percentiles = [0, 0.33333, 0.66667, 1]
                    t_bins = pd.qcut(
                        prices_3h, q=percentiles, labels=None, duplicates="drop"
                    )
                    k = t_bins.cat.categories.size if hasattr(t_bins, "cat") else 0
                    qls = [f"T{i+1}" for i in range(k)]
                    if k > 0:
                        t_bins = t_bins.cat.rename_categories(qls)
                        temps_grouped = (
                            temps_3h.groupby(
                                pd.Series(t_bins.values, index=temps_3h.index),
                                observed=True,
                            )
                            .mean()
                            .reindex(qls)
                        )
                        quartile_temps = (
                            temps_grouped.fillna(method="ffill")
                            .fillna(method="bfill")
                            .values
                        )
                        # Add non-NaN values to global collection
                        valid_temps = quartile_temps[~np.isnan(quartile_temps)]
                        all_quartile_temps.extend(valid_temps)
            except Exception:
                pass
        except Exception:
            logger.exception(f"Prepass failed for scenario {scenario}")
            continue

    # Calculate global temperature range from all quartile averages
    if all_quartile_temps:
        global_t_min = float(np.min(all_quartile_temps))
        global_t_max = float(np.max(all_quartile_temps))
        logger.info(
            f"Global temperature range from tercile averages: {global_t_min:.2f} to {global_t_max:.2f} K"
        )
    else:
        global_t_min = None
        global_t_max = None

    # Create grid
    n_cols = len(available_columns)
    fig, axes = plt.subplots(3, n_cols, figsize=(12, 14), sharex=False, sharey=False)

    # Plot
    all_handles, all_labels = [], []
    last_norm, last_cmap = None, None

    for r, supply_temp in enumerate(supply_temp_groups):
        for c, boosting_config in enumerate(available_columns):
            ax = axes[r, c]
            ax.set_title("")  # no pandas default

            # Set boosting configuration headers for top row BEFORE checking for data
            if r == 0:
                names = {
                    "NoPTES": "No PTES",
                    "hpboost": "Heat Pump\nBoosting",
                    "hpboost_10Cbottom": "Heat Pump to\n10°C",
                    "rhboost": "Resistive\nBoosting",
                    "noboost": "No Boosting",
                }
                ax.set_title(
                    names.get(boosting_config, boosting_config),
                    fontsize=14,
                    pad=20,
                    weight="bold",
                )

            scenario = scenario_mapping.get((supply_temp, boosting_config))
            if scenario is None:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            n_obj = networks[scenario]
            n = next(iter(n_obj.values())) if isinstance(n_obj, dict) else n_obj
            try:
                uch = prepare_energy_data(n)
                gen, load, _ = process_generation_and_load(uch, n)
                handles, labels, (norm, cmap) = plot_single_energy_balance(
                    gen,
                    load,
                    ax,
                    colors,
                    scenario,
                    networks=networks,
                    boosting_ratio_files=boosting_ratio_files,
                    dh_supply_temperatures=None,
                    global_y_max=(
                        global_y_max_storage
                        if plot_type == "storage"
                        else global_y_max_supply_demand
                    )
                    or None,
                    global_y_min=(
                        global_y_min_storage
                        if plot_type == "storage"
                        else global_y_min_supply_demand
                    )
                    or None,
                    global_b_max=global_b_max or None,
                    show_bottom_xticklabels=(r == 2),
                    temp_norm=(
                        (global_t_min, global_t_max)
                        if (global_t_min is not None and global_t_max is not None)
                        else None
                    ),
                    marker_size=96,
                    show_left_y_ticks=(c == 0),
                    show_right_y2_ticks=(c == n_cols - 1),
                    set_right_ylabel=(c == n_cols - 1),
                    plot_type=plot_type,
                )
                # Accumulate legend entries across panels to ensure full coverage (e.g., Resistive heater)
                if handles and labels:
                    all_handles += handles
                    all_labels += labels
                if norm is not None and cmap is not None:
                    last_norm, last_cmap = norm, cmap
            except Exception as e:
                logger.exception(f"Failed to process scenario {scenario}")
                ax.text(
                    0.5,
                    0.5,
                    "Error",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="red",
                )

            # Column titles are now set earlier in the loop
            # Row label and left y label (broken into lines to avoid overlap)
            if c == 0:
                row_names = {
                    "LowSupplyTemperature": "Low Supply\nTemperature",
                    "MidSupplyTemperature": "Med Supply\nTemperature",
                    "HighSupplyTemperature": "High Supply\nTemperature",
                }
                # Add supply temperature title (bold weight, further left)
                ax.text(
                    -0.6,
                    0.5,
                    row_names.get(supply_temp, supply_temp),
                    transform=ax.transAxes,
                    fontsize=14,
                    weight="bold",
                    ha="center",
                    va="center",
                    rotation=90,
                )

                # Keep the supply temperature labels only - primary y-axis label is handled by fig.supylabel

    # Legend and colorbar
    import re

    clean_labels = []
    for label in all_labels:
        label = re.sub(
            "urban central heat$",
            "urban central heat for residential and services",
            label,
        )
        label = (
            label.replace("urban central ", "")
            .replace("water pits", "PTES")
            .replace("water tanks", "TTES")
            .replace(" charger", "")
            .replace(" discharger", "")
        )
        label = label.replace("CHP", "Combined heat and power")
        # Keep new heat pump category names as they are
        if label not in ["HT heat pumps", "LT heat pumps", "Booster heat pump"]:
            label = label.replace("Heat pumps", "Heat pumps (without booster)")
        if any(
            x in label.lower()
            for x in [
                "heat for residential and services",
                "low-temperature heat for industry",
                "residential and services",
                "other loads",
            ]
        ):
            label = "Heat demand"
        clean_labels.append(label)
    # Build unique handles preserving first occurrence order of cleaned labels
    unique_labels = []
    unique_handles = []
    for h, raw_label, clean in zip(all_handles, all_labels, clean_labels):
        # Skip resistive heater in storage plots since it doesn't belong there
        if plot_type == "storage" and any(
            x in clean.lower() for x in ["resistive heater", "resistive", "heater"]
        ):
            continue
        if clean not in unique_labels:
            unique_labels.append(clean)
            unique_handles.append(h)

    # Create categorized legend with bold category headers
    import matplotlib.patches as mpatches

    # Define categories based on plot type
    if plot_type == "storage":
        supply_techs = []
        demand_techs = []
        storage_techs = ["PTES", "TTES"]
    else:
        supply_techs = [
            "HT heat pumps",
            "LT heat pumps",
            "Booster heat pump",
            "Combined heat and power",
            "Resistive heater",
            "other supply technologies",
        ]
        demand_techs = ["Heat demand", "other loads"]
        storage_techs = []  # No storage in supply/demand plot

    # Build categorized legend
    final_handles = []
    final_labels = []

    # Supply category (only if there are supply technologies)
    supply_items = [
        (label, handle)
        for label, handle in zip(unique_labels, unique_handles)
        if any(tech.lower() in label.lower() for tech in supply_techs)
    ]
    if supply_items:
        final_handles.append(mpatches.Patch(color="none", label=""))
        final_labels.append(r"$\mathbf{Supply:}$")
        for label, handle in supply_items:
            final_handles.append(handle)
            final_labels.append(label)

    # Demand category (only if there are demand technologies)
    demand_items = [
        (label, handle)
        for label, handle in zip(unique_labels, unique_handles)
        if any(tech.lower() in label.lower() for tech in demand_techs)
    ]
    if demand_items:
        final_handles.append(mpatches.Patch(color="none", label=""))
        final_labels.append(r"$\mathbf{Demand:}$")
        for label, handle in demand_items:
            final_handles.append(handle)
            final_labels.append(label)

    # Storage category (only if there are storage technologies)
    storage_items = [
        (label, handle)
        for label, handle in zip(unique_labels, unique_handles)
        if any(tech.lower() in label.lower() for tech in storage_techs)
    ]
    if storage_items:
        final_handles.append(mpatches.Patch(color="none", label=""))
        final_labels.append(r"$\mathbf{Storage:}$")
        for label, handle in storage_items:
            final_handles.append(handle)
            final_labels.append(label)

    # Add any remaining technologies
    remaining_techs = []
    for label, handle in zip(unique_labels, unique_handles):
        if not any(
            tech.lower() in label.lower()
            for tech in supply_techs + demand_techs + storage_techs
        ):
            remaining_techs.append((handle, label))

    if remaining_techs:
        final_handles.append(mpatches.Patch(color="none", label=""))
        final_labels.append(r"$\mathbf{Other}$")
        for handle, label in remaining_techs:
            final_handles.append(handle)
            final_labels.append(label)

    # Use the categorized legend
    unique_handles = final_handles
    unique_labels = final_labels

    # If resistive heater appears in any scenario but not in legend, try adding a proxy
    # try:
    #     if any("rhboost" in key[1] for key in scenario_mapping.keys()) and not any(
    #         "resistive" in lbl.lower() for lbl in unique_labels
    #     ):
    #         import matplotlib.patches as mpatches

    #         # Heuristic color key candidates
    #         color_key_candidates = [
    #             "resistive heater",
    #             "urban central resistive heater",
    #             "DE urban central resistive heater",
    #         ]
    #         rh_color = None
    #         for k in color_key_candidates:
    #             if k in colors:
    #                 rh_color = colors[k]
    #                 break
    #         if rh_color is None:
    #             rh_color = "#888888"
    #         proxy = mpatches.Patch(color=rh_color)
    #         unique_handles.append(proxy)
    #         unique_labels.append("Resistive heater")
    # except Exception:
    #     pass

    # Layout: make plots stretch more and leave space for titles above and labels below
    # Do this BEFORE adding text labels so positions are stable
    fig.subplots_adjust(top=0.88, bottom=0.25, hspace=0.15, wspace=0.08)

    # Add primary y-axis label - conditional on plot type and positioned accordingly
    ylabel_text = (
        "District heating storage balance [TWh]"
        if plot_type == "storage"
        else "District Heating Balance [TWh]"
    )
    ylabel_x = 0.08
    fig.supylabel(
        ylabel_text,
        fontsize=14,
        x=ylabel_x,
        y=0.55,
        ha="center",
        rotation=90,
        weight="normal",
    )

    # Add cumulative boosting energy label close to secondary y-axis
    # Position it close to the rightmost column's right edge
    rightmost_ax = axes[1, -1]  # Middle row, rightmost column for vertical centering
    fig.text(
        rightmost_ax.get_position().x1 + 0.04,  # Close to secondary y-axis
        0.55,  # Center vertically across the entire figure
        "Cumulative boosting energy [TWh]",
        ha="center",
        va="center",
        fontsize=14,
        weight="normal",
        rotation=270,
        transform=fig.transFigure,
    )
    # Shared xlabel - positioned below plots
    fig.supxlabel(
        r"Electricity price terciles [€ MWh$^{-1}$]",
        fontsize=14,
        y=0.2,
    )

    # Legend positioned right below xlabel
    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11),
        ncol=3,
        frameon=False,
        fontsize=14,
    )

    # Colorbar for marker colormap right below legend
    try:
        from matplotlib.cm import ScalarMappable

        if last_norm is not None and last_cmap is not None:
            cax = fig.add_axes([0.2, 0.09, 0.6, 0.018])  # Positioned right below legend
            sm = ScalarMappable(norm=last_norm, cmap=last_cmap)
            sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.ax.tick_params(labelsize=13)  # Larger tick font
            cb.set_label(
                "Average delta T to 90°C [K]", fontsize=14
            )  # Larger label font
    except Exception:
        pass

    fig.savefig(output_file, bbox_inches="tight", pad_inches=0.3, dpi=300)
    logger.info(f"District heating grid plot saved to {output_file}")
    plt.close(fig)


def process_networks(run_name, scenarios, planning_horizons):
    networks = {}
    networks_path = os.path.join("results", run_name)
    found_scenarios = False
    if os.path.exists(networks_path):
        found_scenarios = any(
            os.path.exists(os.path.join(networks_path, s, "networks"))
            for s in scenarios
        )
    if not found_scenarios:
        results_dir = "results"
        if os.path.exists(results_dir):
            available_runs = [
                d
                for d in os.listdir(results_dir)
                if os.path.isdir(os.path.join(results_dir, d))
            ]
            preferred_runs = ["20250911_test_new_boosting"]
            for alt_run in preferred_runs + sorted(available_runs, reverse=True):
                if alt_run not in available_runs:
                    continue
                alt_path = os.path.join(results_dir, alt_run)
                alt_scenarios = [
                    d
                    for d in os.listdir(alt_path)
                    if os.path.isdir(os.path.join(alt_path, d)) and d != "sysgf"
                ]
                if any(
                    os.path.exists(os.path.join(alt_path, s, "networks"))
                    for s in alt_scenarios
                ):
                    networks_path = alt_path
                    run_name = alt_run
                    scenarios = [
                        s
                        for s in alt_scenarios
                        if os.path.exists(os.path.join(alt_path, s, "networks"))
                    ]
                    break
    for scenario in scenarios:
        scenario_path = os.path.join(networks_path, scenario, "networks")
        if not os.path.exists(scenario_path):
            continue
        networks[scenario] = {}
        for year in planning_horizons:
            network_file = None
            for file in os.listdir(scenario_path):
                if file.endswith(f"{year}.nc"):
                    network_file = os.path.join(scenario_path, file)
                    break
            if not network_file:
                continue
            try:
                n = pypsa.Network(network_file)
                networks[scenario][year] = n
            except Exception as e:
                logger.error(f"Failed to load network for {scenario} {year}: {e}")
    return networks if networks else None


def get_colors(networks, override_colors=None):
    if override_colors is None:
        override_colors = {}
    first_network = next(iter(networks.values()))[
        next(iter(networks[next(iter(networks))]))
    ]
    colors = first_network.carriers.color
    extended_index = colors.index.union(list(override_colors.keys()))
    colors = colors.reindex(extended_index).fillna("black").to_dict()
    colors.update(override_colors)
    return colors


def main(snakemake):
    configure_logging(snakemake)
    run_name = snakemake.params.run
    scenarios = snakemake.params.scenarios
    planning_horizons = snakemake.params.planning_horizons
    override_colors = {}
    try:
        override_colors = snakemake.params.plotting["override_tech_colors"]
    except Exception:
        pass

    networks = process_networks(run_name, scenarios, planning_horizons)
    if not networks:
        logger.error("No networks could be loaded")
        os.makedirs(os.path.dirname(snakemake.output.dh_grid_plot), exist_ok=True)
        with open(snakemake.output.dh_grid_plot, "w") as f:
            f.write("No data available for plotting\n")
        return

    colors = get_colors(networks, override_colors)

    boosting_ratio_files = {}
    try:
        for scenario in scenarios:
            resources_path = os.path.join("resources", run_name, scenario)
            if os.path.exists(resources_path):
                candidates = [
                    f
                    for f in os.listdir(resources_path)
                    if f.startswith(
                        "ptes_discharger_temperature_boosting_ratio_profiles"
                    )
                ]
                if candidates:
                    boosting_ratio_files[scenario] = os.path.join(
                        resources_path, candidates[0]
                    )
    except Exception:
        pass

    plot_dh_grid_over_elec_prices(
        networks,
        snakemake.output.dh_grid_plot,
        colors,
        run_name,
        boosting_ratio_files=boosting_ratio_files if boosting_ratio_files else None,
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        os.chdir(os.path.join(os.path.dirname(__file__), "..", ".."))
        snakemake = mock_snakemake(
            "plot_dh_over_elec_prices",
            configfiles=["config/config.sysgf.yaml", "config/scenarios.sysgf.yaml"],
        )
    main(snakemake)
