# SPDX-FileCopyrightText: : 2024- The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
This script modifies district heating shares based on eGo^N data for NUTS3
regions in Germany.

Inputs:
    - resources/heating_technologies_nuts3.geojson: Path to the GeoJSON file containing heating technologies data for NUTS3 regions.
    - resources/regions_onshore.geojson: Path to the GeoJSON file containing onshore regions data.
    - resources/district_heat_share.csv: Path to the CSV file containing district heating shares.

Outputs:
    - resources/updated_district_heat_share.csv: Path to the CSV file where the updated district heating shares will be saved.

Parameters
----------
    - sector.district_heating["potential"]: Maximum potential district heating share.
    - sector.district_heating["progress"]: Progress of district heating share over planning horizons.
    - wildcards.planning_horizons: Planning horizon year.
"""

import logging

import geopandas as gpd
import pandas as pd

from scripts._helpers import mock_snakemake

logger = logging.getLogger(__name__)


def cluster_egon(heat_techs, regions_onshore):
    """
    Map NUTS3 regions of egon data to corresponding clusters according to
    maximum overlap.

    Inputs:
        - heat_techs (GeoDataFrame): GeoDataFrame containing heating technologies data for NUTS3 regions.
        - regions_onshore (GeoDataFrame): GeoDataFrame containing onshore regions data of network clusters.

    Outputs:
        - GeoDataFrame: Updated GeoDataFrame with NUTS3 regions aggregated according to cluster structure.
    """

    regions_onshore.set_index("name", inplace=True)

    # Map NUTS3 regions of egon data to corresponding clusters according to maximum overlap

    heat_techs["cluster"] = heat_techs.apply(
        lambda x: regions_onshore.geometry.intersection(x.geometry).area.idxmax(),
        axis=1,
    )

    # Group and aggregate by cluster
    heat_techs_clustered = heat_techs.groupby("cluster").sum(numeric_only=True)

    return heat_techs_clustered


def update_district_heat_share(
    heat_techs_clustered: gpd.GeoDataFrame,
    dh_shares: pd.DataFrame,
    urban_fraction: pd.Series,
    max_dh_share: float,
    progress: float,
) -> pd.DataFrame:
    """
    Update district heating demands of clusters according to shares in eGo^N
    data on NUTS3 level for Germany taking into account expansion of systems.

    Parameters
    ----------
    heat_techs_clustered : geopandas.GeoDataFrame
        GeoDataFrame containing clustered heating technologies data.
    dh_shares : pandas.DataFrame
        DataFrame containing district heating shares and urban fractions to be updated.
    urban_fraction : pandas.Series
        Series representing the urban fraction of district heating shares.
    max_dh_share : float
        Maximum potential district heating share.
    progress : float
        Progress factor for district heating share expansion.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with adjusted district heating shares and urban fractions.
    """

    nodal_dh_shares = heat_techs_clustered[
        "Fernwaerme"
    ] / heat_techs_clustered.drop(  # Fernwaerme is the German term for district heating
        "pop", axis=1
    ).sum(
        axis=1
    )

    diff = ((urban_fraction * max_dh_share) - nodal_dh_shares).clip(lower=0).dropna()
    nodal_dh_shares += diff * progress
    nodal_dh_shares = nodal_dh_shares.filter(regex="DE")
    dh_shares.loc[nodal_dh_shares.index, "district fraction of node"] = nodal_dh_shares
    dh_shares.loc[nodal_dh_shares.index, "urban fraction"] = pd.concat(
        [urban_fraction.loc[nodal_dh_shares.index], nodal_dh_shares], axis=1
    ).max(axis=1)

    return dh_shares


if __name__ == "__main__":
    if "snakemake" not in globals():
        snakemake = mock_snakemake(
            "modify_district_heat_share",
            simpl="",
            clusters=44,
            opts="",
            ll="vopt",
            sector_opts="none",
            planning_horizons="2020",
            run="KN2045_Mix",
        )

    logging.basicConfig(level=snakemake.config["logging"]["level"])
    logger.info("Updating district heating shares with egon data")

    heat_techs = gpd.read_file(snakemake.input.heating_technologies_nuts3)
    regions_onshore = gpd.read_file(snakemake.input.regions_onshore)
    dh_shares = pd.read_csv(snakemake.input.district_heat_share, index_col=0)

    heat_techs_clustered = cluster_egon(heat_techs, regions_onshore)

    urban_fraction = dh_shares["urban fraction"]
    max_dh_share = snakemake.params.district_heating["potential"]
    pop_layout = pd.read_csv(snakemake.input.pop_layout, index_col=0)
    if isinstance(max_dh_share, dict):
        other_countries = set(pop_layout.ct.unique()).difference(max_dh_share.keys())
        if other_countries:
            default_value = max_dh_share.get("default")
            if default_value is None:
                raise ValueError(
                    "No default district heating potential was provided in the config."
                )
            logger.warning(
                "Some countries do not have a district heating potential defined. "
                f"Using default value {default_value:.2%} for these countries."
            )
            max_dh_share = {
                **max_dh_share,
                **{ct: default_value for ct in other_countries},
            }
        max_dh_share = pop_layout.ct.map(max_dh_share)
    progress = snakemake.params.district_heating["progress"][
        int(snakemake.wildcards.planning_horizons)
    ]

    dh_shares = update_district_heat_share(
        heat_techs_clustered, dh_shares, urban_fraction, max_dh_share, progress
    )

    dh_shares.to_csv(snakemake.output.district_heat_share)
