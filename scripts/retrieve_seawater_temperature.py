# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Retrieve seawater temperature data from Copernicus Marine Service.

For the default workflow this downloads one modeled weather year per file.
For the small 2013 test cutout, it falls back to the bundled Zenodo file.
"""

import logging
import os

import copernicusmarine
import requests

from scripts._helpers import (
    configure_logging,
    set_scenario_config,
    update_config_from_wildcards,
)

logger = logging.getLogger(__name__)

# Set these once here to avoid relying on external Copernicus login state.
COPERNICUSMARINE_USERNAME = ""
COPERNICUSMARINE_PASSWORD = ""


def get_copernicus_credentials() -> dict[str, str]:
    username = COPERNICUSMARINE_USERNAME or None
    password = COPERNICUSMARINE_PASSWORD or None

    if (username is None) != (password is None):
        raise ValueError(
            "Set both COPERNICUSMARINE_USERNAME and "
            "COPERNICUSMARINE_PASSWORD, or leave both empty."
        )

    if username is None:
        return {}

    return {"username": username, "password": password}


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "retrieve_seawater_temperature",
            year="2019",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    update_config_from_wildcards(snakemake.config, snakemake.wildcards)

    if snakemake.params.default_cutout == "be-03-2013-era5":
        logger.info("Retrieving test-cutout seawater temperature data.")

        response = requests.get(snakemake.params.test_data_url, stream=True)
        response.raise_for_status()

        with open(snakemake.output.seawater_temperature, "wb") as output_file:
            for chunk in response.iter_content(chunk_size=8192):
                output_file.write(chunk)

        logger.info(
            "Successfully downloaded test-cutout seawater temperature data to %s",
            snakemake.output.seawater_temperature,
        )
    else:
        logger.info(
            "Downloading seawater temperature data for year %s",
            snakemake.wildcards.year,
        )

        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            **get_copernicus_credentials(),
            start_datetime=f"{snakemake.wildcards.year}-01-01",
            end_datetime=f"{int(snakemake.wildcards.year)}-12-31",
            minimum_longitude=-12,
            maximum_longitude=42,
            minimum_latitude=33,
            maximum_latitude=72,
            variables=["thetao"],
            minimum_depth=5,
            maximum_depth=15,
            output_filename=snakemake.output.seawater_temperature,
        )

        if not os.path.exists(snakemake.output.seawater_temperature):
            raise FileNotFoundError(
                "Failed to retrieve seawater temperature data and save to "
                f"{snakemake.output.seawater_temperature}. One reason might be "
                "missing Copernicus Marine login info."
            )

        logger.info(
            "Successfully downloaded seawater temperature data to %s",
            snakemake.output.seawater_temperature,
        )
