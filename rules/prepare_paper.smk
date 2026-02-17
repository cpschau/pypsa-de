# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT


rule create_paper_plots:
    input:
        delta_system_costs=expand(
            "results/" + PREFIX + "/plots/delta_system_costs_{planning_horizons}.pdf",
            **config["scenario"],
        )

