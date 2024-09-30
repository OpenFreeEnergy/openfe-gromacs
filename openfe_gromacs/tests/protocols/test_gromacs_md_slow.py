# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs

import gufe
import pytest
from openff.units import unit as off_unit

from openfe_gromacs.protocols.gromacs_md.md_methods import GromacsMDProtocol


@pytest.mark.integration
def test_protein_ligand_md(toluene_complex_system, tmp_path_factory):
    settings = GromacsMDProtocol.default_settings()
    settings.simulation_settings_em.nsteps = 10
    settings.simulation_settings_nvt.nsteps = 10
    settings.simulation_settings_npt.nsteps = 10000
    protocol = GromacsMDProtocol(
        settings=settings,
    )

    dag = protocol.create(
        stateA=toluene_complex_system,
        stateB=toluene_complex_system,
        mapping=None,
    )

    shared_temp = tmp_path_factory.mktemp("shared")
    scratch_temp = tmp_path_factory.mktemp("scratch")
    gufe.protocols.execute_DAG(
        dag,
        shared_basedir=shared_temp,
        scratch_basedir=scratch_temp,
        keep_shared=False,
        n_retries=3,
    )
