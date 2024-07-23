# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
from unittest import mock

import gmxapi as gmx
import gufe
import pytest

from openfe_gromacs.protocols.gromacs_md.md_methods import (
    GromacsMDProtocol,
    GromacsMDProtocolResult,
    GromacsMDSetupUnit,
)


def test_create_default_settings():
    settings = GromacsMDProtocol.default_settings()

    assert settings


def test_create_default_protocol():
    # this is roughly how it should be created
    protocol = GromacsMDProtocol(
        settings=GromacsMDProtocol.default_settings(),
    )

    assert protocol


def test_serialize_protocol():
    protocol = GromacsMDProtocol(
        settings=GromacsMDProtocol.default_settings(),
    )

    ser = protocol.to_dict()

    ret = GromacsMDProtocol.from_dict(ser)

    assert protocol == ret


def test_create_independent_repeat_ids(benzene_system):
    # if we create two dags each with 3 repeats, they should give 6 repeat_ids
    # this allows multiple DAGs in flight for one Transformation that don't clash on gather
    settings = GromacsMDProtocol.default_settings()
    # Default protocol is 1 repeat, change to 3 repeats
    settings.protocol_repeats = 3
    protocol = GromacsMDProtocol(
        settings=settings,
    )
    dag1 = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag2 = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )

    repeat_ids = set()
    u: GromacsMDSetupUnit
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])

    assert len(repeat_ids) == 6


# Add tests for vacuum simulations?


@pytest.fixture
def solvent_protocol_dag(benzene_system):
    settings = GromacsMDProtocol.default_settings()
    settings.protocol_repeats = 2
    protocol = GromacsMDProtocol(
        settings=settings,
    )

    return protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )


def test_unit_tagging(solvent_protocol_dag, tmpdir):
    # test that executing the Units includes correct generation and repeat info
    dag_units = solvent_protocol_dag.protocol_units
    with mock.patch(
        "openfe_gromacs.protocols.gromacs_md.md_methods.GromacsMDSetupUnit.run",
        return_value={
            "system_gro": "system.gro",
            "system_top": "system.top",
            "mdp_files": ["em.mdp", "nvt.mdp", "npt.mdp"],
        },
    ):
        results = []
        for u in dag_units:
            ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
            results.append(ret)

    repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs["generation"] == 0
        repeats.add(ret.outputs["repeat_id"])
    # repeats are random ints, so check we got 2 individual numbers
    assert len(repeats) == 2


def test_gather(solvent_protocol_dag, tmpdir):
    # check .gather behaves as expected
    with mock.patch(
        "openfe_gromacs.protocols.gromacs_md.md_methods.GromacsMDSetupUnit.run",
        return_value={
            "system_gro": "system.gro",
            "system_top": "system.top",
            "mdp_files": ["em.mdp", "nvt.mdp", "npt.mdp"],
        },
    ):
        dagres = gufe.protocols.execute_DAG(
            solvent_protocol_dag,
            shared_basedir=tmpdir,
            scratch_basedir=tmpdir,
            keep_shared=True,
        )

    settings = GromacsMDProtocol.default_settings()
    settings.protocol_repeats = 2
    prot = GromacsMDProtocol(settings=settings)

    res = prot.gather([dagres])

    assert isinstance(res, GromacsMDProtocolResult)


def test_grompp_on_output(solvent_protocol_dag, tmpdir):
    with mock.patch(
        "openfe_gromacs.protocols.gromacs_md.md_methods.GromacsMDSetupUnit.run",
        return_value={
            "system_gro": "system.gro",
            "system_top": "system.top",
            "mdp_files": ["em.mdp", "nvt.mdp", "npt.mdp"],
        },
    ):
        dagres = gufe.protocols.execute_DAG(
            solvent_protocol_dag,
            shared_basedir=tmpdir,
            scratch_basedir=tmpdir,
            keep_shared=True,
        )

    settings = GromacsMDProtocol.default_settings()
    settings.protocol_repeats = 2
    prot = GromacsMDProtocol(settings=settings)

    res = prot.gather([dagres])
    gro = res.get_gro_filename()
    print(gro)
    assert gro
    grompp_input_files = {
        "-f": res.get_mdp_filenames()[0],
        "-c": res.get_gro_filename(),
        "-p": res.get_top_filename(),
    }

    grompp = gmx.commandline_operation(
        "gmx", "grompp", input_files=grompp_input_files, output_files={"-o": "em.tpr"}
    )
