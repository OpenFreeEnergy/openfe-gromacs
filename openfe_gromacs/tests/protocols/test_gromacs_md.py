# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import json
import pathlib
import sys
from unittest import mock

import gmxapi as gmx
import gufe
import pytest
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_ESPALOMA,
    HAS_NAGL,
    HAS_OPENEYE,
)

import openfe_gromacs
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


def test_no_SolventComponent(benzene_vacuum_system, tmpdir):
    settings = GromacsMDProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    p = GromacsMDProtocol(
        settings=settings,
    )

    dag = p.create(
        stateA=benzene_vacuum_system,
        stateB=benzene_vacuum_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    errmsg = "No SolventComponent provided. This protocol currently"
    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            dag_unit.run(dry=True)


def test_no_constraints(benzene_system, tmpdir):
    settings = GromacsMDProtocol.default_settings()
    settings.forcefield_settings.constraints = "hbonds"

    p = GromacsMDProtocol(
        settings=settings,
    )

    dag = p.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    errmsg = "No constraints are allowed in this step of creating the Gromacs"
    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            dag_unit.run(dry=True)


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


def test_dry_run_ffcache_none(benzene_system, monkeypatch, tmpdir):
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")
    settings = GromacsMDProtocol.default_settings()
    settings.output_settings_em.forcefield_cache = None

    protocol = GromacsMDProtocol(
        settings=settings,
    )
    assert protocol.settings.output_settings_em.forcefield_cache is None

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )
    dag_unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        dag_unit.run(dry=True)


def test_dry_many_molecules_solvent(benzene_many_solv_system, monkeypatch, tmpdir):
    """
    A basic test flushing "will it work if you pass multiple molecules"
    """
    monkeypatch.setenv("INTERCHANGE_EXPERIMENTAL", "1")
    settings = GromacsMDProtocol.default_settings()

    protocol = GromacsMDProtocol(
        settings=settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_many_solv_system,
        stateB=benzene_many_solv_system,
        mapping=None,
    )
    unit = list(dag.protocol_units)[0]

    with tmpdir.as_cwd():
        system = unit.run(dry=True)


class TestProtocolResult:
    @pytest.fixture()
    def protocolresult(self, md_json):
        d = json.loads(md_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = openfe_gromacs.ProtocolResult.from_dict(d["protocol_result"])

        return pr

    def test_reload_protocol_result(self, md_json):
        d = json.loads(md_json, cls=gufe.tokenization.JSON_HANDLER.decoder)

        pr = GromacsMDProtocolResult.from_dict(d["protocol_result"])

        assert pr

    def test_get_estimate(self, protocolresult):
        est = protocolresult.get_estimate()

        assert est is None

    def test_get_uncertainty(self, protocolresult):
        est = protocolresult.get_uncertainty()

        assert est is None

    def test_get_gro_filename(self, protocolresult):
        gro = protocolresult.get_gro_filename()

        assert isinstance(gro, list)
        assert isinstance(gro[0], pathlib.Path)

    def test_get_top_filename(self, protocolresult):
        top = protocolresult.get_top_filename()

        assert isinstance(top, list)
        assert isinstance(top[0], pathlib.Path)

    def test_get_mdp_filenames(self, protocolresult):
        mdps = protocolresult.get_mdp_filenames()

        assert isinstance(mdps, list)
        assert isinstance(mdps[0], list)
        assert isinstance(mdps[0][0], pathlib.Path)


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
    assert gro
    grompp_input_files = {
        "-f": res.get_mdp_filenames()[0],
        "-c": res.get_gro_filename(),
        "-p": res.get_top_filename(),
    }

    grompp = gmx.commandline_operation(
        "gmx", "grompp", input_files=grompp_input_files, output_files={"-o": "em.tpr"}
    )
