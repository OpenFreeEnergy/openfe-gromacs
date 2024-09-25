# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import json
import pathlib
from unittest import mock

import gufe
import pytest
from openff.units import unit as off_unit

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
    for u in dag1.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])
    for u in dag2.protocol_units:
        repeat_ids.add(u.inputs["repeat_id"])
    # This should result in 4 repeat ids per DAG, 1 GromacsMDSetupUnit and
    # 3 GromacsMDRunUnits
    assert len(repeat_ids) == 8


def test_no_SolventComponent(benzene_vacuum_system, tmpdir):
    settings = GromacsMDProtocol.default_settings()
    settings.forcefield_settings.nonbonded_method = "nocutoff"
    p = GromacsMDProtocol(
        settings=settings,
    )

    errmsg = "No SolventComponent provided. This protocol currently"
    with tmpdir.as_cwd():
        with pytest.raises(ValueError, match=errmsg):
            p.create(
                stateA=benzene_vacuum_system,
                stateB=benzene_vacuum_system,
                mapping=None,
            )


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


def test_unit_tagging_setup_unit(solvent_protocol_dag, tmpdir):
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
            if type(u) is GromacsMDSetupUnit:
                ret = u.execute(context=gufe.Context(tmpdir, tmpdir))
                results.append(ret)
    repeats = set()
    for ret in results:
        assert isinstance(ret, gufe.ProtocolUnitResult)
        assert ret.outputs["generation"] == 0
        repeats.add(ret.outputs["repeat_id"])
    # repeats are random ints, so check we got 1 individual number
    # (there's only one Setup Unit, even with two repeats)
    assert len(repeats) == 1


def test_dry_run_ffcache_none(benzene_system, tmp_path_factory):
    settings = GromacsMDProtocol.default_settings()
    settings.output_settings_em.forcefield_cache = None
    settings.simulation_settings_em.nsteps = 10
    settings.simulation_settings_nvt.nsteps = 10
    settings.simulation_settings_npt.nsteps = 1
    settings.simulation_settings_em.rcoulomb = 1.0 * off_unit.nanometer
    settings.simulation_settings_nvt.rcoulomb = 1.0 * off_unit.nanometer
    settings.simulation_settings_npt.rcoulomb = 1.0 * off_unit.nanometer
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

    shared_temp = tmp_path_factory.mktemp("shared")
    scratch_temp = tmp_path_factory.mktemp("scratch")
    gufe.protocols.execute_DAG(
        dag,
        shared_basedir=shared_temp,
        scratch_basedir=scratch_temp,
        keep_shared=False,
        n_retries=3,
    )


def test_dry_many_molecules_solvent(
benzene_many_solv_system, tmp_path_factory
):
    """
    A basic test flushing "will it work if you pass multiple molecules"
    """
    with pytest.warns(UserWarning, match="Environment variabl"):
        settings = GromacsMDProtocol.default_settings()
        # Only run an EM, no MD, to make the test faster
        settings.simulation_settings_em.nsteps = 1
        settings.simulation_settings_nvt.nsteps = 0
        settings.simulation_settings_npt.nsteps = 0
        settings.simulation_settings_em.rcoulomb = 1.0 * off_unit.nanometer
        protocol = GromacsMDProtocol(
            settings=settings,
        )

        # create DAG from protocol and take first (and only) work unit from within
        dag = protocol.create(
            stateA=benzene_many_solv_system,
            stateB=benzene_many_solv_system,
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


def test_gather(benzene_system, tmp_path_factory):
    # check .gather behaves as expected
    # Can't use mock.path since we need the outputs from the Setup Unit
    # to execute the Run Unit
    settings = GromacsMDProtocol.default_settings()
    settings.output_settings_em.forcefield_cache = None
    settings.simulation_settings_em.nsteps = 10
    settings.simulation_settings_nvt.nsteps = 10
    settings.simulation_settings_npt.nsteps = 1
    settings.simulation_settings_em.rcoulomb = 1.0 * off_unit.nanometer
    settings.simulation_settings_nvt.rcoulomb = 1.0 * off_unit.nanometer
    settings.simulation_settings_npt.rcoulomb = 1.0 * off_unit.nanometer
    protocol = GromacsMDProtocol(
        settings=settings,
    )

    # create DAG from protocol and take first (and only) work unit from within
    dag = protocol.create(
        stateA=benzene_system,
        stateB=benzene_system,
        mapping=None,
    )

    shared_temp = tmp_path_factory.mktemp("shared")
    scratch_temp = tmp_path_factory.mktemp("scratch")
    dagres = gufe.protocols.execute_DAG(
        dag,
        shared_basedir=shared_temp,
        scratch_basedir=scratch_temp,
        keep_shared=False,
        n_retries=3,
    )
    prot = GromacsMDProtocol(
        settings=settings
    )

    res = prot.gather([dagres])

    assert isinstance(res, GromacsMDProtocolResult)


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
        assert isinstance(mdps[0], dict)
        assert all(isinstance(mdp, pathlib.Path) for mdp in mdps[0].values())

    def test_get_filenames_em(self, protocolresult):
        dict_file_path = protocolresult.get_filenames_em()
        assert isinstance(dict_file_path, dict)
        assert len(dict_file_path) == 7
        for name, file_path in dict_file_path.items():
            assert isinstance(file_path, list)
            assert len(file_path) == 1
            assert isinstance(file_path[0], pathlib.Path)

    def test_get_gro_em_filename(self, protocolresult):
        file_path = protocolresult.get_gro_em_filename()

        assert isinstance(file_path, list)
        assert isinstance(file_path[0], pathlib.Path)

    def test_get_xtc_em_filename(self, protocolresult):
        file_path = protocolresult.get_xtc_em_filename()

        assert isinstance(file_path, list)
        assert isinstance(file_path[0], pathlib.Path)

    def test_get_filenames_nvt(self, protocolresult):
        dict_file_path = protocolresult.get_filenames_nvt()
        assert isinstance(dict_file_path, dict)
        assert len(dict_file_path) == 7
        for name, file_path in dict_file_path.items():
            assert isinstance(file_path, list)
            assert len(file_path) == 1
            assert isinstance(file_path[0], pathlib.Path)

    def test_get_gro_nvt_filename(self, protocolresult):
        file_path = protocolresult.get_gro_nvt_filename()

        assert isinstance(file_path, list)
        assert isinstance(file_path[0], pathlib.Path)

    def test_get_xtc_nvt_filename(self, protocolresult):
        file_path = protocolresult.get_xtc_nvt_filename()

        assert isinstance(file_path, list)
        assert isinstance(file_path[0], pathlib.Path)

    def test_get_filenames_npt(self, protocolresult):
        dict_file_path = protocolresult.get_filenames_npt()
        assert isinstance(dict_file_path, dict)
        assert len(dict_file_path) == 7
        for name, file_path in dict_file_path.items():
            assert isinstance(file_path, list)
            assert len(file_path) == 1
            assert isinstance(file_path[0], pathlib.Path)

    def test_get_gro_npt_filename(self, protocolresult):
        file_path = protocolresult.get_gro_npt_filename()

        assert isinstance(file_path, list)
        assert isinstance(file_path[0], pathlib.Path)

    def test_get_xtc_npt_filename(self, protocolresult):
        file_path = protocolresult.get_xtc_npt_filename()

        assert isinstance(file_path, list)
        assert isinstance(file_path[0], pathlib.Path)
