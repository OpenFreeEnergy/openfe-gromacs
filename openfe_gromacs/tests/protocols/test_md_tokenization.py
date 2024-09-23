# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import json

import gufe
import openfe
import pytest
from gufe.tests.test_tokenization import GufeTokenizableTestsMixin

from openfe_gromacs.protocols import gromacs_md


@pytest.fixture
def protocol():
    return gromacs_md.GromacsMDProtocol(gromacs_md.GromacsMDProtocol.default_settings())


@pytest.fixture
def protocol_units(protocol, benzene_system):
    pus = protocol.create(
        stateA=benzene_system,
        stateB=openfe.ChemicalSystem({"solvent": openfe.SolventComponent()}),
        mapping=None,
    )
    return list(pus.protocol_units)


@pytest.fixture
def setup_unit(protocol_units):
    for pu in protocol_units:
        if isinstance(pu, gromacs_md.GromacsMDSetupUnit):
            return pu


@pytest.fixture
def protocol_result(md_json):
    d = json.loads(md_json, cls=gufe.tokenization.JSON_HANDLER.decoder)
    pr = gromacs_md.GromacsMDProtocolResult.from_dict(d["protocol_result"])
    return pr


class TestGromacsMDProtocol(GufeTokenizableTestsMixin):
    cls = gromacs_md.GromacsMDProtocol
    key = "GromacsMDProtocol-d550ac68c7de2d38e12ee574358ae6c8"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol):
        return protocol


class TestMDSetupUnit(GufeTokenizableTestsMixin):
    cls = gromacs_md.GromacsMDSetupUnit
    repr = "GromacsMDSetupUnit(Solvent MD SmallMoleculeComponent: benzene)"
    key = None

    @pytest.fixture()
    def instance(self, setup_unit):
        return setup_unit

    def test_key_stable(self):
        pytest.skip()


class TestGromacsMDProtocolResult(GufeTokenizableTestsMixin):
    cls = gromacs_md.GromacsMDProtocolResult
    key = "GromacsMDProtocolResult-d9a0913916079579a217826054941b4f"
    repr = f"<{key}>"

    @pytest.fixture()
    def instance(self, protocol_result):
        return protocol_result
