# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import json
import logging
import pathlib
import sys
from unittest import mock

import gufe
import openfe
import pytest
from numpy.testing import assert_allclose
from openfe.protocols import openmm_md
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_ESPALOMA,
    HAS_NAGL,
    HAS_OPENEYE,
)
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmm import MonteCarloBarostat, NonbondedForce
from openmm import unit as omm_unit
from openmmtools.states import ThermodynamicState

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
