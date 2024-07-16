# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import sys
import gufe
import pytest
from unittest import mock
from numpy.testing import assert_allclose
from openff.units import unit
from openmm import unit as omm_unit
from openmm import NonbondedForce
from openff.units.openmm import to_openmm, from_openmm
from openmmtools.states import ThermodynamicState
from openmm import MonteCarloBarostat
from openfe_gromacs.protocols.gromacs_md.md_methods import (
    GromacsMDProtocol, GromacsMDSetupUnit, GromacsMDProtocolResult,
)
from openfe.protocols.openmm_utils.charge_generation import (
    HAS_NAGL, HAS_OPENEYE, HAS_ESPALOMA
)
import json
import openfe
from openfe.protocols import openmm_md
import pathlib
import logging

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