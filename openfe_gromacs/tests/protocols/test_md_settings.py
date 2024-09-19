# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import json
import pathlib
from unittest import mock

import gufe
import pytest
from openff.units import unit

import openfe_gromacs
from openfe_gromacs.protocols.gromacs_md.md_methods import (
    GromacsMDProtocol,
    GromacsMDProtocolResult,
    GromacsMDSetupUnit,
)


def test_settings_pos_or_zero_error():
    errmsg = (
        "Settings nsteps, nstlist, rlist, rcoulomb, rvdw, ewald_rtol, "
        "shake_tol, lincs_order, and lincs_iter must be zero or posi"
    )
    settings = GromacsMDProtocol.default_settings()
    with pytest.raises(ValueError, match=errmsg):
        settings.simulation_settings_em.nsteps = -100


def test_must_be_positive_error():
    errmsg = "mass_repartition_factor must be positive value"
    settings = GromacsMDProtocol.default_settings()
    with pytest.raises(ValueError, match=errmsg):
        settings.simulation_settings_nvt.dt = 0 * unit.picosecond


def test_must_be_between_3_12_error():
    errmsg = "pme_order must be between 3 and 12"
    settings = GromacsMDProtocol.default_settings()
    with pytest.raises(ValueError, match=errmsg):
        settings.simulation_settings_nvt.pme_order = 13
