# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import gufe
import numpy as np
import pytest
from openmm import NonbondedForce
from openmm.app import GromacsGroFile, GromacsTopFile

from openfe_gromacs.protocols.gromacs_md.md_methods import GromacsMDProtocol
from openfe_gromacs.protocols.gromacs_utils import system_creation

# def test_interchange_gromacs(T4_protein_component, tmpdir):
#     solvent = gufe.SolventComponent()
#     smc_components = {}
#     settings = GromacsMDProtocol.default_settings()
#     omm_system, omm_topology, omm_positions = system_creation.create_openmm_system(
#         settings, solvent, T4_protein_component, smc_components, tmpdir
#     )
#     omm_atom_names = [atom.name for atom in omm_topology.atoms()]
#     interchange = system_creation.create_interchange(
#         omm_system, omm_topology, omm_positions, smc_components
#     )
#     interchange_atom_names = [atom.name for atom in interchange.topology.atoms]
#     interchange.to_gro(f"{tmpdir}/test.gro")
#     gro_atom_names = GromacsGroFile(f"{tmpdir}/test.gro").atomNames
#     assert len(omm_atom_names) == len(interchange_atom_names) == len(gro_atom_names)
#     assert omm_atom_names == interchange_atom_names == gro_atom_names
#
#     # check a few atom names to ensure these are not empty sets
#     for atom_name in ("HA", "CH3", "CA", "CB"):
#         assert atom_name in interchange_atom_names


def test_user_charges(ethane, tmpdir):
    solvent = gufe.SolventComponent()
    off_ethane = ethane.to_openff()
    off_ethane.assign_partial_charges(partial_charge_method="am1bcc")
    off_charges = off_ethane.partial_charges
    settings = GromacsMDProtocol.default_settings()
    smc_components = {ethane: off_ethane}
    omm_system, omm_topology, omm_positions = system_creation.create_openmm_system(
        settings, solvent, None, smc_components, tmpdir
    )

    interchange = system_creation.create_interchange(
        omm_system, omm_topology, omm_positions, smc_components
    )
    # Save to Gromacs .top file
    interchange.to_top(f"{tmpdir}/test.top")
    # Load Gromacs .top file back in as OpenMM system
    gromacs_system = GromacsTopFile(f"{tmpdir}/test.top").createSystem()
    # Get the partial charges of the ligand atoms
    nonbonded = [
        f for f in gromacs_system.getForces() if isinstance(f, NonbondedForce)
    ][0]
    gro_charges = []
    for i in range(len(off_charges)):
        charge, sigma, epsilon = nonbonded.getParticleParameters(i)
        gro_charges.append(charge._value)
    np.testing.assert_almost_equal(off_charges.m, gro_charges, decimal=5)
