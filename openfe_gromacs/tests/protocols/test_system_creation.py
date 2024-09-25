# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import gufe
import pytest
from openfe_gromacs.protocols.gromacs_md.md_methods import GromacsMDProtocol
from openfe_gromacs.protocols.gromacs_utils import system_creation
from openmm.app import GromacsGroFile


def test_interchange_gromacs(T4_protein_component, tmpdir):
    solvent = gufe.SolventComponent()
    smc_components = {}
    settings = GromacsMDProtocol.default_settings()
    omm_system, omm_topology, omm_positions = system_creation.create_openmm_system(
        settings, solvent, T4_protein_component, smc_components, tmpdir)
    omm_atom_names = [atom.name for atom in omm_topology.atoms()]
    interchange = system_creation.create_interchange(omm_system, omm_topology,
                                                     omm_positions,
                                                     smc_components)
    interchange_atom_names = [atom.name for atom in interchange.topology.atoms]
    interchange.to_gro(f"{tmpdir}/test.gro")
    gro_atom_names = GromacsGroFile(f"{tmpdir}/test.gro").atomNames
    assert len(omm_atom_names) == len(interchange_atom_names) == len(
        gro_atom_names)
    assert omm_atom_names == interchange_atom_names == gro_atom_names

    # check a few atom names to ensure these are not empty sets
    for atom_name in ("HA", "CH3", "CA", "CB"):
        assert atom_name in interchange_atom_names
