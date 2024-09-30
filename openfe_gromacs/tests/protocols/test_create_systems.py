# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
import gufe
import numpy as np
from openmm import NonbondedForce
from openmm.app import GromacsGroFile, GromacsTopFile

from openfe_gromacs.protocols.gromacs_md.md_methods import GromacsMDProtocol
from openfe_gromacs.protocols.gromacs_utils import create_systems


def test_interchange_gromacs(alanine_dipeptide_component, tmpdir):
    solvent = gufe.SolventComponent()
    smc_components = {}
    prot_settings = GromacsMDProtocol.default_settings()

    omm_system, omm_topology, omm_positions = create_systems.create_openmm_system(
        solvent,
        alanine_dipeptide_component,
        smc_components,
        prot_settings.partial_charge_settings,
        prot_settings.forcefield_settings,
        prot_settings.integrator_settings,
        prot_settings.thermo_settings,
        prot_settings.solvation_settings,
        prot_settings.output_settings_em,
        tmpdir,
    )
    omm_atom_names = [atom.name for atom in omm_topology.atoms()]
    interchange = create_systems.create_interchange(
        omm_system, omm_topology, omm_positions, smc_components
    )
    interchange_atom_names = [atom.name for atom in interchange.topology.atoms]

    interchange.to_gro(f"{tmpdir}/test.gro")
    gro_atom_names = GromacsGroFile(f"{tmpdir}/test.gro").atomNames

    # Testing that the number of atoms is the same
    assert len(omm_atom_names) == len(interchange_atom_names) == len(gro_atom_names)
    # Testing that atom names are the same
    assert omm_atom_names == interchange_atom_names == gro_atom_names
    # Testing that residue names are the same
    interchange_res_names = [atom.metadata["residue_name"] for atom in
                             interchange.topology.atoms]
    gro_res_names = GromacsGroFile(f"{tmpdir}/test.gro").residueNames
    combined_res_names = interchange_res_names + gro_res_names
    assert len(set(combined_res_names)) == len(set(gro_res_names))
    # Testing that residue numbers are the same
    interchange_res_numbers = [atom.metadata["residue_number"] for atom in
                               interchange.topology.atoms]
    with open(f"{tmpdir}/test.gro") as f:
        gromacs_res_numbers = [int(line[:5].strip()) for line in f.readlines()[2:-1]]
    assert interchange_res_numbers == gromacs_res_numbers


    # check a few atom names to ensure these are not empty sets
    for atom_name in ("HA", "CH3", "CA", "CB"):
        assert atom_name in interchange_atom_names


# def test_user_charges(CN_molecule, tmpdir):
#     solvent = gufe.SolventComponent()
#     off_cn = CN_molecule.to_openff()
#     off_cn.assign_partial_charges(partial_charge_method="am1-mulliken")
#     off_charges = off_cn.partial_charges
#     prot_settings = GromacsMDProtocol.default_settings()
#     smc_components = {CN_molecule: off_cn}
#     omm_system, omm_topology, omm_positions = create_systems.create_openmm_system(
#         solvent,
#         None,
#         smc_components,
#         prot_settings.partial_charge_settings,
#         prot_settings.forcefield_settings,
#         prot_settings.integrator_settings,
#         prot_settings.thermo_settings,
#         prot_settings.solvation_settings,
#         prot_settings.output_settings_em,
#         tmpdir,
#     )
#
#     interchange = create_systems.create_interchange(
#         omm_system, omm_topology, omm_positions, smc_components
#     )
#     # Save to Gromacs .top file
#     interchange.to_top(f"{tmpdir}/test.top")
#     # Load Gromacs .top file back in as OpenMM system
#     gromacs_system = GromacsTopFile(f"{tmpdir}/test.top").createSystem()
#     # Get the partial charges of the ligand atoms
#     nonbonded = [
#         f for f in gromacs_system.getForces() if isinstance(f, NonbondedForce)
#     ][0]
#     gro_charges = []
#     for i in range(len(off_charges)):
#         charge, sigma, epsilon = nonbonded.getParticleParameters(i)
#         gro_charges.append(charge._value)
#     np.testing.assert_almost_equal(off_charges.m, gro_charges, decimal=5)
