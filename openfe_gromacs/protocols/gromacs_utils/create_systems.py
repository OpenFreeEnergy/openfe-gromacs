import os
import warnings

import gufe
from openfe.protocols.openmm_utils import charge_generation, system_creation
from openfe.utils import without_oechem_backend
from openff.interchange import Interchange
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units.openmm import from_openmm, to_openmm
from openmmtools import forces

from openfe_gromacs.protocols.gromacs_md.md_settings import OpenFFPartialChargeSettings


def assign_partial_charges(
    charge_settings: OpenFFPartialChargeSettings,
    smc_components: dict[gufe.SmallMoleculeComponent, OFFMolecule],
) -> None:
    """
    Assign partial charges to SMCs.

    Parameters
    ----------
    charge_settings : OpenFFPartialChargeSettings
      Settings for controlling how the partial charges are assigned.
    smc_components : dict[gufe.SmallMoleculeComponent, openff.toolkit.Molecule]
      Dictionary of OpenFF Molecules to add, keyed by
      SmallMoleculeComponent.
    """
    for mol in smc_components.values():
        charge_generation.assign_offmol_partial_charges(
            offmol=mol,
            overwrite=False,
            method=charge_settings.partial_charge_method,
            toolkit_backend=charge_settings.off_toolkit_backend,
            generate_n_conformers=charge_settings.number_of_conformers,
            nagl_model=charge_settings.nagl_model,
        )


def create_openmm_system(
    solvent_comp,
    protein_comp,
    smc_components,
    partial_charge_settings,
    forcefield_settings,
    integrator_settings,
    thermo_settings,
    solvation_settings,
    output_settings_em,
    shared_basepath,
):
    """
    Creates the OpenMM system.

    Parameters
    ----------
    solvent_comp: gufe.SolventComponent
      The SolventComponent for the system
    protein_comp: gufe.ProteinComponent
      The ProteinComponent for the system
    smc_components: dict[gufe.SmallMoleculeComponent, OFFMolecule]
      A dictionary with SmallMoleculeComponents as keys and OpenFF molecules
      as values
    partial_charge_settings: OpenFFPartialChargeSettings
    forcefield_settings: FFSettingsOpenMM
    integrator_settings: IntegratorSettings
    thermo_settings: gufe.settings.ThermoSettings
    solvation_settings: SolvationSettings
    output_settings_em: EMOutputSettings
    shared_basepath : Pathlike, optional
      Where to run the calculation, defaults to current working directory

    Returns
    -------
    stateA_system: OpenMM system
    stateA_topology: OpenMM topology
    stateA_positions: Positions
    """
    # a. assign partial charges to smcs
    assign_partial_charges(partial_charge_settings, smc_components)

    # b. get a system generator
    if output_settings_em.forcefield_cache is not None:
        ffcache = shared_basepath / output_settings_em.forcefield_cache
    else:
        ffcache = None

    # Note: we block out the oechem backend for all systemgenerator
    # linked operations to avoid any smiles operations that can
    # go wrong when doing rdkit->OEchem roundtripping
    with without_oechem_backend():
        system_generator = system_creation.get_system_generator(
            forcefield_settings=forcefield_settings,
            integrator_settings=integrator_settings,
            thermo_settings=thermo_settings,
            cache=ffcache,
            has_solvent=solvent_comp is not None,
        )

        # Force creation of smc templates so we can solvate later
        for mol in smc_components.values():
            system_generator.create_system(
                mol.to_topology().to_openmm(), molecules=[mol]
            )

        # c. get OpenMM Modeller + a resids dictionary for each component
        stateA_modeller, comp_resids = system_creation.get_omm_modeller(
            protein_comp=protein_comp,
            solvent_comp=solvent_comp,
            small_mols=smc_components,
            omm_forcefield=system_generator.forcefield,
            solvent_settings=solvation_settings,
        )

        # d. get topology & positions
        # Note: roundtrip positions to remove vec3 issues
        stateA_topology = stateA_modeller.getTopology()
        stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

        # e. create the stateA System
        stateA_system = system_generator.create_system(
            stateA_topology,
            molecules=smc_components.values(),
        )
    return stateA_system, stateA_topology, stateA_positions


def create_interchange(
    stateA_system,
    stateA_topology,
    stateA_positions,
    smc_components,
):
    """
    Creates the Interchange object for the system.

    Parameters
    ----------
    stateA_system: OpenMM system
    stateA_topology: OpenMM topology
    stateA_positions: Positions
    smc_components: dict[gufe.SmallMoleculeComponent, OFFMolecule]
      A dictionary with SmallMoleculeComponents as keys and OpenFF molecules
      as values

    Returns
    -------
    stateA_interchange: Interchange object
      The interchange object of the system.
    """
    # Set the environment variable for using the experimental interchange
    # functionality `from_openmm` and raise a warning
    os.environ["INTERCHANGE_EXPERIMENTAL"] = "1"
    war = (
        "Environment variable INTERCHANGE_EXPERIMENTAL=1 is set for using "
        "the interchange functionality 'from_openmm' which is not well "
        "tested yet."
    )
    warnings.warn(war)

    try:
        barostat_idx, barostat = forces.find_forces(
            stateA_system, ".*Barostat.*", only_one=True
        )
        stateA_system.removeForce(barostat_idx)

    except forces.NoForceFoundError:
        pass

    stateA_interchange = Interchange.from_openmm(
        topology=stateA_topology,
        system=stateA_system,
        positions=stateA_positions,
    )

    for molecule_index, molecule in enumerate(stateA_interchange.topology.molecules):
        for atom in molecule.atoms:
            atom.metadata["residue_number"] = molecule_index + 1
        if len([*smc_components.values()]) > 0:
            if molecule.n_atoms == [*smc_components.values()][0].n_atoms:
                for atom in molecule.atoms:
                    atom.metadata["residue_name"] = "UNK"

    return stateA_interchange
