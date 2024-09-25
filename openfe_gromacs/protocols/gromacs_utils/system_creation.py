import os
import warnings
from openfe_gromacs.protocols.gromacs_md.md_settings import (
    IntegratorSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)
from openff.interchange import Interchange
from openff.toolkit.topology import Molecule as OFFMolecule
from gufe import ChemicalSystem, SmallMoleculeComponent, settings
from openff.units.openmm import from_openmm, to_openmm
from openmmtools import forces
from openfe.protocols.openmm_utils import charge_generation, system_creation
from openfe.utils import without_oechem_backend


def assign_partial_charges(
    charge_settings: OpenFFPartialChargeSettings,
    smc_components: dict[SmallMoleculeComponent, OFFMolecule],
) -> None:
    """
    Assign partial charges to SMCs.

    Parameters
    ----------
    charge_settings : OpenFFPartialChargeSettings
      Settings for controlling how the partial charges are assigned.
    smc_components : dict[SmallMoleculeComponent, openff.toolkit.Molecule]
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


def create_interchange(settings, solvent_comp, protein_comp, small_mols, shared_basepath):
    """
    Creates the Interchange object for the system.

    Parameters
    ----------
    settings: dict
      Dictionary of all the settings
    components: dict
      Dictionary of the components of the settings, solvent, protein, and
      small molecules
    shared_basepath : Pathlike, optional
      Where to run the calculation, defaults to current working directory

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
    # Create the stateA system
    # Create a dictionary of OFFMol for each SMC for bookkeeping
    smc_components: dict[SmallMoleculeComponent, OFFMolecule]
    smc_components = {i: i.to_openff() for i in small_mols}

    # a. assign partial charges to smcs
    assign_partial_charges(settings["charge_settings"], smc_components)

    # b. get a system generator
    if settings["output_settings_em"].forcefield_cache is not None:
        ffcache = shared_basepath / settings[
            "output_settings_em"].forcefield_cache
    else:
        ffcache = None

    # Note: we block out the oechem backend for all systemgenerator
    # linked operations to avoid any smiles operations that can
    # go wrong when doing rdkit->OEchem roundtripping
    with without_oechem_backend():
        system_generator = system_creation.get_system_generator(
            forcefield_settings=settings["forcefield_settings"],
            integrator_settings=settings["integrator_settings"],
            thermo_settings=settings["thermo_settings"],
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
            solvent_settings=settings["solvation_settings"],
        )

        # d. get topology & positions
        # Note: roundtrip positions to remove vec3 issues
        stateA_topology = stateA_modeller.getTopology()
        stateA_positions = to_openmm(
            from_openmm(stateA_modeller.getPositions()))

        # e. create the stateA System
        stateA_system = system_generator.create_system(
            stateA_topology,
            molecules=[s.to_openff() for s in small_mols],
        )

        # 3. Create the Interchange object
        barostat_idx, barostat = forces.find_forces(
            stateA_system, ".*Barostat.*", only_one=True
        )
        stateA_system.removeForce(barostat_idx)

        stateA_interchange = Interchange.from_openmm(
            topology=stateA_topology,
            system=stateA_system,
            positions=stateA_positions,
        )

        for molecule_index, molecule in enumerate(
                stateA_interchange.topology.molecules
        ):
            for atom in molecule.atoms:
                atom.metadata["residue_number"] = molecule_index + 1
            if len([*smc_components.values()]) > 0:
                if molecule.n_atoms == [*smc_components.values()][0].n_atoms:
                    for atom in molecule.atoms:
                        atom.metadata["residue_name"] = "UNK"

    return stateA_interchange
