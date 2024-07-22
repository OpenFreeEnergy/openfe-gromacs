# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Gromacs MD Protocol --- :mod:`openfe.gromacs.protocols.gromacs_md.md_methods`
================================================================================

This module implements the necessary methodology tools to run an MD
simulation using OpenMM tools and Gromacs.

"""
from __future__ import annotations

import logging
import pathlib
import uuid
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional

import gufe
import openmm
import pint
from gufe import (
    ChemicalSystem,
    ProteinComponent,
    SmallMoleculeComponent,
    SolventComponent,
    settings,
)
from openfe.protocols.openmm_utils import (
    charge_generation,
    settings_validation,
    system_creation,
    system_validation,
)
from openfe.protocols.openmm_utils.omm_settings import BasePartialChargeSettings
from openfe.utils import log_system_probe, without_oechem_backend
from openff.interchange import Interchange
from openff.toolkit import Topology
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmmtools import forces

from openfe_gromacs.protocols.gromacs_md.md_settings import (
    EMOutputSettings,
    EMSimulationSettings,
    GromacsMDProtocolSettings,
    IntegratorSettings,
    NPTOutputSettings,
    NPTSimulationSettings,
    NVTOutputSettings,
    NVTSimulationSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
)

logger = logging.getLogger(__name__)


def _dict2mdp(settings_dict: dict, shared_basepath):
    """
    Write out a Gromacs .mdp file given a settings dictionary
    :param settings_dict: dict
          Dictionary of settings
    :param shared_basepath: Pathlike
          Where to save the .mdp files
    """
    filename = shared_basepath / settings_dict["mdp_file"]
    settings_dict.pop("forcefield_cache")
    settings_dict.pop("mdp_file")
    with open(filename, "w") as f:
        for key, value in settings_dict.items():
            # Get rid of units
            # ToDo: Add unit validation/conversion somewhere
            if isinstance(value, pint.Quantity):
                value = value.magnitude
            f.write("{} = {}\n".format(key, value))
    return filename


class GromacsMDProtocolResult(gufe.ProtocolResult):
    """
    Dict-like container for the output of a Gromacs MDProtocol.

    Provides access to simulation outputs including the pre-minimized
    system PDB and production trajectory files.
    """

    def __init__(self, **data):
        super().__init__(**data)
        # data is mapping of str(repeat_id): list[protocolunitresults]
        if any(len(pur_list) > 2 for pur_list in self.data.values()):
            raise NotImplementedError("Can't stitch together results yet")

    def get_estimate(self):
        """Since no results as output --> returns None

        Returns
        -------
        None
        """

        return None

    def get_uncertainty(self):
        """Since no results as output --> returns None"""

        return None

    # TODO: Change this to return the actual outputs

    def get_gro_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file

        Returns
        -------
        traj : list[pathlib.Path]
          list of paths (pathlib.Path) to the simulation trajectory
        """
        gro = [pus[0].outputs["system_gro"] for pus in self.data.values()]

        return gro

    def get_top_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file

        Returns
        -------
        traj : list[pathlib.Path]
          list of paths (pathlib.Path) to the simulation trajectory
        """
        top = [pus[0].outputs["system_top"] for pus in self.data.values()]

        return top

    def get_mdp_filenames(self) -> list[list[pathlib.Path]]:
        """
        Get a list of paths to the .mdp files

        Returns
        -------
        mdps : list[list[pathlib.Path]]
          list of paths (pathlib.Path) to the mdp files for energy minimization,
          NVT and NPT MD runs
        """
        mdps = [pus[0].outputs["mdp_files"] for pus in self.data.values()]

        return mdps


class GromacsMDProtocol(gufe.Protocol):
    """
    Protocol for running Molecular Dynamics simulations using Gromacs.

    See Also
    --------
    :mod:`openfe.gromacs.protocols`
    :class:`openfe.gromacs.protocols.gromacs_md.GromacsMDProtocolSettings`
    :class:`openfe.gromacs.protocols.gromacs_md.GromacsMDProtocolUnit`
    :class:`openfe.gromacs.protocols.gromacs_md.GromacsMDProtocolResult`
    """

    result_cls = GromacsMDProtocolResult
    _settings: GromacsMDProtocolSettings

    @classmethod
    def _default_settings(cls):
        """A dictionary of initial settings for this creating this Protocol

        These settings are intended as a suitable starting point for creating
        an instance of this protocol.  It is recommended, however that care is
        taken to inspect and customize these before performing a Protocol.

        Returns
        -------
        Settings
          a set of default settings
        """
        return GromacsMDProtocolSettings(
            forcefield_settings=settings.OpenMMSystemGeneratorFFSettings(
                constraints=None
            ),
            thermo_settings=settings.ThermoSettings(
                temperature=298.15 * unit.kelvin,
                pressure=1 * unit.bar,
            ),
            partial_charge_settings=OpenFFPartialChargeSettings(),
            solvation_settings=OpenMMSolvationSettings(),
            engine_settings=OpenMMEngineSettings(),
            integrator_settings=IntegratorSettings(),
            simulation_settings_em=EMSimulationSettings(
                integrator="steep",
                nsteps=5000,
                tcoupl="no",
                pcoupl="no",
                gen_vel="no",
            ),
            simulation_settings_nvt=NVTSimulationSettings(
                nsteps=50000,  # 100ps
                pcoupl="no",
                gen_vel="yes",
            ),
            simulation_settings_npt=NPTSimulationSettings(
                nsteps=500000,  # 1ns
                pcoupl="Parrinello-Rahman",
                pcoupltype="isotropic",
                ref_p=1.01325 * unit.bar,
                gen_vel="no",  # If continuation from NVT simulation
            ),
            output_settings_em=EMOutputSettings(
                mdp_file="em.mdp",
            ),
            output_settings_nvt=NVTOutputSettings(
                mdp_file="nvt.mdp",
            ),
            output_settings_npt=NPTOutputSettings(
                mdp_file="npt.mdp",
                nstxout=5000,
                nstvout=5000,
                nstfout=5000,
                nstxout_compressed=5000,
            ),
            protocol_repeats=1,
            gro="system.gro",
            top="system.top",
        )

    def _create(
        self,
        stateA: ChemicalSystem,
        stateB: ChemicalSystem,
        mapping: dict[str, gufe.ComponentMapping] | None = None,
        extends: gufe.ProtocolDAGResult | None = None,
    ) -> list[gufe.ProtocolUnit]:
        # TODO: Extensions?
        if extends:
            raise NotImplementedError("Can't extend simulations yet")

        # Validate solvent component
        # TODO: Maybe only allow PME as nonbonded_method since Interchange requires PME?
        nonbond = self.settings.forcefield_settings.nonbonded_method
        # TODO: Add check that there has to be solvent since
        #  Gromacs doesn't support vacuum MD
        system_validation.validate_solvent(stateA, nonbond)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
        # TODO: Deal with multiple ProteinComponents
        solvent_comp, protein_comp, small_mols = system_validation.get_components(
            stateA
        )

        system_name = "Solvent MD" if solvent_comp is not None else "Vacuum MD"

        for comp in [protein_comp] + small_mols:
            if comp is not None:
                comp_type = comp.__class__.__name__
                if len(comp.name) == 0:
                    comp_name = "NoName"
                else:
                    comp_name = comp.name
                system_name += f" {comp_type}: {comp_name}"

        # our DAG has no dependencies, so just list units
        n_repeats = self.settings.protocol_repeats
        units = [
            GromacsMDSetupUnit(
                protocol=self,
                stateA=stateA,
                generation=0,
                repeat_id=int(uuid.uuid4()),
                name=f"{system_name} repeat {i} generation 0",
            )
            for i in range(n_repeats)
        ]

        return units

    def _gather(
        self, protocol_dag_results: Iterable[gufe.ProtocolDAGResult]
    ) -> dict[str, Any]:
        # result units will have a repeat_id and generations within this
        # repeat_id
        # first group according to repeat_id
        unsorted_repeats = defaultdict(list)
        for d in protocol_dag_results:
            pu: gufe.ProtocolUnitResult
            for pu in d.protocol_unit_results:
                if not pu.ok():
                    continue

                unsorted_repeats[pu.outputs["repeat_id"]].append(pu)

        # then sort by generation within each repeat_id list
        repeats: dict[str, list[gufe.ProtocolUnitResult]] = {}
        for k, v in unsorted_repeats.items():
            repeats[str(k)] = sorted(v, key=lambda x: x.outputs["generation"])

        # returns a dict of repeat_id: sorted list of ProtocolUnitResult
        return repeats


class GromacsMDSetupUnit(gufe.ProtocolUnit):
    """
    Protocol unit for settings up plain MD simulations (NonTransformation).
    """

    def __init__(
        self,
        *,
        protocol: GromacsMDProtocol,
        stateA: ChemicalSystem,
        generation: int,
        repeat_id: int,
        name: str | None = None,
    ):
        """
        Parameters
        ----------
        protocol : GromacsMDProtocol
          protocol used to create this Unit. Contains key information such
          as the settings.
        stateA : ChemicalSystem
          the chemical system for the MD simulation
        repeat_id : int
          identifier for which repeat (aka replica/clone) this Unit is
        generation : int
          counter for how many times this repeat has been extended
        name : str, optional
          human-readable identifier for this Unit

        Notes
        -----
        The mapping used must not involve any elemental changes.  A check for
        this is done on class creation.
        """
        super().__init__(
            name=name,
            protocol=protocol,
            stateA=stateA,
            repeat_id=repeat_id,
            generation=generation,
        )

    @staticmethod
    def _assign_partial_charges(
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

    def run(
        self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None
    ) -> dict[str, Any]:
        """Run the MD simulation.

        Parameters
        ----------
        dry : bool
          Do a dry run of the calculation, creating all necessary hybrid
          system components (topology, system, sampler, etc...) but without
          running the simulation.
        verbose : bool
          Verbose output of the simulation progress. Output is provided via
          INFO level logging.
        scratch_basepath: Pathlike, optional
          Where to store temporary files, defaults to current working directory
        shared_basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.

        Raises
        ------
        error
          Exception if anything failed
        """
        if verbose:
            self.logger.info("Creating system")
        if shared_basepath is None:
            # use cwd
            shared_basepath = pathlib.Path(".")

        # 0. General setup and settings dependency resolution step

        # Extract relevant settingsprotocol_settings:
        protocol_settings: GromacsMDProtocolSettings = self._inputs["protocol"].settings

        stateA = self._inputs["stateA"]

        forcefield_settings: settings.OpenMMSystemGeneratorFFSettings = (
            protocol_settings.forcefield_settings
        )
        thermo_settings: settings.ThermoSettings = protocol_settings.thermo_settings
        solvation_settings: OpenMMSolvationSettings = (
            protocol_settings.solvation_settings
        )
        charge_settings: BasePartialChargeSettings = (
            protocol_settings.partial_charge_settings
        )
        sim_settings_em: EMSimulationSettings = protocol_settings.simulation_settings_em
        sim_settings_nvt: NVTSimulationSettings = (
            protocol_settings.simulation_settings_nvt
        )
        sim_settings_npt: NPTSimulationSettings = (
            protocol_settings.simulation_settings_npt
        )
        output_settings_em: EMOutputSettings = protocol_settings.output_settings_em
        output_settings_nvt: NVTOutputSettings = protocol_settings.output_settings_nvt
        output_settings_npt: NPTOutputSettings = protocol_settings.output_settings_npt
        integrator_settings = protocol_settings.integrator_settings

        # # is the timestep good for the mass?
        # settings_validation.validate_timestep(
        #     forcefield_settings.hydrogen_mass, timestep
        # )

        solvent_comp, protein_comp, small_mols = system_validation.get_components(
            stateA
        )

        # Write out .mdp files
        mdps = []
        if protocol_settings.simulation_settings_em.nsteps > 0:
            settings_dict = sim_settings_em.dict() | output_settings_em.dict()
            mdp = _dict2mdp(settings_dict, shared_basepath)
            mdps.append(mdp)
        if protocol_settings.simulation_settings_nvt.nsteps > 0:
            settings_dict = sim_settings_nvt.dict() | output_settings_nvt.dict()
            mdp = _dict2mdp(settings_dict, shared_basepath)
            mdps.append(mdp)
        if protocol_settings.simulation_settings_npt.nsteps > 0:
            settings_dict = sim_settings_npt.dict() | output_settings_npt.dict()
            mdp = _dict2mdp(settings_dict, shared_basepath)
            mdps.append(mdp)

        # 1. Create stateA system
        # Create a dictionary of OFFMol for each SMC for bookeeping
        smc_components: dict[SmallMoleculeComponent, OFFMolecule]

        smc_components = {i: i.to_openff() for i in small_mols}

        # a. assign partial charges to smcs
        self._assign_partial_charges(charge_settings, smc_components)

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
                molecules=[s.to_openff() for s in small_mols],
            )

            # f. Create interchange object
            barostat_idx, barostat = forces.find_forces(
                stateA_system, ".*Barostat.*", only_one=True
            )
            stateA_system.removeForce(barostat_idx)

            water = OFFMolecule.from_mapped_smiles("[O:1]([H:2])[H:3]")
            cl = OFFMolecule.from_smiles("[Cl-]", name="Cl")
            na = OFFMolecule.from_smiles("[Na+]", name="Na")
            unique_molecules = [water, cl, na]
            # if protein_comp:
            #     unique_molecules.append(protein_comp.to_openmm_topology())
            for mol in smc_components.values():
                unique_molecules.append(mol)
            topology = Topology.from_openmm(
                stateA_topology, unique_molecules=unique_molecules
            )

            stateA_interchange = Interchange.from_openmm(
                topology=topology, system=stateA_system, positions=stateA_positions
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
                # molecules don't know their residue metadata, so need to
                # set on each atom
                # https://github.com/openforcefield/openff-toolkit/issues/1554
                elif molecule.is_isomorphic_with(water):
                    for atom in molecule.atoms:
                        atom.metadata["residue_name"] = "WAT"
                elif molecule.is_isomorphic_with(na):
                    for atom in molecule.atoms:
                        atom.metadata["residue_name"] = "Na"
                elif molecule.is_isomorphic_with(cl):
                    for atom in molecule.atoms:
                        atom.metadata["residue_name"] = "Cl"
            # g. Save .gro and .top file of the entire system
            stateA_interchange.to_gro(shared_basepath / protocol_settings.gro)
            stateA_interchange.to_top(shared_basepath / protocol_settings.top)

        output = {
            "system_gro": shared_basepath / protocol_settings.gro,
            "system_top": shared_basepath / protocol_settings.top,
            "mdp_files": mdps,
        }

        return output

    def _execute(
        self,
        ctx: gufe.Context,
        **kwargs,
    ) -> dict[str, Any]:
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        outputs = self.run(scratch_basepath=ctx.scratch, shared_basepath=ctx.shared)

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            **outputs,
        }
