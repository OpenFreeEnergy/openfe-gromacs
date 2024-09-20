# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe

"""Gromacs MD Protocol --- :mod:`openfe_gromacs.protocols.gromacs_md.md_methods`
================================================================================

This module implements the necessary methodology tools to run an MD
simulation using OpenMM tools and Gromacs.

"""
from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import gmxapi as gmx
import gufe
import pint
from gufe import ChemicalSystem, SmallMoleculeComponent, settings
from openfe.protocols.openmm_utils import (
    charge_generation,
    settings_validation,
    system_creation,
    system_validation,
)
from openfe.protocols.openmm_utils.omm_settings import BasePartialChargeSettings
from openfe.utils import log_system_probe, without_oechem_backend
from openff.interchange import Interchange
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm
from openmmtools import forces

from openfe_gromacs.protocols.gromacs_md.md_settings import (
    EMOutputSettings,
    EMSimulationSettings,
    FFSettingsOpenMM,
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


PRE_DEFINED_SETTINGS = {
    "tinit": 0 * unit.picosecond,
    "init_step": 0,
    "simulation_part": 0,
    "comm_mode": "Linear",
    "nstcomm": 100,
    "comm_grps": "system",
    "pbc": "xyz",
    "verlet_buffer_tolerance": 0.005 * unit.kilojoule / (unit.mole * unit.picosecond),
    "verlet_buffer_pressure_tolerance": 0.5 * unit.bar,
    "coulomb_modifier": "Potential-shift",
    "epsilon_r": 1,
    "epsilon_rf": 0,
    "fourierspacing": 0.12 * unit.nanometer,
    "lj_pme_comb_rule": "Geometric",
    "ewald_geometry": "3d",
    "epsilon_surface": 0,
    "nsttcouple": -1,
    "tc_grps": "system",
    "tau_t": 2.0 * unit.picosecond,
    "pcoupltype": "isotropic",
    "nstpcouple": -1,
    "tau_p": 5 * unit.picosecond,
    "compressibility": 4.5e-05 / unit.bar,
    "lincs_warnangle": 30 * unit.degree,
    "continuation": "no",
    "morse": "no",
}

PRE_DEFINED_SETTINGS_EM = {
    "emtol": 10.0 * unit.kilojoule / (unit.mole * unit.nanometer),
    "emstep": 0.01 * unit.nanometer,
}


def _dict2mdp(settings_dict: dict, shared_basepath):
    """
    Write out a Gromacs .mdp file given a settings dictionary
    :param settings_dict: dict
          Dictionary of settings
    :param shared_basepath: Pathlike
          Where to save the .mdp files
    """
    filename = shared_basepath / settings_dict["mdp_file"]
    # Remove non-mdp settings from the dictionary
    non_mdps = [
        "forcefield_cache",
        "mdp_file",
        "tpr_file",
        "trr_file",
        "xtc_file",
        "gro_file",
        "edr_file",
        "log_file",
        "cpt_file",
    ]
    for setting in non_mdps:
        settings_dict.pop(setting)
    with open(filename, "w") as f:
        for key, value in settings_dict.items():
            # First convert units to units in the mdp file, then remove units
            if isinstance(value, pint.Quantity):
                if value.is_compatible_with(unit.nanometer):
                    value = value.to(unit.nanometer)
                if value.is_compatible_with(unit.picosecond):
                    value = value.to(unit.picosecond)
                if value.is_compatible_with(unit.kelvin):
                    value = value.to(unit.kelvin)
                if value.is_compatible_with(unit.bar):
                    value = value.to(unit.bar)
                value = value.magnitude
            # Write out all the setting, value pairs
            f.write(f"{key} = {value}\n")
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
        gro = [
            pus[0].outputs["system_gro"]
            for pus in self.data.values()
            if "GromacsMDSetupUnit" in pus[0].source_key
        ]

        return gro

    def get_top_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file

        Returns
        -------
        traj : list[pathlib.Path]
          list of paths (pathlib.Path) to the simulation trajectory
        """
        top = [
            pus[0].outputs["system_top"]
            for pus in self.data.values()
            if "GromacsMDSetupUnit" in pus[0].source_key
        ]

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
        mdps = [
            pus[0].outputs["mdp_files"]
            for pus in self.data.values()
            if "GromacsMDSetupUnit" in pus[0].source_key
        ]

        return mdps

    def get_filenames_em(self) -> dict[str, list[pathlib.Path]]:
        """
        Get a list of paths to the files of the
        energy minimization
        Following file formats are stored in the dictionary under these keys:
        - "gro_em"
        - "tpr_em"
        - "trr_em"
        - "xtc_em"
        - "edr_em"
        - "log_em"
        - "cpt_em"

        Returns
        -------
        dict_em : dict[str, list[pathlib.Path]]
          dictionary containing list of paths (pathlib.Path)
          to the output files
        """
        file_keys = [
            "gro_em",
            "tpr_em",
            "trr_em",
            "xtc_em",
            "edr_em",
            "log_em",
            "cpt_em",
        ]
        dict_em = {}
        for file in file_keys:

            file_path = [
                pus[0].outputs[file]
                for pus in self.data.values()
                if "GromacsMDRunUnit" in pus[0].source_key
            ]
            dict_em[file] = file_path
        return dict_em

    def get_gro_em_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file, last frame of the
        energy minimization

        Returns
        -------
        gro : list[pathlib.Path]
          list of paths (pathlib.Path) to the output .gro file
        """
        gro = [
            pus[0].outputs["gro_em"]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key
        ]

        return gro

    def get_xtc_em_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .xtc file of the
        energy minimization

        Returns
        -------
        file_path : list[pathlib.Path]
          list of paths (pathlib.Path) to the output .xtc file
        """
        file_path = [
            pus[0].outputs["xtc_em"]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key
        ]

        return file_path

    def get_filenames_nvt(self) -> dict[str, list[pathlib.Path]]:
        """
        Get a list of paths to the files of the
        NVT equilibration
        Following file formats are stored in the dictionary under these keys:
        - "gro_nvt"
        - "tpr_nvt"
        - "trr_nvt"
        - "xtc_nvt"
        - "edr_nvt"
        - "log_nvt"
        - "cpt_nvt"

        Returns
        -------
        dict_nvt : dict[str, list[pathlib.Path]]
          dictionary containing list of paths (pathlib.Path)
          to the output files
        """
        file_keys = [
            "gro_nvt",
            "tpr_nvt",
            "trr_nvt",
            "xtc_nvt",
            "edr_nvt",
            "log_nvt",
            "cpt_nvt",
        ]
        dict_nvt = {}
        for file in file_keys:

            file_path = [
                pus[0].outputs[file]
                for pus in self.data.values()
                if "GromacsMDRunUnit" in pus[0].source_key
            ]
            dict_nvt[file] = file_path
        return dict_nvt

    def get_gro_nvt_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file, last frame of the
        NVT equilibration

        Returns
        -------
        gro : list[pathlib.Path]
          list of paths (pathlib.Path) to the output .gro file
        """
        gro = [
            pus[0].outputs["gro_nvt"]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key
        ]

        return gro

    def get_xtc_nvt_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .xtc file of the
        NVT equilibration

        Returns
        -------
        file_path : list[pathlib.Path]
          list of paths (pathlib.Path) to the output .xtc file
        """
        file_path = [
            pus[0].outputs["xtc_nvt"]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key
        ]

        return file_path

    def get_filenames_npt(self) -> dict[str, list[pathlib.Path]]:
        """
        Get a list of paths to the files of the
        NPT MD simulation
        Following file formats are stored in the dictionary under these keys:
        - "gro_npt"
        - "tpr_npt"
        - "trr_npt"
        - "xtc_npt"
        - "edr_npt"
        - "log_npt"
        - "cpt_npt"

        Returns
        -------
        dict_npt : dict[str, list[pathlib.Path]]
          dictionary containing list of paths (pathlib.Path)
          to the output files
        """
        file_keys = [
            "gro_npt",
            "tpr_npt",
            "trr_npt",
            "xtc_npt",
            "edr_npt",
            "log_npt",
            "cpt_npt",
        ]
        dict_npt = {}
        for file in file_keys:

            file_path = [
                pus[0].outputs[file]
                for pus in self.data.values()
                if "GromacsMDRunUnit" in pus[0].source_key
            ]
            dict_npt[file] = file_path
        return dict_npt

    def get_gro_npt_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file, last frame of the
        NPT MD simulation

        Returns
        -------
        gro : list[pathlib.Path]
          list of paths (pathlib.Path) to the output .gro file
        """
        gro = [
            pus[0].outputs["gro_npt"]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key
        ]

        return gro

    def get_xtc_npt_filename(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .xtc file of the
        NPT MD simulation

        Returns
        -------
        file_path : list[pathlib.Path]
          list of paths (pathlib.Path) to the output .xtc file
        """
        file_path = [
            pus[0].outputs["xtc_npt"]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key
        ]

        return file_path


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
            forcefield_settings=FFSettingsOpenMM(
                nonbonded_method="PME",
                constraints=None,
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
            ),
            simulation_settings_nvt=NVTSimulationSettings(
                nsteps=50000,  # 100ps
                pcoupl="no",
                gen_vel="yes",
            ),
            simulation_settings_npt=NPTSimulationSettings(
                nsteps=500000,  # 1ns
                pcoupl="C-rescale",
                gen_vel="no",  # If continuation from NVT simulation
            ),
            output_settings_em=EMOutputSettings(
                mdp_file="em.mdp",
                tpr_file="em.tpr",
                gro_file="em.gro",
                trr_file="em.trr",
                xtc_file="em.xtc",
                edr_file="em.edr",
                cpt_file="em.cpt",
                log_file="em.log",
            ),
            output_settings_nvt=NVTOutputSettings(
                mdp_file="nvt.mdp",
                tpr_file="nvt.tpr",
                gro_file="nvt.gro",
                trr_file="nvt.trr",
                xtc_file="nvt.xtc",
                edr_file="nvt.edr",
                cpt_file="nvt.cpt",
                log_file="nvt.log",
            ),
            output_settings_npt=NPTOutputSettings(
                mdp_file="npt.mdp",
                tpr_file="npt.tpr",
                gro_file="npt.gro",
                trr_file="npt.trr",
                xtc_file="npt.xtc",
                edr_file="npt.edr",
                cpt_file="npt.cpt",
                log_file="npt.log",
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

        # Validate solvent component
        nonbond = self.settings.forcefield_settings.nonbonded_method
        system_validation.validate_solvent(stateA, nonbond)

        # Validate protein component
        system_validation.validate_protein(stateA)

        # actually create and return Units
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
        setup = GromacsMDSetupUnit(
            protocol=self,
            stateA=stateA,
            generation=0,
            repeat_id=int(uuid.uuid4()),
            name=f"{system_name}",
        )
        run = [
            GromacsMDRunUnit(
                protocol=self,
                setup=setup,
                generation=0,
                repeat_id=int(uuid.uuid4()),
                name=f"{system_name} repeat {i} generation 0",
            )
            for i in range(n_repeats)
        ]
        units = [setup] + run

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

    def _handle_settings(self) -> dict[str, gufe.settings.SettingsBaseModel]:
        """
        Extract the relevant settings for an MD simulation.

        Returns
        -------
        settings : dict[str, SettingsBaseModel]
          A dictionary with the following entries:
            * forcefield_settings : FFSettingsOpenMM
            * thermo_settings : ThermoSettings
            * solvation_settings : SolvationSettings
            * charge_settings: BasePartialChargeSetting
            * sim_settings_em: EMSimulationSettings
            * sim_settings_nvt: NVTSimulationSettings
            * sim_settings_npt: NPTSimulationSettings
            * output_settings_em: EMOutputSettings
            * output_settings_nvt: NVTOutputSettings
            * output_settings_npt: NPTOutputSettings
            * integrator_settings: IntegratorSettings
        """
        prot_settings: GromacsMDProtocolSettings = self._inputs["protocol"].settings

        settings = {}
        settings["forcefield_settings"] = prot_settings.forcefield_settings
        settings["thermo_settings"] = prot_settings.thermo_settings
        settings["solvation_settings"] = prot_settings.solvation_settings
        settings["charge_settings"] = prot_settings.partial_charge_settings
        settings["sim_settings_em"] = prot_settings.simulation_settings_em
        settings["sim_settings_nvt"] = prot_settings.simulation_settings_nvt
        settings["sim_settings_npt"] = prot_settings.simulation_settings_npt
        settings["output_settings_em"] = prot_settings.output_settings_em
        settings["output_settings_nvt"] = prot_settings.output_settings_nvt
        settings["output_settings_npt"] = prot_settings.output_settings_npt
        settings["integrator_settings"] = prot_settings.integrator_settings
        settings["gro"] = prot_settings.gro
        settings["top"] = prot_settings.top

        return settings

    def _write_mdp_files(self, settings: dict, shared_basepath):
        """
        Writes out the .mdp files for running a Gromacs MD simulation.

        Parameters
        ----------
        settings: dict
          Dictionary of all the settings
        shared_basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory

        Returns
        -------
        mdps: list
          List of file paths to mdp files.
        """

        mdps = []
        if settings["sim_settings_em"].nsteps > 0:
            settings_dict = (
                settings["sim_settings_em"].dict()
                | settings["output_settings_em"].dict()
                | PRE_DEFINED_SETTINGS
                | PRE_DEFINED_SETTINGS_EM
            )
            mdp = _dict2mdp(settings_dict, shared_basepath)
            mdps.append(mdp)
        if settings["sim_settings_nvt"].nsteps > 0:
            settings_dict = (
                settings["sim_settings_nvt"].dict()
                | settings["output_settings_nvt"].dict()
                | PRE_DEFINED_SETTINGS
            )
            mdp = _dict2mdp(settings_dict, shared_basepath)
            mdps.append(mdp)
        if settings["sim_settings_npt"].nsteps > 0:
            settings_dict = (
                settings["sim_settings_npt"].dict()
                | settings["output_settings_npt"].dict()
                | PRE_DEFINED_SETTINGS
            )
            mdp = _dict2mdp(settings_dict, shared_basepath)
            mdps.append(mdp)

        return mdps

    def _create_interchange(self, settings, components, shared_basepath):
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
        # Create the stateA system
        # Create a dictionary of OFFMol for each SMC for bookkeeping
        smc_components: dict[SmallMoleculeComponent, OFFMolecule]

        # ToDo: Make this more general, check if there is a smc_component,
        #       allow ProteinComponents, ...
        smc_components = {i: i.to_openff() for i in components["small_mols"]}

        # a. assign partial charges to smcs
        self._assign_partial_charges(settings["charge_settings"], smc_components)

        # b. get a system generator
        if settings["output_settings_em"].forcefield_cache is not None:
            ffcache = shared_basepath / settings["output_settings_em"].forcefield_cache
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
                has_solvent=components["solvent_comp"] is not None,
            )

            # Force creation of smc templates so we can solvate later
            for mol in smc_components.values():
                system_generator.create_system(
                    mol.to_topology().to_openmm(), molecules=[mol]
                )

            # c. get OpenMM Modeller + a resids dictionary for each component
            stateA_modeller, comp_resids = system_creation.get_omm_modeller(
                protein_comp=components["protein_comp"],
                solvent_comp=components["solvent_comp"],
                small_mols=smc_components,
                omm_forcefield=system_generator.forcefield,
                solvent_settings=settings["solvation_settings"],
            )

            # d. get topology & positions
            # Note: roundtrip positions to remove vec3 issues
            stateA_topology = stateA_modeller.getTopology()
            stateA_positions = to_openmm(from_openmm(stateA_modeller.getPositions()))

            # e. create the stateA System
            stateA_system = system_generator.create_system(
                stateA_topology,
                molecules=[s.to_openff() for s in components["small_mols"]],
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

    def run(
        self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None
    ) -> dict[str, Any]:
        """
        Write out all the files necessary to run the Gromacs MD simulation.
        MDP files, Gromacs coordinate and topology files are written.
        This SetupUnit does not actually run the MD simulation.
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
        # Extract relevant protocol_settings:
        settings = self._handle_settings()

        stateA = self._inputs["stateA"]

        # Get the different components
        solvent_comp, protein_comp, small_mols = system_validation.get_components(
            stateA
        )
        components = {
            "solvent_comp": solvent_comp,
            "protein_comp": protein_comp,
            "small_mols": small_mols,
        }

        # Raise an error when no SolventComponent is provided as this Protocol
        # currently does not support vacuum simulations
        if not solvent_comp:
            errmsg = (
                "No SolventComponent provided. This protocol currently does"
                " not support vacuum simulations."
            )
            raise ValueError(errmsg)

        # 1. Write out .mdp files
        mdps = self._write_mdp_files(settings, shared_basepath)

        # 2. Create the Interchange object
        stateA_interchange = self._create_interchange(
            settings, components, shared_basepath
        )

        # 4. Save .gro and .top file of the entire system
        stateA_interchange.to_gro(shared_basepath / settings["gro"])
        stateA_interchange.to_top(shared_basepath / settings["top"])

        output = {
            "system_gro": shared_basepath / settings["gro"],
            "system_top": shared_basepath / settings["top"],
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


class GromacsMDRunUnit(gufe.ProtocolUnit):
    """
    Protocol unit for running plain MD simulations (NonTransformation)
    in Gromacs.
    """

    @staticmethod
    def _run_gromacs(
        mdp: pathlib.Path,
        in_gro: pathlib.Path,
        top: pathlib.Path,
        tpr: pathlib.Path,
        out_gro: str,
        xtc: str,
        trr: str,
        cpt: str,
        log: str,
        edr: str,
        shared_basebath: pathlib.Path,
    ):
        """
        Running Gromacs gmx grompp and gmx mdrun commands using subprocess.

        Parameters
        ----------
        :param mdp: pathlib.Path
          Path to the mdp file
        :param in_gro: pathlib.Path
        :param top: pathlib.Path
        :param tpr: pathlib.Path
        :param out_gro: str
        :param xtc: str
        :param trr: str
        :param cpt: str
        :param log: str
        :param edr: str
        :param shared_basebath: Pathlike, optional
          Where to run the calculation, defaults to current working directory
        """
        assert os.path.exists(in_gro)
        assert os.path.exists(top)
        assert os.path.exists(mdp)
        p = subprocess.Popen(
            [
                "gmx",
                "grompp",
                "-f",
                mdp,
                "-c",
                in_gro,
                "-p",
                top,
                "-o",
                tpr,
            ],
            stdin=subprocess.PIPE,
        )
        p.wait()
        assert os.path.exists(tpr)
        p = subprocess.Popen(
            [
                "gmx",
                "mdrun",
                "-s",
                tpr.name,
                "-cpo",
                cpt,
                "-o",
                trr,
                "-x",
                xtc,
                "-c",
                out_gro,
                "-e",
                edr,
                "-g",
                log,
            ],
            stdin=subprocess.PIPE,
            cwd=shared_basebath,
        )
        p.wait()
        return

    def _execute(
        self,
        ctx: gufe.Context,
        *,
        protocol,
        setup,
        verbose=True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute the simulation part of the Gromacs MD protocol.

        Parameters
        ----------
        ctx : gufe.protocols.protocolunit.Context
            The gufe context for the unit.
        protocol : gufe.protocols.Protocol
            The Protocol used to create this Unit. Contains key
            information
            such as the settings.
        setup : gufe.protocols.ProtocolUnit
            The SetupUnit

        Returns
        -------
        dict : dict[str, str]
            Dictionary with paths to ...
        """
        log_system_probe(logging.INFO, paths=[ctx.scratch])

        if ctx.shared is None:
            # use cwd
            shared_basepath = pathlib.Path(".")
        else:
            shared_basepath = ctx.shared

        # ToDo: Figure out how to specify the order in which to run things
        # ToDo: Add output settings, e.g. name of output files
        protocol_settings: GromacsMDProtocolSettings = self._inputs["protocol"].settings
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

        input_gro = setup.outputs["system_gro"]
        input_top = setup.outputs["system_top"]
        mdp_files = setup.outputs["mdp_files"]

        # Run energy minimization
        if sim_settings_em.nsteps > 0:
            if verbose:
                self.logger.info("Running energy minimization")
            mdp = [
                x
                for x in mdp_files
                if str(x).split("/")[-1] == output_settings_em.mdp_file
            ]
            tpr = pathlib.Path(ctx.shared / output_settings_em.tpr_file)
            assert len(mdp) == 1
            # ToDo: If no traj should be written out, don't write empty file?
            self._run_gromacs(
                mdp[0],
                input_gro,
                input_top,
                tpr,
                output_settings_em.gro_file,
                output_settings_em.xtc_file,
                output_settings_em.trr_file,
                output_settings_em.cpt_file,
                output_settings_em.log_file,
                output_settings_em.edr_file,
                ctx.shared,
            )

        # ToDo: Should we disallow running MD without EM?
        # Run NVT
        if sim_settings_nvt.nsteps > 0:
            if verbose:
                self.logger.info("Running an NVT MD simulation")
            mdp = [
                x
                for x in mdp_files
                if str(x).split("/")[-1] == output_settings_nvt.mdp_file
            ]
            tpr = pathlib.Path(ctx.shared / output_settings_nvt.tpr_file)
            assert len(mdp) == 1
            # ToDo: Change .gro to output from EM if we do EM first,
            # else original .gro file
            self._run_gromacs(
                mdp[0],
                input_gro,
                input_top,
                tpr,
                output_settings_nvt.gro_file,
                output_settings_nvt.xtc_file,
                output_settings_nvt.trr_file,
                output_settings_nvt.cpt_file,
                output_settings_nvt.log_file,
                output_settings_nvt.edr_file,
                ctx.shared,
            )

        # Run NPT MD simulation
        if sim_settings_npt.nsteps > 0:
            if verbose:
                self.logger.info("Running an NPT MD simulation")
            mdp = [
                x
                for x in mdp_files
                if str(x).split("/")[-1] == output_settings_npt.mdp_file
            ]
            tpr = pathlib.Path(ctx.shared / output_settings_npt.tpr_file)
            assert len(mdp) == 1
            self._run_gromacs(
                mdp[0],
                input_gro,
                input_top,
                tpr,
                output_settings_npt.gro_file,
                output_settings_npt.xtc_file,
                output_settings_npt.trr_file,
                output_settings_npt.cpt_file,
                output_settings_npt.log_file,
                output_settings_npt.edr_file,
                ctx.shared,
            )

        return {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
            "gro_em": shared_basepath / output_settings_em.gro_file,
            "tpr_em": shared_basepath / output_settings_em.tpr_file,
            "trr_em": shared_basepath / output_settings_em.trr_file,
            "xtc_em": shared_basepath / output_settings_em.xtc_file,
            "edr_em": shared_basepath / output_settings_em.edr_file,
            "log_em": shared_basepath / output_settings_em.log_file,
            "cpt_em": shared_basepath / output_settings_em.cpt_file,
            "gro_nvt": shared_basepath / output_settings_nvt.gro_file,
            "tpr_nvt": shared_basepath / output_settings_nvt.tpr_file,
            "trr_nvt": shared_basepath / output_settings_nvt.trr_file,
            "xtc_nvt": shared_basepath / output_settings_nvt.xtc_file,
            "edr_nvt": shared_basepath / output_settings_nvt.edr_file,
            "log_nvt": shared_basepath / output_settings_nvt.log_file,
            "cpt_nvt": shared_basepath / output_settings_nvt.cpt_file,
            "gro_npt": shared_basepath / output_settings_npt.gro_file,
            "tpr_npt": shared_basepath / output_settings_npt.tpr_file,
            "trr_npt": shared_basepath / output_settings_npt.trr_file,
            "xtc_npt": shared_basepath / output_settings_npt.xtc_file,
            "edr_npt": shared_basepath / output_settings_npt.edr_file,
            "log_npt": shared_basepath / output_settings_npt.log_file,
            "cpt_npt": shared_basepath / output_settings_npt.cpt_file,
        }
