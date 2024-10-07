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

import gufe
from gufe import ChemicalSystem, settings
from openfe.protocols.openmm_utils import system_validation
from openfe.utils import log_system_probe
from openff.toolkit.topology import Molecule as OFFMolecule
from openff.units import unit

from openfe_gromacs.protocols.gromacs_md.md_settings import (
    EMOutputSettings,
    EMSimulationSettings,
    FFSettingsOpenMM,
    GromacsEngineSettings,
    GromacsMDProtocolSettings,
    IntegratorSettings,
    NPTOutputSettings,
    NPTSimulationSettings,
    NVTOutputSettings,
    NVTSimulationSettings,
    OpenFFPartialChargeSettings,
    SolvationSettings,
)
from openfe_gromacs.protocols.gromacs_utils import create_systems, write_mdp

logger = logging.getLogger(__name__)


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

    def _get_filenames(self, file_type) -> list[pathlib.Path]:
        """
        Get a list of paths to files

        Parameters
        ----------
        file_type: str
          Str of the dictionary entry for which the file paths should be
          returned
        Returns
        -------
        files : list[pathlib.Path]
          list of paths (pathlib.Path) to the files
        """
        file_paths = [
            pus[0].outputs[file_type]
            for pus in self.data.values()
            if "GromacsMDRunUnit" in pus[0].source_key and file_type in pus[0].outputs
        ]
        if not file_paths:
            return None

        return file_paths

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

    def get_gro_filename(self) -> pathlib.Path:
        """
        Get the path to the input coordinate .gro file.
        This returns a single path even if multiple repeats are run since
        the GromacsMDSetupUnit is only run once.

        Returns
        -------
        gro : pathlib.Path
          Path to the input coordinate .gro file
        """
        gro = [
            pus[0].outputs["system_gro"]
            for pus in self.data.values()
            if "GromacsMDSetupUnit" in pus[0].source_key
        ][0]

        return gro

    def get_top_filename(self) -> pathlib.Path:
        """
        Get the path to the .top file.
        This returns a single path even if multiple repeats are run since
        the GromacsMDSetupUnit is only run once.

        Returns
        -------
        top : pathlib.Path
          Path to the input topology .top file
        """
        top = [
            pus[0].outputs["system_top"]
            for pus in self.data.values()
            if "GromacsMDSetupUnit" in pus[0].source_key
        ][0]

        return top

    def get_mdp_filenames(self) -> dict[str, pathlib.Path]:
        """
        Get a dictionary of paths to the .mdp files.
        This returns a single dictionary with paths even if multiple repeats
        are run since the GromacsMDSetupUnit is only run once.

        Returns
        -------
        mdps : dict[str, pathlib.Path]
          dictionary of paths (pathlib.Path) to the mdp files for energy minimization,
          NVT and NPT MD runs
        """

        mdps = [
            pus[0].outputs["mdp_files"]
            for pus in self.data.values()
            if "GromacsMDSetupUnit" in pus[0].source_key
        ][0]

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
        - "grompp_mdp_em"

        Returns
        -------
        dict_em : Optional[dict[str, list[pathlib.Path]]]
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
            "grompp_mdp_em",
        ]
        dict_npt = {}
        for file in file_keys:
            file_path = self._get_filenames(file)
            dict_npt[file] = file_path

        return dict_npt

    def get_gro_em_filenames(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file, last frame of the
        energy minimization

        Returns
        -------
        gro : Optional[list[pathlib.Path]]
          list of paths (pathlib.Path) to the output .gro file
        """
        file_type = "gro_em"

        return self._get_filenames(file_type)

    def get_xtc_em_filenames(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .xtc file of the
        energy minimization

        Returns
        -------
        file_path : Optional[list[pathlib.Path]]
          list of paths (pathlib.Path) to the output .xtc file
        """
        file_type = "xtc_em"

        return self._get_filenames(file_type)

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
        - "grompp_mdp_nvt"

        Returns
        -------
        dict_nvt : Optional[dict[str, list[pathlib.Path]]]
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
            "grompp_mdp_nvt",
        ]
        dict_npt = {}
        for file in file_keys:
            file_path = self._get_filenames(file)
            dict_npt[file] = file_path

        return dict_npt

    def get_gro_nvt_filenames(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file, last frame of the
        NVT equilibration

        Returns
        -------
        gro : Optional[list[pathlib.Path]]
          list of paths (pathlib.Path) to the output .gro file
        """
        file_type = "gro_nvt"

        return self._get_filenames(file_type)

    def get_xtc_nvt_filenames(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .xtc file of the
        NVT equilibration

        Returns
        -------
        file_path : Optional[list[pathlib.Path]]
          list of paths (pathlib.Path) to the output .xtc file
        """
        file_type = "xtc_nvt"

        return self._get_filenames(file_type)

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
        - "grompp_mdp_npt"

        Returns
        -------
        dict_npt : Optional[dict[str, list[pathlib.Path]]]
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
            "grompp_mdp_em",
        ]
        dict_npt = {}
        for file in file_keys:
            file_path = self._get_filenames(file)
            dict_npt[file] = file_path

        return dict_npt

    def get_gro_npt_filenames(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .gro file, last frame of the
        NPT MD simulation

        Returns
        -------
        gro : Optional[list[pathlib.Path]]
          list of paths (pathlib.Path) to the output .gro file
        """
        file_type = "gro_npt"

        return self._get_filenames(file_type)

    def get_xtc_npt_filenames(self) -> list[pathlib.Path]:
        """
        Get a list of paths to the .xtc file of the
        NPT MD simulation

        Returns
        -------
        file_path : Optional[list[pathlib.Path]]
          list of paths (pathlib.Path) to the output .xtc file
        """
        file_type = "xtc_npt"

        return self._get_filenames(file_type)


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
            solvation_settings=SolvationSettings(),
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
            engine_settings=GromacsEngineSettings(),
            output_settings_em=EMOutputSettings(
                mdp_file="em.mdp",
                grompp_mdp_file="mdout_em.mdp",
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
                grompp_mdp_file="mdout_nvt.mdp",
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
                grompp_mdp_file="mdout_npt.mdp",
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

        # Raise an error when no SolventComponent is provided as this Protocol
        # currently does not support vacuum simulations
        if solvent_comp is None:
            errmsg = (
                "No SolventComponent provided. This protocol currently does"
                " not support vacuum simulations."
            )
            raise ValueError(errmsg)

        system_name = "Solvent MD"

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

    def run(
        self, *, dry=False, verbose=True, scratch_basepath=None, shared_basepath=None
    ) -> dict[str, Any]:
        """
        Write out all the files necessary to run the Gromacs MD simulation.
        MDP files, Gromacs coordinate and topology files are written.

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
          Where to store temporary files, defaults to current working directory.
        shared_basepath : Pathlike, optional
          Where to run the calculation, defaults to current working directory.

        Returns
        -------
        dict
          Outputs created in the basepath directory or the debug objects
          (i.e. sampler) if ``dry==True``.

        Raises
        ------
        error
          Exception if anything failed.

        Notes
        -----
        * This SetupUnit does not actually run the MD simulation.
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

        # 1. Write out .mdp files
        mdps = write_mdp.write_mdp_files(settings, shared_basepath)

        # 2. Create the OpenMM system

        # Create a dictionary of OFFMol for each SMC for bookkeeping
        smc_components: dict[gufe.SmallMoleculeComponent, OFFMolecule]
        smc_components = {i: i.to_openff() for i in small_mols}

        (
            stateA_system,
            stateA_topology,
            stateA_positions,
        ) = create_systems.create_openmm_system(
            solvent_comp,
            protein_comp,
            smc_components,
            settings["charge_settings"],
            settings["forcefield_settings"],
            settings["integrator_settings"],
            settings["thermo_settings"],
            settings["solvation_settings"],
            settings["output_settings_em"],
            shared_basepath,
        )
        # 3. Create the Interchange object
        stateA_interchange = create_systems.create_interchange(
            stateA_system,
            stateA_topology,
            stateA_positions,
            smc_components,
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
        out_mdp: str,
        engine_settings: GromacsEngineSettings,
        shared_basebath: pathlib.Path,
    ):
        """
        Running Gromacs gmx grompp and gmx mdrun commands using subprocess.

        Parameters
        ----------
        mdp: pathlib.Path
          Path to the mdp file
        in_gro: pathlib.Path
        top: pathlib.Path
        tpr: pathlib.Path
        out_gro: str
        xtc: str
        trr: str
        cpt: str
        log: str
        edr: str
        out_mdp: str
        engine_settings: GromacsEngineSettings
        shared_basebath: Pathlike, optional
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
                "-po",
                shared_basebath / out_mdp,
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
                "-ntmpi",
                "1",
                "-ntomp",
                str(engine_settings.ntomp),
                "-pme",
                str(engine_settings.pme),
                "-pmefft",
                str(engine_settings.pmefft),
                "-bonded",
                str(engine_settings.bonded),
                "-nb",
                str(engine_settings.nb),
                "-update",
                str(engine_settings.update),
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

        protocol_settings: GromacsMDProtocolSettings = self._inputs["protocol"].settings
        sim_settings_em: EMSimulationSettings = protocol_settings.simulation_settings_em
        sim_settings_nvt: NVTSimulationSettings = (
            protocol_settings.simulation_settings_nvt
        )
        sim_settings_npt: NPTSimulationSettings = (
            protocol_settings.simulation_settings_npt
        )
        engine_settings: GromacsEngineSettings = protocol_settings.engine_settings
        output_settings_em: EMOutputSettings = protocol_settings.output_settings_em
        output_settings_nvt: NVTOutputSettings = protocol_settings.output_settings_nvt
        output_settings_npt: NPTOutputSettings = protocol_settings.output_settings_npt

        input_gro = setup.outputs["system_gro"]
        input_top = setup.outputs["system_top"]
        mdp_files = setup.outputs["mdp_files"]

        # Create a dictionary for the run output
        output_dict = {
            "repeat_id": self._inputs["repeat_id"],
            "generation": self._inputs["generation"],
        }

        # Run energy minimization
        if sim_settings_em.nsteps > 0:
            if verbose:
                self.logger.info("Running energy minimization")
            mdp = mdp_files["em"]
            tpr = pathlib.Path(ctx.shared / output_settings_em.tpr_file)
            assert mdp.exists()
            self._run_gromacs(
                mdp,
                input_gro,
                input_top,
                tpr,
                output_settings_em.gro_file,
                output_settings_em.xtc_file,
                output_settings_em.trr_file,
                output_settings_em.cpt_file,
                output_settings_em.log_file,
                output_settings_em.edr_file,
                output_settings_em.grompp_mdp_file,
                engine_settings,
                ctx.shared,
            )
            output_dict["gro_em"] = shared_basepath / output_settings_em.gro_file
            output_dict["tpr_em"] = shared_basepath / output_settings_em.tpr_file
            output_dict["trr_em"] = shared_basepath / output_settings_em.trr_file
            output_dict["xtc_em"] = shared_basepath / output_settings_em.xtc_file
            output_dict["edr_em"] = shared_basepath / output_settings_em.edr_file
            output_dict["log_em"] = shared_basepath / output_settings_em.log_file
            output_dict["cpt_em"] = shared_basepath / output_settings_em.cpt_file
            output_dict["grompp_mdp_em"] = (
                shared_basepath / output_settings_em.grompp_mdp_file
            )

        # ToDo: Should we disallow running MD without EM?
        # Run NVT
        if sim_settings_nvt.nsteps > 0:
            if verbose:
                self.logger.info("Running an NVT MD simulation")
            mdp = mdp_files["nvt"]
            tpr = pathlib.Path(ctx.shared / output_settings_nvt.tpr_file)
            assert mdp.exists()
            # If EM was run, use the output from that to run NVT MD
            if sim_settings_em.nsteps > 0:
                gro = pathlib.Path(ctx.shared / output_settings_em.gro_file)
            else:
                gro = input_gro
            self._run_gromacs(
                mdp,
                gro,
                input_top,
                tpr,
                output_settings_nvt.gro_file,
                output_settings_nvt.xtc_file,
                output_settings_nvt.trr_file,
                output_settings_nvt.cpt_file,
                output_settings_nvt.log_file,
                output_settings_nvt.edr_file,
                output_settings_nvt.grompp_mdp_file,
                engine_settings,
                ctx.shared,
            )
            output_dict["gro_nvt"] = shared_basepath / output_settings_nvt.gro_file
            output_dict["tpr_nvt"] = shared_basepath / output_settings_nvt.tpr_file
            output_dict["trr_nvt"] = shared_basepath / output_settings_nvt.trr_file
            output_dict["xtc_nvt"] = shared_basepath / output_settings_nvt.xtc_file
            output_dict["edr_nvt"] = shared_basepath / output_settings_nvt.edr_file
            output_dict["log_nvt"] = shared_basepath / output_settings_nvt.log_file
            output_dict["cpt_nvt"] = shared_basepath / output_settings_nvt.cpt_file
            output_dict["grompp_mdp_nvt"] = (
                shared_basepath / output_settings_nvt.grompp_mdp_file
            )

        # Run NPT MD simulation
        if sim_settings_npt.nsteps > 0:
            if verbose:
                self.logger.info("Running an NPT MD simulation")
            mdp = mdp_files["npt"]
            tpr = pathlib.Path(ctx.shared / output_settings_npt.tpr_file)
            assert mdp.exists()
            # If EM and/or NVT MD was run, use the output coordinate file
            # from that to run NPT MD
            if sim_settings_em.nsteps > 0:
                if sim_settings_nvt.nsteps > 0:
                    gro = pathlib.Path(ctx.shared / output_settings_nvt.gro_file)
                else:
                    gro = pathlib.Path(ctx.shared / output_settings_em.gro_file)
            else:
                gro = input_gro
            self._run_gromacs(
                mdp,
                gro,
                input_top,
                tpr,
                output_settings_npt.gro_file,
                output_settings_npt.xtc_file,
                output_settings_npt.trr_file,
                output_settings_npt.cpt_file,
                output_settings_npt.log_file,
                output_settings_npt.edr_file,
                output_settings_npt.grompp_mdp_file,
                engine_settings,
                ctx.shared,
            )
            output_dict["gro_npt"] = shared_basepath / output_settings_npt.gro_file
            output_dict["tpr_npt"] = shared_basepath / output_settings_npt.tpr_file
            output_dict["trr_npt"] = shared_basepath / output_settings_npt.trr_file
            output_dict["xtc_npt"] = shared_basepath / output_settings_npt.xtc_file
            output_dict["edr_npt"] = shared_basepath / output_settings_npt.edr_file
            output_dict["log_npt"] = shared_basepath / output_settings_npt.log_file
            output_dict["cpt_npt"] = shared_basepath / output_settings_npt.cpt_file
            output_dict["grompp_mdp_nvt"] = (
                shared_basepath / output_settings_nvt.grompp_mdp_file
            )

        return output_dict
