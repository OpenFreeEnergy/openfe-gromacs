# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs

"""Settings class for plain MD Protocols using Gromacs + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.gromacs_md.md_methods.py`

"""
from typing import Literal, Optional

from gufe.settings import OpenMMSystemGeneratorFFSettings, SettingsBaseModel
from openfe.protocols.openmm_utils.omm_settings import (
    IntegratorSettings,
    OpenFFPartialChargeSettings,
    OpenMMEngineSettings,
    OpenMMSolvationSettings,
    Settings,
)
from openff.models.types import FloatQuantity
from openff.units import unit

try:
    from pydantic.v1 import validator
except ImportError:
    from pydantic import validator  # type: ignore[assignment]


class SimulationSettings(SettingsBaseModel):
    """
    Settings for simulations in Gromacs.
    """

    class Config:
        arbitrary_types_allowed = True

    # # # Run control # # #
    integrator: Literal["md", "sd", "steep"] = "sd"
    """
    MD integrators and other algorithms (steep for energy minimization).
    Allowed values are:
    'md': a leap-frog integrator
    'sd': A leap-frog stochastic dynamics integrator. Note that when using this
          integrator the parameters tcoupl and nsttcouple are ignored.
    'steep': A steepest descent algorithm for energy minimization.
    """
    dt: FloatQuantity["picosecond"] = 0.002 * unit.picosecond
    """
    Time step for integration (only makes sense for time-based integrators).
    Default 0.002 * unit.picosecond
    """
    nsteps: int = 0
    """
    Maximum number of steps to integrate or minimize, -1 is no maximum
    Default 0
    """
    mass_repartition_factor: int = 1
    """
    Scales the masses of the lightest atoms in the system by this factor to
    the mass mMin. All atoms with a mass lower than mMin also have their mass
    set to that mMin. Default 1 (no mass scaling)
    """

    # # # Langevin dynamics # # #
    ld_seed: int = -1
    """
    Integer used to initialize random generator for thermal noise for
    stochastic and Brownian dynamics. When ld-seed is set to -1,
    a pseudo random seed is used. Default -1.
    """

    # # # Neighbor searching # # #
    cutoff_scheme: Literal["verlet"] = "verlet"
    """
    Only allowed option:
    'verlet': Generate a pair list with buffering. The buffer size is
              automatically set based on verlet-buffer-tolerance, unless this
              is set to -1, in which case rlist will be used.
    """
    nstlist: int = 10
    """
    Frequency to update the neighbor list. When dynamics and
    verlet-buffer-tolerance set, nstlist is actually a minimum value and
    gmx mdrun might increase it, unless it is set to 1.
    If set to zero the neighbor list is only constructed once and never updated.
    Default 10.
    """
    pbc: Literal["xyz", "no", "xy"] = "xyz"
    """
    Treatment of periodic boundary conditions.
    Allowed values are:
    'xyz': Use periodic boundary conditions in all directions.
    'no': Use no periodic boundary conditions, ignore the box.
    'xy': Use periodic boundary conditions in x and y directions only.
          This can be used in combination with walls.
    Default 'xyz'.
    """
    rlist: FloatQuantity["nanometer"] = 1 * unit.nanometer
    """
    Cut-off distance for the short-range neighbor list. With dynamics, this is
    by default set by the verlet-buffer-tolerance and
    verlet-buffer-pressure-tolerance options and the value of rlist is ignored.
    """

    # # # Electrostatics # # #
    coulombtype: Literal["cut-off", "ewald", "PME", "P3M-AD", "Reaction-Field"] = "PME"
    """
    Treatment of electrostatics
    Allowed values are:
    'cut-off': Plain cut-off with pair list radius rlist and Coulomb cut-off
           rcoulomb, where rlist >= rcoulomb.
    'ewald': Classical Ewald sum electrostatics.
           The real-space cut-off rcoulomb should be equal to rlist.
    'PME': Fast smooth Particle-Mesh Ewald (SPME) electrostatics. Direct space
           is similar to the Ewald sum, while the reciprocal part is performed
           with FFTs. Grid dimensions are controlled with fourierspacing and
           the interpolation order with pme-order.
    'P3M-AD': Particle-Particle Particle-Mesh algorithm with analytical
           derivative for for long range electrostatic interactions. The
           method and code is identical to SPME, except that the influence
           function is optimized for the grid. This gives a slight increase
           in accuracy.
    'Reaction-Field': Reaction field electrostatics with Coulomb cut-off
           rcoulomb, where rlist >= rvdw. The dielectric constant beyond the
           cut-off is epsilon-rf. The dielectric constant can be set to
           infinity by setting epsilon-rf =0.
    Default 'PME'
    """
    rcoulomb: FloatQuantity["nanometer"] = 1.2 * unit.nanometer
    """"
    The distance for the Coulomb cut-off. Note that with PME this value can be
    increased by the PME tuning in gmx mdrun along with the PME grid spacing.
    Default 1.2 * unit.nanometer
    """

    # # # Van der Waals # # #
    vdwtype: Literal["Cut-off", "PME"] = "Cut-off"
    """
    Treatment of vdW interactions. Allowed options are:
    'Cut-off': Plain cut-off with pair list radius rlist and VdW cut-off rvdw,
    where rlist >= rvdw.
    'PME: Fast smooth Particle-mesh Ewald (SPME) for VdW interactions.
    Default 'Cut-off'.
    """
    vdw_modifier: Literal["Potential-shift", "None"] = "Potential-shift"
    """
    Allowed values are:
    'Potential-shift': Shift the Van der Waals potential by a constant such
        that it is zero at the cut-off.
    'None': Use an unmodified Van der Waals potential. This can be useful when
        comparing energies with those computed with other software.
    """
    rvdw: FloatQuantity["nanometer"] = 1.0 * unit.nanometer
    """
    Distance for the LJ or Buckingham cut-off. Default 1 * unit.nanometer
    """
    DispCorr: Literal["no", "EnerPres", "Ener"] = "EnerPres"
    """
    Allowed values are:
    'no': Don’t apply any correction
    'EnerPres': Apply long range dispersion corrections for Energy and Pressure.
    'Ener': Apply long range dispersion corrections for Energy only.
    Default 'EnerPres'
    """

    # # # Ewald # # #
    pme_order: int = 4
    """
    The number of grid points along a dimension to which a charge is mapped.
    The actual order of the PME interpolation is one less, e.g. the default of
    4 gives cubic interpolation. Supported values are 3 to 12 (max 8 for
    P3M-AD). When running in parallel, it can be worth to switch to 5 and
    simultaneously increase the grid spacing. Note that on the CPU only values
    4 and 5 have SIMD acceleration and GPUs only support the value 4.
    Default 4.
    """
    ewald_rtol: float = 1e-5
    """
    The relative strength of the Ewald-shifted direct potential at rcoulomb is
    given by ewald-rtol. Decreasing this will give a more accurate direct sum,
    but then you need more wave vectors for the reciprocal sum.
    Default 1e-5
    """

    # # # Temperature coupling # # #
    tcoupl: Literal[
        "no", "berendsen", "nose-hoover", "andersen", "andersen-massive", "v-rescale"
    ] = "no"
    """
    Temperature coupling options. Note that tcoupl will be ignored when the
    'sd' integrator is used.
    Allowed values are:
    'no': No temperature coupling.
    'berendsen': Temperature coupling with a Berendsen thermostat to a bath
        with temperature ref-t, with time constant tau-t. Several groups can be
        coupled separately, these are specified in the tc-grps field separated
        by spaces. This is a historical thermostat needed to be able to
        reproduce previous simulations, but we strongly recommend not to use
        it for new production runs.
    'nose-hoover': Temperature coupling using a Nose-Hoover extended ensemble.
        The reference temperature and coupling groups are selected as above,
        but in this case tau-t controls the period of the temperature
        fluctuations at equilibrium, which is slightly different from a
        relaxation time. For NVT simulations the conserved energy quantity is
        written to the energy and log files.
    'andersen': Temperature coupling by randomizing a fraction of the particle
        velocities at each timestep. Reference temperature and coupling groups
        are selected as above. tau-t is the average time between randomization
        of each molecule. Inhibits particle dynamics somewhat, but little or no
        ergodicity issues. Currently only implemented with velocity Verlet, and
        not implemented with constraints.
    'andersen-massive': Temperature coupling by randomizing velocities of all
        particles at infrequent timesteps. Reference temperature and coupling
        groups are selected as above. tau-t is the time between randomization
        of all molecules. Inhibits particle dynamics somewhat, but little or
        no ergodicity issues. Currently only implemented with velocity Verlet.
    'v-rescale': Temperature coupling using velocity rescaling with a
        stochastic term (JCP 126, 014101). This thermostat is similar to
        Berendsen coupling, with the same scaling using tau-t, but the
        stochastic term ensures that a proper canonical ensemble is generated.
        The random seed is set with ld-seed. This thermostat works correctly
        even for tau-t =0. For NVT simulations the conserved energy quantity
        is written to the energy and log file.
    Default 'no' (for the default integrator, 'sd', this option is ignored).
    """
    ref_t: FloatQuantity["kelvin"] = 298.15 * unit.kelvin
    """
    Reference temperature for coupling (one for each group in tc-grps).
    Default 298.15 * unit.kelvin
    """

    # # # Pressure coupling # # #
    pcoupl: Literal["no", "berendsen", "C-rescale", "Parrinello-Rahman"] = "no"
    """
    Options for pressure coupling (barostat). Allowed values are:
    'no': No pressure coupling. This means a fixed box size.
    'berendsen': Exponential relaxation pressure coupling with time constant
        tau-p. The box is scaled every nstpcouple steps. This barostat does not
        yield a correct thermodynamic ensemble; it is only included to be able
        to reproduce previous runs, and we strongly recommend against using it
        for new simulations.
    'C-rescale': Exponential relaxation pressure coupling with time constant
        tau-p, including a stochastic term to enforce correct volume
        fluctuations. The box is scaled every nstpcouple steps. It can be used
        for both equilibration and production.
    'Parrinello-Rahman': Extended-ensemble pressure coupling where the box
        vectors are subject to an equation of motion. The equation of motion
        for the atoms is coupled to this. No instantaneous scaling takes place.
        As for Nose-Hoover temperature coupling the time constant tau-p is the
        period of pressure fluctuations at equilibrium.
    Default 'no'.
    """
    ref_p: FloatQuantity["bar"] = 1.01325 * unit.bar
    """
    The reference pressure for coupling. The number of required values is
    implied by pcoupltype. Default 1.01325 * unit.bar.
    """
    refcoord_scaling: Literal["no", "all", "com"] = "no"
    """
    Allowed values are:
    'no': The reference coordinates for position restraints are not modified.
    'all': The reference coordinates are scaled with the scaling matrix of the
        pressure coupling.
    'com': Scale the center of mass of the reference coordinates with the
        scaling matrix of the pressure coupling. The vectors of each reference
        coordinate to the center of mass are not scaled. Only one COM is used,
        even when there are multiple molecules with position restraints.
        For calculating the COM of the reference coordinates in the starting
        configuration, periodic boundary conditions are not taken into account.
    Default 'no'.
    """

    # # # Velocity generation # # #
    gen_vel: Literal["no", "yes"] = "yes"
    """
    Velocity generation. Allowed values are:
    'no': Do not generate velocities. The velocities are set to zero when there
        are no velocities in the input structure file.
    'yes': Generate velocities in gmx grompp according to a Maxwell
        distribution at temperature gen-temp, with random seed gen-seed.
        This is only meaningful with integrator=md.
    Default 'yes'.
    """
    gen_temp: FloatQuantity["kelvin"] = 298.15 * unit.kelvin
    """
    Temperature for Maxwell distribution. Default 298.15 * unit.kelvin.
    """
    gen_seed: int = -1
    """
    Used to initialize random generator for random velocities, when gen-seed is
    set to -1, a pseudo random seed is used. Default -1.
    """

    # # # Bonds # # #
    constraints: Literal[
        "none", "h-bonds", "all-bonds", "h-angles", "all-angles"
    ] = "h-bonds"
    """
    Controls which bonds in the topology will be converted to rigid holonomic
    constraints. Note that typical rigid water models do not have bonds, but
    rather a specialized [settles] directive, so are not affected by this
    keyword. Allowed values are:
    'none': No bonds converted to constraints.
    'h-bonds': Convert the bonds with H-atoms to constraints.
    'all-bonds': Convert all bonds to constraints.
    'h-angles': Convert all bonds to constraints and convert the angles that
        involve H-atoms to bond-constraints.
    'all-angles': Convert all bonds to constraints and all angles to
         bond-constraints.
    Default 'h-bonds'
    """
    constraint_algorithm: Literal["lincs", "shake"] = "lincs"
    """
    Chooses which solver satisfies any non-SETTLE holonomic constraints.
    Allowed values are:
    'lincs': LINear Constraint Solver. With domain decomposition the parallel
        version P-LINCS is used. The accuracy in set with lincs-order, which
        sets the number of matrices in the expansion for the matrix inversion.
        After the matrix inversion correction the algorithm does an iterative
        correction to compensate for lengthening due to rotation. The number of
        such iterations can be controlled with lincs-iter. The root mean square
        relative constraint deviation is printed to the log file every nstlog
        steps. If a bond rotates more than lincs-warnangle in one step, a
        warning will be printed both to the log file and to stderr.
        LINCS should not be used with coupled angle constraints.
    'shake': SHAKE is slightly slower and less stable than LINCS, but does work
        with angle constraints. The relative tolerance is set with shake-tol,
        0.0001 is a good value for “normal” MD. SHAKE does not support
        constraints between atoms on different decomposition domains, so it can
        only be used with domain decomposition when so-called update-groups are
        used, which is usally the case when only bonds involving hydrogens are
        constrained. SHAKE can not be used with energy minimization.
    Default 'lincs'
    """
    continuation: Literal["no", "yes"] = "no"
    """
    This option was formerly known as unconstrained-start.
    Allowed values are:
    'no': Apply constraints to the start configuration and reset shells.
    'yes': Do not apply constraints to the start configuration and do not reset
    shells, useful for exact coninuation and reruns.
    """
    shake_tol: float = 0.0001
    """ Relative tolerance for SHAKE. Default 0.0001. """
    lincs_order: int = 12
    """
    Highest order in the expansion of the constraint coupling matrix. When
    constraints form triangles, an additional expansion of the same order is
    applied on top of the normal expansion only for the couplings within such
    triangles. For “normal” MD simulations an order of 4 usually suffices, 6 is
    needed for large time-steps with virtual sites or BD. For accurate energy
    minimization in double precision an order of 8 or more might be required.
    Note that in single precision an order higher than 6 will often lead to
    worse accuracy due to amplification of rounding errors. Default 12.
    """
    lincs_iter: int = 1
    """
    Number of iterations to correct for rotational lengthening in LINCS.
    Default 1.
    """

    @validator(
        "nsteps",
        "nstlist",
        "rlist",
        "rcoulomb",
        "rvdw",
        "ewald_rtol",
        "ref_t",
        "gen_temp",
        "shake_tol",
        "lincs_order",
        "lincs_iter",
    )
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = (
                "Settings nsteps, nstlist, rlist, rcoulomb, rvdw, ewald_rtol, "
                "ref_t, gen_temp, shake_tol, lincs_order, and lincs_iter"
                f" must be zero or positive values, got {v}."
            )
            raise ValueError(errmsg)
        return v

    @validator("dt", "mass_repartition_factor")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = (
                "timestep dt, and mass_repartition_factor "
                f"must be positive values, got {v}."
            )
            raise ValueError(errmsg)
        return v

    @validator("pme_order")
    def must_be_between_3_12(cls, v):
        if not 3 <= v <= 12:
            errmsg = "pme_order " f"must be between 3 and 12, got {v}."
            raise ValueError(errmsg)
        return v

    @validator("dt")
    def is_time(cls, v):
        if not v.is_compatible_with(unit.picosecond):
            raise ValueError("dt must be in time units " "(i.e. picoseconds)")

    @validator("rlist", "rcoulomb", "rvdw")
    def is_distance(cls, v):
        if not v.is_compatible_with(unit.nanometer):
            raise ValueError(
                "rlist, rcoulomb, and rvdw must be in distance "
                "units (i.e. nanometers)"
            )

    @validator("ref_t", "gen_temp")
    def is_temperature(cls, v):
        if not v.is_compatible_with(unit.kelvin):
            raise ValueError(
                "ref_t and gen_temp must be in temperature units (i.e. kelvin)"
            )

    @validator("ref_p")
    def is_pressure(cls, v):
        if not v.is_compatible_with(unit.bar):
            raise ValueError("ref_p must be in pressure units (i.e. bar)")

    @validator("integrator")
    def supported_integrator(cls, v):
        supported = ["md", "sd", "steep"]
        if v.lower() not in supported:
            errmsg = (
                "Only the following sampler_method values are "
                f"supported: {supported}, got {v}"
            )
            raise ValueError(errmsg)
        return v


class OutputSettings(SettingsBaseModel):
    """ "
    Output Settings for simulations run using Gromacs
    """

    forcefield_cache: Optional[str] = "db.json"
    """
    Filename for caching small molecule residue templates so they can be
    later reused.
    """
    mdp_file: str = "em.mdp"
    """
    Filename for the mdp file for running simulations in Gromacs.
    Default 'em.mdp'
    """
    nstxout: int = 0
    """
    Number of steps that elapse between writing coordinates to the output
    trajectory file (trr), the last coordinates are always written unless 0,
    which means coordinates are not written into the trajectory file.
    Default 0.
    """
    nstvout: int = 0
    """
    Number of steps that elapse between writing velocities to the output
    trajectory file (trr), the last velocities are always written unless 0,
    which means velocities are not written into the trajectory file.
    Default 0.
    """
    nstfout: int = 0
    """
    Number of steps that elapse between writing forces to the output trajectory
    file (trr), the last forces are always written, unless 0, which means
    forces are not written into the trajectory file.
    Default 0.
    """
    nstlog: int = 1000
    """
    Number of steps that elapse between writing energies to the log file, the
    last energies are always written. Default 1000.
    """
    nstcalcenergy: int = 100
    """
    Number of steps that elapse between calculating the energies, 0 is never.
    This option is only relevant with dynamics. This option affects the
    performance in parallel simulations, because calculating energies requires
    global communication between all processes which can become a bottleneck at
    high parallelization. Default 100.
    """
    nstenergy: int = 1000
    """"
    Number of steps that elapse between writing energies to energy file, the
    last energies are always written, should be a multiple of nstcalcenergy.
    Note that the exact sums and fluctuations over all MD steps modulo
    nstcalcenergy are stored in the energy file, so gmx energy can report
    exact energy averages and fluctuations also when nstenergy > 1
    """
    nstxout_compressed: int = 0
    """
    Number of steps that elapse between writing position coordinates using
    lossy compression (xtc file), 0 for not writing compressed coordinates
    output. Default 0.
    """
    compressed_x_precision: int = 1000
    """
    Precision with which to write to the compressed trajectory file.
    Default 1000.
    """
    compressed_x_grps: str = ""
    """
    Group(s) to write to the compressed trajectory file, by default the whole
    system is written (if nstxout-compressed > 0).
    """
    energygrps: str = ""
    """
    Group(s) for which to write to write short-ranged non-bonded potential
    energies to the energy file (not supported on GPUs)
    """

    @validator(
        "nstxout",
        "nstvout",
        "nstfout",
        "nstlog",
        "nstcalcenergy",
        "nstenergy",
        "nstxout_compressed",
        "compressed_x_precision",
    )
    def must_be_positive_or_zero(cls, v):
        if v < 0:
            errmsg = (
                "nstxout, nstvout, nstfout, nstlog, nstcalcenergy, nstenergy, "
                "nstxout_compressed, and compressed_x_precision must be zero "
                f"or positive values, got {v}."
            )
            raise ValueError(errmsg)
        return v


class EMSimulationSettings(SimulationSettings):
    """
    Settings for energy minimization.
    """

    @validator("integrator")
    def is_steep(cls, v):
        # EM should have 'steep' integrator
        if v != "steep":
            errmsg = (
                "For energy minimization, only the integrator=steep "
                f"is supported, got integrator={v}."
            )
            raise ValueError(errmsg)
        return v


class EMOutputSettings(OutputSettings):
    """
    Output Settings for the energy minimization.
    """


class NVTSimulationSettings(SimulationSettings):
    """
    Settings for MD simulation in the NVT ensemble.
    """

    @validator("integrator")
    def is_not_steep(cls, v):
        # needs an MD integrator
        if v == "steep":
            errmsg = (
                "Molecular Dynamics settings need an MD integrator, "
                f"not integrator={v}."
            )
            raise ValueError(errmsg)
        return v

    @validator("pcoupl")
    def has_no_barostat(cls, v):
        # NVT cannot have a barostat
        if v != "no":
            errmsg = f"NVT settings cannot have a barostat, got pcoupl={v}."
            raise ValueError(errmsg)
        return v


class NVTOutputSettings(OutputSettings):
    """
    Output Settings for the MD simulation in the NVT ensemble.
    """


class NPTSimulationSettings(SimulationSettings):
    """
    Settings for MD simulation in the NPT ensemble.
    """

    @validator("integrator")
    def is_not_steep(cls, v):
        # needs an MD integrator
        if v == "steep":
            errmsg = (
                "Molecular Dynamics settings need an MD integrator, "
                f"not integrator={v}."
            )
            raise ValueError(errmsg)
        return v

    @validator("pcoupl")
    def has_barostat(cls, v):
        # NPT needs a barostat
        if v == "no":
            errmsg = f"NPT settings need a barostat, got pcoupl={v}."
            raise ValueError(errmsg)
        return v


class NPTOutputSettings(OutputSettings):
    """
    Output Settings for the MD simulation in the NPT ensemble.
    """


class GromacsMDProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    protocol_repeats: int
    """
    Number of independent MD runs to perform.
    """

    @validator("protocol_repeats")
    def must_be_positive(cls, v):
        if v <= 0:
            errmsg = f"protocol_repeats must be a positive value, got {v}."
            raise ValueError(errmsg)
        return v

    # File names for .gro and .top file
    gro: str
    top: str

    # Things for creating the systems
    forcefield_settings: OpenMMSystemGeneratorFFSettings
    partial_charge_settings: OpenFFPartialChargeSettings
    solvation_settings: OpenMMSolvationSettings

    # MD Engine things
    engine_settings: OpenMMEngineSettings

    # Sampling State defining things
    integrator_settings: IntegratorSettings

    # Simulation run settings
    simulation_settings_em: EMSimulationSettings
    simulation_settings_nvt: NVTSimulationSettings
    simulation_settings_npt: NPTSimulationSettings

    # Simulations output settings
    output_settings_em: EMOutputSettings
    output_settings_nvt: NVTOutputSettings
    output_settings_npt: NPTOutputSettings
