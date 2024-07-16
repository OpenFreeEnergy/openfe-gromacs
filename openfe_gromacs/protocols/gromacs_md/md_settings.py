# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs

"""Settings class for plain MD Protocols using Gromacs + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.gromacs_md.md_methods.py`

"""
from typing import Literal

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
    integrator: Literal["md", "md-vv", "md-vv-avek", "sd", "steep"] = "sd"
    """
    MD integrators and other algorithms (steep for energy minimization).
    Allowed values are:
    'md': a leap-frog integrator
    'md-vv': a velocity Verlet integrator
    'md-vv-avek': A velocity Verlet algorithm identical to integrator=md-vv,
                  except that the kinetic energy is determined as the average
                  of the two half step kinetic energies as in the
                  integrator=md integrator, and this thus more accurate.
    'sd': A leap-frog stochastic dynamics integrator. Note that when using this
          integrator the parameters tcoupl and nsttcouple are ignored.
    'steep': A steepest descent algorithm for energy minimization.
    """
    tinit: FloatQuantity["picosecond"] = 0 * unit.picosecond
    """
    Starting time for the MD run.
    This only makes sense for time-based integrators.
    Default 0 * unit.picosecond
    """
    dt: FloatQuantity["picosecond"] = 0.001 * unit.picosecond
    """
    Time step for integration (only makes sense for time-based integrators).
    Default 0.001 * unit.picosecond
    """
    nsteps: int = 0
    """
    Maximum number of steps to integrate or minimize, -1 is no maximum
    Default 0
    """
    init_step: int = 0
    """
    The starting step. The time at step i in a run is calculated as:
    t = tinit + dt * (init-step + i).
    Default 0
    """
    simulation_part: int = 0
    """
    A simulation can consist of multiple parts, each of which has a part number.
     This option specifies what that number will be, which helps keep track of
     parts that are logically the same simulation. This option is generally
     useful to set only when coping with a crashed simulation where files were
     lost. Default 0
    """
    mass_repartition_factor: int = 1
    """
    Scales the masses of the lightest atoms in the system by this factor to
    the mass mMin. All atoms with a mass lower than mMin also have their mass
    set to that mMin. Default 1 (no mass scaling)
    """
    comm_mode: Literal[
        "Linear", "Angular", "Linear-acceleration-correction", "None"
    ] = "Linear"
    """
    Settings for center of mass treatmeant
    Allowed values are:
    'Linear': Remove center of mass translational velocity
    'Angular': Remove center of mass translational and rotational velocity
    'Linear-acceleration-correction': Remove center of mass translational
    velocity. Correct the center of mass position assuming linear acceleration
    over nstcomm steps.
    'None': No restriction on the center of mass motion
    """
    nstcomm: int = 100
    """
    Frequency for center of mass motion removal in unit [steps].
    Default 100
    """
    comm_grps: str
    """
    Group(s) for center of mass motion removal, default is the whole system.
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
    verlet_buffer_tolerance: FloatQuantity["kilojoule / (mole * picosecond)"] = (
        0.005 * unit.kilojoule / (unit.mole * unit.picosecond)
    )
    """"
    Used when performing a simulation with dynamics. This sets the maximum
    allowed error for pair interactions per particle caused by the Verlet
    buffer, which indirectly sets rlist.
    Default 0.005 * unit.kilojoule / (unit.mole * unit.picosecond)
    """
    verlet_buffer_pressure_tolerance: FloatQuantity["bar"] = 0.5 * unit.bar
    """
    Used when performing a simulation with dynamics and only active when
    verlet-buffer-tolerance is positive. This sets the maximum tolerated error
    in the average pressure due to missing Lennard-Jones interactions of
    particle pairs that are not in the pair list, but come within rvdw range
    as the pair list ages.
    Default 0.5 * unit.bar
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
    coulomb_modifier: Literal["Potential-shift", "None"] = "Potential-shift"
    """
    Allowed options are:
    'Potential-shift': Shift the Coulomb potential by a constant such that it
          is zero at the cut-off.
    'None': Use an unmodified Coulomb potential. This can be useful when
          comparing energies with those computed with other software.
    Default 'Potential-shift'
    """
    rcoulomb_switch: FloatQuantity["nanometer"] = 0 * unit.nanometer
    """
    Where to start switching the Coulomb potential, only relevant when force
    or potential switching is used.
    Default 0 * unit.nanometer
    """
    rcoulomb: FloatQuantity["nanometer"] = 1.2 * unit.nanometer
    """"
    The distance for the Coulomb cut-off. Note that with PME this value can be
    increased by the PME tuning in gmx mdrun along with the PME grid spacing.
    Default 1.2 * unit.nanometer
    """
    epsilon_r: int = 1
    """
    The relative dielectric constant. A value of 0 means infinity. Default 1.
    """
    epsilon_rf: int = 0
    """
    The relative dielectric constant of the reaction field. This is only used
    with reaction-field electrostatics. A value of 0 means infinity. Default 0
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
    vdw_modifier: Literal[
        "Potential-shift", "None", "Force-switch", "Potential-switch"
    ] = "Potential-shift"
    """
    Allowed values are:
    'Potential-shift': Shift the Van der Waals potential by a constant such
        that it is zero at the cut-off.
    'None': Use an unmodified Van der Waals potential. This can be useful when
        comparing energies with those computed with other software.
    'Force-switch': Smoothly switches the forces to zero between rvdw-switch
        and rvdw. This shifts the potential shift over the whole range and
        switches it to zero at the cut-off. Note that this is more expensive
        to calculate than a plain cut-off and it is not required for energy
        conservation, since Potential-shift conserves energy just as well.
    'Potential-switch': Smoothly switches the potential to zero between
        rvdw-switch and rvdw. Note that this introduces articifically large
        forces in the switching region and is much more expensive to calculate.
        This option should only be used if the force field you are using
        requires this.
    """
    rvdw_switch: FloatQuantity["nanometer"] = 0 * unit.nanometer
    """"
    Where to start switching the LJ force and possibly the potential, only
    relevant when force or potential switching is used.
    Default 0 * unit.nanometer
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
    fourierspacing: FloatQuantity["nanometer"] = 0.12 * unit.nanometer
    """
    For ordinary Ewald, the ratio of the box dimensions and the spacing
    determines a lower bound for the number of wave vectors to use in each
    (signed) direction. For PME and P3M, that ratio determines a lower bound
    for the number of Fourier-space grid points that will be used along that
    axis. In all cases, the number for each direction can be overridden by
    entering a non-zero value for that fourier-nx direction.
    Default 0.12 * unit.nanometer
    """
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
    lj_pme_comb_rule: Literal["Geometric", "Lorentz-Berthelot"] = "Geometric"
    """
    The combination rules used to combine VdW-parameters in the reciprocal part
    of LJ-PME. Geometric rules are much faster than Lorentz-Berthelot and
    usually the recommended choice, even when the rest of the force field uses
    the Lorentz-Berthelot rules. Allowed values are:
    'Geometric': Apply geometric combination rules.
    'Lorentz-Berthelot': Apply Lorentz-Berthelot combination rules.
    Default 'Geometric'.
    """
    ewald_geometry: Literal["3d", "3dc"] = "3d"
    """
    Allowed values are:
    '3d': The Ewald sum is performed in all three dimensions.
    '3dc': The reciprocal sum is still performed in 3D, but a force and
    potential correction applied in the z dimension to produce a pseudo-2D
    summation.
    Default '3d'.
    """
    epsilon_surface: float = 0
    """
    This controls the dipole correction to the Ewald summation in 3D.
    The default value of zero means it is turned off. Turn it on by setting it
    to the value of the relative permittivity of the imaginary surface around
    your infinite system. Be careful - you shouldn’t use this if you have free
    mobile charges in your system. This value does not affect the slab 3DC
    variant of the long range corrections. Default 0.
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
    nsttcouple: int = -1
    """
    The frequency for coupling the temperature. The default value of -1 sets
    nsttcouple equal to 100, or fewer steps if required for accurate
    integration (5 steps per tau for first order coupling, 20 steps per tau for
    second order coupling). Note that the default value is large in order to
    reduce the overhead of the additional computation and communication
    required for obtaining the kinetic energy. For velocity Verlet integrators
    nsttcouple is set to 1.
    Default -1.
    """
    tc_grps: str = "system"
    """
    Groups to couple to separate temperature baths. Default 'system'.
    """
    tau_t: FloatQuantity["picosecond"] = 2.0 * unit.picosecond
    """"
    Time constant for coupling (one for each group in tc-grps), -1 means no
    temperature coupling. Default 2.0 * unit.picosecond.
    """
    ref_t: FloatQuantity["Kelvin"] = 298.15 * unit.kelvin
    """
    Reference temperature for coupling (one for each group in tc-grps).
    Default 298.15 * unit.kelvin
    """

    # # # Pressure coupling # # #
    pcoupl: Literal["no", "berendsen", "C-rescale", "Parrinello-Rahman", "MTTK"] = "no"
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
    'MTTK': Martyna-Tuckerman-Tobias-Klein implementation, only useable with
        integrator=md-vv or integrator=md-vv-avek, very similar to
        Parrinello-Rahman. As for Nose-Hoover temperature coupling the time
        constant tau-p is the period of pressure fluctuations at equilibrium.
        This is probably a better method when you want to apply pressure
        scaling during data collection, but beware that you can get very large
        oscillations if you are starting from a different pressure.
        This requires a constant ensemble temperature for the system.
        Currently it only supports isotropic scaling, and only works without
        constraints.
    Default 'no'.
    """
    pcoupltype: Literal[
        "isotropic", "semiisotropic", "anisotropic", "surface-tension"
    ] = "isotropic"
    """
    Specifies the kind of isotropy of the pressure coupling used. Each kind
    takes one or more values for compressibility and ref-p. Only a single value
    is permitted for tau-p. Allowed values are:
    'isotropic': sotropic pressure coupling with time constant tau-p. One value
        each for compressibility and ref-p is required.
    'semiisotropic': Pressure coupling which is isotropic in the x and y
        direction, but different in the z direction. This can be useful for
        membrane simulations. Two values each for compressibility and ref-p are
        required, for x/y and z directions respectively.
    'anisotropic': Same as before, but 6 values are needed for xx, yy, zz,
        xy/yx, xz/zx and yz/zy components, respectively. When the off-diagonal
        compressibilities are set to zero, a rectangular box will stay
        rectangular. Beware that anisotropic scaling can lead to extreme
        deformation of the simulation box.
    'surface-tension': Surface tension coupling for surfaces parallel to the
        xy-plane. Uses normal pressure coupling for the z-direction, while the
        surface tension is coupled to the x/y dimensions of the box. The first
        ref-p value is the reference surface tension times the number of
        surfaces bar nm, the second value is the reference z-pressure bar.
        The two compressibility values are the compressibility in the x/y and z
        direction respectively. The value for the z-compressibility should be
        reasonably accurate since it influences the convergence of the
        surface-tension, it can also be set to zero to have a box with constant
        height.
    Default 'isotropic'
    """
    nstpcouple: int = -1
    """
    The frequency for coupling the pressure. The default value of -1 sets
    nstpcouple equal to 100, or fewer steps if required for accurate
    integration (5 steps per tau for first order coupling, 20 steps per tau for
    second order coupling). Note that the default value is large in order to
    reduce the overhead of the additional computation and communication
    required for obtaining the virial and kinetic energy. For velocity Verlet
    integrators nsttcouple is set to 1. Default -1.
    """
    tau_p: FloatQuantity["picosecond"] = 5 * unit.picosecond
    """
    The time constant for pressure coupling (one value for all directions).
    Default 5 * unit.picosecond.
    """
    compressibility: FloatQuantity["1/bar"] = 4.5e-05 / unit.bar
    """
    The compressibility. For water at 1 atm and 300 K the compressibility is
    4.5e-5 bar-1. The number of required values is implied by pcoupltype.
    Default 4.5e-05 / unit.bar
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
    lincs_warnangle: FloatQuantity["deg"] = 30 * unit.degree
    """
    Maximum angle that a bond can rotate before LINCS will complain.
    Default 30 * unit.degree.
    """
    morse: Literal["no", "yes"] = "no"
    """
    Allowed options are:
    'no': Bonds are represented by a harmonic potential.
    'yes': Bonds are represented by a Morse potential.
    Default 'no'.
    """


class OutputSettings(SettingsBaseModel):
    """ "
    Output Settings for simulations run using Gromacs
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
    compressed_x_grps: str
    """
    Group(s) to write to the compressed trajectory file, by default the whole
    system is written (if nstxout-compressed > 0).
    """
    energygrps: str
    """
    Group(s) for which to write to write short-ranged non-bonded potential
    energies to the energy file (not supported on GPUs)
    """


class EMSimulationSettings(SimulationSettings):
    """
    Settings for energy minimization.
    """

    integrator = "steep"
    nsteps = 5000

    emtol: FloatQuantity["kilojoule / (mole * nanometer)"] = (
        10.0 * unit.kilojoule / (unit.mole * unit.nanometer)
    )
    """
    The minimization is converged when the maximum force is smaller than this
    value. Default 10.0 * unit.kilojoule / (unit.mole * unit.nanometer)
    """
    emstep: FloatQuantity["nanometer"] = 0.01 * unit.nanometer
    """
    Initial step size. Default 0.01 * unit.nanometer
    """


class EMOutputSettings(OutputSettings):
    """
    Output Settings for the energy minimization.
    """


class NVTSimulationSettings(SimulationSettings):
    """
    Settings for MD simulation in the NVT ensemble.
    """


class NVTOutputSettings(OutputSettings):
    """
    Output Settings for the MD simulation in the NVT ensemble.
    """


class NPTSimulationSettings(SimulationSettings):
    """
    Settings for MD simulation in the NPT ensemble.
    """


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
