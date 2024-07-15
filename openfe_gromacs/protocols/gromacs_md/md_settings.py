# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs

"""Settings class for plain MD Protocols using Gromacs + OpenMMTools

This module implements the settings necessary to run MD simulations using
:class:`openfe.protocols.gromacs_md.md_methods.py`

"""
from typing import Optional, Literal
from openff.units import unit
from openff.models.types import FloatQuantity
from openfe.protocols.openmm_utils.omm_settings import (
    Settings,
    OpenMMSolvationSettings,
    OpenMMEngineSettings,
    IntegratorSettings,
    OpenFFPartialChargeSettings,
)
from gufe.settings import (
    SettingsBaseModel,
    OpenMMSystemGeneratorFFSettings
)
try:
    from pydantic.v1 import validator
except ImportError:
    from pydantic import validator  # type: ignore[assignment]

class MDSimulationSettings(SettingsBaseModel):
    """
    Settings for MD simulations in Gromacs.
    """
    class Config:
        arbitrary_types_allowed = True

    ### Run control ###
    integrator: Literal['md', 'md-vv', 'md-vv-avek', 'sd', 'steep'] = 'sd'
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
    tinit: FloatQuantity['picosecond'] = 0 * unit.picosecond
    """
    Starting time for the MD run. 
    This only makes sense for time-based integrators.
    Default 0 * unit.picosecond 
    """
    dt: FloatQuantity['picosecond'] = 0.001 * unit.picosecond
    """
    Time step for integration (only makes sense for time-based integrators).
    Default 0.001 * unit.picosecond
    """
    nsteps: int = 0
    """
    Maximum number of steps to integrate or minimize, -1 is no maximum
    Default 0
    """
    init-step: int = 0
    """
    The starting step. The time at step i in a run is calculated as: 
    t = tinit + dt * (init-step + i).
    Default 0
    """
    simulation-part: int=0
    """
    A simulation can consist of multiple parts, each of which has a part number.
     This option specifies what that number will be, which helps keep track of 
     parts that are logically the same simulation. This option is generally 
     useful to set only when coping with a crashed simulation where files were 
     lost. Default 0
    """
    mass-repartition-factor: int = 1
    """
    Scales the masses of the lightest atoms in the system by this factor to 
    the mass mMin. All atoms with a mass lower than mMin also have their mass 
    set to that mMin. Default 1 (no mass scaling)
    """
    comm-mode: Literal['Linear', 'Angular', 'Linear-acceleration-correction', 'None'] = 'Linear'
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
    comm-grps: str
    """
    Group(s) for center of mass motion removal, default is the whole system.
    """

    ### Langevin dynamics ###
    ld-seed: int = -1
    """
    Integer used to initialize random generator for thermal noise for 
    stochastic and Brownian dynamics. When ld-seed is set to -1, 
    a pseudo random seed is used. Default -1.
    """

    ### Energy minimization ###
    emtol: FloatQuantity['kilojoule / (mole * nanometer)'] = 10.0 * unit.kilojoule / (unit.mole * unit.nanometer)
    """
    The minimization is converged when the maximum force is smaller than this 
    value. Default 10.0 * unit.kilojoule / (unit.mole * unit.nanometer)
    """
    emstep: FloatQuantity['nanometer'] = 0.01 * unit.nanometer
    """
    Initial step size. Default 0.01 * unit.nanometer
    """

    ### Neighbor searching ###
    cutoff-scheme: Literal['verlet'] = 'verlet'
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
    pbc: Literal['xyz', 'no', 'xy'] = 'xyz'
    """
    Treatment of periodic boundary conditions.
    Allowed values are:
    'xyz': Use periodic boundary conditions in all directions.
    'no': Use no periodic boundary conditions, ignore the box.
    'xy': Use periodic boundary conditions in x and y directions only. 
          This can be used in combination with walls.
    Default 'xyz'.
    """
    verlet-buffer-tolerance: FloatQuantity['kilojoule / (mole * picosecond)'] = 0.005 * unit.kilojoule / (unit.mole * unit.picosecond)
    """"
    Used when performing a simulation with dynamics. This sets the maximum 
    allowed error for pair interactions per particle caused by the Verlet 
    buffer, which indirectly sets rlist.
    Default 0.005 * unit.kilojoule / (unit.mole * unit.picosecond)
    """
    verlet-buffer-pressure-tolerance: FloatQuantity['bar'] = 0.5 * unit.bar
    """
    Used when performing a simulation with dynamics and only active when 
    verlet-buffer-tolerance is positive. This sets the maximum tolerated error 
    in the average pressure due to missing Lennard-Jones interactions of 
    particle pairs that are not in the pair list, but come within rvdw range 
    as the pair list ages. 
    Default 0.5 * unit.bar
    """
    rlist: FloatQuantity['nanometer'] = 1 * unit.nanometer
    """
    Cut-off distance for the short-range neighbor list. With dynamics, this is 
    by default set by the verlet-buffer-tolerance and verlet-buffer-pressure-tolerance 
    options and the value of rlist is ignored.
    """

    ### Electrostatics ##
    

class MDOutputSettings(SettingsBaseModel):
    """"
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
    nstxout-compressed: int = 0
    """
    Number of steps that elapse between writing position coordinates using 
    lossy compression (xtc file), 0 for not writing compressed coordinates 
    output. Default 0.
    """
    compressed-x-precision: int = 1000
    """
    Precision with which to write to the compressed trajectory file.
    Default 1000.
    """
    compressed-x-grps: str
    """
    Group(s) to write to the compressed trajectory file, by default the whole 
    system is written (if nstxout-compressed > 0).
    """
    energygrps: str
    """
    Group(s) for which to write to write short-ranged non-bonded potential 
    energies to the energy file (not supported on GPUs)
    """


class GromacsMDProtocolSettings(Settings):
    class Config:
        arbitrary_types_allowed = True

    protocol_repeats: int
    """
    Number of independent MD runs to perform.
    """

    @validator('protocol_repeats')
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
    simulation_settings: MDSimulationSettings

    # Simulations output settings
    output_settings: MDOutputSettings
