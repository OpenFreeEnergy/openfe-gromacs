import pint
from openff.units import unit

# Settings that are not exposed to the user
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
    "lincs_warnangle": 30 * unit.degree,
    "continuation": "no",
    "morse": "no",
}

PRE_DEFINED_SETTINGS_EM = {
    "emtol": 10.0 * unit.kilojoule / (unit.mole * unit.nanometer),
    "emstep": 0.01 * unit.nanometer,
}

PRE_DEFINED_SETTINGS_MD = {
    "nsttcouple": -1,
    "tc_grps": "system",
    "tau_t": 2.0 * unit.picosecond,
    "pcoupltype": "isotropic",
    "nstpcouple": -1,
    "tau_p": 5 * unit.picosecond,
    "compressibility": 4.5e-05 / unit.bar,
}


def dict2mdp(settings_dict: dict, shared_basepath):
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
        "ntomp",
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


def write_mdp_files(settings: dict, shared_basepath):
    """
    Writes out the .mdp files for running a Gromacs simulation.

    Parameters
    ----------
    settings: dict
      Dictionary of all the settings
    shared_basepath : Pathlike, optional
      Where to run the calculation, defaults to current working directory

    Returns
    -------
    mdps: dict
      Dictionary of file paths to mdp files.
    """

    mdps = {}
    if settings["sim_settings_em"].nsteps > 0:
        settings_dict = (
            settings["sim_settings_em"].dict()
            | settings["output_settings_em"].dict()
            | PRE_DEFINED_SETTINGS
            | PRE_DEFINED_SETTINGS_EM
        )
        mdp = dict2mdp(settings_dict, shared_basepath)
        mdps["em"] = mdp
    if settings["sim_settings_nvt"].nsteps > 0:
        settings_dict = (
            settings["sim_settings_nvt"].dict()
            | settings["output_settings_nvt"].dict()
            | PRE_DEFINED_SETTINGS
            | PRE_DEFINED_SETTINGS_MD
        )
        mdp = dict2mdp(settings_dict, shared_basepath)
        mdps["nvt"] = mdp
    if settings["sim_settings_npt"].nsteps > 0:
        settings_dict = (
            settings["sim_settings_npt"].dict()
            | settings["output_settings_npt"].dict()
            | PRE_DEFINED_SETTINGS
            | PRE_DEFINED_SETTINGS_MD
        )
        mdp = dict2mdp(settings_dict, shared_basepath)
        mdps["npt"] = mdp

    return mdps
