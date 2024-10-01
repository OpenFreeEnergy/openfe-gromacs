openfe-gromacs
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/OpenFreeEnergy/openfe-gromacs/workflows/CI/badge.svg)](https://github.com/OpenFreeEnergy/openfe-gromacs/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/OpenFreeEnergy/openfe-gromacs/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenFreeEnergy/openfe-gromacs/branch/main)
[![documentation](https://readthedocs.org/projects/openfe-gromacs/badge/?version=latest)](https://openfe-gromacs.readthedocs.io/en/latest/?badge=latest)

# `openfe-gromacs` - A Python package for GROMACS-based Protocols

The `openfe-gromacs` package provides protocols for running simulations in GROMACS.

## License

This library is made available under the [MIT](https://opensource.org/licenses/MIT) open source license.

## Install

### Development version

The development version of `openfe_gromacs` can be installed directly from the `main` branch of this repository.

First install the package dependencies using `mamba`:

```bash
mamba env create -f environment.yml
```

The openfe-gromacs library can then be installed via:

```
python -m pip install --no-deps .
```

## Authors

The OpenFE development team.

## Acknowledgements

OpenFE is an [Open Molecular Software Foundation](https://omsf.io/) hosted project.
Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
