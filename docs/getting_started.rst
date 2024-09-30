Getting Started
===============

This page details how to get started with openfe-gromacs.


Developer Install
*****************

To install the development version of the OpenFE Gromacs protocols, you should
do a source installation in the following manner::

    git clone https://github.com/OpenFreeEnergy/openfe-gromacs.git

    cd openfe-gromacs
    mamba env create -f environment.yml

    mamba activate openfe_gromacs
    python -m pip install -e .
