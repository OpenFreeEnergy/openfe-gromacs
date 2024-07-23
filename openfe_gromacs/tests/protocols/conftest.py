# This code is part of OpenFE and is licensed under the MIT license.
# For details, see https://github.com/OpenFreeEnergy/openfe-gromacs
from importlib import resources

import openfe
import pytest
from openff.units import unit
from rdkit import Chem
from rdkit.Geometry import Point3D


@pytest.fixture
def benzene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {"ligand": benzene_modifications["benzene"]},
    )


@pytest.fixture(scope="session")
def benzene_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na",
                negative_ion="Cl",
                ion_concentration=0.15 * unit.molar,
            ),
        },
    )


@pytest.fixture
def benzene_complex_system(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["benzene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na",
                negative_ion="Cl",
                ion_concentration=0.15 * unit.molar,
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture
def toluene_vacuum_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {"ligand": benzene_modifications["toluene"]},
    )


@pytest.fixture(scope="session")
def toluene_system(benzene_modifications):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na",
                negative_ion="Cl",
                ion_concentration=0.15 * unit.molar,
            ),
        },
    )


@pytest.fixture
def toluene_complex_system(benzene_modifications, T4_protein_component):
    return openfe.ChemicalSystem(
        {
            "ligand": benzene_modifications["toluene"],
            "solvent": openfe.SolventComponent(
                positive_ion="Na",
                negative_ion="Cl",
                ion_concentration=0.15 * unit.molar,
            ),
            "protein": T4_protein_component,
        }
    )


@pytest.fixture
def benzene_charges():
    files = {}
    with resources.files("openfe.tests.data.openmm_rfe") as d:
        fn = str(d / "charged_benzenes.sdf")
        supp = Chem.SDMolSupplier(str(fn), removeHs=False)
        for rdmol in supp:
            files[rdmol.GetProp("_Name")] = openfe.SmallMoleculeComponent(rdmol)
    return files


@pytest.fixture
def benzene_many_solv_system(benzene_modifications):

    rdmol_phenol = benzene_modifications["phenol"].to_rdkit()
    rdmol_benzo = benzene_modifications["benzonitrile"].to_rdkit()

    conf_phenol = rdmol_phenol.GetConformer()
    conf_benzo = rdmol_benzo.GetConformer()

    for atm in range(rdmol_phenol.GetNumAtoms()):
        x, y, z = conf_phenol.GetAtomPosition(atm)
        conf_phenol.SetAtomPosition(atm, Point3D(x + 30, y, z))

    for atm in range(rdmol_benzo.GetNumAtoms()):
        x, y, z = conf_benzo.GetAtomPosition(atm)
        conf_benzo.SetAtomPosition(atm, Point3D(x, y + 30, z))

    phenol = openfe.SmallMoleculeComponent.from_rdkit(rdmol_phenol, name="phenol")

    benzo = openfe.SmallMoleculeComponent.from_rdkit(rdmol_benzo, name="benzonitrile")

    return openfe.ChemicalSystem(
        {
            "whatligand": benzene_modifications["benzene"],
            "foo": phenol,
            "bar": benzo,
            "solvent": openfe.SolventComponent(),
        },
    )


@pytest.fixture
def toluene_many_solv_system(benzene_modifications):

    rdmol_phenol = benzene_modifications["phenol"].to_rdkit()
    rdmol_benzo = benzene_modifications["benzonitrile"].to_rdkit()

    conf_phenol = rdmol_phenol.GetConformer()
    conf_benzo = rdmol_benzo.GetConformer()

    for atm in range(rdmol_phenol.GetNumAtoms()):
        x, y, z = conf_phenol.GetAtomPosition(atm)
        conf_phenol.SetAtomPosition(atm, Point3D(x + 30, y, z))

    for atm in range(rdmol_benzo.GetNumAtoms()):
        x, y, z = conf_benzo.GetAtomPosition(atm)
        conf_benzo.SetAtomPosition(atm, Point3D(x, y + 30, z))

    phenol = openfe.SmallMoleculeComponent.from_rdkit(rdmol_phenol, name="phenol")

    benzo = openfe.SmallMoleculeComponent.from_rdkit(rdmol_benzo, name="benzonitrile")
    return openfe.ChemicalSystem(
        {
            "whatligand": benzene_modifications["toluene"],
            "foo": phenol,
            "bar": benzo,
            "solvent": openfe.SolventComponent(),
        },
    )
