"""
Unit and regression test for the openfe_gromacs package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import openfe_gromacs


def test_openfe_gromacs_imported():
    """Sample test, will always pass so long as import statement worked."""
    print("importing ", openfe_gromacs.__name__)
    assert "openfe_gromacs" in sys.modules


# Assert that a certain exception is raised
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()
