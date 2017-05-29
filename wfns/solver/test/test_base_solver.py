"""Test wfns.solver.base_solver."""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
from wfns.solver.base_solver import BaseSolver
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


class TestBaseSolver(BaseSolver):
    """BaseSolver that skips initialization."""
    def __init__(self):
        pass


class TestBaseWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abstract class structure."""
    def __init__(self):
        pass

    def get_overlap(self):
        pass

    @property
    def spin(self):
        return None

    @property
    def seniority(self):
        return None

    @property
    def template_params(self):
        return None


class TestChemicalHamiltonian(ChemicalHamiltonian):
    """ChemicalHamiltonian that skips initialization."""
    def __init__(self):
        pass


def test_assign_wavefunction():
    """Test BaseSolver.assign_wavefunction."""
    test = TestBaseSolver()
    test_wfn = TestBaseWavefunction()
    test.assign_wavefunction(test_wfn)
    assert test.wavefunction == test_wfn
    assert_raises(TypeError, test.assign_wavefunction, None)


def test_assign_hamiltonian():
    """Test BaseSolver.assign_hamiltonian."""
    test = TestBaseSolver()
    test_ham = TestChemicalHamiltonian()
    test.assign_hamiltonian(test_ham)
    assert test.hamiltonian == test_ham
    assert_raises(TypeError, test.assign_hamiltonian, None)
