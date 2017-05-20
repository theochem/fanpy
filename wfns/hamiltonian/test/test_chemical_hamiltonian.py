"""Test wfns.hamiltonian.chemical_hamiltonian."""
import numpy as np
from nose.tools import assert_raises
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


class Empty:
    pass


def test_assign_orbtype():
    """Test ChemicalHamiltonian.assign_orbtype."""
    # default option
    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test)
    assert test.orbtype == 'restricted'

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, None)
    assert test.orbtype == 'restricted'

    # bad option
    test = Empty()
    assert_raises(TypeError, ChemicalHamiltonian.assign_orbtype, test, 'Restricted')
    assert_raises(TypeError, ChemicalHamiltonian.assign_orbtype, test, 'unrestricteD')
    assert_raises(TypeError, ChemicalHamiltonian.assign_orbtype, test, 'sdf')

    # explicit option
    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'restricted')
    assert test.orbtype == 'restricted'

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'unrestricted')
    assert test.orbtype == 'unrestricted'

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'generalized')
    assert test.orbtype == 'generalized'


def test_assign_energy_nuc_nuc():
    """Test ChemicalHamiltonian.assign_energy_nuc_nuc."""
    # default option
    test = Empty()
    ChemicalHamiltonian.assign_energy_nuc_nuc(test)
    assert test.energy_nuc_nuc == 0.0

    test = Empty()
    ChemicalHamiltonian.assign_energy_nuc_nuc(test, None)
    assert test.energy_nuc_nuc == 0.0

    # explicit option
    test = Empty()
    ChemicalHamiltonian.assign_energy_nuc_nuc(test, 0)
    assert test.energy_nuc_nuc == 0.0

    test = Empty()
    ChemicalHamiltonian.assign_energy_nuc_nuc(test, 1.5)
    assert test.energy_nuc_nuc == 1.5

    test = Empty()
    ChemicalHamiltonian.assign_energy_nuc_nuc(test, np.inf)
    assert test.energy_nuc_nuc == np.inf

    # bad option
    test = Empty()
    assert_raises(TypeError, ChemicalHamiltonian.assign_energy_nuc_nuc, test, [-2])
    assert_raises(TypeError, ChemicalHamiltonian.assign_energy_nuc_nuc, test, '2')


def test_assign_integrals():
    """Test ChemicalHamiltonian.assign_integrals."""
    # good input
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)
    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'restricted')
    ChemicalHamiltonian.assign_integrals(test, one_int, two_int)
    assert np.allclose(test.one_int.integrals, (one_int, ))
    assert np.allclose(test.two_int.integrals, (two_int, ))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'unrestricted')
    ChemicalHamiltonian.assign_integrals(test, 2*(one_int, ), 3*(two_int, ))
    assert np.allclose(test.one_int.integrals, 2*(one_int, ))
    assert np.allclose(test.two_int.integrals, 3*(two_int, ))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'generalized')
    ChemicalHamiltonian.assign_integrals(test, one_int, two_int)
    assert np.allclose(test.one_int.integrals, (one_int, ))
    assert np.allclose(test.two_int.integrals, (two_int, ))

    # bad input
    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, ChemicalHamiltonian.assign_integrals, test, np.random.rand(4, 4),
                  np.random.rand(3, 3, 3, 3))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, ChemicalHamiltonian.assign_integrals, test,
                  np.random.rand(4, 4).astype(float), np.random.rand(4, 4, 4, 4).astype(complex))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, ChemicalHamiltonian.assign_integrals, test, 2*(np.random.rand(4, 4), ),
                  np.random.rand(4, 4, 4, 4))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, ChemicalHamiltonian.assign_integrals, test, np.random.rand(4, 4),
                  3*(np.random.rand(4, 4, 4, 4), ))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'unrestricted')
    assert_raises(TypeError, ChemicalHamiltonian.assign_integrals, test, np.random.rand(4, 4),
                  np.random.rand(4, 4, 4, 4))

    test = Empty()
    ChemicalHamiltonian.assign_orbtype(test, 'generalized')
    assert_raises(NotImplementedError, ChemicalHamiltonian.assign_integrals, test, np.random.rand(3, 3),
                  np.random.rand(3, 3, 3, 3))


def test_nspin():
    """Test ChemicalHamiltonian.dtype."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = ChemicalHamiltonian(one_int, two_int, 'restricted')
    assert test.nspin == 4
    test = ChemicalHamiltonian(2*[one_int], 3*[two_int], 'unrestricted')
    assert test.nspin == 4
    test = ChemicalHamiltonian(one_int, two_int, 'generalized')
    assert test.nspin == 2


def test_dtype():
    """Test ChemicalHamiltonian.dtype."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = ChemicalHamiltonian(one_int, two_int, 'restricted')
    assert test.dtype == float

    one_int = np.arange(1, 5, dtype=complex).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=complex).reshape(2, 2, 2, 2)
    test = ChemicalHamiltonian(one_int, two_int, 'restricted')
    assert test.dtype == complex
