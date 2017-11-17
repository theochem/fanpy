"""Test wfns.ham.base."""
import numpy as np
from nose.tools import assert_raises
from wfns.ham.base import BaseHamiltonian


class Empty:
    """Empty container class."""
    pass


class TestBaseHamiltonian(BaseHamiltonian):
    """BaseHamiltonian class that bypasses the abstract methods."""
    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        """Abstract method."""
        pass

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        """Abstract method."""
        pass


def test_assign_orbtype():
    """Test BaseHamiltonian.assign_orbtype."""
    # default option
    test = Empty()
    BaseHamiltonian.assign_orbtype(test)
    assert test.orbtype == 'restricted'

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, None)
    assert test.orbtype == 'restricted'

    # bad option
    test = Empty()
    assert_raises(TypeError, BaseHamiltonian.assign_orbtype, test, 'Restricted')
    assert_raises(TypeError, BaseHamiltonian.assign_orbtype, test, 'unrestricteD')
    assert_raises(TypeError, BaseHamiltonian.assign_orbtype, test, 'sdf')

    # explicit option
    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'restricted')
    assert test.orbtype == 'restricted'

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'unrestricted')
    assert test.orbtype == 'unrestricted'

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'generalized')
    assert test.orbtype == 'generalized'


def test_assign_energy_nuc_nuc():
    """Test BaseHamiltonian.assign_energy_nuc_nuc."""
    # default option
    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test)
    assert test.energy_nuc_nuc == 0.0

    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, None)
    assert test.energy_nuc_nuc == 0.0

    # explicit option
    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, 0)
    assert test.energy_nuc_nuc == 0.0

    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, 1.5)
    assert test.energy_nuc_nuc == 1.5

    test = Empty()
    BaseHamiltonian.assign_energy_nuc_nuc(test, np.inf)
    assert test.energy_nuc_nuc == np.inf

    # bad option
    test = Empty()
    assert_raises(TypeError, BaseHamiltonian.assign_energy_nuc_nuc, test, [-2])
    assert_raises(TypeError, BaseHamiltonian.assign_energy_nuc_nuc, test, '2')


def test_assign_integrals():
    """Test BaseHamiltonian.assign_integrals."""
    # good input
    one_int = np.random.rand(4, 4)
    two_int = np.random.rand(4, 4, 4, 4)
    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'restricted')
    BaseHamiltonian.assign_integrals(test, one_int, two_int)
    assert np.allclose(test.one_int.integrals, (one_int, ))
    assert np.allclose(test.two_int.integrals, (two_int, ))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'unrestricted')
    BaseHamiltonian.assign_integrals(test, 2*(one_int, ), 3*(two_int, ))
    assert np.allclose(test.one_int.integrals, 2*(one_int, ))
    assert np.allclose(test.two_int.integrals, 3*(two_int, ))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'generalized')
    BaseHamiltonian.assign_integrals(test, one_int, two_int)
    assert np.allclose(test.one_int.integrals, (one_int, ))
    assert np.allclose(test.two_int.integrals, (two_int, ))

    # bad input
    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, BaseHamiltonian.assign_integrals, test, np.random.rand(4, 4),
                  np.random.rand(3, 3, 3, 3))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, BaseHamiltonian.assign_integrals, test,
                  np.random.rand(4, 4).astype(float), np.random.rand(4, 4, 4, 4).astype(complex))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, BaseHamiltonian.assign_integrals, test, 2*(np.random.rand(4, 4), ),
                  np.random.rand(4, 4, 4, 4))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'restricted')
    assert_raises(TypeError, BaseHamiltonian.assign_integrals, test, np.random.rand(4, 4),
                  3*(np.random.rand(4, 4, 4, 4), ))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'unrestricted')
    assert_raises(TypeError, BaseHamiltonian.assign_integrals, test, np.random.rand(4, 4),
                  np.random.rand(4, 4, 4, 4))

    test = Empty()
    BaseHamiltonian.assign_orbtype(test, 'generalized')
    assert_raises(NotImplementedError, BaseHamiltonian.assign_integrals, test, np.random.rand(3, 3),
                  np.random.rand(3, 3, 3, 3))


def test_nspin():
    """Test BaseHamiltonian.nspin."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = TestBaseHamiltonian(one_int, two_int, 'restricted')
    assert test.nspin == 4
    test = TestBaseHamiltonian(2*[one_int], 3*[two_int], 'unrestricted')
    assert test.nspin == 4
    test = TestBaseHamiltonian(one_int, two_int, 'generalized')
    assert test.nspin == 2
    # hack in bad orbital type
    test.orbtype = 'bad orbital type'
    assert_raises(NotImplementedError, lambda: test.nspin)


def test_dtype():
    """Test BaseHamiltonian.dtype."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = TestBaseHamiltonian(one_int, two_int, 'restricted')
    assert test.dtype == float

    one_int = np.arange(1, 5, dtype=complex).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=complex).reshape(2, 2, 2, 2)
    test = TestBaseHamiltonian(one_int, two_int, 'restricted')
    assert test.dtype == complex


def test_orb_rotate_jacobi():
    """Test BaseHamiltonian.orb_rotate_jacobi."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)
    ham = TestBaseHamiltonian(one_int, two_int, 'restricted')

    theta = 2 * np.pi * (np.random.random() - 0.5)
    p, q = 0, 3
    jacobi_matrix = np.identity(4)
    jacobi_matrix[p, p] = np.cos(theta)
    jacobi_matrix[p, q] = np.sin(theta)
    jacobi_matrix[q, p] = -np.sin(theta)
    jacobi_matrix[q, q] = np.cos(theta)
    one_answer = np.copy(one_int)
    one_answer = np.einsum('ij,ia->aj', one_answer, jacobi_matrix)
    one_answer = np.einsum('aj,jb->ab', one_answer, jacobi_matrix)
    two_answer = np.copy(two_int)
    two_answer = np.einsum('ijkl,ia->ajkl', two_answer, jacobi_matrix)
    two_answer = np.einsum('ajkl,jb->abkl', two_answer, jacobi_matrix)
    two_answer = np.einsum('abkl,kc->abcl', two_answer, jacobi_matrix)
    two_answer = np.einsum('abcl,ld->abcd', two_answer, jacobi_matrix)

    ham.orb_rotate_jacobi((p, q), theta)
    assert np.allclose(ham.one_int.integrals[0], one_answer)
    assert np.allclose(ham.two_int.integrals[0], two_answer)


def test_orb_rotate_matrix():
    """Test BaseHamiltonian.orb_rotate_matrix."""
    one_int = np.arange(1, 17, dtype=float).reshape(4, 4)
    two_int = np.arange(1, 257, dtype=float).reshape(4, 4, 4, 4)

    random = np.random.rand(4, 4)
    transform = np.linalg.eigh(random + random.T)[1]

    one_answer = np.copy(one_int)
    one_answer = np.einsum('ij,ia->aj', one_answer, transform)
    one_answer = np.einsum('aj,jb->ab', one_answer, transform)
    two_answer = np.copy(two_int)
    two_answer = np.einsum('ijkl,ia->ajkl', two_answer, transform)
    two_answer = np.einsum('ajkl,jb->abkl', two_answer, transform)
    two_answer = np.einsum('abkl,kc->abcl', two_answer, transform)
    two_answer = np.einsum('abcl,ld->abcd', two_answer, transform)

    ham = TestBaseHamiltonian(one_int, two_int, 'restricted')
    ham.orb_rotate_matrix(transform)
    assert np.allclose(ham.one_int.integrals[0], one_answer)
    assert np.allclose(ham.two_int.integrals[0], two_answer)
    ham = TestBaseHamiltonian(one_int, two_int, 'restricted')
    ham.orb_rotate_matrix([transform])
    assert np.allclose(ham.one_int.integrals[0], one_answer)
    assert np.allclose(ham.two_int.integrals[0], two_answer)

    ham = TestBaseHamiltonian((one_int, one_int), (two_int, two_int, two_int), 'unrestricted')
    ham.orb_rotate_matrix(transform)
    assert np.allclose(ham.one_int.integrals[0], one_answer)
    assert np.allclose(ham.one_int.integrals[1], one_answer)
    assert np.allclose(ham.two_int.integrals[0], two_answer)
    assert np.allclose(ham.two_int.integrals[1], two_answer)
    assert np.allclose(ham.two_int.integrals[2], two_answer)
    ham = TestBaseHamiltonian((one_int, one_int), (two_int, two_int, two_int), 'unrestricted')
    ham.orb_rotate_matrix([transform])
    assert np.allclose(ham.one_int.integrals[0], one_answer)
    assert np.allclose(ham.one_int.integrals[1], one_answer)
    assert np.allclose(ham.two_int.integrals[0], two_answer)
    assert np.allclose(ham.two_int.integrals[1], two_answer)
    assert np.allclose(ham.two_int.integrals[2], two_answer)
    ham = TestBaseHamiltonian((one_int, one_int), (two_int, two_int, two_int), 'unrestricted')
    ham.orb_rotate_matrix([transform, transform])
    assert np.allclose(ham.one_int.integrals[0], one_answer)
    assert np.allclose(ham.one_int.integrals[1], one_answer)
    assert np.allclose(ham.two_int.integrals[0], two_answer)
    assert np.allclose(ham.two_int.integrals[1], two_answer)
    assert np.allclose(ham.two_int.integrals[2], two_answer)
