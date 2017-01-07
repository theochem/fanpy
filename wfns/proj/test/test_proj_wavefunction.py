"""Tests wfns.proj.proj_wavefunctions
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.proj.proj_wavefunction import ProjectedWavefunction
from wfns.sd_list import sd_list
from wfns import slater


class TestProjectedWavefunction(ProjectedWavefunction):
    """ Child of ProjectedWavefunction used to test ProjectedWavefunction

    Because ProjectedWavefunction is an abstract class
    """
    def __init__(self, nelec, one_int, two_int, dtype=None, nuc_nuc=None, orbtype=None):
        super(ProjectedWavefunction, self).__init__(nelec, one_int, two_int, dtype=dtype,
                                                    nuc_nuc=nuc_nuc, orbtype=orbtype)
        self.cache = {}
        self.d_cache = {}

    @property
    def template_coeffs(self):
        return np.arange(4, dtype=float)

    def compute_overlap(self, sd, deriv=None):
        if deriv is None:
            if sd == 0b0101:
                return 5
            elif sd == 0b1010:
                return 6
            return 7
        else:
            return 8

    def normalize(self, ref_sds=None):
        pass


def test_nconstraints():
    """ Tests ProjectedWavefunction._nconstraints
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test._nconstraints == 1


def test_seniority():
    """ Tests ProjectedWavefunction._seniority
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test._seniority is None


def test_spin():
    """ Tests ProjectedWavefunction._spin
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test._spin is None


def test_nparams():
    """ Tests ProjectedWavefunction.nparams
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test.nparams == 4 + 1

    # different template_coeffs
    class TestWfn(TestProjectedWavefunction):
        @property
        def template_coeffs(self):
            return np.array(np.arange(100).reshape(2, 2, 5, 5))
    test = TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test.nparams == 100 + 1


def test_nproj():
    """ Tests ProjectedWavefunction.nproj
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    test.pspace = [1, 2, 3, 4]
    assert test.nproj == 4


def test_generate_pspace():
    """ Tests ProjectedWavefunction.generate_pspace

    Taken from wfns.sd_list.sd_list
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=47, spin=None,
                                             seniority=None)
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=15, spin=None,
                                             seniority=None)

    # change number of electrons and spatial orbitals
    test = TestProjectedWavefunction(5, np.ones((10, 10)), np.ones((10, 10, 10, 10)))
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=47, spin=None,
                                             seniority=None)

    # change number of parameters
    test.params = np.array([])
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=47, spin=None,
                                             seniority=None)
    # NOTE: THIS DOES NOT CHANGE THE test.nparams BECAUSE nparams ONLY DEPENDS ON template_coeffs

    # different template_coeffs
    class TestWfn(TestProjectedWavefunction):
        @property
        def template_coeffs(self):
            return np.array([])
    test = TestWfn(5, np.ones((10, 10)), np.ones((10, 10, 10, 10)))
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=43, spin=None,
                                             seniority=None)

    # change spin
    test._spin = 1
    test._seniority = None
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=43, spin=1,
                                             seniority=None)

    # change seniority
    test._spin = None
    test._seniority = 2
    assert test.generate_pspace() == sd_list(test.nelec, test.nspatial, num_limit=43, spin=None,
                                             seniority=2)


def test_assign_pspace():
    """ Tests ProjectedWavefunction.assign_pspace
    """
    test = TestProjectedWavefunction(5, np.ones((10, 10)), np.ones((10, 10, 10, 10)))
    # check errors (bad input)
    assert_raises(TypeError, lambda: test.assign_pspace({1:2, 3:4}))
    assert_raises(TypeError, lambda: test.assign_pspace('123'))
    assert_raises(TypeError, lambda: test.assign_pspace(np.array[1, 2, 3]))

    # default assignment
    test.assign_pspace()
    assert test.pspace == tuple(sd_list(test.nelec, test.nspatial, num_limit=47, spin=None,
                                        seniority=None))

    test.assign_pspace(None)
    assert test.pspace == tuple(sd_list(test.nelec, test.nspatial, num_limit=47, spin=None,
                                        seniority=None))

    # assign pspace
    test.assign_pspace([0b01111, 0b10111])
    assert test.pspace == (0b01111, 0b10111)
    test.assign_pspace((0b10111, 0b01111))
    assert test.pspace == (0b10111, 0b01111)
    # check errors (bad sd)
    assert_raises(TypeError, lambda: test.assign_pspace(['0b01111', '0b10111']))

    # custom spin
    test._spin = 1
    test._seniority = None
    test.assign_pspace([3 + 3*2**10, 1 + 7*2**10, 7 + 1*2**10])
    assert test.pspace == (slater.internal_sd(7 + 1*2**10), )
    # custom seniority
    test._spin = None
    test._seniority = 2
    test.assign_pspace([3 + 3*2**10, 1 + 7*2**10, 7 + 1*2**10])
    assert test.pspace == (slater.internal_sd(1 + 7*2**10), slater.internal_sd(7 + 1*2**10), )
    # custom seniority
    test._spin = 1
    test._seniority = 2
    test.assign_pspace([3 + 3*2**10, 1 + 7*2**10, 7 + 1*2**10])
    assert test.pspace == (slater.internal_sd(7 + 1*2**10), )
    test._spin = 3
    test._seniority = 2
    test.assign_pspace([3 + 3*2**10, 1 + 7*2**10, 7 + 1*2**10])
    assert test.pspace == ()

def test_assign_ref_sds():
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    test.pspace = (0b001001, 0b010001)
    # check error
    assert_raises(TypeError, lambda: test.assign_ref_sds(1.0))
    assert_raises(TypeError, lambda: test.assign_ref_sds({}))
    assert_raises(TypeError, lambda: test.assign_ref_sds('0b111'))
    assert_raises(ValueError, lambda: test.assign_ref_sds(0b011001))
    test._spin = 1
    test._seniority = None
    assert_raises(ValueError, lambda: test.assign_ref_sds(0b001001))
    test._spin = None
    test._seniority = 2
    assert_raises(ValueError, lambda: test.assign_ref_sds(0b001001))
    # single integer
    test._spin = None
    test._seniority = None
    test.assign_ref_sds(0b001001)
    assert test.ref_sds == (0b001001, )
    # single long
    test.assign_ref_sds(long(0b001001))
    assert test.ref_sds == (0b001001, )
    # single mpz
    test.assign_ref_sds(slater.internal_sd(0b001001))
    assert test.ref_sds == (0b001001, )
    # tuple
    test.assign_ref_sds((0b001001, ))
    assert test.ref_sds == (0b001001, )
    # list
    test.assign_ref_sds([0b001001, ])
    assert test.ref_sds == (0b001001, )
    # check default
    test.assign_ref_sds()
    assert test.ref_sds == (0b001001, )
    test.assign_ref_sds(None)
    assert test.ref_sds == (0b001001, )


def test_assign_params():
    """ Tests ProjectedWavefunction.assign_params
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # check errors
    test.ref_sds = (0b11111, 0b11)
    assert_raises(ValueError, test.assign_params)
    test.ref_sds = (0b11111, )
    assert_raises(TypeError, test.assign_params, [1, 2])
    assert_raises(ValueError, test.assign_params, np.arange(5).reshape(1, 5))
    assert_raises(ValueError, test.assign_params, np.arange(5).reshape(1, 5))
    assert_raises(TypeError, test.assign_params, np.arange(5).astype(bool))
    assert_raises(TypeError, test.assign_params, np.arange(5).astype(np.complex128))

    # default
    test.ref_sds = (0b001001, )
    test.assign_params()
    assert np.allclose(test.params, np.hstack((np.arange(4), 721/49))) # energy is 721/49
    # Assign random float (default)
    test.dtype = np.float64
    test.assign_params(add_noise=True)
    assert np.allclose(test.params, np.hstack((np.arange(4), 721/49)), atol=0.2/4*0.5)
    # Assign params (float), default energy
    test.assign_params(np.hstack((np.arange(4, dtype=float), 0)), add_noise=False)
    assert np.allclose(test.params, np.hstack((np.arange(4), 721/49))) # energy is 721/49
    # Assign params (float), custom energy
    test.assign_params(np.hstack((np.arange(4, dtype=float), 3)), add_noise=False)
    assert np.allclose(test.params, np.hstack((np.arange(4), 3)))
    # Assign params (np.float64), default energy
    test.assign_params(np.hstack((np.arange(4, dtype=np.float64), 0)), add_noise=False)
    assert np.allclose(test.params, np.hstack((np.arange(4), 721/49))) # energy is 721/49
    # Assign params (np.float64), custom energy
    test.assign_params(np.hstack((np.arange(4, dtype=np.float64), 3)), add_noise=False)
    assert np.allclose(test.params, np.hstack((np.arange(4), 3)))
    # Assign random float (default)
    test.dtype = np.complex128
    test.assign_params(add_noise=True)
    assert np.allclose(np.real(test.params), np.hstack((np.arange(4), 721/49)), atol=0.2/4*0.5)
    assert np.allclose(np.imag(test.params), np.hstack((np.zeros(4), 0)), atol=0.001*0.2/4*0.5)
    # Assign params (complex), default energy
    test.assign_params(np.hstack((np.arange(4, dtype=complex), 0)), add_noise=False)
    assert np.allclose(np.real(test.params), np.hstack((np.arange(4), 721/49)))
    assert np.allclose(np.imag(test.params), np.hstack((np.zeros(4), 0)))
    # Assign params (complex), custom energy
    test.assign_params(np.hstack((np.arange(4, dtype=complex), 3)), add_noise=False)
    assert np.allclose(np.real(test.params), np.hstack((np.arange(4), 3)))
    assert np.allclose(np.imag(test.params), np.hstack((np.zeros(4), 0)))
    # Assign params (np.complex128), default energy
    test.assign_params(np.hstack((np.arange(4, dtype=np.complex128), 0)), add_noise=False)
    assert np.allclose(np.real(test.params), np.hstack((np.arange(4), 721/49)))
    assert np.allclose(np.imag(test.params), np.hstack((np.zeros(4), 0)))
    # Assign params (np.complex128), custom energy
    test.assign_params(np.hstack((np.arange(4, dtype=np.complex128), 3)), add_noise=False)
    assert np.allclose(np.real(test.params), np.hstack((np.arange(4), 3)))
    assert np.allclose(np.imag(test.params), np.hstack((np.zeros(4), 0)))


def test_get_overlap():
    """ Tests ProjectedWavefunction.get_overlap
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    test.cache = {0b1010:1, 0b0101:2}
    test.d_cache = {(0b1010, 0):3, (0b0101, 1):4}
    # in cache
    assert test.get_overlap(0b1010, deriv=None) == 1
    assert test.get_overlap(0b0101, deriv=None) == 2
    assert test.get_overlap(0b1010, deriv=0) == 3
    assert test.get_overlap(0b0101, deriv=1) == 4
    # not in cache
    assert test.get_overlap(0b0011, deriv=None) == 7
    assert test.get_overlap(0b1100, deriv=None) == 7
    assert test.get_overlap(0b1010, deriv=1) == 8
    assert test.get_overlap(0b0011, deriv=1) == 8


def test_compute_norm():
    """ Tests ProjectedWavefunction.compute_norm
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)))
    # single integer
    test.ref_sds = (0b0011, )
    # default
    assert test.compute_norm() == 7**2
    assert test.compute_norm(deriv=0) == 2 * 7 * 8
    # custom sds
    assert test.compute_norm(ref_sds=0b0011) == 7**2
    assert test.compute_norm(ref_sds=0b0011, deriv=0) == 2 * 7 * 8
    assert test.compute_norm(ref_sds=(0b0011,)) == 7**2
    assert test.compute_norm(ref_sds=(0b0011,), deriv=0) == 2 * 7 * 8
    assert test.compute_norm(ref_sds=(0b0011, 0b0101)) == 7**2 + 5**2
    assert test.compute_norm(ref_sds=[0b0011, 0b0101], deriv=0) == 2*7*8 + 2*5*8
    # bad input
    assert_raises(TypeError, lambda: test.compute_norm(ref_sds=1.0))
    assert_raises(TypeError, lambda: test.compute_norm(ref_sds={}))
    assert_raises(TypeError, lambda: test.compute_norm(ref_sds='0b111'))


def test_get_energy():
    """ Tests ProjectedWavefunction.get_energy
    """
    test = TestProjectedWavefunction(2, np.ones((3, 3)), np.ones((3, 3, 3, 3)), nuc_nuc=4.0)
    test.params = np.arange(5)
    # default
    assert test.get_energy() == 4
    # include_nuc
    assert test.get_energy(include_nuc=True) == 8
    # derivatized
    assert test.get_energy(include_nuc=True, deriv=4) == 1
    assert test.get_energy(include_nuc=False, deriv=4) == 1
    assert test.get_energy(include_nuc=True, deriv=3) == 0
    assert test.get_energy(include_nuc=False, deriv=-1) == 0


def test_compute_hamiltonian():
    """ Tests ProjectedWavefunction.compute_hamiltonian
    """
    test = TestProjectedWavefunction(2, np.arange(1, 5, dtype=float).reshape(2, 2),
                                     np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2))
    # no seniority
    test._seniority = None
    one_energy, coulomb, exchange = test.compute_hamiltonian(0b0101)
    assert one_energy == 5*2 + 7*4
    assert coulomb == 5*5 + 7*7 + 7*7 + 6*8
    assert exchange == 0
    # seniority 0
    test._seniority = 0
    one_energy, coulomb, exchange = test.compute_hamiltonian(0b0101)
    assert one_energy == 5*2
    assert coulomb == 5*5 + 6*8
    assert exchange == 0
    # other seniority
    test._seniority = 1
    one_energy, coulomb, exchange = test.compute_hamiltonian(0b0101)
    assert one_energy == 5*2 + 7*4
    assert coulomb == 5*5 + 7*7 + 7*7 + 6*8
    assert exchange == 0


def test_compute_energy():
    """ Tests ProjectedWavefunction.compute_energy
    """
    test = TestProjectedWavefunction(2, np.arange(1, 5, dtype=float).reshape(2, 2),
                                     np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2),
                                     nuc_nuc=4.0)
    # single sd
    test.ref_sds = (0b0011, )
    assert test.compute_energy() == 7*28/49
    assert test.compute_energy(ref_sds=(0b0011, )) == 7*28/49
    assert test.compute_energy(ref_sds=0b0011) == 7*28/49
    assert test.compute_energy(include_nuc=True) == 7*28/49 + 4
    # single sd, derivatized
    assert test.compute_energy(deriv=0) == (8*28 + 7*32)/49 - 28*7*112/49**2
    assert test.compute_energy(ref_sds=(0b0011, ), deriv=0) == (8*28 + 7*32)/49 - 28*7*112/49**2
    assert test.compute_energy(ref_sds=0b0011, deriv=0) == (8*28 + 7*32)/49 - 28*7*112/49**2
    assert test.compute_energy(deriv=0, include_nuc=True) == (8*28 + 7*32)/49 - 28*7*112/49**2 + 4
    # multiple sds
    test.ref_sds = (0b0011, 0b1010)
    assert test.compute_energy() == (7*28 + 6*547)/85
    assert test.compute_energy(ref_sds=(0b0011, 0b1010)) == (7*28 + 6*547)/85
    assert test.compute_energy(include_nuc=True) == (7*28 + 6*547)/85 + 4
    # multiple sds, derivatized
    assert test.compute_energy(deriv=0) == (8*28+8*547+7*32+6*696)/85 - (28*7+547*6)*208/85**2
    assert test.compute_energy(ref_sds=(0b0011, 0b1010), deriv=0) == ((8*28+8*547+7*32+6*696)/85
                                                                      - (28*7+547*6)*208/85**2)
    assert test.compute_energy(deriv=0, include_nuc=True) == ((8*28+8*547+7*32+6*696)/85
                                                              - (28*7+547*6)*208/85**2 + 4)
    # bad sds
    assert_raises(TypeError, lambda: test.compute_energy(ref_sds=1.0))
    assert_raises(TypeError, lambda: test.compute_energy(ref_sds={}))
    assert_raises(TypeError, lambda: test.compute_energy(ref_sds='0b111'))
    # bad norm
    test.compute_overlap = lambda sd, deriv=None: 0
    assert_raises(ValueError, test.compute_energy)
    test.compute_overlap = lambda sd, deriv=None: 1j
    assert_raises(ValueError, test.compute_energy)


def test_objective():
    """ Tests ProjectedWavefunction.objective
    """
    test = TestProjectedWavefunction(2, np.arange(1, 5, dtype=float).reshape(2, 2),
                                     np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2))
    test.params = np.array([1, 2, 3])
    test.pspace = (0b0101, 0b1010, 0b1100)
    test.ref_sds = (0b0101, )
    # weighted norms
    assert np.allclose(test.objective(test.params), np.array([209 - 3*5, 547 - 3*6, 28 - 3*7,
                                                              (25 - 1)*(3 + 1)]))
    assert np.allclose(test.objective(test.params, weigh_constraints=True),
                       np.array([209 - 3*5, 547 - 3*6, 28 - 3*7, (25 - 1)*(3 + 1)]))
    # non weighted norms
    assert np.allclose(test.objective(test.params, weigh_constraints=False),
                       np.array([209 - 3*5, 547 - 3*6, 28 - 3*7, (25 - 1)]))


def test_jacobian():
    """ Tests ProjectedWavefunction.jacobian
    """
    test = TestProjectedWavefunction(2, np.arange(1, 5, dtype=float).reshape(2, 2),
                                     np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2))
    test.params = np.array([1, 2, 3, 4, 5])
    test.pspace = (0b0101, 0b1010, 0b1100, 0b0011, 0b1001)
    test.ref_sds = (0b0101, )
    # note that template coeffs has 4 elements, plus one for energy (need at least 4 projections)
    # weighted norms
    assert np.allclose(test.jacobian(test.params),
                       np.array([[264-5*8-0*5, 264-5*8-0*5, 264-5*8-0*5, 264-5*8-0*5, 264-5*8-1*5],
                                 [696-5*8-0*6, 696-5*8-0*6, 696-5*8-0*6, 696-5*8-0*6, 696-5*8-1*6],
                                 [32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-1*7],
                                 [32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-1*7],
                                 [448-5*8-0*7, 448-5*8-0*7, 448-5*8-0*7, 448-5*8-0*7, 448-5*8-1*7],
                                 [(80-1)*(5+1), (80-1)*(5+1), (80-1)*(5+1), (80-1)*(5+1),
                                  (80-1)*(5+1)]]))
    assert np.allclose(test.jacobian(test.params, weigh_constraints=True),
                       np.array([[264-5*8-0*5, 264-5*8-0*5, 264-5*8-0*5, 264-5*8-0*5, 264-5*8-1*5],
                                 [696-5*8-0*6, 696-5*8-0*6, 696-5*8-0*6, 696-5*8-0*6, 696-5*8-1*6],
                                 [32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-1*7],
                                 [32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-1*7],
                                 [448-5*8-0*7, 448-5*8-0*7, 448-5*8-0*7, 448-5*8-0*7, 448-5*8-1*7],
                                 [(80-1)*(5+1), (80-1)*(5+1), (80-1)*(5+1), (80-1)*(5+1),
                                  (80-1)*(5+1)]]))
    # non weighted norms
    assert np.allclose(test.jacobian(test.params, weigh_constraints=False),
                       np.array([[264-5*8-0*5, 264-5*8-0*5, 264-5*8-0*5, 264-5*8-0*5, 264-5*8-1*5],
                                 [696-5*8-0*6, 696-5*8-0*6, 696-5*8-0*6, 696-5*8-0*6, 696-5*8-1*6],
                                 [32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-1*7],
                                 [32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-0*7, 32-5*8-1*7],
                                 [448-5*8-0*7, 448-5*8-0*7, 448-5*8-0*7, 448-5*8-0*7, 448-5*8-1*7],
                                 [(80-1), (80-1), (80-1), (80-1), (80-1)]]))


def test_abstract():
    """ Check for abstract methods and properties
    """
    # both template_coeffs and compute_overlap not defined
    assert_raises(TypeError, lambda: ProjectedWavefunction(2, np.ones((3, 3)),
                                                           np.ones((3, 3, 3, 3))))
    # template_coeffs not defined
    class TestWfn(ProjectedWavefunction):
        def compute_overlap(self):
            pass

        def normalize(self):
            pass
    assert_raises(TypeError, lambda: TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))
    # compute_overlap not defined
    class TestWfn(ProjectedWavefunction):
        @property
        def template_coeffs(self):
            pass

        def normalize(self):
            pass
    assert_raises(TypeError, lambda: TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))
    # normalize not defined
    class TestWfn(ProjectedWavefunction):
        @property
        def template_coeffs(self):
            pass

        def compute_overlap(self):
            pass
    assert_raises(TypeError, lambda: TestWfn(2, np.ones((3, 3)), np.ones((3, 3, 3, 3))))
