"""Test wfns.solver.equation_solver."""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
import scipy.optimize
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
import wfns.solver.equation_solver as equation_solver


class TestBaseWavefunction(BaseWavefunction):
    """Base wavefunction that bypasses abc structure and overwrite properties and attributes."""
    nelec = 4
    nspin = 8
    dtype = np.float64
    params = np.random.rand(10)
    nparams = 10
    _spin = None
    _seniority = None

    def __init__(self):
        pass

    def get_overlap(self, sd, deriv=None):
        if sd == 0b0011:
            if deriv is None:
                return (self.params[0]-3)*(self.params[1]-2)
            elif deriv == 0:
                return (self.params[1]-2)
            elif deriv == 1:
                return (self.params[0]-3)
            else:
                return 0
        elif sd == 0b1100:
            if deriv is None:
                return self.params[0]**3 + self.params[1]**2
            elif deriv == 0:
                return 3*self.params[0]**2
            elif deriv == 1:
                return 2*self.params[1]
            else:
                return 0
        else:
            return 0

    @property
    def spin(self):
        return self._spin

    @property
    def seniority(self):
        return self._seniority

    @property
    def template_params(self):
        return None


class TestChemicalHamiltonian(ChemicalHamiltonian):
    """ChemicalHamiltonian that bypasses abc structure and overwrite properties and attributes."""
    dtype = np.float64
    nspin = 8

    def __init__(self):
        pass


def test_equation_solver_initialize():
    """Test input checks of wfns.solver.equation_solver.optimize_wfn_variational.

    Note
    ----
    Properties and attributesof the base classes are overwritten. This means that you shouldn't use
    the following code as a reference for setting wavefunction and hamiltonian instances
    """
    # check wfn and hamiltonian
    test_wfn = TestBaseWavefunction()
    test_ham = TestChemicalHamiltonian()
    assert_raises(TypeError, equation_solver.optimize_wfn_variational, test_wfn, None, None, None,
                  '', None, None)
    assert_raises(TypeError, equation_solver.optimize_wfn_variational, None, test_ham, None, None,
                  '', None, None)
    test_wfn = TestBaseWavefunction()
    test_ham = TestChemicalHamiltonian()
    test_ham.dtype = np.complex128
    assert_raises(ValueError, equation_solver.optimize_wfn_variational, test_wfn, test_ham,
                  None, None, '', None, None)
    test_wfn = TestBaseWavefunction()
    test_ham = TestChemicalHamiltonian()
    test_ham.nspin = 6
    assert_raises(ValueError, equation_solver.optimize_wfn_variational, test_wfn, test_ham,
                  None, None, '', None, None)

    # save_files
    test_wfn = TestBaseWavefunction()
    test_ham = TestChemicalHamiltonian()
    assert_raises(TypeError, equation_solver.optimize_wfn_variational, test_wfn, test_ham, None,
                  None, None, None, None, None)

    # solver
    test_wfn = TestBaseWavefunction()
    test_ham = TestChemicalHamiltonian()
    assert_raises(TypeError, equation_solver.optimize_wfn_variational, test_wfn, test_ham, None,
                  None, '', None, None, [])


def trial_run_equation_solver(num_runs, rtol_num_errors, atol=None, init_guess=None,
                              solver_type=None, solver_kwargs=None):
    """ Runs a test multiple times, keeping track of number of errors

    Parameters
    ----------
    num_runs : int
        Number of runs
    rtol_num_errors : float
        Percentage of the runs that are allowed to fail
    atol : float
        Tolerance for the absolute difference with the answer
        Default is 1e-8 (numpy default)
    init_guess : function
        Function that generates a (random) guess for the initial parameters
    solver_type : function
        Solver to use to solve the system of nonlinear equations
    solver_kwargs : dict
        Keyword arguments for the solver
    """
    wfn = TestBaseWavefunction()
    wfn.nelec = 2
    wfn.nspin = 4
    wfn.dtype = np.float64
    wfn.nparams = 2
    wfn.params = 10*(np.random.rand(2) - 0.5)
    wfn._cache_fns = {}
    ham = TestChemicalHamiltonian()
    ham.assign_orbtype(None)
    ham.assign_energy_nuc_nuc(None)
    ham.assign_integrals(np.ones((2, 2)), np.ones((2, 2, 2, 2)))
    pspace = [0b0011, 0b1100]

    num_errors = 0
    for i in range(num_runs):
        try:
            wfn.params = init_guess()
            result = equation_solver.optimize_wfn_variational(wfn, ham, left_pspace=pspace,
                                                              right_pspace=None, solver=solver_type,
                                                              solver_kwargs=solver_kwargs)
            if result['success']:
                assert np.allclose(result['fun'], 2, atol=atol)
                assert np.allclose((wfn.params[0] - 3)**2 * (wfn.params[1] - 2)**2, 1,
                                   atol=atol)
        except (AssertionError, ValueError):
            num_errors += 1
    if (num_runs - num_errors)/num_runs < rtol_num_errors:
        raise AssertionError('Optimizer, {0}, given by cannot solve the problem'
                             ''.format(solver_type))


def test_optimize_wfn_variational():
    """Test wfns.solver.equation_solver.optimize_wfn_variational."""
    # minimize
    # bad init guess
    trial_run_equation_solver(20, 0.5, atol=1e-8, init_guess=lambda: 10*(np.random.rand(2) - 0.5),
                              solver_type=scipy.optimize.minimize, solver_kwargs={'jac': None})
    trial_run_equation_solver(20, 0.5, atol=1e-8, init_guess=lambda: 10*(np.random.rand(2) - 0.5),
                              solver_type=scipy.optimize.minimize, solver_kwargs={})
    # less bad init guess
    trial_run_equation_solver(20, 0.5, atol=1e-8, init_guess=lambda: 2*(np.random.rand(2) - 0.5),
                              solver_type=scipy.optimize.minimize, solver_kwargs={'jac': None})
    trial_run_equation_solver(20, 0.5, atol=1e-8, init_guess=lambda: 2*(np.random.rand(2) - 0.5),
                              solver_type=scipy.optimize.minimize, solver_kwargs={})
    # good init guess
    trial_run_equation_solver(20, 0.5, atol=1e-8, init_guess=lambda: 0.5*(np.random.rand(2) - 0.5),
                              solver_type=scipy.optimize.minimize, solver_kwargs={'jac': None})
    trial_run_equation_solver(20, 0.5, atol=1e-8, init_guess=lambda: 0.5*(np.random.rand(2) - 0.5),
                              solver_type=scipy.optimize.minimize, solver_kwargs={})

    # root need equal number of projections as there are parameters
    # cma
    try:
        import paraopt
        trial_run_equation_solver(20, 0.5, atol=1e-8,
                                  init_guess=lambda: 10*(np.random.rand(2) - 0.5),
                                  solver_type=paraopt.cma, solver_kwargs={})
        trial_run_equation_solver(20, 0.5, atol=1e-8,
                                  init_guess=lambda: 2*(np.random.rand(2) - 0.5),
                                  solver_type=paraopt.cma, solver_kwargs={})
        trial_run_equation_solver(20, 0.5, atol=1e-8, energy_is_param=False,
                                  init_guess=lambda: 0.5*(np.random.rand(2) - 0.5),
                                  solver_type=paraopt.cma, solver_kwargs={})
    except ModuleNotFoundError:
        print('Module, paraopt, is not available.')
