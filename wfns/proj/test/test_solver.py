""" Tests wfns.proj.solver
"""
from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises
import numpy as np
from wfns.proj.proj_wavefunction import ProjectedWavefunction
from wfns.proj.solver import solve


class TestProjWavefunction(ProjectedWavefunction):
    """ Child of ProjectedWavefunction used to test solve

    Hard code the nonlinear problem with 3 equations:
    ..math::
        (2-z)(x-3)(y-2) = 0
        (2-z)(x^3 + y^2) = 0
        ((x-3)^2 (y-2)^2 - 1)*k = 0
    where x, y and z are parameters and k is the weight for the constraints

    Solution: z=2 and (x-3)^2 (y-2)^2 = 1
    """
    @property
    def template_coeffs(self):
        return np.array([1, 0])

    def compute_overlap(self, sd, deriv=None):
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

    def normalize(self):
        pass


def check_solver_type(test_wfn, init_guess, solver_type, use_jac, atol=1e-8):
    """ Checks that given solver type is able to produce a solution when run multiple times

    Parameters
    ----------
    test_wfn : TestProjWavefunction
        Specific wavefunction being tested
    init_guess : function
        Function that generates a (random) guess for the initial parameters
    solver_type : {'least_squares', 'root', 'cma', 'cma_guess'}
        Solver to use to solve
    use_jac : bool
        Flag for using Jacobian
    atol : float
        Tolerance for the absolute difference with the answer
        Default is 1e-8 (numpy default)
    """
    test_wfn.assign_params(init_guess().astype(test_wfn.dtype))
    result = solve(test_wfn, solver_type=solver_type, use_jac=use_jac)
    if result['success']:
        assert np.allclose(test_wfn.params[2], 2, atol=atol)
        assert np.allclose((test_wfn.params[0] - 3)**2 * (test_wfn.params[1] - 2)**2, 1, atol=atol)


def multiple_run(num_runs, rtol_num_errors, *args, **kwargs):
    """ Runs a test multiple times, keeping track of number of errors

    Parameters
    ----------
    num_runs : int
        Number of runs
    rtol_num_errors : float
        Percentage of the runs that are allowed to fail
    args : list
        List of arguments to feed into check_solver_type:
        test_wfn : TestProjWavefunction
            Specific wavefunction being tested
        init_guess : function
            Function that generates a (random) guess for the initial parameters
        solver_type : {'least_squares', 'root', 'cma', 'cma_guess'}
            Solver to use to solve
        use_jac : bool
            Flag for using Jacobian
    kwargs : dict
        Dictionary of keyword arguments to feed into check_solver_type:
        atol : float
            Tolerance for the absolute difference with the answer
            Default is 1e-8 (numpy default)
    """
    num_errors = 0
    for i in range(num_runs):
        try:
            check_solver_type(*args, **kwargs)
        except (AssertionError, ValueError):
            num_errors += 1
    if (num_runs - num_errors)/num_runs < rtol_num_errors:
        raise AssertionError('Optimizer given by {0} and {1} cannot solve the problem')


def test_solver_float():
    """ Tests wfns.proj.solver.solve with float wavefunction
    """
    test = TestProjWavefunction(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), dtype=float,
                                params=np.array([1.0, 1.0, 2.0]), pspace=[0b0011, 0b1100])
    # check
    assert_raises(TypeError, solve, int, 'least_squares')
    assert_raises(TypeError, solve, 'sdfsdf', 'least_squares')
    assert_raises(TypeError, solve, test, 'least squares')
    assert_raises(TypeError, solve, test, 'random')

    # answers (bad init guess)
    init_guess = lambda: (np.random.rand(3) - 0.5)*10
    multiple_run(20, 0.5, test, init_guess, 'root', False, atol=1e-8)
    multiple_run(20, 0.5, test, init_guess, 'root', True, atol=1e-8)
    multiple_run(20, 0.5, test, init_guess, 'least_squares', False, atol=1e-8)
    multiple_run(20, 0.5, test, init_guess, 'least_squares', True, atol=1e-8)
    multiple_run(20, 0.5, test, init_guess, 'cma', False, atol=1e-8)
    multiple_run(20, 0.5, test, init_guess, 'cma_guess', False, atol=1e-4)
    # answers (better init guess)
    init_guess = lambda: (np.random.rand(3) - 0.5)*2 + np.array([0, 0, 2])
    multiple_run(20, 0.8, test, init_guess, 'root', False, atol=1e-8)
    multiple_run(20, 0.8, test, init_guess, 'root', True, atol=1e-8)
    multiple_run(20, 0.8, test, init_guess, 'least_squares', False, atol=1e-8)
    multiple_run(20, 0.8, test, init_guess, 'least_squares', True, atol=1e-8)
    multiple_run(20, 0.8, test, init_guess, 'cma', False, atol=1e-8)
    multiple_run(20, 0.8, test, init_guess, 'cma_guess', False, atol=1e-4)
    # answers (good init guess)
    init_guess = lambda: (np.random.rand(3) - 0.5)*0.5 + np.array([0, 0, 2])
    multiple_run(20, 0.9, test, init_guess, 'root', False, atol=1e-8)
    multiple_run(20, 0.9, test, init_guess, 'root', True, atol=1e-8)
    multiple_run(20, 0.9, test, init_guess, 'least_squares', False, atol=1e-8)
    multiple_run(20, 0.9, test, init_guess, 'least_squares', True, atol=1e-8)
    multiple_run(20, 0.9, test, init_guess, 'cma', False, atol=1e-8)
    multiple_run(20, 0.9, test, init_guess, 'cma_guess', False, atol=1e-4)

def test_solver_complex():
    """ Tests wfns.proj.solver.solve with complex wavefunction
    """
    test = TestProjWavefunction(2, np.ones((2, 2)), np.ones((2, 2, 2, 2)), dtype=complex,
                                params=np.array([1.0, 1.0, 2.0]), pspace=[0b0011, 0b1100])
    # check
    assert_raises(NotImplementedError, solve, test, 'root')
    assert_raises(NotImplementedError, solve, test, 'cma')
    assert_raises(NotImplementedError, solve, test, 'cma_guess')

    # answers (bad init guess)
    init_guess = lambda: (np.random.rand(3) - 0.5)*10
    multiple_run(20, 0.5, test, init_guess, 'least_squares', False, atol=1e-8)
    multiple_run(20, 0.5, test, init_guess, 'least_squares', True, atol=1e-8)
    # answers (better init guess)
    init_guess = lambda: (np.random.rand(3) - 0.5)*2 + np.array([0, 0, 2])
    multiple_run(20, 0.8, test, init_guess, 'least_squares', False, atol=1e-8)
    multiple_run(20, 0.8, test, init_guess, 'least_squares', True, atol=1e-8)
    # answers (good init guess)
    init_guess = lambda: (np.random.rand(3) - 0.5)*0.5 + np.array([0, 0, 2])
    multiple_run(20, 0.9, test, init_guess, 'least_squares', False, atol=1e-8)
    multiple_run(20, 0.9, test, init_guess, 'least_squares', True, atol=1e-8)
