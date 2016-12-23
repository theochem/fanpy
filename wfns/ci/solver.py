from __future__ import absolute_import, division, print_function
from scipy.linalg import eigh

def solve(wavefunction, solver_type='eigh', **kwargs):
    """
    Optimize `wavefunction.objective(params)` to solve the coefficient vector.

    Parameters
    ----------
    wavefunction : instance of ProjectionWavefunction
        Wavefunction to solve
    solver_type : str
        solver type
        One of 'eigh'
        Default is Least Squares Nonlinear system of equation solver
    kwargs : dict, optional
        Keywords to pass to the internal solver, `scipy.optimize.root`.

    Returns
    -------
    result : tuple
    """
    # check solver_type
    if solver_type not in ['eigh', 'davidson']:
        raise TypeError('Given solver type, {0}, is not supported'.format(solver_type))

    ci_matrix = wavefunction.compute_ci_matrix()
    min_exc = min(wavefunction.dict_exc_index.iterkeys())
    max_exc = max(wavefunction.dict_exc_index.iterkeys())
    result = eigh(ci_matrix, eigvals=(min_exc, max_exc), **kwargs)
    del ci_matrix

    # NOTE: overwrites last sd_coeffs
    for exc, index in wavefunction.dict_exc_index.iteritems():
        wavefunction.sd_coeffs[:, index] = result[1][:, exc - min_exc]
        wavefunction.energies[index] = result[0][exc - min_exc]

    return result
