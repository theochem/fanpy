""" Solver for the CIWavefunction

Function
--------
solve(ci_wfn, solver_type='eigh', **kwargs)
    Solves a given CIWavefunction
"""
from __future__ import absolute_import, division, print_function
from scipy.linalg import eigh

def solve(ci_wfn, solver_type='eigh', **kwargs):
    """ Solves CI wavefunction for the coefficients

    Parameters
    ----------
    ci_wfn : instance of CIWavefunction
        Wavefunction to solve
    solver_type : {'eigh', 'davidson'}
        Algorithm for solving wavefunction
        Default is 'eigh'
        Use 'eigh' for eigenvalue decomposition of Hermitian matrices (explicitly constructs
        Hamiltonian matrix)
        Use 'davidson' for Davidson algorithm
    kwargs : dict, optional
        Keywords to pass to the internal solver
        For more info on 'eigh' solver, see `scipy.linalg.eigh`

    Returns
    -------
    result : dict
        'success' : bool
            Describes whether optimization was sucessful

    Raises
    ------
    TypeError
        If solver_type is not one of 'eigh' and 'davidson'
    """
    if solver_type not in ['eigh', 'davidson']:
        raise TypeError('Given solver type, {0}, is not supported'.format(solver_type))

    result = {}
    if solver_type == 'eigh':
        ci_matrix = ci_wfn.compute_ci_matrix()
        min_exc = min(ci_wfn.dict_exc_index.iterkeys())
        max_exc = max(ci_wfn.dict_exc_index.iterkeys())
        eigval, eigvec = eigh(ci_matrix, eigvals=(min_exc, max_exc), **kwargs)
        del ci_matrix

        # NOTE: overwrites last sd_coeffs
        for exc, index in ci_wfn.dict_exc_index.iteritems():
            ci_wfn.sd_coeffs[:, index] = eigvec[:, exc - min_exc]
            ci_wfn.energies[index] = eigval[exc - min_exc]

        result['success'] = True
    else:
        raise NotImplementedError

    return result
