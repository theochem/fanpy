"""Single equation solver for wavefunctions."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
from wfns.solver import energies
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
import wfns.backend.slater as slater
import wfns.backend.sd_list as sd_list
from wfns.wrapper.docstring import docstring


# FIXME: copies most of system_solver for initialization and docstring
@docstring(indent_level=1)
def optimize_wfn_variational(wfn, ham, left_pspace=None, right_pspace=None, ref_sds=None,
                             save_file='', solver=None, solver_kwargs=None, norm_constrained=False):
    r"""Optimize the wavefunction with the given Hamiltonian as a single equation for energy.

    Solves the following equation:

    .. math::

        \bra{\Psi}
        \sum_{\mathbf{m} \in P_{left}} \ket{\mathbf{m}} \bra{\mathbf{m}}
        \hat{H}
        \sum_{\mathbf{n} \in P_{right}} \ket{\mathbf{n}} \bra{\mathbf{n}}
        \ket{\Psi}
        &= E \sum_{\mathbf{m} \in P_{ref}}
        \braket{ \Psi | \mathbf{m}} \braket{\mathbf{m} | \Psi}\\
        \sum_{\mathbf{m} \in P_{left}} \sum_{\mathbf{n} \in P_{right}}
        \braket{\Psi | \mathbf{m}}
        \braket{\mathbf{m} | \hat{H} | \mathbf{n}}
        \braket{\mathbf{n} | \Psi}
        &= E \sum_{\mathbf{m} \in P_{ref}}
        \braket{ \Psi | \mathbf{m}} \braket{\mathbf{m} | \Psi}

    where :math:`P_{left}` and :math:`P_{right}` are the set of Slater determinants for which the
    left and right sides of the Schrodinger equation are projected.

    Wavefunction is optimized by minimizing the energy.

    .. math::

        E =
        \frac{
            \sum_{\mathbf{m} \in P_{left}} \sum_{\mathbf{n} \in P_{right}}
            \braket{\Psi | \mathbf{m}}
            \braket{\mathbf{m} | \hat{H} | \mathbf{n}}
            \braket{\mathbf{n} | \Psi}
        }{
            \sum_{\mathbf{m} \in P_{left} \cap P_{right}}
            \braket{ \Psi | \mathbf{m}} \braket{\mathbf{m} | \Psi}
        }

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study.
    left_pspace : {tuple/list, None}
        Slater determinants onto which the left side of the Schrodinger equation is projected.
        Tuple of objects that are compatible with the internal_sd format.
        By default, the largest space is used.
    right_pspace : {tuple/list, None}
        Slater determinants onto which the right side of the Schrodinger equation is projected.
        Tuple of objects that are compatible with the internal_sd format.
        By default, the largest space is used.
    ref_sds : {tuple/list of int, None}
        One or more Slater determinant with respect to which the energy and the norm are calculated.
        Default is ground state HF.
    save_file : str
        Name of the numpy file that contains the wavefunction parameters of the last optimization
        step.
        By default, does not save.
    energy_is_param : bool
        Flag to control whether energy is calculated with respect to the reference Slater
        determinants or is optimized as a parameter.
        By default, energy is not a parameter.
    energy_guess : float
        Starting guess for the energy of the wavefunction.
        By default, energy is calculated with respect to the reference Slater determinants.
        Energy must be a parameter.
    solver : {function, None}
        Solver that will solve the objective function (system of equations).
        By default scipy's least_squares function will be used.
    solver_kwargs : dict
        Keyword arguments for the solver.
        In order to disable default keyword arguments, the appropriate key need to be created with
        value `None`.
        Default keyword arguments depend on the solver.

    Returns
    -------
    output : dict
        Output of the solver.

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian.
        If save_file is not a string.
        If solver_kwargs is not a dictionary or `None`.
    ValueError
        If wavefunction and Hamiltonian do not have the same data type.
        If wavefunction and Hamiltonian do not have the same number of spin orbitals.

    Notes
    -----
    Optimized wavefunction may not be variational; especially if the left and right projection space
    are not equal.

    """
    # Preprocess variables
    if not isinstance(wfn, BaseWavefunction):
        raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                        'child).')
    if not isinstance(ham, ChemicalHamiltonian):
        raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its '
                        'child).')
    if wfn.dtype != ham.dtype:
        raise ValueError('Wavefunction and Hamiltonian do not have the same data type.')
    if wfn.nspin != ham.nspin:
        raise ValueError('Wavefunction and Hamiltonian do not have the same number of '
                         'spin orbitals')

    if not isinstance(save_file, str):
        raise TypeError('save_file must be a string.')

    # FIXME: incredibly slow implementation
    def _objective(params):
        """Energy of the Schrodinger equation after projecting out the left and right sides."""
        # update wavefunction
        if not np.allclose(params, wfn.params.flat, atol=1e-12, rtol=0):
            wfn.params = np.array(params).reshape(wfn.params_shape)
            wfn.clear_cache()

        # save params
        if save_file != '':
            np.save('{0}_temp.npy'.format(save_file), wfn.params)

        if left_pspace is not None and right_pspace is not None:
            return energies.get_energy_two_proj(wfn, ham, l_pspace_energy=left_pspace,
                                                r_pspace_energy=right_pspace, pspace_norm=ref_sds,
                                                use_norm=not norm_constrained, return_grad=False)
        elif right_pspace is None:
            return energies.get_energy_one_proj(wfn, ham, pspace_energy=left_pspace,
                                                pspace_norm=ref_sds, use_norm=not norm_constrained,
                                                return_grad=False)
        else:
            return energies.get_energy_one_proj(wfn, ham, pspace_energy=right_pspace,
                                                pspace_norm=ref_sds, use_norm=not norm_constrained,
                                                return_grad=False)

    # gradiant
    def _gradient(params):
        """Gradient of the energy of the Schrodinger equation after projection."""
        if not np.allclose(params, wfn.params.flat, atol=1e-12, rtol=0):
            wfn.params = np.array(params).reshape(wfn.params_shape)
            wfn.clear_cache()

        if left_pspace is not None and right_pspace is not None:
            return energies.get_energy_two_proj(wfn, ham, l_pspace_energy=left_pspace,
                                                r_pspace_energy=right_pspace, pspace_norm=ref_sds,
                                                use_norm=not norm_constrained, return_grad=True)
        elif right_pspace is None:
            return energies.get_energy_one_proj(wfn, ham, pspace_energy=left_pspace,
                                                pspace_norm=ref_sds, use_norm=not norm_constrained,
                                                return_grad=True)
        else:
            return energies.get_energy_one_proj(wfn, ham, pspace_energy=right_pspace,
                                                pspace_norm=ref_sds, use_norm=not norm_constrained,
                                                return_grad=True)

    # check solver
    if solver is None:
        solver = scipy.optimize.minimize
        default_kwargs = {'method': 'BFGS', 'jac': _gradient, 'options': {'gtol': 1e-8}}
    elif solver.lower() == 'powell':
        solver = scipy.optimize.minimize
        default_kwargs = {'method': 'Powell', 'options': {'xtol': 1e-9, 'ftol': 1e-9}}
    elif solver == 'cma':
        import cma
        solver = cma.fmin
        default_kwargs = {'sigma0': 0.01, 'gradf': _gradient, 'options': {'ftarget': 1e-7}}
    else:
        default_kwargs = {}

    # check keyword arguments
    if solver_kwargs is None:
        solver_kwargs = {}
    elif not isinstance(solver_kwargs, dict):
        raise TypeError('solver_kwargs must be a dictionary or None')

    # overwrite default keyword arguments
    default_kwargs.update(solver_kwargs)
    solver_kwargs = default_kwargs

    # add energy to results
    results = solver(_objective, wfn.params.flat, **solver_kwargs)

    return results

# TODO: add support for least squares problem
