"""System of nonlinear equations solver for wavefunctions."""
from __future__ import absolute_import, division, print_function
import numpy as np
import scipy.optimize
from ..backend import slater, sd_list
from ..wavefunction.base_wavefunction import BaseWavefunction
from ..hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


def optimize_wfn_system(wfn, ham, pspace=None, ref_sds=None, save_file='', energy_is_param=False,
                        energy_guess=None, eqn_weights=None, solver=None, solver_kwargs=None):
    r"""Optimize the wavefunction with the given Hamiltonian as a system of nonlinear equations.

    Reference Slater determinants are used to calculate the norm and possibly the energy of the
    wavefunction.
    ..math::

        E = \sum_{\Phi \in S_{ref}} \braket{\Psi | \Phi} \braket{\Phi | \hat{H} | \Psi}
        N = \sum_{\Phi \in S_{ref}} \braket{\Psi | \Phi} \braket{\Phi | \Psi}

    where :math:`E` is the energy and :math:`N` is the norm. If the energy is used as a parameter,
    then it is not calculated, rather updated through the optimization process.

    Parameters
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state)
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study
    pspace : tuple/list, None
        Slater determinants onto which the wavefunction is projected
        Tuple of objects that are compatible with the internal_sd format
        By default, the largest space is used
    ref_sds : tuple/list of int, None
        One or more Slater determinant with respect to which the energy and the norm are calculated
        Default is ground state HF
    save_file : str
        Name of the numpy file that contains the wavefunction parameters of the last optimization
        step
        By default, does not save
    energy_is_param : bool
        Flag to control whether energy is calculated with respect to the reference Slater
        determinants or is optimized as a parameter
        By default, energy is not a parameter
    energy_guess : float
        Starting guess for the energy of the wavefunction
        By default, energy is calculated with respect to the reference Slater determinants
        Energy must be a parameter
    eqn_weights : np.ndarray
        Weights of each equation
        By default, all equations are given weight of 1 except for the normalization constraint,
        which is weighed by the number of equations.
    solver : function, None
        Solver that will solve the objective function (system of equations)
        By default scipy's least_squares function will be used
    solver_kwargs : dict
        Keyword arguments for the solver
        In order to disable default keyword arguments, the appropriate key need to be created with
        value `None`
        Default keyword arguments depend on the solver.

    Returns
    -------
    Output of the solver

    Raises
    ------
    TypeError
        If wavefunction is not an instance (or instance of a child) of BaseWavefunction
        If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian
        If wavefunction and Hamiltonian do not have the same data type
        If save_file is not a string
        If energy_is_param is not a boolean
        If energy_guess is not a float
        If eqn_weights is not a numpy array
        If eqn_weights do not have the same data type as wavefunction and Hamiltonian
        If solver_kwargs is not a dictionary or None
    ValueError
        If wavefunction and Hamiltonian do not have the same data type
        If wavefunction and Hamiltonian do not have the same number of spin orbitals
        If eqn_weights do not have the correct shape
        If energy_guess is given and energy is not a parameter.

    Note
    ----
    Assumes only one constraint (normalization constraint)
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

    if pspace is None:
        pspace = sd_list.sd_list(wfn.nelec, wfn.nspatial, spin=wfn.spin, seniority=wfn.seniority)
    else:
        pspace = [slater.internal_sd(sd) for sd in pspace]
    nproj = len(pspace)

    if ref_sds is None:
        ref_sds = [slater.ground(wfn.nelec, wfn.nspin)]
    else:
        ref_sds = [slater.internal_sd(sd) for sd in ref_sds]

    if not isinstance(save_file, str):
        raise TypeError('save_file must be a string.')

    if not isinstance(energy_is_param, bool):
        raise TypeError('energy_is_param must be a boolean.')

    if not energy_is_param and energy_guess is not None:
        raise ValueError('energy_guess cannot be given if energy is not a parameter.')
    if energy_guess is not None and not isinstance(energy_guess, float):
        raise TypeError('energy_guess must be a float')

    nparams = wfn.nparams
    if energy_is_param:
        nparams += 1

    if eqn_weights is None:
        eqn_weights = np.ones(nproj + 1)
        eqn_weights[-1] *= (nproj + 1)
    elif not isinstance(eqn_weights, np.ndarray):
        raise TypeError('Weights of the equations must be given as a numpy array')
    elif eqn_weights.dtype != wfn.dtype:
        raise TypeError('Weights of the equations must have the same dtype as the wavefunction '
                        'and Hamiltonian.')
    elif eqn_weights.shape != (nproj+1, ):
        raise ValueError('Weights of the equations must be given as a one-dimensional array of '
                         'shape, {0}.'.format((nproj+1, )))

    def _objective(params):
        r"""System of equations that corresponds to the (projected) Schrodinger equation.

        ..math::
            f_1(x) &= \braket{\Phi_1 | H | \Psi} - E \braket{\Phi_1 | \Psi}
            &\vdots
            f_K(x) &= \braket{\Phi_K | H | \Psi} - E \braket{\Phi_K | \Psi}
            f_{K+1}(x) &= constraint_1
            &\vdots

        where :math:`K` is the number of Slater determinant onto which the wavefunction is projected
        Equations after the :math:`K`th index are the constraints on the system of equations.
        The constraints, hopefully, will move out into their own module some time in the future.
        By default, the normalization constraint
        ..math::
            f_{K+1} = norm - 1
        is present where :math:`norm` is defined by ProjectedWavefunction.compute_norm.

        Parameters
        ----------
        params : 1-index np.ndarray
            Wavefunction parameters

        Returns
        -------
        objective : np.ndarray(nproj+1, )

        Note
        ----
        Wavefunction and Hamiltonian objects are updated iteratively.
        """
        # Update the wavefunction parameters
        if energy_is_param and not np.allclose(params[:-1], wfn.params.flat, atol=1e-14, rtol=0):
            wfn.params = params[:-1].reshape(wfn.params_shape)
            wfn.clear_cache()
        elif not energy_is_param and not np.allclose(params, wfn.params.flat, atol=1e-14, rtol=0):
            wfn.params = params.reshape(wfn.params_shape)
            wfn.clear_cache()

        # Save params
        if save_file != '':
            np.save('{0}_temp.npy'.format(save_file), wfn.params)

        # define norm and energy
        norm = sum(wfn.get_overlap(sd)**2 for sd in ref_sds)
        if energy_is_param:
            energy = params[-1]
        else:
            energy = sum(sum(ham.integrate_wfn_sd(wfn, sd)) * wfn.get_overlap(sd) for sd in ref_sds)
            energy /= norm

        obj = np.empty(nproj + 1, dtype=wfn.dtype)
        # <SD|H|Psi> - E<SD|Psi> == 0
        for i, sd in enumerate(pspace):
            obj[i] = sum(ham.integrate_wfn_sd(wfn, sd)) - energy * wfn.get_overlap(sd)
        # Add constraints
        obj[nproj] = norm - 1

        # weigh equations
        obj *= eqn_weights

        return obj

    def _jacobian(params):
        r"""Jacobian of the objective function.

        If :math:`\(f_1(\vec{x}), f_2(\vec{x}), \dots\)` is the objective function, the Jacobian is
        ..math::
            J_{ij}(\vec{x}) = \frac{\partial f_i(\vec{x})}{\partial x_j}
        where :math:`\vec{x}` is the parameters at a given iteration.

        Parameters
        ----------
        params : 1-index np.ndarray
            Wavefunction parameters

        Returns
        -------
        jac : np.ndarray(nproj+1, nparams)
            Value of the Jacobian :math:`J_{ij}`
        """
        # update wavefunction
        if energy_is_param and not np.allclose(params[:-1], wfn.params.flat, atol=1e-14, rtol=0):
            wfn.params = params[:-1].reshape(wfn.params_shape)
            wfn.clear_cache()
        elif not energy_is_param and not np.allclose(params, wfn.params.flat, atol=1e-14, rtol=0):
            wfn.params = params.reshape(wfn.params_shape)
            wfn.clear_cache()

        # define norm and energy
        norm = sum(wfn.get_overlap(sd)**2 for sd in ref_sds)
        if energy_is_param:
            energy = params[-1]
        else:
            energy = sum(sum(ham.integrate_wfn_sd(wfn, sd)) * wfn.get_overlap(sd) for sd in ref_sds)
        jac = np.empty((nproj + 1, nparams), dtype=wfn.dtype)

        for j in range(jac.shape[1]):
            if energy_is_param:
                if j == nparams - 1:
                    d_norm = 0.0
                    d_energy = 1.0
                else:
                    d_norm = sum(2 * wfn.get_overlap(sd) * wfn.get_overlap(sd, deriv=j)
                                 for sd in ref_sds)
                    d_energy = 0.0
            else:
                d_norm = sum(2 * wfn.get_overlap(sd) * wfn.get_overlap(sd, deriv=j)
                             for sd in ref_sds)
                d_energy = sum(sum(ham.integrate_wfn_sd(wfn, sd, deriv=j))*wfn.get_overlap(sd)/norm
                               + (sum(ham.integrate_wfn_sd(wfn, sd))*wfn.get_overlap(sd, deriv=j)
                                  / norm)
                               - (d_norm*sum(ham.integrate_wfn_sd(wfn, sd))*wfn.get_overlap(sd)
                                  / (norm**2))
                               for sd in ref_sds)
            for i, sd in enumerate(pspace):
                # <SD|H|Psi> - E<SD|Psi> = 0
                jac[i, j] = (sum(ham.integrate_wfn_sd(wfn, sd, deriv=j))
                             - energy*wfn.get_overlap(sd, deriv=j)
                             - d_energy*wfn.get_overlap(sd))

            # Add normalization constraint
            jac[-1, j] = d_norm

        # weigh equations
        jac *= eqn_weights[:, np.newaxis]
        return jac

    # check solver
    if solver is None:
        solver = scipy.optimize.least_squares
    elif (solver.__name__ == 'root' and 'scipy.optimize' in solver.__module__
          and nproj + 1 != nparams):
        raise ValueError('Cannot use root solver if the number of equations (including constraints)'
                         ' does not match with the number of parameters')

    # check keyword arguments
    if solver_kwargs is None:
        solver_kwargs = {}
    elif not isinstance(solver_kwargs, dict):
        raise TypeError('solver_kwargs must be a dictionary or None')

    # set default keyword arguments
    if solver.__name__ == 'least_squares' and 'scipy.optimize' in solver.__module__:
        default_kwargs = {'xtol': 1.0e-15, 'ftol': 1.0e-15, 'gtol': 1.0e-15,
                          'max_nfev': 1000*nparams, 'jac': _jacobian}
        # if no jacobian
        if 'jac' in solver_kwargs and solver_kwargs['jac'] in [None, False]:
            del solver_kwargs['jac']
            if wfn.dtype == np.float64:
                default_kwargs['jac'] = '3-point'
            elif wfn.dtype == np.complex128:
                default_kwargs['jac'] = 'cs'
    elif solver.__name__ == 'root' and 'scipy.optimize' in solver.__module__:
        default_kwargs = {'method': 'hybr', 'jac': _jacobian, 'options': {'xtol': 1.0e-9}}
        # if no jacobian
        if 'jac' in solver_kwargs and solver_kwargs['jac'] in [None, False]:
            del solver_kwargs['jac']
            del default_kwargs['jac']
            default_kwargs.update({'method': 'krylov',
                                   'options': {'fatol': 1.0e-9, 'xatol': 1.0e-7}})
    elif solver.__name__ == 'paraopt.cma':
        default_kwargs = {'npop': 2*nparams, 'verbose': False, 'sigma0': 0.01, 'max_iter': 1000,
                          'wtol': 1e-10}
    else:
        default_kwargs = {}

    # overwrite default keyword arguments
    default_kwargs.update(solver_kwargs)
    solver_kwargs = default_kwargs

    # apply solver
    if energy_is_param:
        if energy_guess is None:
            # compute starting energy
            energy_guess = sum(sum(ham.integrate_wfn_sd(wfn, sd)) * wfn.get_overlap(sd)
                               for sd in ref_sds)
            energy_guess /= sum(wfn.get_overlap(sd)**2 for sd in ref_sds)
        results = solver(_objective, np.hstack([wfn.params.flat, energy_guess]), **solver_kwargs)
        results['energy'] = results['x'][-1]
    else:
        results = solver(_objective, wfn.params.flat, **solver_kwargs)
        results['energy'] = sum(sum(ham.integrate_wfn_sd(wfn, sd)) * wfn.get_overlap(sd)
                                for sd in ref_sds)
        results['energy'] /= sum(wfn.get_overlap(sd)**2 for sd in ref_sds)

    return results
