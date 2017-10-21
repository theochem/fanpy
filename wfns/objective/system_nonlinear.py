"""Schrodinger equation as a system of equations."""
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.objective.base_objective import BaseObjective
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.backend import slater, sd_list


@docstring_class(indent_level=1)
class SystemEquations(BaseObjective):
    r"""Schrodinger equation as a system of equations.

    .. math::

        \braket{\Phi_1 | \hat{H} | \Psi} - E \braket{\Phi_1 | \Psi} &= 0\\
        \braket{\Phi_2 | \hat{H} | \Psi} - E \braket{\Phi_2 | \Psi} &= 0\\
        &\vdots\\
        \braket{\Phi_K | \hat{H} | \Psi} - E \braket{\Phi_K | \Psi} &= 0\\

    Energy can be a constant, a parameter that gets optimized, or a function of the wavefunction and
    hamiltonian parameters.

    .. math::

        E = \frac{\braket{\Phi_{ref} | \hat{H} | \Psi}{\braket{\Phi_{ref} | \Psi}}

    Additionally, the normalization constraint is added with respect to the reference state.

    Attributes
    ----------
    pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected.
        By default, the largest space is used.
    ref_state : {tuple/list of int, CIWavefunction, None}
        State with respect to which the energy and the norm are computed.
        If a list/tuple of Slater determinants are given, then the reference state is the given
        wavefunction truncated by the provided Slater determinants.
        Default is ground state HF.
    eqn_weights : np.ndarray
        Weights of each equation.
        By default, all equations are given weight of 1 except for the normalization constraint,
        which is weighed by the number of equations.
    energy_type : {'fixed', 'param', 'compute'}
        Type of the energy used in the Schrodinger equation.
        If 'fixed', the energy of the Schrodinger equation is fixed to the given value.
        If 'param', the energy of the Schrodinger equation is given by a parameter. Specifically,
        the last parameter will be the energy.
        If 'compute', the energy of the Schrodinger equation is computed with respect to the
        reference.
        By default, the energy is computed with respect to the reference.
    energy : {float, None}
        Energy used in the Schrodinger equation.
        By default, the energy is calculated with respect to the reference.

    """
    def __init__(self, wfn, ham, pspace=None, ref_state=None, eqn_weights=None,
                 energy_type='compute', energy=None):
        """

        Parameters
        ----------
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the Schrodinger equation is projected.
            By default, the largest space is used.
        ref_state : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground state HF.
        eqn_weights : np.ndarray
            Weights of each equation.
            By default, all equations are given weight of 1 except for the normalization constraint,
            which is weighed by the number of equations.
        energy_type : {'fixed', 'param', 'compute'}
            Type of the energy used in the Schrodinger equation.
            If 'fixed', the energy of the Schrodinger equation is fixed to the given value.
            If 'param', the energy of the Schrodinger equation is given by a parameter.
            Specifically, the last parameter will be the energy.
            If 'compute', the energy of the Schrodinger equation is computed with respect to the
            reference.
            By default, the energy is computed with respect to the reference.
        energy : {float, None}
            Energy of the Schrodinger equation.
            If provided, the energy within the Schrodinger equation is fixed to the given value.
            If not provided, the energy within the Schrodinger equation is given as a parameter (if
            energy is a parameter) or is computed when needd (if energy is not a parameter).
            By default, `fixed_energy` is not provided.

        Raises
        ------
        ValueError
            If fixed_energy is a parameter and `energy` is not None.
        TypeError
            If fixed_energy is not None or float.

        """
        super().__init__(wfn, ham)
        self.assign_pspace(pspace)
        self.assign_ref_state(ref_state)
        self.assign_eqn_weights(eqn_weights)

        if energy_type not in ['fixed', 'param', 'compute']:
            raise ValueError("`energy_type` must be one of 'fixed', 'param', or 'compute'.")
        if not (energy is None or isinstance(energy, (float, complex))):
            raise TypeError('`energy` must be a float or None.')

        if energy_type in ['fixed', 'param'] and energy is None:
            energy = self.get_energy_one_proj(self.ref_state)
        elif energy_type == 'compute' and energy is not None:
            raise ValueError("`energy` cannot be given when the `energy_type` is 'compute'.")
        self.energy_type = energy_type
        self.energy = energy

    @property
    def params(self):
        """

        Notes
        -----
        The last parameter will be the energy.

        """
        old_params = super().params
        if self.energy_type == 'param':
            return np.hstack((old_params, self.energy))
        else:
            return old_params

    def assign_params(self, params):
        if self.energy_type == 'param':
            self.energy = params[-1]
        super().assign_params(params[:-1])

    def assign_pspace(self, pspace=None):
        """Set the projection space.

        Parameters
        ----------
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the Schrodinger equation is projected.
            By default, the largest space is used.

        Raises
        ------
        TypeError
            If list/tuple of incompatible objects.
            If not list/tuple or None.

        """
        if pspace is None:
            pspace = sd_list.sd_list(self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin,
                                     seniority=self.wfn.seniority)
        elif (isinstance(pspace, (list, tuple)) and
              all(slater.is_sd_compatible(state) or isinstance(state, CIWavefunction)
                  for state in pspace)):
            pspace = tuple(pspace)
        else:
            raise TypeError('Projected space must be given as a list/tuple of Slater determinants. '
                            'See `backend.slater` for compatible Slater determinant formats.')
        self.pspace = pspace

    def assign_ref_state(self, ref_state=None):
        """Set the reference state.

        Parameters
        ----------
        ref_state : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground state HF.

        """
        if ref_state is None:
            ref_state = (slater.ground(self.wfn.nelec, self.wfn.nspin), )
        elif slater.is_sd_compatible(ref_state):
            ref_state = (ref_state, )
        elif (isinstance(ref_state, (list, tuple)) and
              all(slater.is_sd_compatible(sd) for sd in ref_state)):
            ref_state = tuple(slater.internal_sd(sd) for sd in ref_state)
        elif isinstance(ref_state, CIWavefunction):
            ref_state = ref_state
        else:
            raise TypeError('Reference state must be given as a Slater determinant, a CI '
                            'Wavefunction, or a list/tuple of Slater determinants. See '
                            '`backend.slater` for compatible representations of the Slater '
                            'determinants.')

    def assign_eqn_weights(self, eqn_weights=None):
        """Set the weights of each equation.

        Parameters
        ----------
        eqn_weights : {np.ndarray, None}
            Weights of each equation.
            By default, all equations are given weight of 1 except for the normalization constraint,
            which is weighed by the number of equations.

        Raises
        ------
        Type Error
            If eqn_weights is not a numpy array.
            If eqn_weights do not have the same data type as wavefunction and Hamiltonian.
        ValueError
            If eqn_weights do not have the correct shape.

        """
        if eqn_weights is None:
            eqn_weights = np.ones(self.nproj + 1)
            eqn_weights[-1] *= self.nproj + 1
        elif not isinstance(eqn_weights, np.ndarray):
            raise TypeError('Weights of the equations must be given as a numpy array.')
        elif eqn_weights.dtype != self.wfn.dtype:
            raise TypeError('Weights of the equations must have the same dtype as the wavefunction '
                            'and Hamiltonian.')
        elif eqn_weights.shape != (self.nproj+1, ):
            raise ValueError('Weights of the equations must be given as a one-dimensional array of '
                             'shape, {0}.'.format((self.nproj+1, )))
        self.eqn_weights = eqn_weights

    @property
    def nproj(self):
        """Return the number of projected states.

        Returns
        -------
        nproj : int
            Number of equations.

        """
        return len(self.pspace)

    def objective(self, params):
        r"""System of equations that corresponds to the Projected Schrodinger equation.

        .. math::

            f_1(x) &= \braket{\Phi_1 | \hat{H} | \Psi} - E \braket{\Phi_1 | \Psi}\\
            &\vdots\\
            f_K(x) &= \braket{\Phi_K | \hat{H} | \Psi} - E \braket{\Phi_K | \Psi}\\
            f_{K+1}(x) &= \braket{\Phi_{ref} | \Psi} - 1\\

        where :math:`K` is the number of Slater determinant onto which the wavefunction is
        projected. The :math:`K+1`th equation is the normalization constraint. The norm is computed
        with respect to the reference state. The energy can be a constant, a parameter that gets
        optimized, or a function of the wavefunction and hamiltonian parameters.

        .. math::

            E = \frac{\braket{\Phi_{ref} | \hat{H} | \Psi}{\braket{\Phi_{ref} | \Psi}}

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters that describe the system of equations.

        Returns
        -------
        objective : np.ndarray(nproj+1, )
            Output of the function that will be optimized.

        """
        params = np.array(params)
        # Assign params to wavefunction, hamiltonian, and other involved objects.
        self.assign_params(params)
        # Save params
        self.save_params(params)

        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_wfn_sd = np.vectorize(self.wrapped_integrate_wfn_sd)

        # define reference
        if isinstance(self.ref_state[0], CIWavefunction):
            ref_sds = self.ref_state[0].sd_vec
            ref_coeffs = self.ref_state[0].params
        else:
            ref_sds = self.ref_state
            ref_coeffs = get_overlap(ref_sds)

        # reference values
        norm = np.sum(ref_coeffs * get_overlap(ref_sds))
        if self.energy_type == 'param':
            energy = params[-1]
        elif self.energy_type == 'fixed':
            energy = self.energy
        elif self.energy_type == 'compute':
            energy = np.sum(ref_coeffs * integrate_wfn_sd(ref_sds)) / norm

        # objective
        obj = np.empty(self.nproj + 1)
        # <SD|H|Psi> - E<SD|Psi> == 0
        # FIXME: does not support pspace that is linear combination of sd's
        obj[:self.nproj] = integrate_wfn_sd(self.pspace)
        obj[:self.nproj] -= energy * get_overlap(self.pspace)
        # Add constraints
        obj[self.nproj] = norm - 1
        # weigh equations
        obj *= self.eqn_weights

        return obj

    def jacobian(self, params):
        r"""Jacobian of the objective function.

        If :math:`\(f_1(\vec{x}), f_2(\vec{x}), \dots\)` is the objective function, the Jacobian is

        .. math::

            J_{ij}(\vec{x}) = \frac{\partial f_i(x_1, \dots, x_N)}{\partial x_j}

        where :math:`\vec{x}` is the parameters at a given iteration.

        Parameters
        ----------
        params : np.ndarray(N,)
            Parameters that describe the system of equations.

        Returns
        -------
        jac : np.ndarray(nproj+1, nparams)
            Value of the Jacobian :math:`J_{ij}`.

        Notes
        -----
        Much of the code is copied from `BaseObjective.get_energy_one_proj` to compute the energy.
        It is not called because the norm is still needed for the constraint (meaning that much of
        the code will be copied over anyways), and the derivative of the energy uses some of the
        the same matrices.

        """
        params = np.array(params)
        # Assign params
        self.assign_params(params)
        # Save params
        self.save_params(params)

        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_wfn_sd = np.vectorize(self.wrapped_integrate_wfn_sd)

        # indices with respect to which objective is derivatized
        derivs = np.arange(self.nproj + 1)
        # need to reshape to allow for summing over slater determinants
        derivs = derivs[np.newaxis, :]

        # define reference
        if isinstance(self.ref_state[0], CIWavefunction):
            ref_sds = self.ref_state[0].sd_vec
            ref_coeffs = self.ref_state[0].params
            d_ref_coeffs = np.zeros((len(ref_sds), derivs.size))
        else:
            ref_sds = self.ref_state
            ref_coeffs = get_overlap(ref_sds)
            d_ref_coeffs = get_overlap(ref_sds[:, np.newaxis], derivs)

        # overlaps and integrals
        ref_olps = get_overlap(ref_sds)
        d_ref_olps = get_overlap(ref_sds[:, np.newaxis], derivs)
        ref_ints = integrate_wfn_sd(ref_sds)
        d_ref_ints = integrate_wfn_sd(ref_sds[:, np.newaxis], derivs)
        # NOTE: d_ref_olps and d_ref_ints are two dimensional tensors (axis 0 corresponds to the
        # reference Slater determinants and 2 to the index of the parameter with respect to which
        # the value is derivatized).

        # norm
        norm = np.sum(ref_olps * ref_coeffs)
        d_norm = np.sum(ref_olps[:, np.newaxis]*d_ref_coeffs + d_ref_olps*ref_coeffs[:, np.newaxis],
                        axis=0)

        # energy
        if self.energy_type == 'param':
            energy = params[-1]
            d_energy = np.zeros(params.size)
            d_energy[-1] = 1.0
        elif self.energy_type == 'fixed':
            energy = self.energy
            d_energy = np.zeros(params.size)
        elif self.energy_type == 'compute':
            energy = np.sum(ref_coeffs * ref_ints) / norm
            d_energy = np.sum(d_ref_ints * ref_olps[:, np.newaxis] +
                              ref_ints[:, np.newaxis] * d_ref_olps -
                              d_norm[np.newaxis, :] * energy, axis=0) / norm

        # reshape for broadcasting
        pspace = np.array(self.pspace)[:, np.newaxis]

        # jacobian
        # FIXME: does not support pspace that is linear combination of sd's
        jac = np.empty((self.nproj+1, params.size))
        jac[:self.nproj, :] = integrate_wfn_sd(pspace, derivs)
        jac[:self.nproj, :] -= energy * get_overlap(pspace, derivs)
        jac[:self.nproj, :] -= d_energy[np.newaxis, :] * get_overlap(pspace)
        # Add normalization constraint
        jac[self.nproj, :] = d_norm
        # weigh equations
        jac *= self.eqn_weights[:, np.newaxis]

        return jac
