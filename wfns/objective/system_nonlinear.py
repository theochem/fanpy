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

        \braket{\Phi_1 | \hat{H} | \Psi} &= E \braket{\Phi_1 | \Psi}\\
        \braket{\Phi_2 | \hat{H} | \Psi} &= E \braket{\Phi_2 | \Psi}\\
        &\vdots\\
        \braket{\Phi_K | \hat{H} | \Psi} &= E \braket{\Phi_K | \Psi}\\

    Energy is calculated with respect to the reference state.

    .. math::

        E = \braket{\Phi_{ref} | \hat{H} | \Psi}

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

    """
    def __init__(self, wfn, ham, pspace=None, ref_state=None, eqn_weights=None):
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

        """
        super().__init__(wfn, ham)
        self.assign_pspace(pspace)
        self.assign_ref_state(ref_state)
        self.assign_eqn_weights(eqn_weights)

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
        # FIXME: hard codes int format of Slater determinant
        elif (isinstance(pspace, (list, tuple)) and
              all(slater.is_internal_sd(state) or isinstance(state, (int, CIWavefunction))
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
        # FIXME: hard codes int format of Slater determinant
        elif slater.is_internal_sd(ref_state) or isinstance(ref_state, (int, CIWavefunction)):
            ref_state = (ref_state, )
        # FIXME: hard codes int format of Slater determinant
        elif (isinstance(ref_state, (list, tuple)) and
              all(slater.is_internal_sd(sd) or isinstance(sd, int) for sd in ref_state)):
            ref_state = tuple(slater.internal_sd(sd) for sd in ref_state)
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
        with respect to the reference state. The energy is computed with respect to the reference
        states:

        .. math::
            E = \braket{\Phi_{ref} | \hat{H} | \Psi}

        Parameters
        ----------
        params : np.ndarray(N,)
            Wavefunction parameters.

        Returns
        -------
        objective : np.ndarray(nproj+1, )
            Output of the function that will be optimized.

        Notes
        -----
        Wavefunction and Hamiltonian objects are updated iteratively.

        """
        wfn = self.wfn
        params = np.array(params)
        # Assign params
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
        norm = np.sum(get_overlap(ref_sds) * ref_coeffs)
        energy = np.sum(integrate_wfn_sd(ref_sds) * ref_coeffs)
        energy /= norm

        # objective
        obj = np.empty(self.nproj + 1, dtype=wfn.dtype)
        # <SD|H|Psi> - E<SD|Psi> == 0
        obj[:self.nproj] = integrate_wfn_sd(self.pspace) - energy * get_overlap(self.pspace)
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
            Wavefunction parameters.

        Returns
        -------
        jac : np.ndarray(nproj+1, nparams)
            Value of the Jacobian :math:`J_{ij}`.

        """
        wfn = self.wfn
        params = np.array(params)
        # Assign params
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

        # indices with respect to which objective is derivatized
        derivs, _ = np.meshgrid(np.arange(params.size), np.arange(params.size))
        # need to reshape to allow for summing over slater determinants
        derivs = derivs[np.newaxis, :, :]

        # reference values
        ref_olps = get_overlap(ref_sds)
        ref_ints = integrate_wfn_sd(ref_sds)
        d_ref_olps = get_overlap(ref_sds, derivs)
        d_ref_ints = integrate_wfn_sd(ref_sds, derivs)

        norm = np.sum(ref_olps * ref_coeffs)
        energy = np.sum(ref_ints * ref_coeffs)
        energy /= norm

        # reshape for broadcasting
        ref_olps = get_overlap(ref_sds)[:, np.newaxis, np.newaxis]
        ref_ints = integrate_wfn_sd(ref_sds)[:, np.newaxis, np.newaxis]

        # sum over reference Slater determinants
        d_norm = np.sum(2 * ref_olps * get_overlap(ref_sds, derivs), axis=0)
        d_energy = np.sum(d_ref_ints * ref_olps / norm +
                          ref_ints * d_ref_olps / norm -
                          d_norm[np.newaxis, :, :] * ref_ints * ref_olps / (norm**2),
                          axis=0)

        # reshape for broadcasting
        derivs = np.squeeze(derivs, axis=0)
        pspace = np.array(self.pspace)[:, np.newaxis]

        # jacobian
        jac = np.empty((self.nproj+1, params.size), dtype=wfn.dtype)
        jac[:self.nproj, :] = (integrate_wfn_sd(pspace, derivs) -
                               energy * get_overlap(pspace, derivs) -
                               d_energy * get_overlap(pspace)[:, np.newaxis])
        # Add normalization constraint
        jac[self.nproj, :] = d_norm
        # weigh equations
        jac *= self.eqn_weights[:, np.newaxis]

        return jac
