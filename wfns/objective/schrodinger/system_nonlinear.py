"""Schrodinger equation as a system of equations."""
import numpy as np
from wfns.backend import sd_list, slater
from wfns.objective.base import BaseObjective
from wfns.objective.constraints.norm import NormConstraint
from wfns.objective.schrodinger.base import BaseSchrodinger
from wfns.param import ParamContainer
from wfns.wfn.ci.base import CIWavefunction


class SystemEquations(BaseSchrodinger):
    r"""Schrodinger equation as a system of equations.

    .. math::

        \left< \Phi_1 \middle| \hat{H} \middle| \Psi \right> - E \left< \Phi_1 \middle| \Psi \right>
        &= 0\\
        \left< \Phi_2 \middle| \hat{H} \middle| \Psi \right> - E \left< \Phi_2 \middle| \Psi \right>
        &= 0\\
        &\vdots\\
        \left< \Phi_K \middle| \hat{H} \middle| \Psi \right> - E \left< \Phi_K \middle| \Psi \right>
        &= 0\\
        f_{constraint}(\Psi, \hat{H}) &= 0

    Energy can be a constant, a parameter that gets optimized, or a function of the wavefunction and
    hamiltonian parameters.

    .. math::

        E = \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                 {\left< \Phi_{ref} \middle| \Psi \right>}

    Additionally, the normalization constraint is added with respect to the reference state.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : BaseHamiltonian
        Hamiltonian that defines the system under study.
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        By default, the parameter values are not stored.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    param_selection : ParamMask
        Selection of parameters that will be used in the objective.
        Default selects the wavefunction parameters.
        Any subset of the wavefunction, composite wavefunction, and Hamiltonian parameters can be
        selected.
    pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
        States onto which the Schrodinger equation is projected.
        By default, the largest space is used.
    refwfn : {tuple/list of int, CIWavefunction}
        State with respect to which the energy and the norm are computed.
        If a list/tuple of Slater determinants are given, then the reference state is the given
        wavefunction truncated by the provided Slater determinants.
        Default is ground state HF.
    eqn_weights : np.ndarray
        Weights of each equation.
        By default, all equations are given weight of 1 except for the normalization constraint,
        which is weighed by the number of equations.
    energy : ParamContainer
        Energy used in the Schrodinger equation.
        Used to store/cache the energy.
    energy_type : {'fixed', 'variable', 'compute'}
        Type of the energy used in the Schrodinger equation.
        If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
        If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
        If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect to
        the reference.

    Properties
    ----------
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.
    nproj : int
        Number of states onto which the Schrodinger equation is projected.
    num_eqns : int
        Number of equations in the objective.

    Methods
    -------
    __init__(self, param_selection=None, tmpfile='')
        Initialize the objective.
    assign_param_selection(self, param_selection=None)
        Select parameters that will be active in the objective.
    assign_params(self, params)
        Assign the parameters to the wavefunction and/or hamiltonian.
    save_params(self)
        Save all of the parameters in the `param_selection` to the temporary file.
    wrapped_get_overlap(self, sd, deriv=None)
        Wrap `get_overlap` to be derivatized with respect to the parameters of the objective.
    wrapped_integrate_wfn_sd(self, sd, deriv=None)
        Wrap `integrate_wfn_sd` to be derivatized wrt the parameters of the objective.
    wrapped_integrate_sd_sd(self, sd1, sd2, deriv=None)
        Wrap `integrate_sd_sd` to be derivatized wrt the parameters of the objective.
    get_energy_one_proj(self, refwfn, deriv=None)
        Return the energy of the Schrodinger equation with respect to a reference wavefunction.
    get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=None)
        Return the energy of the Schrodinger equation after projecting out both sides.
    assign_pspace(self, pspace=None)
        Assign the projection space.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_constraints(self, constraints=None)
        Assign the constraints on the objective.
    assign_eqn_weights(self, eqn_weights=None)
        Assign the weights of each equation.
    objective(self, params) : np.ndarray(self.num_eqns, )
        Return the values of the system of equations.
    jacobian(self, params) : np.ndarray(self.num_eqns, self.nparams.size)
        Return the Jacobian of the objective function.

    """

    def __init__(
        self,
        wfn,
        ham,
        tmpfile="",
        param_selection=None,
        pspace=None,
        refwfn=None,
        eqn_weights=None,
        energy_type="compute",
        energy=None,
        constraints=None,
    ):
        """Initialize the objective instance.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction.
        ham : BaseHamiltonian
            Hamiltonian that defines the system under study.
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        param_selection : {list, tuple, ParamMask, None}
            Selection of parameters that will be used to construct the objective.
            If list/tuple, then each entry is a 2-tuple of the parameter object and the numpy
            indexing array for the active parameters. See `ParamMask.__init__` for details.
        pspace : {tuple/list of int, tuple/list of CIWavefunction, None}
            States onto which the Schrodinger equation is projected.
            By default, the largest space is used.
        refwfn : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground state HF.
        eqn_weights : np.ndarray
            Weights of each equation.
            By default, all equations are given weight of 1 except for the normalization constraint,
            which is weighed by the number of equations.
        energy_type : {'fixed', 'variable', 'compute'}
            Type of the energy used in the Schrodinger equation.
            If 'fixed', the energy of the Schrodinger equation is fixed at the given value.
            If 'variable', the energy of the Schrodinger equation is optimized as a parameter.
            If 'compute', the energy of the Schrodinger equation is computed on-the-fly with respect
            to the reference.
            By default, the energy is computed on-the-fly.
        energy : {float, None}
            Energy of the Schrodinger equation.
            If not provided, energy is computed with respect to the reference.
            By default, energy is computed with respect to the reference.
            Note that this parameter is not used at all if `energy_type` is 'compute'.
        constraints : list/tuple of BaseObjective
            Constraints that will be imposed on the optimization process.
            By default, the normalization constraint used.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If save_file is not a string.
            If `energy` is not float, complex or None.
        ValueError
            If wavefunction and Hamiltonian do not have the same data type.
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.
            If `energy_type` is not one of 'fixed', 'variable', or 'compute'.

        """
        super().__init__(wfn, ham, tmpfile=tmpfile, param_selection=param_selection)
        self.assign_pspace(pspace)
        self.assign_refwfn(refwfn)

        if energy_type not in ["fixed", "variable", "compute"]:
            raise ValueError("`energy_type` must be one of 'fixed', 'variable', or 'compute'.")
        self.energy_type = energy_type

        if energy is None:
            energy = self.get_energy_one_proj(self.refwfn)
        elif not isinstance(energy, (float, complex)):
            raise TypeError("Energy must be given as a float or complex.")
        self.energy = ParamContainer(energy)
        if energy_type in ["fixed", "variable"]:
            self.param_selection.load_mask_container_params(self.energy, energy_type == "variable")
            self.param_selection.load_masks_objective_params()

        self.assign_constraints(constraints)
        self.assign_eqn_weights(eqn_weights)

    @property
    def nproj(self):
        """Return the number of projected states.

        Returns
        -------
        nproj : int
            Number of states onto which the Schrodinger equation is projected.

        """
        return len(self.pspace)

    @property
    def num_eqns(self):
        """Return the number of equations in the objective.

        Returns
        -------
        num_eqns : int
            Number of equations in the objective.

        """
        return self.nproj + sum(cons.num_eqns for cons in self.constraints)

    def assign_pspace(self, pspace=None):
        """Assign the projection space.

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
            pspace = sd_list.sd_list(
                self.wfn.nelec, self.wfn.nspatial, spin=self.wfn.spin, seniority=self.wfn.seniority
            )
        elif isinstance(pspace, (list, tuple)) and all(
            slater.is_sd_compatible(state) or isinstance(state, CIWavefunction) for state in pspace
        ):
            pspace = tuple(pspace)
        else:
            raise TypeError(
                "Projected space must be given as a list/tuple of Slater determinants. "
                "See `backend.slater` for compatible Slater determinant formats."
            )
        self.pspace = pspace

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction.

        Parameters
        ----------
        refwfn : {tuple/list of int, CIWavefunction, None}
            State with respect to which the energy and the norm are computed.
            If a list/tuple of Slater determinants are given, then the reference state is the given
            wavefunction truncated by the provided Slater determinants.
            Default is ground state HF.

        """
        if refwfn is None:
            self.refwfn = (slater.ground(self.wfn.nelec, self.wfn.nspin),)
        elif slater.is_sd_compatible(refwfn):
            self.refwfn = (refwfn,)
        elif isinstance(refwfn, (list, tuple)) and all(
            slater.is_sd_compatible(sd) for sd in refwfn
        ):
            self.refwfn = tuple(slater.internal_sd(sd) for sd in refwfn)
        elif isinstance(refwfn, CIWavefunction):
            self.refwfn = refwfn
        else:
            raise TypeError(
                "Reference state must be given as a Slater determinant, a list/tuple of"
                " Slater determinants, or a CIWavefunction. See `backend.slater` for "
                "compatible representations of the Slater determinants."
            )

    def assign_constraints(self, constraints=None):
        """Assign the constraints on the objective.

        Parameters
        ----------
        constraints : {list/tuple of BaseObjective, BaseObjective, None}
            Constraints that will be imposed on the optimization process.
            By default, the normalization constraint used.

        Raises
        ------
        TypeError
            If constraints are not given as a BaseObjective instance or list/tuple of BaseObjective
            instances.
            If a constraint is not a BaseObjective or its subclass.
        ValueError
            If a constraint does not have the same param_selection as the objective class.

        """
        if constraints is None:
            constraints = [
                NormConstraint(self.wfn, refwfn=self.refwfn, param_selection=self.param_selection)
            ]
        elif isinstance(constraints, BaseObjective):
            constraints = [constraints]
        elif not isinstance(constraints, (list, tuple)):
            raise TypeError(
                "Constraints must be given as a BaseObjective instance or list/tuple "
                "of BaseObjective instances."
            )

        for constraint in constraints:
            if not isinstance(constraint, BaseObjective):
                raise TypeError(
                    "Each constraint must be an instance of BaseObjective or its " "child."
                )
            if constraint.param_selection != self.param_selection:
                raise ValueError(
                    "The given constraint must have the same parameter selection (in "
                    "the form of ParamMask) as the objective."
                )
        self.constraints = constraints

    def assign_eqn_weights(self, eqn_weights=None):
        """Assign the weights of each equation.

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
        num_constraints = sum(cons.num_eqns for cons in self.constraints)
        if eqn_weights is None:
            eqn_weights = np.ones(self.nproj + num_constraints)
            eqn_weights[self.nproj :] *= self.nproj
        elif not isinstance(eqn_weights, np.ndarray):
            raise TypeError("Weights of the equations must be given as a numpy array.")
        elif eqn_weights.dtype != self.wfn.dtype:
            raise TypeError(
                "Weights of the equations must have the same dtype as the wavefunction "
                "and Hamiltonian."
            )
        elif eqn_weights.shape != (self.nproj + num_constraints,):
            raise ValueError(
                "Weights of the equations must be given as a one-dimensional array of "
                "shape, {0}.".format((self.nproj + num_constraints,))
            )
        self.eqn_weights = eqn_weights

    def objective(self, params):
        r"""Return the values of the system of equations.

        .. math::

            f_1(x) &= \left< \Phi_1 \middle| \hat{H} \middle| \Psi \right>
                      - E \left< \Phi_1 \middle| \Psi \right>\\
            &\vdots\\
            f_K(x) &= \left< \Phi_K \middle| \hat{H} \middle| \Psi \right>
                      - E \left< \Phi_K \middle| \Psi \right>\\
            f_{K+1}(x) &= \left< \Phi_{ref} \middle| \Psi \right> - 1\\

        where :math:`K` is the number of Slater determinant onto which the wavefunction is
        projected. The :math:`K+1`th equation is the normalization constraint. The norm is computed
        with respect to the reference state. The energy can be a constant, a parameter that gets
        optimized, or a function of the wavefunction and hamiltonian parameters.

        .. math::

            E = \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                     {\left< \Phi_{ref} \middle| \Psi \right>}

        In general, :math:`\Phi_i` can be some linear combination of Slater determinants,
        :math:`SD_j`.

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
        self.save_params()

        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_wfn_sd = np.vectorize(self.wrapped_integrate_wfn_sd)

        # reference values
        if self.energy_type in ["variable", "fixed"]:
            energy = self.energy.params
        elif self.energy_type == "compute":
            # define reference
            if isinstance(self.refwfn, CIWavefunction):
                ref_sds = self.refwfn.sd_vec
                ref_coeffs = self.refwfn.params
            else:
                ref_sds = self.refwfn
                ref_coeffs = get_overlap(ref_sds)

            norm = np.sum(ref_coeffs * get_overlap(ref_sds))
            energy = self.get_energy_one_proj(self.refwfn)
            energy = np.sum(ref_coeffs * integrate_wfn_sd(ref_sds)) / norm
            # can be replaced with
            energy = self.get_energy_one_proj(self.refwfn)
            self.energy.assign_params(energy)

        # objective
        obj = np.empty(self.num_eqns)
        # <SD|H|Psi> - E<SD|Psi> == 0
        obj[: self.nproj] = integrate_wfn_sd(self.pspace) - energy * get_overlap(self.pspace)
        # Add constraints
        if self.nproj < self.num_eqns:
            obj[self.nproj :] = np.hstack([cons.objective(params) for cons in self.constraints])
        # weigh equations
        obj *= self.eqn_weights

        return obj

    def jacobian(self, params):
        r"""Return the Jacobian of the objective function.

        If :math:`(f_1(\vec{x}), f_2(\vec{x}), \dots)` is the objective function, the Jacobian is

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
        self.save_params()

        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_wfn_sd = np.vectorize(self.wrapped_integrate_wfn_sd)

        # indices with respect to which objective is derivatized
        derivs = np.arange(params.size)
        # need to reshape (to a row of a matrix) to allow for summing over slater determinants
        derivs = derivs[np.newaxis, :]

        # define reference
        if isinstance(self.refwfn, CIWavefunction):
            ref_sds = np.array(self.refwfn.sd_vec)[:, np.newaxis]
            ref_coeffs = self.refwfn.params[:, np.newaxis]
            d_ref_coeffs = np.zeros((ref_sds.size, params.size))
            try:
                # pylint: disable=W0212
                objective_indices = self.param_selection._masks_objective_params[self.refwfn]
                container_indices = self.param_selection._masks_container_params[self.refwfn]
            except KeyError:
                pass
            else:
                d_ref_coeffs[container_indices, objective_indices] = 1.0
        else:
            ref_sds = np.array(self.refwfn)[:, np.newaxis]
            ref_coeffs = get_overlap(ref_sds)
            d_ref_coeffs = get_overlap(ref_sds, derivs)

        # overlaps of each Slater determinant in reference <SD_i | Psi>
        ref_sds_olps = get_overlap(ref_sds)
        d_ref_sds_olps = get_overlap(ref_sds, derivs)
        # NOTE: d_ref_olps and d_ref_ints are two dimensional matrices (axis 0 corresponds to the
        # reference Slater determinants and 1 to the index of the parameter with respect to which
        # the value is derivatized).

        # norm <ref | Psi>
        d_norm = np.sum(d_ref_coeffs * ref_sds_olps, axis=0)
        d_norm += np.sum(ref_coeffs * d_ref_sds_olps, axis=0)

        # energy
        if self.energy_type in ["variable", "fixed"]:
            energy = self.energy.params
            d_energy = np.array(
                [
                    self.param_selection.derivative_index(self.energy, i) is not None
                    for i in range(params.size)
                ],
                dtype=float,
            )
        elif self.energy_type == "compute":
            # norm
            norm = np.sum(ref_coeffs * ref_sds_olps)
            # integral <SD | H | Psi>
            ref_sds_ints = integrate_wfn_sd(ref_sds)
            d_ref_sds_ints = integrate_wfn_sd(ref_sds, derivs)
            # integral <ref | H | Psi>
            ref_int = np.sum(ref_coeffs * ref_sds_ints)
            d_ref_int = np.sum(d_ref_coeffs * ref_sds_ints, axis=0)
            d_ref_int += np.sum(ref_coeffs * d_ref_sds_ints, axis=0)

            energy = ref_int / norm
            d_energy = (d_ref_int - d_norm * energy) / norm
            self.energy.assign_params(energy)

        # reshape for broadcasting
        pspace = np.array(self.pspace)[:, np.newaxis]

        # jacobian
        jac = np.empty((self.num_eqns, params.size))
        jac[: self.nproj, :] = integrate_wfn_sd(pspace, derivs)
        jac[: self.nproj, :] -= energy * get_overlap(pspace, derivs)
        jac[: self.nproj, :] -= d_energy[np.newaxis, :] * get_overlap(pspace)
        # Add constraints
        if self.nproj < self.num_eqns:
            jac[self.nproj :] = np.vstack([cons.gradient(params) for cons in self.constraints])
        # weigh equations
        jac *= self.eqn_weights[:, np.newaxis]

        return jac
