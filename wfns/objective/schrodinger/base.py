"""Base class for objectives related to solving the Schrodinger equation."""
import os
import numpy as np
import wfns.backend.slater as slater
from wfns.ham.base import BaseHamiltonian
from wfns.objective.base import BaseObjective
from wfns.param import ParamMask
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.ci.base import CIWavefunction

from wfns.objective.schrodinger.cext import get_energy_one_proj, get_energy_one_proj_deriv


class BaseSchrodinger(BaseObjective):
    """Base class for objectives related to solving the Schrodinger equation.

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

    Properties
    ----------
    params : {np.ndarray(K, )}
        Parameters of the objective at the current state.

    Abstract Properties
    -------------------
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

    Abstract Methods
    ----------------
    objective(self, params) : float
        Return the value of the objective for the given parameters.

    """

    # pylint: disable=W0223
    def __init__(self, wfn, ham, tmpfile="", param_selection=None):
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

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of BaseHamiltonian.
            If save_file is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same data type.
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

        """
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError(
                "Given wavefunction is not an instance of BaseWavefunction (or its " "child)."
            )
        if not isinstance(ham, BaseHamiltonian):
            raise TypeError(
                "Given Hamiltonian is not an instance of BaseWavefunction (or its " "child)."
            )
        if wfn.nspin != ham.nspin:
            raise ValueError(
                "Wavefunction and Hamiltonian do not have the same number of spin " "orbitals"
            )
        self.wfn = wfn
        self.ham = ham

        if param_selection is None:
            param_selection = ParamMask((self.wfn, None))

        super().__init__(param_selection, tmpfile=tmpfile)

    # FIXME: there are problems when wfn is a composite wavefunction (wfn must distinguish between
    #        the different )
    def wrapped_get_overlap(self, sd, deriv=None):
        """Wrap `get_overlap` to be derivatized with respect to the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        """
        # pylint: disable=C0103
        if deriv is None:
            return self.wfn.get_overlap(sd)

        # change derivative index
        deriv = self.param_selection.derivative_index(self.wfn, deriv)
        if deriv is None:
            return 0.0
        return self.wfn.get_overlap(sd, deriv)

    # FIXME: there are problems when wfn is a composite wavefunction (wfn must distinguish between
    #        the different deriv's) and when ham is a composite hamiltonian (ham must distinguish
    #        between different derivs)
    def wrapped_integrate_wfn_sd(self, sd, deriv=None):
        r"""Wrap `integrate_wfn_sd` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, np.ndarray, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral :math:`\left< \Phi \middle| \hat{H} \middle| \Psi \right>`.

        Notes
        -----
        Since `integrate_wfn_sd` depends on both the Hamiltonian and the wavefunction, it can be
        derivatized with respect to the paramters of the hamiltonian and of the wavefunction.

        """
        # pylint: disable=C0103
        if deriv is None:
            return np.sum(self.ham.integrate_sd_wfn(sd, self.wfn))

        # change derivative index
        if isinstance(deriv, np.ndarray):
            wfn_deriv = [self.param_selection.derivative_index(self.wfn, i) for i in deriv]
            wfn_mask = np.array([i is not None for i in wfn_deriv])
            wfn_deriv = np.array([i for i in wfn_deriv if i is not None])

            ham_deriv = [self.param_selection.derivative_index(self.ham, i) for i in deriv]
            ham_mask = np.array([i is not None for i in ham_deriv])
            ham_deriv = np.array([i for i in ham_deriv if i is not None])
            ham_deriv = ham_deriv.astype(int)

            results = np.zeros(deriv.size)
            if wfn_deriv is not None:
                results[wfn_mask] = np.array(
                    [sum(self.ham.integrate_sd_wfn(sd, self.wfn, wfn_deriv=i)) for i in wfn_deriv]
                )
            if ham_deriv is not None:
                results[ham_mask] = np.sum(
                    self.ham.integrate_sd_wfn_deriv(sd, self.wfn, ham_deriv), axis=0
                )
            return results
        else:
            wfn_deriv = self.param_selection.derivative_index(self.wfn, deriv)
            ham_deriv = self.param_selection.derivative_index(self.ham, deriv)
            if wfn_deriv is not None:
                return sum(self.ham.integrate_sd_wfn(sd, self.wfn, wfn_deriv=wfn_deriv))
            # b/c the integral cannot be derivatized wrt both wfn and ham
            if ham_deriv is not None:
                return np.sum(
                    self.ham.integrate_sd_wfn_deriv(sd, self.wfn, ham_derivs=np.array([ham_deriv]))
                )
        return 0.0

    # FIXME: there are problems when ham is a composite hamiltonian (ham must distinguish between
    #        different derivs)
    def wrapped_integrate_sd_sd(self, sd1, sd2, deriv=None):
        r"""Wrap `integrate_sd_sd` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd1 : int
            Slater determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater determinant against which the Hamiltonian is integrated.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral :math:`\left< \Phi_i \middle| \hat{H} \middle| \Phi_j \right>`.

        """
        if deriv is None:
            return sum(self.ham.integrate_sd_sd(sd1, sd2))

        # change derivative index
        deriv = self.param_selection.derivative_index(self.ham, deriv)
        if deriv is None:
            return 0.0
        return sum(self.ham.integrate_sd_sd(sd1, sd2, deriv=deriv))

    def get_energy_one_proj(self, refwfn, deriv=None):
        r"""Return the energy of the Schrodinger equation with respect to a reference wavefunction.

        .. math::

            E \approx \frac{\left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>}
                           {\left< \Phi_{ref} \middle| \Psi \right>}

        where :math:`\Phi_{ref}` is some reference wavefunction. Let

        .. math::

            \left| \Phi_{ref} \right> = \sum_{\mathbf{m} \in S}
                                        g(\mathbf{m}) \left| \mathbf{m} \right>

        Then,

        .. math::

            \left< \Phi_{ref} \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S}
              g^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

        and

        .. math::

            \left< \Phi_{ref} \middle| \Psi \right> =
            \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \left< \mathbf{m} \middle| \Psi \right>

        Ideally, we want to use the actual wavefunction as the reference, but, without further
        simplifications, :math:`\Psi` uses too many Slater determinants to be computationally
        tractible. Then, we can truncate the Slater determinants as a subset, :math:`S`, such that
        the most significant Slater determinants are included, while the energy can be tractibly
        computed. This is equivalent to inserting a projection operator on one side of the integral

        .. math::

            \left< \Psi \right| \sum_{\mathbf{m} \in S}
            \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S}
              f^*(\mathbf{m}) \left< \mathbf{m} \middle| \hat{H} \middle| \Psi \right>

        Parameters
        ----------
        refwfn : {CIWavefunction, list/tuple of int}
            Reference wavefunction used to calculate the energy.
            If list/tuple of Slater determinants are given, then the reference wavefunction will be
            the truncated form (according to the given Slater determinants) of the provided
            wavefunction.
        deriv : {int, None}
            Index of the selected parameters with respect to which the energy is derivatized.

        Returns
        -------
        energy : float
            Energy of the wavefunction with the given Hamiltonian.

        Raises
        ------
        TypeError
            If `refwfn` is not a CIWavefunction, int, or list/tuple of int.

        """
        # define reference
        if isinstance(refwfn, CIWavefunction):
            # ref_sds = refwfn.sd_vec
            # ref_coeffs = refwfn.params
            # if deriv is not None:
            #     ref_deriv = self.param_selection.derivative_index(refwfn, deriv)
            #     if ref_deriv is None:
            #         d_ref_coeffs = 0.0
            #     else:
            #         d_ref_coeffs = np.zeros(refwfn.nparams, dtype=float)
            #         d_ref_coeffs[ref_deriv] = 1
            raise ValueError("CI wavefunction as a reference wavefunction is not supported.")
        elif isinstance(refwfn, int):
            refwfn = [refwfn]

        refwfn = np.array(refwfn)
        # if deriv is None:
        #     return get_energy_one_proj(self.wfn, self.ham, refwfn)

        if isinstance(deriv, np.ndarray):
            wfn_deriv = [self.param_selection.derivative_index(self.wfn, i) for i in deriv]
            wfn_mask = np.array([i is not None for i in wfn_deriv])
            wfn_deriv = np.array([i for i in wfn_deriv if i is not None])

            ham_deriv = [self.param_selection.derivative_index(self.ham, i) for i in deriv]
            ham_mask = np.array([i is not None for i in ham_deriv])
            ham_deriv = np.array([i for i in ham_deriv if i is not None])
            ham_deriv = ham_deriv.astype(int)

            all_derivs = get_energy_one_proj_deriv(self.wfn, self.ham, refwfn)[1]
            results = np.zeros(deriv.size)
            if wfn_deriv is not None:
                results[wfn_mask] = all_derivs[wfn_deriv]
            if ham_deriv is not None:
                results[ham_mask] = all_derivs[ham_deriv + self.wfn.nparams]
            return results

        # old code
        get_overlap = self.wrapped_get_overlap
        integrate_wfn_sd = self.wrapped_integrate_wfn_sd

        # define reference
        if isinstance(refwfn, CIWavefunction):
            ref_sds = refwfn.sd_vec
            ref_coeffs = refwfn.params
            if deriv is not None:
                ref_deriv = self.param_selection.derivative_index(refwfn, deriv)
                if ref_deriv is None:
                    d_ref_coeffs = 0.0
                else:
                    d_ref_coeffs = np.zeros(refwfn.nparams, dtype=float)
                    d_ref_coeffs[ref_deriv] = 1
        elif slater.is_sd_compatible(refwfn) or (
            isinstance(refwfn, (list, tuple, np.ndarray)) and all(slater.is_sd_compatible(sd) for sd in refwfn)
        ):
            if slater.is_sd_compatible(refwfn):
                refwfn = [refwfn]
            ref_sds = refwfn
            ref_coeffs = np.array([get_overlap(i) for i in refwfn])
            if deriv is not None:
                d_ref_coeffs = np.array([get_overlap(i, deriv) for i in refwfn])
        else:
            raise TypeError(
                "Reference state must be given as a Slater determinant, a CI "
                "Wavefunction, or a list/tuple of Slater determinants. See "
                "`backend.slater` for compatible representations of the Slater "
                "determinants."
            )

        # overlaps and integrals
        overlaps = np.array([get_overlap(i) for i in ref_sds])
        integrals = np.array([integrate_wfn_sd(i) for i in ref_sds])

        # norm
        norm = np.sum(ref_coeffs * overlaps)

        # energy
        energy = np.sum(ref_coeffs * integrals) / norm

        if deriv is None:
            return energy

        d_norm = np.sum(d_ref_coeffs * overlaps)
        d_norm += np.sum(ref_coeffs * np.array([get_overlap(i, deriv) for i in ref_sds]))
        d_energy = np.sum(d_ref_coeffs * integrals) / norm
        d_energy += (
            np.sum(ref_coeffs * np.array([integrate_wfn_sd(i, deriv) for i in ref_sds])) / norm
        )
        d_energy -= d_norm * energy / norm
        return d_energy

    def get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=None):
        r"""Return the energy of the Schrodinger equation after projecting out both sides.

        .. math::

            E = \frac{\left< \Psi \middle| \hat{H} \middle| \Psi \right>}
                     {\left< \Psi \middle| \Psi \right>}

        Then, the numerator can be approximated by inserting projection operators:

        .. math::

            \left< \Psi \middle| \hat{H} \middle| \Psi \right> &\approx \left< \Psi \right|
            \sum_{\mathbf{m} \in S_l} \left| \mathbf{m} \middle> \middle< \mathbf{m} \right|
            \hat{H}
            \sum_{\mathbf{n} \in S_r}
            \left| \mathbf{n} \middle> \middle< \mathbf{n} \middle| \Psi_\mathbf{n} \right>\\
            &\approx \sum_{\mathbf{m} \in S_l} \sum_{\mathbf{n} \in S_r}
            \left< \Psi \middle| \mathbf{m} \right>
            \left< \mathbf{m} \middle| \hat{H} \middle| \mathbf{n} \right>
            \left< \mathbf{n} \middle| \Psi \right>\\

        Likewise, the denominator can be approximated by inserting a projection operator:

        .. math::

            \left< \Psi \middle| \Psi \right> &\approx \left< \Psi \right|
            \sum_{\mathbf{m} \in S_{norm}} \left| \mathbf{m} \middle> \middle< \mathbf{m} \middle|
            \middle| \Psi \right>\\
            &\approx \sum_{\mathbf{m} \in S_{norm}} \left< \Psi \middle| \mathbf{m} \right>^2

        Parameters
        ----------
        pspace_l : list/tuple of int
            Projection space used to truncate the numerator of the energy evaluation from the left.
        pspace_r : {list/tuple of int, None}
            Projection space used to truncate the numerator of the energy evaluation from the right.
            By default, the same space as `l_pspace` is used.
        pspace_norm : {list/tuple of int, None}
            Projection space used to truncate the denominator of the energy evaluation
            By default, the same space as `l_pspace` is used.
        deriv : {int, None}
            Index with respect to which the energy is derivatized.

        Returns
        -------
        energy : float
            Energy of the wavefunction with the given Hamiltonian.

        Raises
        ------
        TypeError
            If projection space is not a list/tuple of int.

        """
        if pspace_r is None:
            pspace_r = pspace_l
        if pspace_norm is None:
            pspace_norm = pspace_l

        for pspace in [pspace_l, pspace_r, pspace_norm]:
            if not (
                slater.is_sd_compatible(pspace)
                or (
                    isinstance(pspace, (list, tuple))
                    and all(slater.is_sd_compatible(sd) for sd in pspace)
                )
            ):
                raise TypeError(
                    "Projection space must be given as a Slater determinant or a "
                    "list/tuple of Slater determinants. See `backend.slater` for "
                    "compatible representations of the Slater determinants."
                )

        if slater.is_sd_compatible(pspace_l):
            pspace_l = [pspace_l]
        if slater.is_sd_compatible(pspace_r):
            pspace_r = [pspace_r]
        if slater.is_sd_compatible(pspace_norm):
            pspace_norm = [pspace_norm]
        pspace_l = np.array(pspace_l)
        pspace_r = np.array(pspace_r)
        pspace_norm = np.array(pspace_norm)

        get_overlap = self.wrapped_get_overlap
        integrate_sd_sd = self.wrapped_integrate_sd_sd

        # overlaps and integrals
        overlaps_l = np.array([[get_overlap(i)] for i in pspace_l])
        overlaps_r = np.array([[get_overlap(i) for i in pspace_r]])
        ci_matrix = np.array([[integrate_sd_sd(i, j) for j in pspace_r] for i in pspace_l])
        overlaps_norm = np.array([get_overlap(i) for i in pspace_norm])

        # norm
        norm = np.sum(overlaps_norm ** 2)

        # energy
        if deriv is None:
            return np.sum(overlaps_l * ci_matrix * overlaps_r) / norm

        d_norm = 2 * np.sum(overlaps_norm * np.array([get_overlap(i, deriv) for i in pspace_norm]))
        d_energy = (
            np.sum(np.array([[get_overlap(i, deriv)] for i in pspace_l]) * ci_matrix * overlaps_r)
            / norm
        )
        d_energy += (
            np.sum(
                overlaps_l
                * np.array([[integrate_sd_sd(i, j, deriv) for j in pspace_r] for i in pspace_l])
                * overlaps_r
            )
            / norm
        )
        d_energy += (
            np.sum(overlaps_l * ci_matrix * np.array([[get_overlap(i, deriv) for i in pspace_r]]))
            / norm
        )
        d_energy -= d_norm * np.sum(overlaps_l * ci_matrix * overlaps_r) / norm ** 2
        return d_energy

    def save_params(self):
        """Save all of the parameters in the `param_selection` to the temporary file.

        All of the parameters are saved, even if it was frozen in the objective.

        """
        if self.tmpfile != "":
            np.save(self.tmpfile, self.param_selection.all_params)
            np.save('{}_um{}'.format(*os.path.splitext(self.tmpfile)), self.ham._prev_unitary)
