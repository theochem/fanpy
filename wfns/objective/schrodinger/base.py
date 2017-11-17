"""Base class for objectives related to solving the Schrodinger equation."""
import numpy as np
from wfns.param import ParamMask
from wfns.objective.base import BaseObjective
from wfns.wfn.base import BaseWavefunction
from wfns.wfn.ci.base import CIWavefunction
from wfns.ham.base import BaseHamiltonian
import wfns.backend.slater as slater


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
    def __init__(self, wfn, ham, tmpfile='', param_selection=None):
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
            raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                            'child).')
        if not isinstance(ham, BaseHamiltonian):
            raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its '
                            'child).')
        if wfn.dtype != ham.dtype:
            raise ValueError('Wavefunction and Hamiltonian do not have the same data type.')
        if wfn.nspin != ham.nspin:
            raise ValueError('Wavefunction and Hamiltonian do not have the same number of spin '
                             'orbitals')
        self.wfn = wfn
        self.ham = ham

        if param_selection is None:
            param_selection = ParamMask((self.wfn, None))

        super().__init__(param_selection, tmpfile=tmpfile)

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
        if deriv is None:
            return self.wfn.get_overlap(sd)

        # change derivative index
        deriv = self.param_selection.derivative_index(self.wfn, deriv)
        if deriv is None:
            return 0.0
        else:
            return self.wfn.get_overlap(sd, deriv)

    def wrapped_integrate_wfn_sd(self, sd, deriv=None):
        r"""Wrap `integrate_wfn_sd` to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd : {int, np.int64, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral :math:`\braket{\Phi | \hat{H} | \Psi}`.

        Notes
        -----
        Since `integrate_wfn_sd` depends on both the Hamiltonian and the wavefunction, it can be
        derivatized with respect to the paramters of the hamiltonian and of the wavefunction.

        """
        if deriv is None:
            return sum(self.ham.integrate_wfn_sd(self.wfn, sd))

        # change derivative index
        wfn_deriv = self.param_selection.derivative_index(self.wfn, deriv)
        ham_deriv = self.param_selection.derivative_index(self.ham, deriv)
        if wfn_deriv is not None:
            return sum(self.ham.integrate_wfn_sd(self.wfn, sd, wfn_deriv=wfn_deriv))
        # b/c the integral cannot be derivatized wrt both wfn and ham
        elif ham_deriv is not None:
            return sum(self.ham.integrate_wfn_sd(self.wfn, sd, ham_deriv=ham_deriv))
        else:
            return 0.0

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
            Value of the integral :math:`\braket{\Phi_i | \hat{H} | \Phi_j}`.

        """
        if deriv is None:
            return sum(self.ham.integrate_sd_sd(sd1, sd2))

        # change derivative index
        deriv = self.param_selection.derivative_index(self.ham, deriv)
        if deriv is None:
            return 0.0
        else:
            return sum(self.ham.integrate_sd_sd(sd1, sd2, deriv=deriv))

    def get_energy_one_proj(self, refwfn, deriv=None):
        r"""Return the energy of the Schrodinger equation with respect to a reference wavefunction.

        ..math::

            E \approx \frac{\braket{\Phi_{ref} | \hat{H} | \Psi}}{\braket{\Phi_{ref} | \Psi}}

        where :math:`\Phi_{ref}` is some reference wavefunction. Let

        ..math::

            \ket{\Phi_{ref}} = \sum_{\mathbf{m} \in S} g(\mathbf{m}) \ket{\mathbf{m}}

        Then,

        ..math:

            \braket{\Phi_{ref} | \hat{H} | \Psi}
            &= \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \bra{\mathbf{m}} \hat{H} \ket{\Psi}\\

        and

        ..math::

            \braket{\Phi_{ref} | \Psi} &=
            \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \bra{\mathbf{m}} \ket{\Psi}\\

        Ideally, we want to use the actual wavefunction as the reference, but, without further
        simplifications, :math:`\Psi` uses too many Slater determinants to be computationally
        tractible. Then, we can truncate the Slater determinants as a subset, :math:`S`, such that
        the most significant Slater determinants are included, while the energy can be tractibly
        computed. This is equivalent to inserting a projection operator on one side of the integral

        ..math:

            \bra{\Psi} \sum_{\mathbf{m} \in S} \ket{\mathbf{m}} \braket{\mathbf{m} | \hat{H} | \Psi}
            &= \sum_{\mathbf{m} \in S} f^*(\mathbf{m}) \braket{\mathbf{m} | \hat{H} | \Psi}\\

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
        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_wfn_sd = np.vectorize(self.wrapped_integrate_wfn_sd)

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
        elif slater.is_sd_compatible(refwfn) or (isinstance(refwfn, (list, tuple)) and
                                                 all(slater.is_sd_compatible(sd) for sd in refwfn)):
            if slater.is_sd_compatible(refwfn):
                refwfn = [refwfn]
            ref_sds = refwfn
            ref_coeffs = get_overlap(refwfn)
            if deriv is not None:
                d_ref_coeffs = get_overlap(refwfn, deriv)
        else:
            raise TypeError('Reference state must be given as a Slater determinant, a CI '
                            'Wavefunction, or a list/tuple of Slater determinants. See '
                            '`backend.slater` for compatible representations of the Slater '
                            'determinants.')

        # overlaps and integrals
        overlaps = get_overlap(ref_sds)
        integrals = integrate_wfn_sd(ref_sds)

        # norm
        norm = np.sum(ref_coeffs * overlaps)

        # energy
        energy = np.sum(ref_coeffs * integrals) / norm

        if deriv is None:
            return energy
        else:
            d_norm = np.sum(d_ref_coeffs * overlaps)
            d_norm += np.sum(ref_coeffs * get_overlap(ref_sds, deriv))
            d_energy = np.sum(d_ref_coeffs * integrals) / norm
            d_energy += np.sum(ref_coeffs * integrate_wfn_sd(ref_sds, deriv=deriv)) / norm
            d_energy -= d_norm * energy / norm
            return d_energy

    def get_energy_two_proj(self, pspace_l, pspace_r=None, pspace_norm=None, deriv=None):
        r"""Return the energy of the Schrodinger equation after projecting out both sides.

        ..math::

            E = \frac{\braket{\Psi | \hat{H} | \Psi}}{\braket{\Psi | \Psi}}

        Then, the numerator can be approximated by inserting projection operators:

        ..math:

            \braket{\Psi | \hat{H} | \Psi} &\approx \bra{\Psi}
            \sum_{\mathbf{m} \in S_l} \ket{\mathbf{m}} \bra{\mathbf{m}}
            \hat{H}
            \sum_{\mathbf{n} \in S_r} \ket{\mathbf{n}} \braket{\mathbf{n} | \Psi_\mathbf{n}}\\
            &\approx \sum_{\mathbf{m} \in S_l} \sum_{\mathbf{n} \in S_r} \braket{\Psi | \mathbf{m}}
            \braket{\mathbf{m} | \hat{H} | \mathbf{n}} \braket{\mathbf{n} | \Psi}\\

        Likewise, the denominator can be approximated by inserting a projection operator:

        ..math::

            \braket{\Psi | \Psi} &\approx \bra{\Psi}
            \sum_{\mathbf{m} \in S_{norm}} \ket{\mathbf{m}} \bra{\mathbf{m}}
            \ket{\Psi}\\
            &\approx \sum_{\mathbf{m} \in S_{norm}} \braket{\Psi | \mathbf{m}}^2

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
            if not (slater.is_sd_compatible(pspace) or
                    (isinstance(pspace, (list, tuple)) and
                     all(slater.is_sd_compatible(sd) for sd in pspace))):
                raise TypeError('Projection space must be given as a Slater determinant or a '
                                'list/tuple of Slater determinants. See `backend.slater` for '
                                'compatible representations of the Slater determinants.')

        if slater.is_sd_compatible(pspace_l):
            pspace_l = [pspace_l]
        if slater.is_sd_compatible(pspace_r):
            pspace_r = [pspace_r]
        if slater.is_sd_compatible(pspace_norm):
            pspace_norm = [pspace_norm]
        pspace_l = np.array(pspace_l)
        pspace_r = np.array(pspace_r)
        pspace_norm = np.array(pspace_norm)

        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_sd_sd = np.vectorize(self.wrapped_integrate_sd_sd)

        # reshape for broadcasting
        pspace_l = pspace_l[:, np.newaxis]
        pspace_r = pspace_r[np.newaxis, :]

        # overlaps and integrals
        overlaps_l = get_overlap(pspace_l)
        overlaps_r = get_overlap(pspace_r)
        ci_matrix = integrate_sd_sd(pspace_l, pspace_r)
        overlaps_norm = get_overlap(pspace_norm)

        # norm
        norm = np.sum(overlaps_norm**2)

        # energy
        if deriv is None:
            return np.sum(overlaps_l * ci_matrix * overlaps_r) / norm
        else:
            d_norm = 2 * np.sum(overlaps_norm * get_overlap(pspace_norm, deriv))
            d_energy = np.sum(get_overlap(pspace_l, deriv) * ci_matrix * overlaps_r) / norm
            d_energy += np.sum(overlaps_l
                               * integrate_sd_sd(pspace_l, pspace_r, deriv)
                               * overlaps_r) / norm
            d_energy += np.sum(overlaps_l * ci_matrix * get_overlap(pspace_r, deriv)) / norm
            d_energy -= d_norm * np.sum(overlaps_l * ci_matrix * overlaps_r) / norm**2
            return d_energy
