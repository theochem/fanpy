r"""Wavefunction with orbitals rotated by Jacobi matrix."""
import numpy as np
from wfns.backend import slater
from wfns.wfn.composite.base_one import BaseCompositeOneWavefunction
from wfns.wfn.composite.nonorth import NonorthWavefunction


# FIXME: needs refactoring
class JacobiWavefunction(BaseCompositeOneWavefunction):
    r"""Wavefunction with jacobi rotated orbitals expressed with respect to orthonormal orbitals.

    A wavefunction constructed from nonorthonormal orbitals can be written as

    .. math::
        \left| \Psi \right>
        = \sum_{\mathbf{n}} \sum_{\mathbf{m}}
        f(\mathbf{n}) |C(\mathbf{n}, \mathbf{m})|^- \left| \mathbf{m} \right>

    where :math:`\left| \mathbf{m} \right>` and :math:`\left| \mathbf{n} \right>` are Slater
    determinants constructed from orthonormal and nonorthonormal orbitals. The
    :math:`C(\mathbf{n}, \mathbf{m})` is a submatrix of the transformation matrix :math:`C` where
    rows are selected according to :math:`\left| \mathbf{n} \right>` and columns to
    :math:`\left| \mathbf{m} \right>`.

    If the orbitals are transformed with a Jacobi rotation, then many of the determinants are
    simplified.

    .. math::
        \left< \mathbf{m} \middle| J^\dagger_{pq} \middle| \Psi \right>
        &= f(\mathbf{m}) \mbox{ if $p \not\in \mathbf{m}$ and $q \not\in \mathbf{m}$}\\
        &= f(\mathbf{m}) \mbox{ if $p \in \mathbf{m}$ and $q \in \mathbf{m}$}\\
        &= f(\mathbf{m}) (\cos\theta + \sin\theta)
        \mbox{ if $p \in \mathbf{m}$ and $q \not\in \mathbf{m}$}\\
        &= f(\mathbf{m}) (\cos\theta - \sin\theta)
        \mbox{ if $p \not\in \mathbf{m}$ and $q \in \mathbf{m}$}\\

    where :math:`J^\dagger_{pq}` is the orbital rotation operator that mixes the orbitals that
    corresponds to orbitals :math:`p` and :math:`q`.

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction.
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    wfn : BaseWavefunction
        Wavefunction whose orbitals are rotated.
    jacobi_indices : 2-tuple of ints
        Orbitals that are rotated.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction.
    seniority : int
        Seniority of the wavefunction.
    template_params : np.ndarray
        Default parameters of the wavefunction.
    jacobi_rotation : tuple of np.ndarray
        Rotation matrix that corresponds to the given parameter.
        If the orbitals are restricted, then the rotation matrix for the spatial orbitals is
        returned.
        If the orbitals are unrestricted, then the rotation matrices for alpha and beta orbitals are
        returned.
        If the orbitals are generalized, then the rotation matrices for spin orbitals are returned.

    Methods
    -------
    __init__(self, nelec, nspin, wfn, dtype=None, memory=None, params=None, orbtype=None,
             jacobi_indices=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    assign_orbtype(self, orbtype=None)
        Assign the orbital type of the orbital rotation.
    assign_jacobi_indices(self, jacobi_indices)
        Assign the indices of the orbitals that will be rotated.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """

    def __init__(
        self,
        nelec,
        nspin,
        wfn,
        dtype=None,
        memory=None,
        params=None,
        orbtype=None,
        jacobi_indices=None,
    ):
        """Initialize the wavefunction.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that will be built using nonorthnormal orbitals.
        orbtype : str
            Type of orbital used by the wavefunction.
            One of 'restricted', 'unrestricted', and 'generalized'.
        jacobi_indices : 2-tuple/list of ints
            Orbitals that will be rotated.

        """
        super().__init__(nelec, nspin, wfn, dtype=dtype, memory=memory, params=params)
        self.assign_orbtype(orbtype)
        self.assign_jacobi_indices(jacobi_indices)

    # FIXME: copied from NonorthWavefunction
    @property
    def spin(self):
        """Return the spin of the wavefunction.

        If the orbitals are restricted or unrestricted, the spin should be same as the original.
        Otherwise, the orbitals may mix regardless of the spin, the spin of the wavefunction is hard
        to determine.

        Returns
        -------
        spin : float
            Spin of the (composite) wavefunction if the orbitals are restricted or unrestricted.
            None if all spins are allowed.

        """
        return NonorthWavefunction.spin.__get__(self)  # pylint: disable=E1120

    # FIXME: copied from NonorthWavefunction
    @property
    def seniority(self):
        """Return the seniority of the wavefunction.

        If the orbitals are restricted or unrestricted, the seniority should be same as the
        original. Otherwise, the orbitals may mix regardless of the seniority, the seniority of the
        wavefunction is hard to determine.

        Returns
        -------
        seniority : int
            Seniority of the (composite) wavefunction if the orbitals are restricted or
            unrestricted.
            None if all seniority are allowed.

        """
        return NonorthWavefunction.seniority.__get__(self)  # pylint: disable=E1120

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        """
        return ()

    @property
    def template_params(self):
        """Return the template of the parameters of the given wavefunction.

        Returns
        -------
        template_params : np.ndarray(1)
            Angle with which the orbitals are rotated.

        """
        return np.array(0.0)

    @property
    def jacobi_rotation(self):
        """Return the rotation matrix that corresponds to the given parameter.

        Returns
        -------
        jacobi_rotation : tuple of np.ndarray
            Rotation matrix that corresponds to the given parameter.
            If the orbitals are restricted, then the rotation matrix for the spatial orbitals is
            returned.
            If the orbitals are unrestricted, then the rotation matrices for alpha and beta orbitals
            are returned.
            If the orbitals are generalized, then the rotation matrices for spin orbitals are
            returned.

        """
        # pylint: disable=C0103
        if self.orbtype in ["restricted", "unrestricted"]:
            jacobi_rotation = np.identity(self.nspatial, dtype=self.dtype)
        else:
            jacobi_rotation = np.identity(self.nspin, dtype=self.dtype)

        p, q = self.jacobi_indices
        # FIXME: use function that converts spin to spatial
        if self.orbtype == "unrestricted" and p >= self.nspatial <= q:
            p -= self.nspatial
            q -= self.nspatial

        jacobi_rotation[p, p] = np.cos(self.params)
        jacobi_rotation[p, q] = np.sin(self.params)
        jacobi_rotation[q, p] = -np.sin(self.params)
        jacobi_rotation[q, q] = np.cos(self.params)

        # pylint: disable=R1705
        if self.orbtype in ["restricted", "generalized"]:
            return (jacobi_rotation,)
        elif all(i < self.nspatial for i in self.jacobi_indices):
            return (jacobi_rotation, np.identity(self.nspatial, dtype=self.dtype))
        else:
            return (np.identity(self.nspatial, dtype=self.dtype), jacobi_rotation)

    def assign_params(self, params=None):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
        add_noise : bool
            Flag to add noise to the given parameters.

        Raises
        ------
        TypeError
            If `params` is not a numpy array.
            If `params` does not have data type of `float`, `complex`, `np.float64` and
            `np.complex128`.
            If `params` has complex data type and wavefunction has float data type.
        ValueError
            If `params` does not have the same shape as the template_params.

        Notes
        -----
        Depends on dtype, template_params, and nparams.

        """
        if isinstance(params, (int, float)):
            params = np.array(params, dtype=float)
        super().assign_params(params=params)

    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type of the orbital rotation.

        Parameters
        ----------
        orbtype : str
            Type of the orbital that are rotated.
            One of 'restricted', 'unrestricted', and 'generalized'.

        """
        if orbtype is None:
            orbtype = "restricted"

        if orbtype not in ["restricted", "unrestricted", "generalized"]:
            raise ValueError(
                "Orbital type must be one of 'restricted', 'unrestricted', and " "'generalized'."
            )
        self.orbtype = orbtype

    def assign_jacobi_indices(self, jacobi_indices):
        """Assign the indices of the orbitals that will be rotated.

        Parameters
        ----------
        jacobi_indices : tuple/list of ints
            2-tuple/list of indices of the orbitals that will be rotated.

        Raises
        ------
        TypeError
            If `jacobi_indices` is not a tuple or list.
            If `jacobi_indices` does not have two elements.
            If `jacobi_indices` must only contain integers.
        ValueError
            If the indices are negative.
            If the two indices are the same.
            If orbitals are generalized and indices are greater than `nspin`.
            If orbitals are restricted and indices are greater than `nspatial`.
            If orbitals are unrestricted and indices mix alpha and beta orbitals.

        Notes
        -----
        `orbtype` is used in this method.

        """
        if not isinstance(jacobi_indices, (tuple, list)):
            raise TypeError("Indices `jacobi_indices` must be a tuple or list.")
        if len(jacobi_indices) != 2:
            raise TypeError("Indices `jacobi_indices` must be have two elements.")
        if not all(isinstance(i, int) for i in jacobi_indices):
            raise TypeError("Indices `jacobi_indices` must only contain integers.")

        if not all(i >= 0 for i in jacobi_indices):
            raise ValueError("Indices cannot be negative.")
        if jacobi_indices[0] == jacobi_indices[1]:
            raise ValueError("Two indices are the same.")
        if self.orbtype == "generalized" and not all(i < self.nspin for i in jacobi_indices):
            raise ValueError(
                "If the orbitals are generalized, their indices must be less than the "
                "number of spin orbitals."
            )
        if self.orbtype == "restricted" and not all(i < self.nspatial for i in jacobi_indices):
            raise ValueError(
                "If the orbitals are restricted, their indices must be less than the "
                "number of spatial orbitals."
            )
        if self.orbtype == "unrestricted" and (jacobi_indices[0] < self.nspatial) != (
            jacobi_indices[1] < self.nspatial
        ):
            raise ValueError(
                "If the orbitals are unrestricted, the alpha and beta orbitals cannot " "be mixed."
            )

        if jacobi_indices[0] > jacobi_indices[1]:
            jacobi_indices = jacobi_indices[::-1]
        self.jacobi_indices = tuple(jacobi_indices)

    # FIXME: too many return statements, too many branches
    def _olp(self, sd):
        """Calculate the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.

        Returns
        -------
        olp : float
            Overlap of the wavefunction with the Slater determinant.

        """
        # pylint: disable=C0103,R1705,R0911,R0912
        p, q = self.jacobi_indices
        ns = self.nspatial
        sin = np.sin(self.params)
        cos = np.cos(self.params)
        if self.orbtype in ["generalized", "unrestricted"]:
            if slater.occ(sd, p) == slater.occ(sd, q):
                return self.wfn.get_overlap(sd)
            elif slater.occ(sd, p) and not slater.occ(sd, q):
                # get signature for the excited Slater determinant
                sign = slater.sign_swap(sd, p, q)
                return cos * self.wfn.get_overlap(sd) + sign * sin * self.wfn.get_overlap(
                    slater.excite(sd, p, q)
                )
            # elif not slater.occ(sd, p) and slater.occ(sd, q):
            else:
                sign = slater.sign_swap(sd, q, p)
                return cos * self.wfn.get_overlap(sd) - sign * sin * self.wfn.get_overlap(
                    slater.excite(sd, q, p)
                )

        alpha_sd, beta_sd = slater.split_spin(sd, ns)
        # alpha block contains both p and q or neither p and q
        # beta block contains both p and q or neither p and q
        if slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and slater.occ(
            beta_sd, p
        ) == slater.occ(beta_sd, q):
            return self.wfn.get_overlap(sd)
        # beta block contains p and not q
        elif (
            slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p)
            and not slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(beta_sd, p, q)
            return cos * self.wfn.get_overlap(sd) + sin * sign * self.wfn.get_overlap(
                slater.excite(sd, p + ns, q + ns)
            )
        # beta block contains q and not p
        elif (
            slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q)
            and not slater.occ(beta_sd, p)
            and slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(beta_sd, q, p)
            return cos * self.wfn.get_overlap(sd) - sign * sin * self.wfn.get_overlap(
                slater.excite(sd, q + ns, p + ns)
            )
        # alpha block contains p and not q
        # beta block contains both p and q or neither p and q
        elif (
            slater.occ(alpha_sd, p)
            and not slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p) == slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(alpha_sd, p, q)
            return cos * self.wfn.get_overlap(sd) + sign * sin * self.wfn.get_overlap(
                slater.excite(sd, p, q)
            )
        # beta block contains p and not q
        elif (
            slater.occ(alpha_sd, p)
            and not slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p)
            and not slater.occ(beta_sd, q)
        ):
            alpha_sign = slater.sign_swap(alpha_sd, p, q)
            beta_sign = slater.sign_swap(beta_sd, p, q)
            return (
                cos ** 2 * self.wfn.get_overlap(sd)
                + (
                    alpha_sign * self.wfn.get_overlap(slater.excite(sd, p, q))
                    + beta_sign * self.wfn.get_overlap(slater.excite(sd, p + ns, q + ns))
                )
                * cos
                * sin
                + alpha_sign
                * beta_sign
                * sin ** 2
                * self.wfn.get_overlap(slater.excite(sd, p, p + ns, q, q + ns))
            )
        # beta block contains q and not p
        elif (
            slater.occ(alpha_sd, p)
            and not slater.occ(alpha_sd, q)
            and not slater.occ(beta_sd, p)
            and slater.occ(beta_sd, q)
        ):
            alpha_sign = slater.sign_swap(alpha_sd, p, q)
            beta_sign = slater.sign_swap(beta_sd, q, p)
            return (
                cos ** 2 * self.wfn.get_overlap(sd)
                + (
                    alpha_sign * self.wfn.get_overlap(slater.excite(sd, p, q))
                    - beta_sign * self.wfn.get_overlap(slater.excite(sd, q + ns, p + ns))
                )
                * cos
                * sin
                - alpha_sign
                * beta_sign
                * sin ** 2
                * self.wfn.get_overlap(slater.excite(sd, p, q + ns, q, p + ns))
            )
        # alpha block contains q and not p
        # beta block contains both p and q or neither p and q
        elif (
            not slater.occ(alpha_sd, p)
            and slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p) == slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(alpha_sd, q, p)
            return cos * self.wfn.get_overlap(sd) - sign * sin * self.wfn.get_overlap(
                slater.excite(sd, q, p)
            )
        # beta block contains p and not q
        elif (
            not slater.occ(alpha_sd, p)
            and slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p)
            and not slater.occ(beta_sd, q)
        ):
            alpha_sign = slater.sign_swap(alpha_sd, q, p)
            beta_sign = slater.sign_swap(beta_sd, p, q)
            return (
                cos ** 2 * self.wfn.get_overlap(sd)
                + (
                    -alpha_sign * self.wfn.get_overlap(slater.excite(sd, q, p))
                    + beta_sign * self.wfn.get_overlap(slater.excite(sd, p + ns, q + ns))
                )
                * cos
                * sin
                - alpha_sign
                * beta_sign
                * sin ** 2
                * self.wfn.get_overlap(slater.excite(sd, q, p + ns, p, q + ns))
            )
        # beta block contains q and not p
        # elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
        #         not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
        else:
            alpha_sign = slater.sign_swap(alpha_sd, q, p)
            beta_sign = slater.sign_swap(beta_sd, q, p)
            return (
                cos ** 2 * self.wfn.get_overlap(sd)
                + (
                    -alpha_sign * self.wfn.get_overlap(slater.excite(sd, q, p))
                    - beta_sign * self.wfn.get_overlap(slater.excite(sd, q + ns, p + ns))
                )
                * cos
                * sin
                + alpha_sign
                * beta_sign
                * sin ** 2
                * self.wfn.get_overlap(slater.excite(sd, q, q + ns, p, p + ns))
            )

    # FIXME: too many return statements, too many branches
    def _olp_deriv(self, sd, deriv):
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd : int
            Occupation vector of a Slater determinant given as a bitstring.
        deriv : int
            Index of the parameter with respect to which the overlap is derivatized.

        Returns
        -------
        olp_deriv : float
            Derivative of the overlap of the wavefunction with the Slater determinant.

        """
        # pylint: disable=C0103,R1705,R0911,R0912
        p, q = self.jacobi_indices
        ns = self.nspatial
        sin = np.sin(self.params)
        cos = np.cos(self.params)
        if self.orbtype == "restricted":
            alpha_sd, beta_sd = slater.split_spin(sd, ns)

        if self.orbtype in ["generalized", "unrestricted"]:
            if slater.occ(sd, p) == slater.occ(sd, q):
                return 0.0
            elif slater.occ(sd, p) and not slater.occ(sd, q):
                # get signature for the excited Slater determinant
                sign = slater.sign_swap(sd, p, q)
                return -sin * self.wfn.get_overlap(sd) + sign * cos * self.wfn.get_overlap(
                    slater.excite(sd, p, q)
                )
            # elif not slater.occ(sd, p) and slater.occ(sd, q):
            else:
                sign = slater.sign_swap(sd, q, p)
                return -sin * self.wfn.get_overlap(sd) - sign * cos * self.wfn.get_overlap(
                    slater.excite(sd, q, p)
                )

        alpha_sd, beta_sd = slater.split_spin(sd, ns)
        # alpha block contains both p and q or neither p and q
        # beta block contains both p and q or neither p and q
        if slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q) and slater.occ(
            beta_sd, p
        ) == slater.occ(beta_sd, q):
            return 0.0
        # beta block contains p and not q
        elif (
            slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p)
            and not slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(beta_sd, p, q)
            return -sin * self.wfn.get_overlap(sd) + cos * sign * self.wfn.get_overlap(
                slater.excite(sd, p + ns, q + ns)
            )
        # beta block contains q and not p
        elif (
            slater.occ(alpha_sd, p) == slater.occ(alpha_sd, q)
            and not slater.occ(beta_sd, p)
            and slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(beta_sd, q, p)
            return -sin * self.wfn.get_overlap(sd) - sign * cos * self.wfn.get_overlap(
                slater.excite(sd, q + ns, p + ns)
            )
        # alpha block contains p and not q
        # beta block contains both p and q or neither p and q
        elif (
            slater.occ(alpha_sd, p)
            and not slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p) == slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(alpha_sd, p, q)
            return -sin * self.wfn.get_overlap(sd) + sign * cos * self.wfn.get_overlap(
                slater.excite(sd, p, q)
            )
        # beta block contains p and not q
        elif (
            slater.occ(alpha_sd, p)
            and not slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p)
            and not slater.occ(beta_sd, q)
        ):
            alpha_sign = slater.sign_swap(alpha_sd, p, q)
            beta_sign = slater.sign_swap(beta_sd, p, q)
            return (
                -2 * cos * sin * self.wfn.get_overlap(sd)
                + (
                    alpha_sign * self.wfn.get_overlap(slater.excite(sd, p, q))
                    + beta_sign * self.wfn.get_overlap(slater.excite(sd, p + ns, q + ns))
                )
                * (cos * cos - sin * sin)
                + alpha_sign
                * beta_sign
                * 2
                * sin
                * cos
                * self.wfn.get_overlap(slater.excite(sd, p, p + ns, q, q + ns))
            )
        # beta block contains q and not p
        elif (
            slater.occ(alpha_sd, p)
            and not slater.occ(alpha_sd, q)
            and not slater.occ(beta_sd, p)
            and slater.occ(beta_sd, q)
        ):
            alpha_sign = slater.sign_swap(alpha_sd, p, q)
            beta_sign = slater.sign_swap(beta_sd, q, p)
            return (
                -2 * cos * sin * self.wfn.get_overlap(sd)
                + (
                    alpha_sign * self.wfn.get_overlap(slater.excite(sd, p, q))
                    - beta_sign * self.wfn.get_overlap(slater.excite(sd, q + ns, p + ns))
                )
                * (cos * cos - sin * sin)
                - alpha_sign
                * beta_sign
                * 2
                * sin
                * cos
                * self.wfn.get_overlap(slater.excite(sd, p, q + ns, q, p + ns))
            )
        # alpha block contains q and not p
        # beta block contains both p and q or neither p and q
        elif (
            not slater.occ(alpha_sd, p)
            and slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p) == slater.occ(beta_sd, q)
        ):
            sign = slater.sign_swap(alpha_sd, q, p)
            return -sin * self.wfn.get_overlap(sd) - sign * cos * self.wfn.get_overlap(
                slater.excite(sd, q, p)
            )
        # beta block contains p and not q
        elif (
            not slater.occ(alpha_sd, p)
            and slater.occ(alpha_sd, q)
            and slater.occ(beta_sd, p)
            and not slater.occ(beta_sd, q)
        ):
            alpha_sign = slater.sign_swap(alpha_sd, q, p)
            beta_sign = slater.sign_swap(beta_sd, p, q)
            return (
                -2 * cos * sin * self.wfn.get_overlap(sd)
                + (
                    -alpha_sign * self.wfn.get_overlap(slater.excite(sd, q, p))
                    + beta_sign * self.wfn.get_overlap(slater.excite(sd, p + ns, q + ns))
                )
                * (cos * cos - sin * sin)
                - alpha_sign
                * beta_sign
                * 2
                * sin
                * cos
                * self.wfn.get_overlap(slater.excite(sd, q, p + ns, p, q + ns))
            )
        # beta block contains q and not p
        # elif (not slater.occ(alpha_sd, p) and slater.occ(alpha_sd, q) and
        #       not slater.occ(beta_sd, p) and slater.occ(beta_sd, q)):
        else:
            alpha_sign = slater.sign_swap(alpha_sd, q, p)
            beta_sign = slater.sign_swap(beta_sd, q, p)
            return (
                -2 * cos * sin * self.wfn.get_overlap(sd)
                + (
                    -alpha_sign * self.wfn.get_overlap(slater.excite(sd, q, p))
                    - beta_sign * self.wfn.get_overlap(slater.excite(sd, q + ns, p + ns))
                )
                * (cos * cos - sin * sin)
                + alpha_sign
                * beta_sign
                * 2
                * sin
                * cos
                * self.wfn.get_overlap(slater.excite(sd, q, q + ns, p, p + ns))
            )

    # FIXME: derivative wrt wavefunction parameters (not jacobi rotation)?
    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        """
        if deriv is None:
            return self._cache_fns["overlap"](sd)
        # if derivatization
        if not (isinstance(deriv, int) and deriv >= 0):
            raise ValueError("Index for derivatization must be a non-negative integer.")
        return self._cache_fns["overlap derivative"](sd, deriv)
