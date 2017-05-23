"""APIG wavefunction."""
from __future__ import absolute_import, division, print_function
import numpy as np
from .base_geminal import BaseGeminal

__all__ = []


class APIG(BaseGeminal):
    """Antisymmetric Product of Interacting Geminals (APIG) Wavefunction.

    .. math::

        \big| \Psi_{\mathrm{APIG}} \big>
        &= \prod_{p=1}^P a^\dagger_p \big| \theta \big>\\
        &= \sum_{\{\mathbf{m}| m_i \in \{0,1\}, \sum_{p=1}^K m_p = P\}} |C(\mathbf{m})|^+
        \big| \mathbf{m} \big>

    where :math:`P` is the number of electron pairs and :math:`\mathbf{m}` is a seniority-zero
    Slater determinant.

    Attributes
    ----------
    nelec : int
        Number of electrons
    nspin : int
        Number of spin orbitals (alpha and beta)
    dtype : {np.float64, np.complex128}
        Data type of the wavefunction
    params : np.ndarray
        Parameters of the wavefunction
    dict_orbpair_ind : dict of 2-tuple of int to int
        Dictionary of orbital pair (i, j) where i and j are spin orbital indices and i < j
        to the column index of the geminal coefficient matrix
    dict_ind_orbpair : dict of int to 2-tuple of int
        Dictionary of column index of the geminal coefficient matrix to the orbital pair (i, j)
        where i and j are spin orbital indices and i < j
    cache : dict of sd to float
        Cache of the overlaps that are calculated for each Slater determinant encountered
    d_cache : dict of gmpy2.mpz to float
        Cache of the derivative of overlaps that are calculated for each Slater determinant and
        derivative index encountered

    Properties
    ----------
    npairs : int
        Number of electorn pairs
    nspatial : int
        Number of spatial orbitals
    ngem : int
        Number of geminals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters
    template_params : np.ndarray
        Template for the initial guess of geminal coefficient matrix
        Depends on the attributes given

    Methods
    -------
    __init__(self, nelec, one_int, two_int, dtype=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_ref_sds(self, ref_sds=None)
        Assigns the reference Slater determinants from which the initial guess, energy, and norm are
        calculated
        Default is the first Slater determinant of projection space
    assign_orbpairs(self, orbpairs=None)
        Assigns the orbital pairs that will be used to construct geminals
    compute_permanent(self, orbpairs, deriv_row_col=None)
        Compute the permanent that corresponds to the given orbital pairs
    get_overlap(self, sd, deriv_ind=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    generate_possible_orbpairs(self, occ_indices)
        Yields the possible orbital pairs that can construct the given Slater determinant.
    """
    @property
    def spin(self):
        """Spin of the APIG wavefunction

        Note
        ----
        Seniority is not zero if you change the pairing scheme
        """
        return 0.0

    @property
    def seniority(self):
        """Seniority of the APIG wavefunction

        Note
        ----
        Seniority is not zero if you change the pairing scheme
        """
        return 0

    def assign_orbpairs(self, orbpairs=None):
        """Set the orbital pairs that will be used to construct the geminals.

        Parameters
        ----------
        orbpairs : iterable of 2-tuple of ints
            Indices of the orbital pairs that will be used to construct each geminal
            Default is all spatial orbital (alpha beta) pair

        Raises
        ------
        TypeError
            If `orbpairs` is not an iterable
            If an orbital pair is not given as a list or a tuple
            If an orbital pair does not contain exactly two elements
            If an orbital index is not an integer
        ValueError
            If an orbital pair has the same integer
            If an orbital pair occurs more than once
            If any two orbital pair shares an orbital
            If any orbital is not included in any orbital pair

        Note
        ----
        Must have `nspin` defined for the default option
        """
        if orbpairs is None:
            orbpairs = ((i, i+self.nspatial) for i in range(self.nspatial))
        super().assign_orbpairs(orbpairs)

        all_orbs = [j for i in self.dict_orbpair_ind.keys() for j in i]
        if len(all_orbs) != len(set(all_orbs)):
            raise ValueError('At least two orbital pairs share an orbital')
        elif len(all_orbs) != self.nspin:
            raise ValueError('Not all of the orbitals are included in orbital pairs')

    def generate_possible_orbpairs(self, occ_indices):
        """Yield the possible orbital pairs that can construct the given Slater determinant.

        The APIG wavefunction only contains one pairing scheme for each Slater determinant (because
        the orbital pairs are disjoint of one another)

        Parameters
        ----------
        occ_indices : N-tuple of int
            Indices of the orbitals from which the Slater determinant is constructed

        Yields
        ------
        orbpairs : P-tuple of 2-tuple of ints
            Indices of the creation operators (grouped by orbital pairs) that construct the Slater
            determinant.

        Raises
        ------
        ValueError
            If the number of electrons in the Slater determinant does not match up with the number
            of electrons in the wavefunction
            If the Slater determinant cannot be constructed using the APIG pairing scheme
        """
        if len(occ_indices) != self.nelec:
            raise ValueError('The number of electrons in the Slater determinant does not match up '
                             'with the number of electrons in the wavefunction.')
        orbpairs = set()
        for i in occ_indices:
            if i < self.nspatial:
                orbpairs.add((i, i+self.nspatial))
            else:
                orbpairs.add((i-self.nspatial, i))

        if len(orbpairs) != self.npair:
            raise ValueError('This Slater determinant cannot be created using the pairing scheme of'
                             ' APIG wavefunction.')

        yield tuple(orbpairs)

    # TODO: refactor when APr2G is set up
    def to_apr2g(self, rmsd=0.01):
        """Convert APIG wavefunction to APr2G wavefunction.

        Using least squares, the APIG geminal coefficients are converted to the APr2G variant, i.e.
        find the coefficients :math:`\{\lambda_j\}`, :math:`\{\epsilon_i\}`, and :math:`\{\zeta_i\}`
        such that following equation is best satisfied
        ..math::
            C_{ij} &= \frac{\zeta_i}{\epsilon_i + \lambda_j}\\
            0 &= \zeta_i - C_{ij} \epsilon_i - C_{ij} \lambda_j\\

        The least square has the form of :math:`Ax=b`. Given that the :math:`b=0`
        and the unknowns are
        ..math::
            x = \begin{bmatrix}
            \lambda_1 \\ \vdots\\ \lambda_K\\
            \zeta_1 \\ \vdots\\ \zeta_P\\
            \epsilon_1 \\ \vdots\\ \epsilon_P\\
            \end{bmatrix},
        then A must be
        ..math::
            A = \begin{bmatrix}
            -C_{11} & 0 & \dots & 0 & -C_{11} & 0 & \dots & 0 &  & 1 & 0 & \dots & 0\\
            -C_{12} & 0 & \dots & 0 & 0 & -C_{12} & \dots & 0 &  & 0 & 1 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots\\
            -C_{1K} & 0 & \dots & 0 & 0 & 0 & \dots & -C_{1K} &  & 0 & 0 & \dots & 1\\
            0 & -C_{21} & \dots & 0 & -C_{21} & 0 & \dots & 0 &  & 1 & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots\\
            0 & -C_{2K} & \dots & 0 & 0 & 0 & \dots & -C_{2K} &  & 0 & 0 & \dots & 1\\
            0 & 0 & \dots & -C_{PK} & -C_{P1} & 0 & \dots & 0 &  & 1 & 0 & \dots & 0\\
            \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots & \vdots
            & \vdots & \vdots\\
            0 & 0 & \dots & -C_{PK} & 0 & 0 & \dots & -C_{PK} &  & 0 & 0 & \dots & 1\\
            \end{bmatrix}


        Parameters
        ----------
        rmsd : float
            Root mean square deviation allowed for the generated APr2G coefficient matrix (compared
            to the APIG coefficient matrix)

        Returns
        -------
        apr2g : APr2G instance
            APr2G wavefunction that best corresponds to the given APIG wavefunction

        Raises
        ------
        ValueError
            If generate APr2G coefficient matrix has a root mean square deviation with the APIG
            coefficient matrix that is greater than the threshold value

        Example
        -------
        Assuming we have a system with 2 electron pairs and 4 spatial orbitals,
        we have
        ..math::
            C = \begin{bmatrix}
            C_{11} & \dots & C_{1K}\\
            C_{21} & \dots & C_{2K}
            \end{bmatrix}
        ..math::
            A = \begin{bmatrix}
            C_{11} & 0 & -C_{11} & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
            C_{12} & 0 & 0 & -C_{12} & 0 & 0 & 0 & 1 & 0 & 0\\
            C_{13} & 0 & 0 & 0 & -C_{13} & 0 & 0 & 0 & 1 & 0\\
            C_{14} & 0 & 0 & 0 & 0 & -C_{14} & 0 & 0 & 0 & 1\\
            0 & C_{21} & -C_{21} & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
            0 & C_{22} & 0 & -C_{22} & 0 & 0 & 0 & 1 & 0 & 0\\
            0 & C_{23} & 0 & 0 & -C_{23} & 0 & 0 & 0 & 1 & 0\\
            0 & C_{24} & 0 & 0 & 0 & -C_{24} & 0 & 0 & 0 & 1\\
            \end{bmatrix}

        ..math::
            x = \begin{bmatrix}
            \lambda_ 1& \lambda_2
            \epsilon_1 \\ \epsilon_2\ \\ \epsilon_3 \\ \epsilon_4\\
            \zeta_1 \\ \zeta_2\\
            \end{bmatrix}
        """
        # import APr2G (because nasty cyclic import business)
        from .apr2g import APr2G

        # make coefficients
        apig_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # assign least squares matrix by reference
        matrix = np.zeros((apig_coeffs.size, self.npair + 2*self.nspatial), dtype=self.dtype)
        # set up submatrices that references a specific part of the matrix
        lambdas = matrix[:, :self.npair]
        epsilons = matrix[:, self.npair:self.npair + self.nspatial]
        zetas = matrix[:, self.npair + self.nspatial:self.npair + 2*self.nspatial]
        for i in range(self.npair):
            lambdas[i*self.nspatial:(i + 1)*self.nspatial, i] = -apig_coeffs[i, :]
            epsilons[i*self.nspatial:(i + 1)*self.nspatial, :] = np.diag(apig_coeffs[i, :])
            zetas[i*self.nspatial:(i + 1)*self.nspatial, :] = np.identity(self.nspatial)

        # Turn system of equations heterogeneous
        # Select indices that will be assigned values
        indices = np.zeros(self.npair + 2*self.nspatial, dtype=bool)
        vals = np.array([])

        # assign epsilons
        #  with orbital energy
        # indices[self.npair:self.npair + self.nspatial] = True
        # vals = np.diag(self.one_int[0])
        # indices[self.npair] = True
        # vals = np.hstack((vals, np.diag(self.one_int[0])[0]))

        # assign zetas
        # FIXME: assign weights based on orbital coupling
        # indices[self.npair + self.nspatial:2*self.npair + self.nspatial] = True
        # vals = np.hstack((vals, np.ones(self.npair)))
        indices[self.npair + self.nspatial] = True
        vals = np.hstack((vals, 1))

        # find ordinate
        ordinate = -matrix[:, indices].dot(vals)

        # Solve the least-squares system
        sol = np.zeros(indices.size)
        sol[indices] = vals
        sol[-indices] = np.linalg.lstsq(matrix[:, -indices], ordinate)[0].astype(self.dtype)

        # Check
        lambdas = sol[:self.npair][:, np.newaxis]
        epsilons = sol[self.npair:self.npair+self.nspatial]
        zetas = sol[self.npair+self.nspatial:]
        apr2g_coeffs = zetas / (lambdas - epsilons)
        deviation = (np.sum((apig_coeffs - apr2g_coeffs)**2)/apig_coeffs.size)**(0.5)
        if deviation > rmsd or np.isnan(deviation):
            raise ValueError('APr2G coefficient matrix has RMSD of {0} with the APIG coefficient'
                             ' matrix'.format(deviation))
        # Add energy
        params = np.hstack((sol, self.get_energy()))
        # make wavefunction
        return APr2G(self.nelec, self.one_int, self.two_int, dtype=self.dtype, nuc_nuc=self.nuc_nuc,
                     orbtype=self.orbtype, pspace=self.pspace, ref_sds=self.ref_sds,
                     params=params, ngem=self.ngem, orbpairs=self.dict_orbpair_ind.keys())

    # TODO: refactor when APG is set up
    def to_apg(self):
        """Convert APIG wavefunction to APG wavefunction.

        Returns
        -------
        apg : APG instance
            APG wavefunction that corresponds to the given APIG wavefunction
        """
        # import APG (because nasty cyclic import business)
        from .apg import APG

        # make apig coefficients
        apig_coeffs = self.params[:-1].reshape(self.template_coeffs.shape)
        # make apg coefficients
        apg = APG(self.nelec, self.one_int, self.two_int, dtype=self.dtype, nuc_nuc=self.nuc_nuc,
                  orbtype=self.orbtype, ref_sds=self.ref_sds, ngem=self.ngem)
        apg_coeffs = apg.params[:-1].reshape(apg.template_coeffs.shape)
        # assign apg coefficients
        for orbpair, ind in self.dict_orbpair_ind.items():
            apg_coeffs[:, apg.dict_orbpair_ind[orbpair]] = apig_coeffs[:, ind]
        apg.assign_params(np.hstack((apg_coeffs.flat, self.get_energy())))

        # make wavefunction
        return apg
