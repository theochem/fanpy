r"""Hamiltonian object that interacts with the wavefunction."""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from wfns.param import ParamContainer
from wfns.backend.integrals import OneElectronIntegrals, TwoElectronIntegrals
from wfns.backend import slater
from wfns.wrapper.docstring import docstring_class


@docstring_class(indent_level=1)
class ChemicalHamiltonian(ParamContainer):
    r"""Hamiltonian used for a typical chemical system.

    .. math::

        \hat{H} = \sum_{ij} h_{ij} a^\dagger_i a_j
        + \sum_{ijkl} g_{ijkl} a^\dagger_i a^\dagger_j a_k a_l

    where :math:`h_{ik}` is the one-electron integral and :math:`g_{ijkl}` is the two-electron
    integral in Physicists' notation.

    Attributes
    ----------
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used.
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.
    one_int : {1- or 2-tuple np.ndarray(K, K)}
        One electron integrals for restricted, unrestricted, or generalized orbitals.
        1-tuple for spatial (restricted) and generalized orbitals.
        2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components).
    two_int : {1- or 3-tuple np.ndarray(K, K)}
        Two electron integrals for restricted, unrestricted, or generalized orbitals.
        Uses the physicist's notation.
        1-tuple for spatial (restricted) and generalized orbitals.
        3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
        beta-beta-beta-beta components).

    """

    def __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        one_int : {np.ndarray(K, K), 1- or 2-tuple np.ndarray(K, K)}
            One-electron integrals.
            If orbitals are spatial or generalized, then integrals are given as np.ndarray or
            1-tuple of np.ndarray.
            If orbitals are unretricted, then integrals are 2-tuple of np.ndarray.
        two_int : {np.ndarray(K, K, K, K), 1- or 3-tuple np.ndarray(K, K, K, K)}
            If orbitals are spatial or generalized, then integrals are given as np.ndarray or
            1-tuple of np.ndarray.
            If orbitals are unretricted, then integrals are 3-tuple of np.ndarray.
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is `'restricted'`.
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy.
            Default is `0.0`.

        """
        self.assign_orbtype(orbtype)
        self.assign_energy_nuc_nuc(energy_nuc_nuc)
        self.assign_integrals(one_int, two_int)

    @property
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        if self.orbtype in ('restricted', 'unrestricted'):
            return 2*self.one_int.num_orbs
        elif self.orbtype == 'generalized':
            return self.one_int.num_orbs
        else:
            raise NotImplementedError('Unsupported orbital type.')

    @property
    def dtype(self):
        """Return the data type of the integrals.

        Returns
        -------
        dtype : {'restricted', 'unrestricted', 'generalized'}
            Number of spin orbitals.

        """
        return self.one_int.dtype

    # FIXME: getter/setter is not used b/c assign_integrals is a little complicated.
    def assign_orbtype(self, orbtype=None):
        """Assign the orbital type.

        Parameters
        ----------
        orbtype : {'restricted', 'unrestricted', 'generalized', None}
            Type of the orbitals used.
            Default is 'restricted'.

        Raises
        ------
        ValueError
            If orbtype is not one of 'restricted', 'unrestricted', or 'generalized'.

        """
        if orbtype is None:
            orbtype = 'restricted'
        if orbtype not in ['restricted', 'unrestricted', 'generalized']:
            raise TypeError("Orbital type must be one of 'restricted', 'unrestricted', "
                            "and 'generalized'.")
        self.orbtype = orbtype

    def assign_energy_nuc_nuc(self, energy_nuc_nuc=None):
        """Assigns the nuclear nuclear repulsion.

        Parameters
        ----------
        energy_nuc_nuc : {int, float, None}
            Nuclear-nuclear repulsion energy.
            Default is `0.0`.

        Raises
        ------
        TypeError
            If `energy_nuc_nuc` is not int or float or None.

        """
        if energy_nuc_nuc is None:
            energy_nuc_nuc = 0.0
        elif isinstance(energy_nuc_nuc, (int, float)):
            energy_nuc_nuc = float(energy_nuc_nuc)
        else:
            raise TypeError('Nuclear-nuclear repulsion must be given as a int, float, or None.')
        self.energy_nuc_nuc = energy_nuc_nuc

    def assign_integrals(self, one_int, two_int):
        """Assign the one- and two-electron integrals.

        Parameters
        ----------
        one_int : {np.ndarray(K, K), 1- or 2-tuple np.ndarray(K, K), OneElectronIntegrals}
            One electron integrals for restricted, unrestricted, or generalized orbitals.
            1-tuple for spatial (restricted) and generalized orbitals.
            2-tuple for unrestricted orbitals (alpha-alpha and beta-beta components).
        two_int : {np.ndarray(K, K), 1- or 3-tuple np.ndarray(K, K), TwoElectronIntegrals}
            Two electron integrals for restricted, unrestricted, or generalized orbitals.
            Uses physicist's notation.
            1-tuple for spatial (restricted) and generalized orbitals.
            3-tuple for unrestricted orbitals (alpha-alpha-alpha-alpha, alpha-beta-alpha-beta, and
            beta-beta-beta-beta components).

        Raises
        ------
        TypeError
            If integrals are not provided as a numpy array or a tuple of numpy arrays.
            If integral matrices do not have dtype of `float` or `complex`.
            If integral matrices do not have the same dtype.
            If the number of one-electron integral matrices is not 1 or 2.
            If one-electron integral matrices are not two dimensional.
            If one-electron integral matrices are not square.
            If one-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals.
            If the number of two-electron integral matrices is not 1 or 3.
            If two-electron integral matrices are not four dimensional.
            If two-electron integral matrices does not have same dimensionality in all axis.
            If two-electron integrals (for the unrestricted orbitals) has different number of alpha
            and beta orbitals.
            If one- and two-electron integrals do not have the same number of orbitals.
            If one- and two-electron integrals do not have the same data type.
            If one- and two-electron integrals do not correspond to the same orbital type (i.e.
            restricted, unrestricted, generalized).
            If orbital type of the integrals do not match with the given orbital type.
        NotImplementedError
            If generalized orbitals and odd number of spin orbitals.

        Notes
        -----
        Needs `orbype` defined.

        Depending on the orbital type, the form of the integrals can vary. If the `orbtype` is
        'restricted', then `one_int` can be a np.ndarray or a 1-tuple of np.ndarray and `two_int`
        can be a np.ndarray or a 1-tuple of np.ndarray. If the `orbtype` is 'unrestricted', then
        `one_int` can be a 2-tuple of np.ndarray and `two_int` can be a 3-tuple of np.ndarray. If
        the `orbtype` is 'generalized', then `one_int` can be a np.ndarray or a 1-tuple of
        np.ndarray and `two_int` can be a np.ndarray or a 1-tuple of np.ndarray.

        """
        one_int = OneElectronIntegrals(one_int)
        two_int = TwoElectronIntegrals(two_int)
        if one_int.num_orbs != two_int.num_orbs:
            raise TypeError('One- and two-electron integrals do not have the same number of '
                            'orbitals.')
        elif one_int.dtype != two_int.dtype:
            raise TypeError('One- and two-electron integrals do not have the same data type.')
        elif one_int.possible_orbtypes != two_int.possible_orbtypes:
            raise TypeError('One- and two-electron integrals do not correspond to the same orbital '
                            'type (i.e. restricted, unrestricted, generalized).')
        elif self.orbtype not in one_int.possible_orbtypes:
            raise TypeError('Orbital type of the integrals do not match with the given orbital '
                            'type, {0}'.format(self.orbtype))
        elif self.orbtype == 'generalized' and one_int.num_orbs % 2 == 1:
            raise NotImplementedError('Odd number of "spin" orbitals will cause problems when '
                                      ' constructing Slater determinants.')

        self.one_int = one_int
        self.two_int = two_int

    def assign_params(self, params):
        """Assigns parameters of the Hamiltonian.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    # FIXME: need to speed up
    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

        .. math::

            \braket{\Phi | \hat{H} | \Psi}
            &= \sum_{\mathbf{m} \in S_\Phi} f(\mathbf{m}) \braket{\Phi | \hat{H} | \mathbf{m}}

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
        determinants for which :math:`\braket{\Phi | \hat{H} | \mathbf{m}}` is not zero, which are
        the :math:`\Phi` and its first and second order excitations for a chemical Hamiltonian.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction against which the Hamiltonian is integrated.
            Needs to have the following in `__dict__`: `get_overlap`.
        sd : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : {int, None}
            Index of the parameter against which the expectation value is derivatized.
            Default is no derivatization.

        Returns
        -------
        one_electron : float
            One-electron energy.
        coulomb : float
            Coulomb energy.
        exchange : float
            Exchange energy.

        """
        occ_indices = slater.occ_indices(sd)
        vir_indices = slater.vir_indices(sd, self.nspin)

        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        sd_energy = self.integrate_sd_sd(sd, sd)
        one_electron += sd_energy[0]
        coulomb += sd_energy[1]
        exchange += sd_energy[2]
        for counter_i, i in enumerate(occ_indices):
            for counter_a, a in enumerate(vir_indices):
                sd_energy = self.integrate_sd_sd(sd, slater.excite(sd, i, a))
                one_electron += sd_energy[0]
                coulomb += sd_energy[1]
                exchange += sd_energy[2]
                for j in occ_indices[counter_i+1:]:
                    for b in enumerate(vir_indices[counter_a+1:]):
                        sd_energy = self.integrate_sd_sd(sd, slater.excite(sd, i, j, b, a))
                        one_electron += sd_energy[0]
                        coulomb += sd_energy[1]
                        exchange += sd_energy[2]

        return one_electron, coulomb, exchange

    # FIXME: need to speed up
    def integrate_sd_sd(self, sd1, sd2, sign=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{ij} = \braket{\Phi_i | \hat{H} | \Phi_j}

        where :math:`\hat{H}` is the Hamiltonian operator, and :math:`\Phi_1` and :math:`\Phi_2` are
        Slater determinants.

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sign : {1, -1, None}
            Sign change resulting from cancelling out the orbitals shared between the two Slater
            determinants.
            Computes the sign if none is provided.
            Make sure that the provided sign is correct. It will not be checked to see if its
            correct.

        Returns
        -------
        one_electron : float
            One-electron energy.
        coulomb : float
            Coulomb energy.
        exchange : float
            Exchange energy.

        Raises
        ------
        ValueError
            If `sign` is not `1`, `-1` or `None`.

        """
        one_electron = 0.0
        coulomb = 0.0
        exchange = 0.0

        shared_indices = slater.shared_orbs(sd1, sd2)
        diff_sd1, diff_sd2 = slater.diff_orbs(sd1, sd2)
        # if two Slater determinants do not have the same number of electrons
        if len(diff_sd1) != len(diff_sd2):
            return 0.0, 0.0, 0.0
        diff_order = len(diff_sd1)

        # Assume the creators are ordered from smallest to largest (left to right) and
        # annihilators are ordered from largest to smallest.
        # Move all of the creators and annihilators that are shared between the two Slater
        # determinants towards the middle starting from the smallest index.
        # e.g. Given left Slater determinant a_8 a_6 a_3 a_2 a_1 and
        #      right Slater determinant a^\dagger_1 a^\dagger_2 a^dagger_5 a^\dagger_7 a^\dagger_8,
        #      annihilators and creators of orbitals 1, 2, and 8 must move towards one another for
        #      mutual destruction
        #    0    a_8 a_6 a_3 a_2 a_1    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    1    a_8 a_6 a_3 a_1 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    2    a_6 a_8 a_3 a_1 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    3    a_6 a_3 a_8 a_1 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    4    a_6 a_3 a_1 a_8 a_2    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    5    a_6 a_3 a_1 a_2 a_8    a^\dagger_1 a^\dagger_2 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    6    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_1 a^\dagger_5 a^\dagger_7 a^\dagger_8
        #    7    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_1 a^\dagger_5 a^\dagger_8 a^\dagger_7
        #    8    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_1 a^\dagger_8 a^\dagger_5 a^\dagger_7
        #    9    a_6 a_3 a_1 a_2 a_8    a^\dagger_2 a^\dagger_8 a^\dagger_1 a^\dagger_5 a^\dagger_7
        #   10    a_6 a_3 a_1 a_2 a_8    a^\dagger_8 a^\dagger_2 a^\dagger_1 a^\dagger_5 a^\dagger_7
        #   10    a_6 a_3                                                    a^\dagger_5 a^\dagger_7
        # where first column is the number of transpositions, second column is the ordering of the
        # annhilators (left SD) and third column is the ordering of the creators (right SD)

        # NOTE: after moving all of the different creators toward the middle, the unshared creators
        # will be ordered from smallest to largest and the shared creators will be ordered from
        # largest to smallest
        if sign is None:
            num_transpositions_1 = sum(slater.find_num_trans_swap(sd1, i, 0) for i in diff_sd1)
            num_transpositions_2 = sum(slater.find_num_trans_swap(sd2, i, 0) for i in diff_sd2)
            sign = (-1)**(num_transpositions_1 + num_transpositions_2)
        elif sign not in [1, -1]:
            raise ValueError('The sign associated with the integral must be either `1` or `-1`.')

        # two sd's are the same
        if diff_order == 0:
            for count, i in enumerate(shared_indices):
                one_electron += sign * self.one_int.get_value(i, i, self.orbtype)
                for j in shared_indices[count + 1:]:
                    coulomb += sign * self.two_int.get_value(i, j, i, j, self.orbtype)
                    exchange -= sign * self.two_int.get_value(i, j, j, i, self.orbtype)

        # two sd's are different by single excitation
        elif diff_order == 1:
            i, = diff_sd1
            a, = diff_sd2
            one_electron += sign * self.one_int.get_value(i, a, self.orbtype)
            for j in shared_indices:
                coulomb += sign * self.two_int.get_value(i, j, a, j, self.orbtype)
                exchange -= sign * self.two_int.get_value(i, j, j, a, self.orbtype)

        # two sd's are different by double excitation
        elif diff_order == 2:
            i, j = diff_sd1
            a, b = diff_sd2
            coulomb += sign * self.two_int.get_value(i, j, a, b, self.orbtype)
            exchange -= sign * self.two_int.get_value(i, j, b, a, self.orbtype)

        return one_electron, coulomb, exchange

    def orb_rotate_jacobi(self, jacobi_indices, theta):
        """Rotate orbitals using Jacobi matrix.

        Parameters
        ----------
        jacobi_indices : tuple/list of ints
            2-tuple/list of indices of the orbitals that will be rotated
        theta : float
            Angle with which the orbitals are rotated
        """
        self.one_int.rotate_jacobi(jacobi_indices, theta)
        self.two_int.rotate_jacobi(jacobi_indices, theta)

    def orb_rotate_matrix(self, matrix):
        """Rotate orbitals using a transformation matrix.

        Parameters
        ----------
        matrix : np.ndarray
            Transformation matrix.

        """
        if isinstance(matrix, np.ndarray):
            matrix = [matrix]
        if len(matrix) == 1 and self.orbtype == 'unrestricted':
            matrix *= 2

        self.one_int.rotate_matrix(matrix)
        self.two_int.rotate_matrix(matrix)
