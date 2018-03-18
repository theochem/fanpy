"""Hamiltonian that will be used to make the Schrodinger equation."""
import abc
from wfns.param import ParamContainer


class BaseHamiltonian(ParamContainer):
    """Hamiltonian for a Schrodinger equation.

    Attributes
    ----------
    energy_nuc_nuc : float
        Nuclear-nuclear repulsion energy.

    Methods
    -------
    __init__(self, one_int, two_int, orbtype=None, energy_nuc_nuc=None)
        Initialize the Hamiltonian.
    assign_energy_nuc_nuc(self, energy_nuc_nuc=None)
        Assigns the nuclear nuclear repulsion.

    Abstract Properties
    -------------------
    dtype : {np.float64, np.complex128}
        Data type of the Hamiltonian.
    nspin : int
        Number of spin orbitals.

    Abstract Methods
    ----------------
    assign_integrals(self, one_int, two_int)
        Assign the one- and two-electron integrals.
    integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None)
        Integrate the Hamiltonian with against a wavefunction and Slater determinant.
    integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None)
        Integrate the Hamiltonian with against two Slater determinants.

    """

    def __init__(self, energy_nuc_nuc=None):
        """Initialize the Hamiltonian.

        Parameters
        ----------
        energy_nuc_nuc : {float, None}
            Nuclear nuclear repulsion energy.
            Default is `0.0`.

        """
        self.assign_energy_nuc_nuc(energy_nuc_nuc)

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

    @abc.abstractproperty
    def dtype(self):
        """Return the data type of the integrals.

        Returns
        -------
        dtype : {'restricted', 'unrestricted', 'generalized'}
            Number of spin orbitals.

        """
        pass

    @abc.abstractproperty
    def nspin(self):
        """Return the number of spin orbitals.

        Returns
        -------
        nspin : int
            Number of spin orbitals.

        """
        pass

    @abc.abstractmethod
    def assign_integrals(self, one_int, two_int):
        """Assign the one- and two-electron integrals."""
        pass

    @abc.abstractmethod
    def integrate_wfn_sd(self, wfn, sd, wfn_deriv=None, ham_deriv=None):
        r"""Integrate the Hamiltonian with against a wavefunction and Slater determinant.

        .. math::

            \left< \Phi \middle| \hat{H} \middle| \Psi \right>
            = \sum_{\mathbf{m} \in S_\Phi} f(\mathbf{m})
              \left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>

        where :math:`\Psi` is the wavefunction, :math:`\hat{H}` is the Hamiltonian operator, and
        :math:`\Phi` is the Slater determinant. The :math:`S_{\Phi}` is the set of Slater
        determinants for which :math:`\left< \Phi \middle| \hat{H} \middle| \mathbf{m} \right>` is
        not zero, which are the :math:`\Phi` and its first and second order excitations for a
        chemical Hamiltonian.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction against which the Hamiltonian is integrated.
            Needs to have the following in `__dict__`: `get_overlap`.
        sd : int
            Slater Determinant against which the Hamiltonian is integrated.
        wfn_deriv : {int, None}
            Index of the wavefunction parameter against which the integral is derivatized.
            Default is no derivatization.
        ham_deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
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
        pass

    @abc.abstractmethod
    def integrate_sd_sd(self, sd1, sd2, sign=None, deriv=None):
        r"""Integrate the Hamiltonian with against two Slater determinants.

        .. math::

            H_{ij} = \left< \Phi_i \middle| \hat{H} \middle| \Phi_j \right>

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
        deriv : {int, None}
            Index of the Hamiltonian parameter against which the integral is derivatized.
            Default is no derivatization.

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
        pass
