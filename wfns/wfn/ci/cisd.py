"""CI Singles and Doubles Wavefunction."""
from wfns.backend.sd_list import sd_list
from wfns.wfn.ci.base import CIWavefunction


class CISD(CIWavefunction):
    r"""Configuration Interaction Singles and Doubles Wavefunction.

    CI with HF Ground state and all single and double excitations

    Attributes
    ----------
    nelec : int
        Number of electrons.
    nspin : int
        Number of spin orbitals (alpha and beta).
    params : np.ndarray
        Parameters of the wavefunction.
    memory : float
        Memory available for the wavefunction.
    _spin : float
        Total spin of each Slater determinant.
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`.
        Default is no spin (all spins possible).
    _seniority : int
        Number of unpaired electrons in each Slater determinant.
    sd_vec : tuple of int
        List of Slater determinants used to construct the CI wavefunction.
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sd_vec.

    Properties
    ----------
    nparams : int
        Number of parameters.
    nspatial : int
        Number of spatial orbitals
    param_shape : tuple of int
        Shape of the parameters.
    spin : int
        Spin of the wavefunction
    seniority : int
        Seniority of the wavefunction
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, params=None, sd_vec=None, spin=None, seniority=None):
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None):
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    assign_spin(self, spin=None)
        Assign the spin of the wavefunction.
    assign_seniority(self, seniority=None)
        Assign the seniority of the wavefunction.
    assign_sd_vec(self, sd_vec=None)
        Assign the list of Slater determinants from which the CI wavefunction is constructed.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the CI wavefunction with a Slater determinant.

    """

    # pylint:disable=W0223
    def assign_sd_vec(self, sd_vec=None):
        """Assign the list of Slater determinants in the CISD wavefunction.

        Ignores user input and uses the Slater determinants for the FCI wavefunction (within the
        given spin).

        Parameters
        ----------
        sd_vec : iterable of int
            List of Slater determinants (in the form of integers that describe the occupation as a
            bitstring)

        Raises
        ------
        ValueError
            If the sd_vec is not `None` (default value).

        Notes
        -----
        Needs to have `nelec`, `nspin`, `spin`, `seniority`.

        """
        if sd_vec is None:
            super().assign_sd_vec(
                sd_list(
                    self.nelec,
                    self.nspatial,
                    num_limit=None,
                    exc_orders=[1, 2],
                    spin=self.spin,
                    seniority=self.seniority,
                )
            )
        else:
            raise ValueError(
                "Only the default list of Slater determinants is allowed. i.e. sd_vec "
                "is `None`. If you would like to customize your CI wavefunction, use "
                "CIWavefunction instead."
            )
