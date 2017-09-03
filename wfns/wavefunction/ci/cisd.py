"""CI Singles and Doubles Wavefunction."""
from __future__ import absolute_import, division, print_function
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.backend.sd_list import sd_list

__all__ = []


class CISD(CIWavefunction):
    r"""Configuration Interaction Singles and Doubles Wavefunction.

    CI with HF Ground state and all single and double excitations

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
    _spin : float
        Total spin of each Slater determinant
        :math:`\frac{1}{2}(N_\alpha - N_\beta)`
        Default is no spin (all spins possible)
    _seniority : int
        Number of unpaired electrons in each Slater determinant
    sd_vec : tuple of int
        List of Slater determinants used to construct the CI wavefunction
    dict_sd_index : dictionary of int to int
        Dictionary from Slater determinant to its index in sd_vec

    Properties
    ----------
    nspatial : int
        Number of spatial orbitals
    spin : float, None
        Spin of the wavefunction
        :math:`\frac{1}{2}(N_\alpha - N_\beta)` (Note that spin can be negative)
        None means that all spins are allowed
    seniority : int, None
        Seniority (number of unpaired electrons) of the wavefunction
        None means that all seniority is allowed
    template_params : np.ndarray
        Template of the wavefunction parameters
        Depends on the attributes given
    nparams : int
        Number of parameters
    params_shape : 2-tuple of int
        Shape of the parameters

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, params=None, sd_vec=None, spin=None, seniority=None)
        Initializes wavefunction
    assign_nelec(self, nelec)
        Assigns the number of electrons
    assign_nspin(self, nspin)
        Assigns the number of spin orbitals
    assign_params(self, params)
        Assigns the parameters of the wavefunction
    assign_dtype(self, dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_spin(self, spin=None)
        Assigns the spin of the wavefunction
    assign_seniority(self, seniority=None)
        Assigns the seniority of the wavefunction
    assign_sd_vec(self, sd_vec=None)
        Assigns the tuple of Slater determinants used in the CI wavefunction
    get_overlap(self, sd, deriv=None)
        Gets the overlap from cache and compute if not in cache
        Default is no derivatization
    """
    def assign_sd_vec(self, sd_vec=None):
        """Set the list of Slater determinants for the CISD wavefunction.

        Ignores user input and uses the Slater determinants for the FCI wavefunction (within the
        given spin)

        Parameters
        ----------
        sd_vec : iterable of int
            List of Slater determinants (in the form of integers that describe the occupation as a
            bitstring)

        Raises
        ------
        ValueError
            If the sd_vec is not `None` (default value)

        Notes
        -----
        Needs to have `nelec`, `nspin`, `spin`, `seniority`
        """
        if sd_vec is None:
            super().assign_sd_vec(sd_list(self.nelec, self.nspatial, num_limit=None,
                                          exc_orders=[1, 2], spin=self.spin,
                                          seniority=self.seniority))
        else:
            raise ValueError('Only the default list of Slater determinants is allowed. '
                                      'i.e. sd_vec is `None`. '
                                      'If you would like to customize your CI wavefunction, use '
                                      'CIWavefunction instead.')
