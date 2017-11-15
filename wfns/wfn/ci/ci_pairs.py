"""CI Pairs Wavefunction."""
import numpy as np
from wfns.wfn.ci.doci import DOCI
from wfns.wfn.geminal.ap1rog import AP1roG
from wfns.backend.sd_list import sd_list
from wfns.backend import slater


class CIPairs(DOCI):
    r"""CI Pairs Wavefunction.

    DOCI wavefunction with only first order pairwise excitations.

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
    template_params : np.ndarray
        Default parameters of the CI wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, dtype=None, memory=None, params=None, sd_vec=None, spin=None,
             seniority=None):
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

    def assign_sd_vec(self, sd_vec=None):
        """Assign the list of Slater determinants in the CI Pairs wavefunction.

        Ignores user input and uses the Slater determinants for the CI Pairs wavefunction (within
        the given spin).

        Parameters
        ----------
        sd_vec : iterable of int
            List of Slater determinants.

        Raises
        ------
        ValueError
            If the sd_vec is not `None` (default value).

        """
        if sd_vec is None:
            super().assign_sd_vec(sd_list(self.nelec, self.nspatial, num_limit=None, exc_orders=[2],
                                          spin=self.spin, seniority=self.seniority))
        else:
            raise ValueError('Only the default list of Slater determinants is allowed. i.e. sd_vec '
                             'is `None`. If you would like to customize your CI wavefunction, use '
                             'CIWavefunction instead.')

    def to_ap1rog(self):
        """Return the AP1roG wavefunction that corresponds to the CIPairs wavefunction.

        Returns
        -------
        ap1rog : wfns.wavefunction.geminals.ap1rog.AP1roG
            AP1roG wavefunction.

        """
        # select Slater determinant with largest contributor as the reference Slater determinant
        ref_sd_ind = np.argsort(np.abs(self.params))[-1]
        ref_sd = self.sd_vec[ref_sd_ind]
        spatial_ref_sd, _ = slater.split_spin(ref_sd, self.nspatial)
        # use ap1rog normalization scheme
        ci_params = self.params / self.params[ref_sd_ind]

        # create AP1roG
        ap1rog = AP1roG(self.nelec, self.nspin, dtype=self.dtype, ngem=None, orbpairs=None,
                        ref_sd=ref_sd, params=None)
        # fill empty geminal coefficient
        gem_coeffs = np.zeros(ap1rog.params.shape, dtype=self.dtype)
        for occ_ind in slater.occ_indices(spatial_ref_sd):
            for vir_ind in slater.vir_indices(spatial_ref_sd, self.nspatial):
                # excite slater determinant
                sd_exc = slater.excite(ref_sd, occ_ind, vir_ind)
                sd_exc = slater.excite(sd_exc, occ_ind+self.nspatial, vir_ind+self.nspatial)
                # set geminal coefficient (`a` decremented b/c first npair columns are removed)
                row_ind = ap1rog.dict_reforbpair_ind[(occ_ind, occ_ind+self.nspatial)]
                col_ind = ap1rog.dict_orbpair_ind[(vir_ind, vir_ind+self.nspatial)]
                try:
                    gem_coeffs[row_ind, col_ind] = ci_params[self.dict_sd_index[sd_exc]]
                except KeyError:
                    gem_coeffs[row_ind, col_ind] = 0.0
        ap1rog.assign_params(gem_coeffs)

        return ap1rog
