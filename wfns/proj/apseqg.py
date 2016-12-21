from __future__ import absolute_import, division, print_function

import itertools as it
import numpy as np
from gmpy2 import mpz, bit_scan1

from .. import slater
from ..math_tools import permanent_ryser
from ..sd_list import sd_list

from .proj_wavefunction import ProjectionWavefunction
from .proj_hamiltonian import hamiltonian


class APseqG(ProjectionWavefunction):
    """ Antisymmetric Product of Sequantially Interacting Geminals

    Attributes
    ----------
    dtype : {np.float64, np.complex128}
        Numpy data type
    H : np.ndarray(K,K) or tuple np.ndarray(K,K)
        One electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 2), one electron integrals for the (alpha, alpha)
        and the (beta, beta) unrestricted orbitals
    G : np.ndarray(K,K,K,K) or tuple np.ndarray(K,K)
        Two electron integrals for restricted, unrestricted, or generalized orbitals
        If tuple of np.ndarray (length 3), two electron integrals for the
        (alpha, alpha, alpha, alpha), (alpha, beta, alpha, beta), and
        (beta, beta, beta, beta) unrestricted orbitals
    nuc_nuc : float
        Nuclear nuclear repulsion value
    nelec : int
        Number of electrons
    orbtype : {'restricted', 'unrestricted', 'generalized'}
        Type of the orbital used in obtaining the one-electron and two-electron integrals
    params : np.ndarray(K)
        Guess for the parameters
        Iteratively updated during convergence
        Initial guess before convergence
        Coefficients after convergence
    cache : dict of mpz to float
        Cache of the Slater determinant to the overlap of the wavefunction with this
        Slater determinant
    d_cache : dict of (mpz, int) to float
        Cache of the Slater determinant to the derivative(with respect to some index)
        of the overlap of the wavefunction with this Slater determinan

    Properties
    ----------
    _methods : dict of func
        Dictionary of methods that are used to solve the wavefunction
    nspin : int
        Number of spin orbitals (alpha and beta)
    nspatial : int
        Number of spatial orbitals
    npair : int
        Number of electron pairs (rounded down)
    nparam : int
        Number of parameters used to define the wavefunction
    nproj : int
        Number of Slater determinants to project against
    ref_sd : int or list of int
        Reference Slater determinants with respect to which the norm and the energy
        are calculated
        Integer that describes the occupation of a Slater determinant as a bitstring
        Or list of integers
    template_coeffs : np.ndarray(K)
        Default numpy array of parameters.
        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

    Method
    ------
    __init__(nelec=None, H=None, G=None, dtype=None, nuc_nuc=None, orbtype=None)
        Initializes wavefunction
    __call__(method="default", **kwargs)
        Solves the wavefunction
    assign_dtype(dtype)
        Assigns the data type of parameters used to define the wavefunction
    assign_integrals(H, G, orbtype=None)
        Assigns integrals of the one electron basis set used to describe the Slater determinants
        (and the wavefunction)
    assign_nuc_nuc(nuc_nuc=None)
        Assigns the nuclear nuclear repulsion
    assign_nelec(nelec)
        Assigns the number of electrons
    _solve_least_squares(**kwargs)
        Solves the system of nonliear equations (and the wavefunction) using
        least squares method
    assign_params(params=None)
        Assigns the parameters used to describe the wavefunction.
        Adds random noise from the template if necessary
    assign_pspace(pspace=None)
        Assigns projection space
    overlap(sd, deriv=None)
        Retrieves overlap from the cache if available, otherwise compute overlap
    compute_norm(sd=None, deriv=None)
        Computes the norm of the wavefunction
    compute_energy(include_nuc=False, sd=None, deriv=None)
        Computes the energy of the wavefunction
    objective(x)
        The objective (system of nonlinear equations) associated with the projected
        Schrodinger equation
    jacobian(x)
        The Jacobian of the objective
    compute_pspace
        Generates a tuple of Slater determinants onto which the wavefunction is projected
    compute_overlap
        Computes the overlap of the wavefunction with one or more Slater determinants
    compute_hamiltonian
        Computes the hamiltonian of the wavefunction with respect to one or more Slater
        determinants
        By default, the energy is determined with respect to ref_sd
    normalize
        Normalizes the wavefunction (different definitions available)
        By default, the norm should the projection against the ref_sd squared
    """

    def __init__(self,
                 # Mandatory arguments
                 nelec=None,
                 H=None,
                 G=None,
                 # Arguments handled by base Wavefunction class
                 dtype=None,
                 nuc_nuc=None,
                 # Arguments handled by ProjWavefunction class
                 params=None,
                 pspace=None,
                 # sequence list
                 seq_list=None,
                 # Arguments for saving parameters
                 params_save_name=''
                 ):
        # FIXME: this fucking mess
        super(ProjectionWavefunction, self).__init__(nelec=nelec,
                                                     H=H,
                                                     G=G,
                                                     dtype=dtype,
                                                     nuc_nuc=nuc_nuc,
                                                     )
        del self._energy
        self.cache = {}
        self.d_cache = {}
        self.dict_orbpair_gem = {}
        self.dict_gem_orbpair = {}

        self.assign_params_save(params_save_name=params_save_name)
        self.assign_seq_list(seq_list=seq_list)
        self.assign_pspace(pspace=pspace)
        self.config_gem_orbpair()
        # assigned last due to cyclic dependence business
        self.assign_params(params=params)
        if params is None:
            self.params[-1] = self.compute_energy(ref_sds=self.default_ref_sds)
        # normalize
        self.normalize()

    @property
    def ngem(self):
        """ Number of geminals

        Returns
        -------
        int

        """
        return len(self.dict_gem_orbpair)

    @property
    def template_coeffs(self):
        """ Default numpy array of parameters.

        This will be used to determine the number of parameters
        Initial guess, if not provided, will be obtained by adding random noise to
        this template

        Returns
        -------
        template_coeffs : np.ndarray(K, )

        """
        gem_coeffs = np.zeros((self.npair, self.ngem), dtype=self.dtype)
        for i in range(self.npair):
            gem_ind = self.dict_orbpair_gem[(i, i+self.nspatial)]
            gem_coeffs[i, gem_ind] = 1
        return gem_coeffs

    # NOTE: we need to redefine assign_pspace, because of the dependency order
    # normally it is: pspace depends on params
    #                 params depends on template_coeffs
    #                 template_coeffs depends on nothing
    # however in APseqG: pspace depends on params
    #                    params depends on template_coeffs,
    #                    template_coeffs depends on dict_orbpair_gem
    #                    dict_orbpair_gem depends on pspace
    # so pspace was chosen to break this ugly dependence:
    #     params depends on template_coeffs,
    #     template_coeffs depends on ngem
    #     ngem depends on dict_orbpair_gem
    #     dict_orbpair_gem depends on pspace
    #     pspace depends on nothing
    # There is still some ugly cyclic dependence so pspace needs to be truncated
    # after params is assigned
    def assign_pspace(self, pspace=None):
        """ Sets the Slater determinants on which to project against

        Parameters
        ----------
        pspace : int, iterable of int,
            If iterable, then it is a list of Slater determinants (in the form of integers that describe
            the occupation as a bitstring)
            If integer, then it is the number of Slater determinants to be generated

        Note
        ----
        If pspace is None, we take the maximum number of geminals possible for
        the given sequences as the projection space size. However, the number of
        geminals possible will be less for a given pspace. So the pspace will
        need to be truncated afterwards.
        """
        if pspace is None:
            # maximum number of geminals possible for a given sequence (i.e. number
            # of geminals for a given sequence if all of the orbitals are occupied)
            # num_max_gems = sum(self.nspin-1-seq for seq in self.seq_list)
            # select as many projections as possible
            # NOTE: the projections need to be truncated afterwards!
            # pspace = self.compute_pspace(num_max_gems+1)
            # + 1 because we need one equation for normalization

            # choose all singles and doubles
            pspace = self.compute_pspace(9999999999999999)
        # FIXME: this is quite terrible
        if isinstance(pspace, int):
            pspace = self.compute_pspace(pspace)
        elif isinstance(pspace, (list, tuple)):
            if not all(type(i) in [int, type(mpz())] for i in pspace):
                raise ValueError('Each Slater determinant must be an integer or mpz object')
            pspace = [mpz(sd) for sd in pspace]
        else:
            raise TypeError("pspace must be an int, list or tuple")
        self.pspace = tuple(pspace)

    def assign_seq_list(self, seq_list=None):
        """ Assigns the list of sequence orders that will be included

        Parameters
        ----------
        seq_list : None, list of ints
            Order of sequence that will be included in the wavefunction
            Default is 0

        """
        if seq_list is None:
            seq_list = [0]
        if not isinstance(seq_list, (list, tuple)):
            raise TypeError('seq_list must be a list or a tuple')
        if any(i<0 for i in seq_list):
            raise ValueError('All of the sequence order must be positive')
        self.seq_list = seq_list

    def find_gem_indices(self, sd, raise_error=True):
        """ Returns the geminal indices that corresponds to the desired sequences
        and the Slater determinant

        Parameters
        ----------
        sd : mpz
            Slater determinant written as a bitstring
        raise_error : bool
            If True and the orbital pair is not already defined in
            dict_orbpair_gem, raises KeyError
            If False and the orbital pair is not already defined in
            dict_orbpair_gem, creates new key-value pair in dict_orbpair_gem
            and dict_gem_orbpair.

        Returns
        -------
        List of geminal indices that corresponds to the given sequences and the
        Slater determinant

        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = mpz(sd)
        # because orbitals are ordered by alpha orbs then beta orbs,
        # we need to interleave/shuffle them to pair the alpha and the beta orbitals together
        sd = slater.interleave(sd, self.nspatial)
        # find orbital indices that correspond to each sequence
        orb_pairs = {}
        for seq in self.seq_list:
            occ_index = bit_scan1(sd, 0)
            while occ_index is not None:
                next_index = occ_index+seq+1
                if slater.occ(sd, next_index):
                    sd = slater.annihilate(sd, occ_index, next_index)
                    orb_pairs[(occ_index, next_index)] = None
                    if sd is None:
                        break
                occ_index = bit_scan1(sd, occ_index+1)

        # convert orbital indices to geminal indices
        gem_indices = []
        for orbpair in orb_pairs:
            # unshuffle (and sort) the indices
            orbpair = sorted(slater.deinterleave_index(i, self.nspatial) for i in orbpair)
            # convert to tuple (because lists are not hashable)
            orbpair = tuple(orbpair)
            # store indices
            try:
                gem_indices.append(self.dict_orbpair_gem[orbpair])
            except KeyError:
                # if orbpair not in dictionary
                if raise_error:
                    raise KeyError('Cannot find key {0} in dictionary dict_orbpair_gem'.format(orbpair))
                self.dict_orbpair_gem[orbpair] = self.ngem
                self.dict_gem_orbpair[self.ngem] = orbpair
                gem_indices.append(self.ngem)

        return gem_indices

    # FIXME: this part is hamiltonian dependent
    def config_gem_orbpair(self):
        """ Constructs all of the geminal indices and the orbital pairs that are
        encountered with the given projection space and the sequences

        """
        for sd in self.pspace:
            self.find_gem_indices(sd, raise_error=False)
            occ_indices = slater.occ_indices(sd)
            vir_indices = slater.vir_indices(sd, self.nspin)
            for one, i in enumerate(occ_indices):
                for two, a in enumerate(vir_indices):
                    exc_sd = slater.excite(sd, i, a)
                    if exc_sd is None:
                        continue
                    self.find_gem_indices(exc_sd,
                                          raise_error=False)
                    for j in occ_indices[one:]:
                        for b in vir_indices[two:]:
                            exc_sd = slater.excite(sd, i, j, a, b)
                            if exc_sd is None:
                                continue
                            self.find_gem_indices(exc_sd,
                                                  raise_error=False)


    def compute_pspace(self, num_sd):
        """ Generates Slater determinants to project onto

        Parameters
        ----------
        num_sd : int
            Number of Slater determinants to generate

        Returns
        -------
        pspace : list of gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        """
        return sd_list(self.nelec, self.nspatial, num_limit=num_sd, exc_orders=[1, 2])

    def compute_overlap(self, sd, deriv=None):
        """ Computes the overlap between the wavefunction and a Slater determinant

        The results are cached in self.cache and self.d_cache.

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        overlap : float
        """
        # caching is done wrt mpz objects, so you should convert sd to mpz first
        sd = mpz(sd)

        occ_indices = slater.occ_indices(sd)

        # find the relevant geminal indices
        gem_indices = self.find_gem_indices(sd, raise_error=True)
        # FIXME: THIS MUST BE CHECKED
        if len(gem_indices) < self.npair:
            return 0.0

        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.npair, self.ngem)

        val = 0.0
        # if no derivatization
        if deriv is None:
            val = permanent_ryser(gem_coeffs[:, gem_indices])
            self.cache[sd] = val
        # if derivatization
        elif isinstance(deriv, int) and deriv < self.params.size - 1:
            row_to_remove = deriv // self.ngem
            col_to_remove = deriv % self.ngem
            orbs_to_remove = self.dict_gem_orbpair[col_to_remove]
            # both orbitals of the geminal must be present in the Slater determinant
            if orbs_to_remove[0] in occ_indices and orbs_to_remove[1] in occ_indices:
                row_inds = [i for i in range(self.npair) if i != row_to_remove]
                col_inds = [i for i in gem_indices if i != col_to_remove]
                # NOTE: length of row_inds and col_inds must be the same
                if len(row_inds) == 0 and len(col_inds) == 0:
                    val += 1
                else:
                    val += permanent_ryser(gem_coeffs[row_inds][:, col_inds])
                self.d_cache[(sd, deriv)] = val
        return val

    def compute_hamiltonian(self, sd, deriv=None):
        """ Computes the hamiltonian of the wavefunction with respect to a Slater
        determinant

        ..math::
            \big< \Phi_i \big| H \big| \Psi_{mathrm{APIG}} \big>

        Since only Slater determinants from DOCI will be used, we can use the DOCI
        Hamiltonian

        Parameters
        ----------
        sd : int, gmpy2.mpz
            Integer (gmpy2.mpz) that describes the occupation of a Slater determinant
            as a bitstring
        deriv : None, int
            Index of the paramater to derivatize the overlap with respect to
            Default is no derivatization

        Returns
        -------
        float
        """
        return sum(hamiltonian(self, sd, self.orbtype, deriv=deriv))

    def normalize(self):
        """ Normalizes the wavefunction using the norm defined in
        ProjectionWavefunction.compute_norm

        Some of the cache are emptied because the parameters are rewritten
        """
        # build geminal coefficient
        gem_coeffs = self.params[:-1].reshape(self.npair, self.ngem)
        # # normalize the geminals
        # norm = np.sum(gem_coeffs**2, axis=1)
        # gem_coeffs *= np.abs(norm[:, np.newaxis])**(-0.5)
        # # flip the negative norms
        # gem_coeffs[norm < 0, :] *= -1
        # normalize the wavefunction
        norm = self.compute_norm()
        gem_coeffs *= norm**(-0.5 / self.npair)
        # set attributes
        self.params[:-1] = gem_coeffs.flatten()
        # empty cache
        for sd in self.default_ref_sds:
            del self.cache[sd]
            for i in (j for j in self.d_cache.keys() if j[0] == sd):
                del self.d_cache[i]
            # Following requires d_cache to be a dictionary of dictionary
            # self.d_cache[sd] = {}
