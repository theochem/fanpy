"""Pair Coupled Cluster Doubles-AP1roG wavefunction."""
import numpy as np
from itertools import combinations
from collections import Counter
from fanpy.tools import slater
from fanpy.tools import graphs
from fanpy.wfn.cc.base import BaseCC
from fanpy.wfn.ci.base import CIWavefunction


class PCCD(BaseCC):
    r"""Pair CC doubles-AP1roG wavefunction.

    .. math::

        \begin{align}
        & \left| {{\Psi }_{AP1roG}} \right\rangle =\prod\limits_{i}
        {\left( a_{i}^{\dagger }a_{{\bar{i}}}^{\dagger }+
        \sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }}
        \right)}\left| \theta  \right\rangle  \\ & =\prod\limits_{i}
        {\left( 1+\sum\nolimits_{a}{t_{i}^{a}a_{a}^{\dagger }a_{{\bar{a}}}^{\dagger }{{a}_
        {{\bar{i}}}}{{a}_{i}}} \right)}\left| {{\Phi }_{0}} \right\rangle  \\ \end{align}

    In this case the reference wavefunction can only be a single Slater determinant with
    seniority 0.

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
    ranks : list of ints
        Ranks of the excitation operators.
    exops : list of list of int
        Excitation operators given as lists of ints. The first half of indices correspond to
        indices to be annihilated, the second half correspond to indices to be created.
    refwfn : int
        Reference wavefunction upon which the CC operator will act.
    exop_combinations : dict
        dictionary, the keys are tuples with the indices of annihilation and creation
        operators, and the values are the excitation operators that allow to excite from the
        annihilation to the creation operators.

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
    nexops : int
        Number of excitation operators.
    nranks : int
        Number of allowed ranks.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None, ngem=None, orbpairs=None, params=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_dtype(self, dtype)
        Assign the data type of the parameters.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_ranks(self, ranks=None)
        Assign the allowed excitation ranks.
    assign_exops(self, exops=None)
        Assign the allowed excitation operators.
    assign_refwfn(self, refwfn=None)
        Assign the reference wavefunction.
    assign_params(self, params=None, add_noise=False)
        Assign the parameters of the CC wavefunction.
    get_ind(self, exop) : int
        Return the parameter index that corresponds to a given excitation operator.
    get_exop(self, ind) : list of int
        Return the excitation operator that corresponds to a given parameter index.
    product_amplitudes(self, inds, deriv=None) : float
        Return the product of the CC amplitudes of the coefficients corresponding to
        the given indices.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.
    generate_possible_exops(self, a_inds, c_inds):
        Assign the excitation operators that can excite from the given indices to be annihilated
        to the given indices to be created.

    """
    def __init__(self, nelec, nspin, memory=None, ranks=None, indices=None,
                 refwfn=None, params=None, exop_combinations=None, refresh_exops=None):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        dtype : {float, complex, np.float64, np.complex128, None}
            Numpy data type.
            Default is `np.float64`.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).
        ranks : {int, list of int, None}
            Ranks of the excitation operators (in increasing order).
            If int is provided, it takes all the ranks lower than that.
            Default is None, which is equivalent to taking ranks=self.nelec.
        indices : {list of list of ints, None}
            List of lists containing the indices ot the spin-orbitals to annihilate and create.
            The first sub-list contains indices of orbitals to annihilate.
            The second sub-list contains indices of orbitals to create.
            Default generates all possible indices according to the given ranks.
        refwfn: {int, None}
            Reference wavefunction upon which the CC operator will act.
        params : {np.ndarray, BaseCC, None}
            1-vector of CC amplitudes.
        exop_combinations : dict
            dictionary, the keys are tuples with the indices of annihilation and creation
            operators, and the values are the excitation operators that allow to excite from the
            annihilation to the creation operators.

        """
        super().__init__(nelec, nspin, memory=memory, params=params,
                         exop_combinations=exop_combinations, refresh_exops=refresh_exops)
        self.assign_ranks(ranks=ranks)
        self.assign_exops(indices=indices)
        self.assign_refwfn(refwfn=refwfn)

    def assign_nelec(self, nelec):
        """Assign the number of electrons.

        Parameters
        ----------
        nelec : int
            Number of electrons.

        Raises
        ------
        TypeError
            If number of electrons is not an integer.
        ValueError
            If number of electrons is not a positive number.
            If number of electrons is odd.

        """
        super().assign_nelec(nelec)
        if self.nelec % 2 != 0:
            raise ValueError('Odd number of electrons is not supported')

    def assign_ranks(self, ranks=None):
        """Assign the ranks of the excitation operators.

        Parameters
        ----------
        ranks : None
            Ranks of the allowed excitation operators. Set by default to [2].

        Raises
        ------
        ValueError
            If a value that is not the default is provided.
            If the maximum rank is greater than the number of electrons.

        """
        if ranks is not None:
            raise ValueError('Only the default, rank = 2, is allowed')
        if self.nelec <= 1:
            raise ValueError('Only wavefunctions with more than 1 electron can be considered')
        self.ranks = [2]

    def assign_exops(self, indices=None):
        """Assign the excitation operators that will be used to construct the CC operator.

        Parameters
        ----------
        indices : None
            The allowed excitation operators are solely defined by the occupied and virtual
            orbitals of the given reference Slater determinant.

        Raises
        ------
        TypeError
            If `indices` is not None.

        Notes
        -----
        The excitation operators are given as a list of lists of ints.
        Each sub-list corresponds to an excitation operator.
        In each sub-list, the first half of indices corresponds to the indices of the
        spin-orbitals to annihilate, and the second half corresponds to the indices of the
        spin-orbitals to create.
        [a1, a2, ..., aN, c1, c2, ..., cN]

        """
        if indices is not None:
            raise TypeError('Only the excitation operators constructed by default from '
                            'the given reference Slater determinant are allowed')
        else:
            exops = {}
            counter = 0
            ex_from = slater.occ_indices(self.refwfn)
            ex_to = [i for i in range(self.nspin) if i not in ex_from]
            for occ_alpha in ex_from[:len(ex_from) // 2]:
                for virt_alpha in ex_to[:len(ex_to) // 2]:
                    exop = [occ_alpha, occ_alpha + self.nspatial,
                            virt_alpha, virt_alpha + self.nspatial]
                    exops[tuple(exop)] = counter
                    counter += 1
            self.exops = exops

    def assign_refwfn(self, refwfn=None):
        """Assign the reference wavefunction upon which the CC operator will act.

        Parameters
        ----------
        refwfn: {int, None}
            Seniority 0 wavefunction that will be modified by the CC operator.
            Default is the ground-state Slater determinant.

        Raises
        ------
        TypeError
            If refwfn is not a int instance.
        ValueError
            If refwfn does not have the right number of electrons.
            If refwfn does not have the right number of spin orbitals.
            If refwfn is not a seniority-0 wavefunction.

        """
        if refwfn is None:
            self.refwfn = slater.ground(nocc=self.nelec, norbs=self.nspin)
        else:
            if not isinstance(refwfn, int):
                raise TypeError('refwfn must be a int object')
            if slater.total_occ(refwfn) != self.nelec:
                raise ValueError('refwfn must have {} electrons'.format(self.nelec))
            if not all([i + self.nspatial in slater.occ_indices(refwfn) for i in
                        slater.occ_indices(refwfn)[:self.nspatial]] +
                       [i - self.nspatial in slater.occ_indices(refwfn) for i in
                        slater.occ_indices(refwfn)[self.nspatial:]]):
                raise ValueError('refwfn must be a seniority-0 wavefuntion')
            # TODO: check that refwfn has the right number of spin-orbs
            self.refwfn = refwfn

    # Restore old exop api
    # FIXME: BAD structure
    def _olp(self, sd):
        r"""Calculate the matrix element of the CC operator between the Slater determinants.

        .. math::

        \[\left\langle  {{m}_{1}}
        \right|\prod\limits_{u,v}{\left( 1+t_{u}^{v}\hat{\tau }_{u}^{v} \right)}
        \left| {{m}_{2}} \right\rangle \]

        Parameters
        ----------
        sd1 : int
            Occupation vector of the left Slater determinant given as a bitstring.
        sd2 : int
            Occupation vector of the right Slater determinant given as a bitstring.

        Returns
        -------
        olp : {float, complex}
            Matrix element of the CC operator between the given Slater determinant.

        """
        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return 1.0
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()
            # NOTE: Indices of the annihilation (a_inds) and creation (c_inds) operators
            # that need to be applied to sd2 to turn it into sd1

            val = 0.0
            if tuple(a_inds + c_inds) not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)

            # FIXME: exop_list only provides indices (for product_amplitudes) and exop (for sign).
            # This can probably be stored instead of exop_list
            # FIXME: vectorize? not sure if it's possible because all this hinges on generating
            # exops
            for exop_list in self.exop_combinations[tuple(a_inds + c_inds)]:
                if len(exop_list) == 0:
                    continue
                else:
                    inds = np.array([self.get_ind(exop) for exop in exop_list])
                    sign = 1
                    extra_sd = sd2
                    for exop in exop_list:
                        # FIXME: use array sign_excite?
                        try:
                            sign *= slater.sign_excite(extra_sd, exop[:len(exop) // 2],
                                                       exop[len(exop) // 2:])
                        except ValueError:
                            sign = None
                            break
                        extra_sd = slater.excite(extra_sd, *exop)
                    if sign:
                        val += sign * self.product_amplitudes(inds)
                    else:
                        continue
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = 0
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp(sd, self.refwfn)

    def _olp_deriv(self, sd):
        """Calculate the derivative of the overlap with the Slater determinant.

        Parameters
        ----------
        sd1 : int
            Occupation vector of the left Slater determinant given as a bitstring.
        sd2 : int
            Occupation vector of the right Slater determinant given as a bitstring.

        Returns
        -------
        olp : {float, complex}
            Derivative of the overlap with respect to the given parameter.

        """
        def temp_olp(sd1, sd2):
            if sd1 == sd2:
                return np.zeros(self.nparams)
            # FIXME: this should definitely be vectorized
            c_inds, a_inds = slater.diff_orbs(sd1, sd2)
            if isinstance(a_inds, np.ndarray):
                a_inds = a_inds.tolist()
            if isinstance(c_inds, np.ndarray):
                c_inds = c_inds.tolist()
            # NOTE: Indices of the annihilation (a_inds) and creation (c_inds) operators
            # that need to be applied to sd2 to turn it into sd1

            val = np.zeros(self.nparams)
            if tuple(a_inds + c_inds) not in self.exop_combinations:
                self.generate_possible_exops(a_inds, c_inds)
            for exop_list in self.exop_combinations[tuple(a_inds + c_inds)]:
                if len(exop_list) == 0:
                    continue
                else:
                    inds = np.array([self.get_ind(exop) for exop in exop_list])
                    sign = 1
                    extra_sd = sd2
                    for exop in exop_list:
                        try:
                            sign *= slater.sign_excite(extra_sd, exop[:len(exop) // 2],
                                                       exop[len(exop) // 2:])
                        except ValueError:
                            sign = None
                            break
                        extra_sd = slater.excite(extra_sd, *exop)
                    if sign:
                        val += sign * self.product_amplitudes(inds, deriv=True)
                    else:
                        continue
            return val

        if isinstance(self.refwfn, CIWavefunction):
            val = np.zeros(self.nparams)
            for refsd in self.refwfn.sd_vec:
                val += temp_olp(sd, refsd) * self.refwfn.get_overlap(refsd)
            return val
        else:
            return temp_olp(sd, self.refwfn)

    def generate_possible_exops(self, a_inds, c_inds):
        """Assign possible excitation operators from the given creation and annihilation operators.

        Parameters
        ----------
        a_inds : list of int
            Indices of the orbitals of the annihilation operators.
            Must be strictly increasing.
        c_inds : list of int
            Indices of the orbitals of the creation operators.
            Must be strictly increasing.

        Notes
        -----
        The excitation operators are sored as values of the exop_combinations dictionary.
        Each value is a list of lists of possible excitation operators.
        Each sub-list contains excitation operators such that, multiplied together, they allow
        to excite to and from the given indices.

        """
        if self.refresh_exops and len(self.exop_combinations) > self.refresh_exops:
            self.exop_combinations = {}

        exrank = len(a_inds)
        check_ops = []
        # NOTE: Is necessary to invert the results of int_partition_recursive
        # to be consistent with the ordering of operators in the CC operator.
        for partition in list(graphs.int_partition_recursive(self.ranks,
                                                             self.nranks, exrank))[::-1]:
            reduced_partition = Counter(partition)
            bin_size_num = []
            for bin_size in sorted(reduced_partition):
                bin_size_num.append((bin_size, reduced_partition[bin_size]))
            nops = 0
            for b_size in bin_size_num:
                nops += b_size[1]
            for annhs in graphs.generate_unordered_partition(a_inds, bin_size_num):
                for creas in graphs.generate_unordered_partition(c_inds, bin_size_num):
                    combs = []
                    for annh in annhs:
                        for crea in creas:
                            if len(annh) == len(crea):
                                combs.append(annh+crea)
                    for match in combinations(combs, nops):
                        matchs = []
                        for op in match:
                            other_ops = [other_op for other_op in match if other_op != op]
                            matchs += [set(other_op).isdisjoint(op) for other_op in other_ops]
                        if all(matchs):
                            check_ops.append(match)
        self.exop_combinations[tuple(a_inds + c_inds)] = []
        for op_list in check_ops:
            if all(tuple(op) in self.exops for op in op_list):
            # if all(tuple(op) in self.exops for op in op_list):
                self.exop_combinations[tuple(a_inds + c_inds)].append(op_list)

