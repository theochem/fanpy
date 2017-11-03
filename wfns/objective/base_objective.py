"""Parent class of the objective equations.

The objective is used to optimize the wavefunction and/or Hamiltonian.

"""
import abc
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.wavefunction.composite.base_composite_one import BaseCompositeOneWavefunction
from wfns.wavefunction.composite.lin_comb import LinearCombinationWavefunction
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian
import wfns.backend.slater as slater


# parameters
#  passed to objective as a numpy array
#  save parameters
#  update parameters to some object
#  select correct index when derivatizing
#   execute correct derivatization (need to know which object parameter belongs to)
#  freeze certain parameters from optimization
#
#  extract relevant parameters for objective (packing)
#  load relevant parameters to objects (unpacking)
#  derivatization with respect to parameters
# parameter types
#  wavefunction
#  composite wavefunctions
#  hamiltonian
#  energy
# attributes
#  index in the active parameters, active indices, object/label, index within object
#  object type to mask (requires upto 4x memory for the parameters)
#   e.g. wfn to boolean array of wfn parameters that are active
#        wfn to active wfn parameters from all active parameters
#  param type to indices of active parameters (within object)(find active parameters from an object)
#  active param toparam type (find active parameters of a certain object/type)
#  active param index to param type (find the type of the given active parameter)

#  masks_objective
#   dict of obj to boolean array of the parameters (from objective) that corresponds to that object
#  masks_object
#   dict of obj to boolean array of the parameters (from object) that corresponds to that object
#
# methods
#  extract parameters (get values of the parameters that are active)
#   get the correct object (in the right order)
#   get active parameters (in the right order) from each object
#  load parameters (assign values of the parameters to the appropriate object)
#   update only the active parameters (if possible)
#   for each object
#    get all parameters
#    overwrite active parameters
#    update all parameters
#  derivatize
#   get index of parameter within the cor
#   return 0 if the called function does correspond to the selected object
#   call the appropriate function

@docstring_class(indent_level=1)
class ParamMask(abc.ABC):
    """Class for handling subset of a collection of different types of parameters.

    The objective of the Schrodinger equation can depend on more than one type of parameters. For
    example, it can depend on both wavefunction and Hamiltonian parameters. In addition, only some
    of these parameters may be active, because the others were frozen during the optimization.
    Though each object (wavefunction and Hamiltonian instances) is expressed with respect to its own
    parameters, the objective, the objective can be expressed with respect to different combinations
    of these parameters. This class allows different combinations of parameters to be used in the
    objective by acting as the wrapper between the parameters of each object (wavefunction and
    Hamiltonian) and the parameters of the objective.

    Attributes
    ----------
    masks_objective_params : dict of instance to np.ndarray of bool
        Mask of objective parameters for each object (wavfunction or Hamiltonian).
        Shows which parameters of each object are active in the optimization.
    masks_object_params : dict of instance to np.ndarray of bool
        Mask of object parameters for each object (wavfunction or Hamiltonian).
        Shows which parameters of the objective belong to which object.

    Methods
    -------
    extract_params(self) : np.ndarray
        Extract out the active parameters from the objects.
    load_params(self, params) : np.ndarray
        Assign given parameters of the objective to the appropriate objects.
    derivative_index(self, object, index) : np.ndarray
        Return the index within the objective parameters as the index within the given object.
        If the selected objective parameter is not part of the given object, then `None` is
        returned.

    """
    pass


@docstring_class(indent_level=1)
class BaseObjective(abc.ABC):
    """Base objective function.

    Attributes
    ----------
    wfn : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited state).
    ham : ChemicalHamiltonian
        Hamiltonian that defines the system under study.
    tmpfile : str
        Name of the file that will store the parameters used by the objective method.
        By default, the parameter values are not stored.
        If a file name is provided, then parameters are stored upon execution of the objective
        method.
    param_types : {'wfn', 'ham', 'wfn_components'}
        Types of parameters that will be used by the objective.
        Each type must be one of 'wfn', 'ham', and 'wfn_components'.
        Default is 'wfn'.
        Type 'wfn' means that the objective will depend on the wavefunction parameters.
        Type 'ham' means that the objective will depend on the hamiltonian parameters.
        Type 'wfn_components' means that the objective will depend on the parameters of the
        components of a composite wavefunction. Note that if the composite wavefunction has
        components that are also composite wavefunctions, then only the top layer (first layer of
        components) will be considered.
    param_ranges : tuple of {2-tuple of int, tuple of 2-tuple of int}
        Start and end indices (of the objective parameters) for each parameter type.
        If the parameter type is 'wfn_components', then each component should be associated with a
        start and end indices, resulting in tuple of 2-tuple of ints.

    """
    def __init__(self, wfn, ham, tmpfile='', param_types=None):
        """Initialize the objective.

        Parameters
        ----------
        wfn : BaseWavefunction
            Wavefunction that defines the state of the system (number of electrons and excited
            state).
        ham : ChemicalHamiltonian
            Hamiltonian that defines the system under study.
        tmpfile : str
            Name of the file that will store the parameters used by the objective method.
            By default, the parameter values are not stored.
            If a file name is provided, then parameters are stored upon execution of the objective
            method.
        param_types : {list/tuple, str, None}
            Types of parameters that will be used to construct the objective.
            Each type must be one of 'wfn', 'ham', and 'wfn_components'.
            Default is 'wfn'.
            Type 'wfn' means that the objective will depend on the wavefunction parameters.
            Type 'ham' means that the objective will depend on the hamiltonian parameters.
            Type 'wfn_components' means that the objective will depend on the parameters of the
            components of a composite wavefunction. Note that if the composite wavefunction has
            components that are also composite wavefunctions, then only the top layer (first layer
            of components) will be considered.

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian.
            If save_file is not a string.
            If param_types is not one of 'wfn', 'ham', and 'wfn_components' or a tuple/list of these
            strings.
        ValueError
            If wavefunction and Hamiltonian do not have the same data type.
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.
            If param_types has repeated elements.

        """
        if not isinstance(wfn, BaseWavefunction):
            raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                            'child).')
        if not isinstance(ham, ChemicalHamiltonian):
            raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its '
                            'child).')
        if wfn.dtype != ham.dtype:
            raise ValueError('Wavefunction and Hamiltonian do not have the same data type.')
        if wfn.nspin != ham.nspin:
            raise ValueError('Wavefunction and Hamiltonian do not have the same number of spin '
                             'orbitals')
        self.wfn = wfn
        self.ham = ham

        if not isinstance(tmpfile, str):
            raise TypeError('`tmpfile` must be a string.')
        self.tmpfile = tmpfile

        if param_types is None:
            param_types = ('wfn', )
        elif isinstance(param_types, str):
            param_types = (param_types, )

        allowed_types = ('wfn', 'ham', 'wfn_components')
        if (not isinstance(param_types, (tuple, list)) or
                not all(i in allowed_types for i in param_types)):
            raise TypeError("Parameter types must be one of 'wfn', 'ham', and 'wfn_components' or a"
                            " tuple/list of these strings.")
        elif len(set(param_types)) != len(param_types):
            raise ValueError('Parameter types cannot have repeated elements.')
        self.param_types = tuple(param_types)

        # store parameter range
        # it is stored instead of generating with every use b/c it will be used often.
        param_ranges = []
        ind = 0
        for ptype in self.param_types:
            if ptype == 'wfn_components':
                wfn_component_ranges = []
                for wfn in self._wfn_components:
                    nparams = wfn.nparams
                    wfn_component_ranges.append((ind, ind+nparams))
                    ind += nparams
                param_ranges.append(tuple(wfn_component_ranges))
                continue

            if ptype == 'wfn':
                nparams = self.wfn.nparams
            elif ptype == 'ham':
                nparams = self.ham.nparams
            param_ranges.append((ind, ind+nparams))
            ind += nparams
        self.param_ranges = tuple(param_ranges)

    @property
    def params(self):
        """Return the parameters of the objective at the current state.

        Returns
        -------
        params : np.ndarray(K,)
            Parameters of the objective.

        """
        params = np.array([])
        for ptype in self.param_types:
            if ptype == 'wfn':
                params = np.hstack((params, self.wfn.params.flatten()))
            elif ptype == 'ham':
                params = np.hstack((params, self.ham.params.flatten()))
            elif ptype == 'wfn_components':
                for wfn in self._wfn_components:
                    params = np.hstack((params, self.wfn.params.flatten()))
        return params

    @property
    def _wfn_components(self):
        """Return the wavefunction components.

        Parameters
        ----------
        wfn_components : list of BaseWavefunction
            Component wavefunctions of a composite wavefunction.

        Raises
        ------
        TypeError
            If the provided wavefunction is not a composite wavefunction.

        """
        if isinstance(self.wfn, BaseCompositeOneWavefunction):
            return [self.wfn.wfn]
        elif isinstance(self.wfn, LinearCombinationWavefunction):
            return self.wfns
        else:
            raise TypeError('Provide wavefunction is not a composite wavefunction. It does not have'
                            ' any component wavefunctions.')

    def save_params(self, params):
        """Save the given parameters in the temporary file.

        Parameters
        ----------
        params : np.ndarray
            Parameters used by the objective method.

        """
        if self.tmpfile != '':
            np.save(self.tmpfile, params)

    def assign_params(self, params):
        """Assign the parameters to the wavefunction and/or hamiltonian.

        Parameters
        ----------
        params : np.ndarray(K, )
            Parameters used by the objective method.

        Raises
        ------
        TypeError
            If `params` is not a one-dimensional numpy array.

        """
        if not isinstance(params, np.ndarry) or len(params.shape) != 1:
            raise TypeError('`params` must be a one-dimensional numpy array.')

        for ptype, ind_ranges in zip(self.param_types, self.param_ranges):
            if ptype == 'wfn_components':
                for wfn, (ind_start, ind_end) in zip(self._wfn_components, ind_ranges):
                    wfn.assign_params(params[ind_start: ind_end].reshape(wfn.params_shape))
                    wfn.clear_cache()
                continue

            ind_start, ind_end = ind_ranges
            if ptype == 'wfn':
                self.wfn.assign_params(params[ind_start: ind_end].reshape(self.wfn.params_shape))
                self.wfn.clear_cache()
            elif ptype == 'ham':
                self.ham.assign_params(params[ind_start: ind_end].reshape(self.ham.params_shape))

        if params.size != ind_end:
            raise ValueError('Number of parameter does not match the parameters selected by '
                             '`param_types`.')

    def wrapped_func(self, func, param_type, *sd, deriv):
        """Wrap a function to be derivatized with respect to the parameters of the objective.

        Parameters
        ----------
        func : function
            Function that will be derivatized within the context of the parameters of the objective.
            Has an argument `sd` and keyword argument `deriv`.
        param_type : {'wfn', 'ham', 'wfn_components', None}
            Type of the parameter the given function uses.
            `None` means that derivatization of the given function will always produce zero, i.e.
            function is not dependent on the parameters used in the objective.
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Must be provided as a keyword.

        Returns
        -------
        val : float
            Value of the function given arguments `sd` and `deriv`.

        """
        if deriv is None:
            return func(*sd)

        if param_type is None:
            return 0.0

        try:
            type_ind = self.param_types.index(param_type)
            ind_start, ind_end = self.param_ranges[type_ind]
            if ind_start <= deriv < ind_end:
                deriv -= ind_start
            else:
                raise ValueError
        except ValueError:
            return 0.0
        else:
            return func(*sd, deriv=deriv)

    def wrapped_get_overlap(self, sd, deriv=None):
        """Wrap 'get_overlap' to be derivatized with respect to the parameters of the objective.

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        """
        return self.wrapped_func(self.wfn.get_overlap, 'wfn', sd, deriv=deriv)

    def wrapped_integrate_wfn_sd(self, sd, deriv=None):
        """Wrap 'integrate_wfn_sd' to be derivatized wrt the parameters of the objective.

        Parameters
        ----------
        sd : {int, mpz}
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
        Since `integrate_wfn_sd` depends on both the hamiltonian and the wavefunction, it can be
        derivatized with respect to the paramters of the hamiltonian and of the wavefunction.

        """
        def integrate_wfn_sd(sd, wfn_deriv=None, ham_deriv=None):
            return sum(self.ham.integrate_wfn_sd(self.wfn, sd,
                                                 wfn_deriv=wfn_deriv, ham_deriv=ham_deriv))

        integral = self.wrapped_func(lambda sd, deriv: integrate_wfn_sd(sd, ham_deriv=deriv),
                                     'ham', sd, deriv=deriv)
        if deriv is not None:
            # we can add these two together because one of these will be zero (a derivative index
            # only belong to one parameter type)
            integral += self.wrapped_func(lambda sd, deriv: integrate_wfn_sd(sd, wfn_deriv=deriv),
                                          'wfn', sd, deriv=deriv)
        return integral

    def wrapped_integrate_sd_sd(self, sd1, sd2, deriv=None):
        """

        Parameters
        ----------
        sd1 : int
            Slater Determinant against which the Hamiltonian is integrated.
        sd2 : int
            Slater Determinant against which the Hamiltonian is integrated.
        deriv : {int, None}
            Index of the objective parameters with respect to which the overlap is derivatized.
            Default is no derivatization.

        Returns
        -------
        integral : float
            Value of the integral :math:`\braket{\Phi_i | \hat{H} | \Phi_j}`.

        """
        return self.wrapped_func(lambda sd1, sd2, deriv: sum(self.ham.integrate_sd_sd(sd1, sd2,
                                                                                      deriv=deriv)),
                                 'ham', sd1, sd2, deriv=deriv)

    def get_energy_one_proj(self, ref, deriv=None):
        """Return the energy of the Schrodinger equation with respect to a reference wavefunction.

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

            \braket{\Phi_{ref} | \Psi}
            &=
            \sum_{\mathbf{m} \in S} g^*(\mathbf{m}) \bra{\mathbf{m}} \ket{\Psi}\\

        Ideally, we want to use the actual wavefunction as the reference, but, without further
        simplifications, :math:`\Psi` uses too many Slater determinants to be computationally
        tractible. Then, we can truncate the Slater determinants as a subset, :math:`S`, such that
        the most significant Slater determinants are included, while the energy can be tractibly
        computed. This is equivalent to inserting a projection operator

        ..math:

            \braket{\Psi | \sum_{\mathbf{m} \in S} \ket{\mathbf{m}} \bra{\mathbf{m}} \hat{H} | \Psi}
            &= \sum_{\mathbf{m} \in S} f^*(\mathbf{m}) \bra{\mathbf{m}} \hat{H} \ket{\Psi}\\

        Parameters
        ----------
        ref : {CIWavefunction, list/tuple of int}
            Reference wavefunction used to calculate the energy.
            If list/tuple of Slater determinants are given, then the reference wavefunction will be
            the truncated form (according to the given Slater determinants) of the provided
            wavefunction.
        deriv : {int, None}
            Index with respect to which the energy is derivatized.

        Returns
        -------
        energy : float
            Energy of the wavefunction with the given Hamiltonian.

        Raises
        ------
        TypeError
            If `ref` is not a CIWavefunction, int, or list/tuple of int.

        """
        # vectorize functions
        get_overlap = np.vectorize(self.wrapped_get_overlap)
        integrate_wfn_sd = np.vectorize(self.wrapped_integrate_wfn_sd)

        # define reference
        if isinstance(ref, CIWavefunction):
            ref_sds = ref.sd_vec
            ref_coeffs = ref.params
            # FIXME: assumes that the CI wavefunction will always be completely separate from the
            #        given wavefuncion/hamiltonian. It does not support projecting onto one of the
            #        component CI wavefunction of a linear combination of CI wavefunctions.
            if deriv is not None:
                d_ref_coeffs = 0.0
        elif (isinstance(ref, (list, tuple)) and
              all(slater.is_sd_compatible(sd) for sd in ref)):
            ref_sds = ref
            ref_coeffs = get_overlap(ref)
            if deriv is not None:
                d_ref_coeffs = get_overlap(ref, deriv)
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
        """Return the energy of the Schrodinger equation after projecting out both sides.

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
            if not (isinstance(pspace, (list, tuple)) and
                    all(slater.is_sd_compatible(sd) for sd in pspace)):
                raise TypeError('Projection space must be given as a list/tuple of ints. See '
                                '`backend.slater` for compatible representations of the Slater '
                                'determinants.')

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
            d_energy += np.sum(overlaps_l*get_overlap(pspace_l, pspace_r, deriv)*overlaps_r) / norm
            d_energy += np.sum(overlaps_l * ci_matrix * get_overlap(pspace_r, deriv)) / norm
            d_energy -= d_norm * np.sum(overlaps_l * ci_matrix * overlaps_r) / norm**2
            return d_energy

    @abc.abstractmethod
    def objective(self, params):
        """Return the value of the objective for the given parameters.

        Parameters
        ----------
        params : np.ndarray
            Parameter that changes the objective.

        Returns
        -------
        objective_value : float
            Value of the objective for the given parameters.

        """
        pass
