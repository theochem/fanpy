"""Parent class of the objective equations.

The objective is used to optimize the wavefunction and/or Hamiltonian.

"""
import abc
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.wavefunction.base_wavefunction import BaseWavefunction
from wfns.wavefunction.composite.base_composite_one import BaseCompositeOneWavefunction
from wfns.wavefunction.composite.lin_comb import LinearCombinationWavefunction
from wfns.hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


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
                params = np.hstack(params, self.wfn.params.flatten())
            elif ptype == 'ham':
                params = np.hstack(params, self.ham.params.flatten())
            elif ptype == 'wfn_components':
                for wfn in self._wfn_components:
                    params = np.hstack(self.wfn.params.flatten())
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
        """Wrap 'integrate_sd_sd' to be derivatized wrt the parameters of the objective.

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
