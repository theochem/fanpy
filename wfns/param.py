"""Class for handling the parameters."""
import abc
import collections

import numpy as np


class ParamContainer(abc.ABC):
    """Container for parameters.

    This class is used to provide the basic functionality that would be available when updating the
    parameters in a given object. This was essential in constructing a generalized objective class,
    which can take in any ParamContainer instance, select specific parameters that will be used to
    evaluate the objective, and update these selected parameters in the course of the optimization.

    Attributes
    ----------
    params : np.ndarray
        Parameters.

    Properties
    ----------
    nparams : int
        Number of parameters.

    Methods
    -------
    __init__(self, params)
        Initialize.
    assign_params(self, params)
        Assign parameters.
    clear_cache(self)
        Placeholder function that would clear the cache.

    """

    def __init__(self, params):
        """Initialize.

        Parameters
        ----------
        params : np.ndarray
            Parameters.

        """
        self.assign_params(params)

    @property
    def nparams(self):
        """Return the number of parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.params.size

    def assign_params(self, params):
        r"""Assign parameters.

        Parameters
        ----------
        params : {int, float, complex, np.float64, np.complex128, np.ndarray}
            Parameters.

        Raises
        ------
        TypeError
            If given parameters are not given as a numpy array.
            If given parameters does not have data type of `int`, `float`, `complex`, `np.float64`
            or `np.complex128`.

        """
        if isinstance(params, (int, float, complex, np.float64, np.complex128)):
            params = np.array([params])

        if not isinstance(params, np.ndarray):
            raise TypeError("Parameters must be given as a numpy array.")
        if params.dtype not in (int, float, complex, np.float64, np.complex128):
            raise TypeError(
                "Parameters must have data type of `int`, `float`, `complex`, "
                "`np.float64` and `np.complex128`."
            )
        self.params = params

    def clear_cache(self):
        """Clear the cache of the cached functions.

        This function doesn't actually do anything, but exists as a placeholder so that all if the
        cache exists, it can be cleared when updating the parameters in the optimization process. So
        it is essential that if a child of ParamContainer has a method for clearing its cache, then
        it must be called clear_cache.

        """


class ParamMask(abc.ABC):
    """Class for handling subset of a collection of different types of parameters.

    The objective of the Schrodinger equation can depend on more than one type of parameters. For
    example, it can depend on both wavefunction and Hamiltonian parameters. In addition, only some
    of these parameters may be active, because the others were frozen during the optimization.
    Though each `ParameterContainer` (e.g. `BaseWavefunction`) is expressed with respect to its own
    parameters, the objective, the objective can be expressed with respect to different combinations
    of these parameters. This class allows different combinations of parameters to be used in the
    objective by acting as the wrapper between the parameters of each container (wavefunction and
    Hamiltonian) and the parameters of the objective.

    Attributes
    ----------
    _masks_container_params : OrderedDict
        Mask for the parameters in the container.
        Shows which parameters of each container are used in the objective.
        Keys are the ParamContainer instances.
        Values are the integer indices of the parameters in the container that will be used in the
        objective.
    _masks_objective_params : OrderedDict
        Mask for the parameters in the objective.
        Shows which parameters of the objective belong to a given container.
        Keys are the ParamContainer instances.
        Values are the boolean indices of the parameters in the objective that belong to the given
        container.

    Properties
    ----------
    active_params : np.ndarray
        Parameters that will be used for the objective.
    all_params : np.ndarray
        All of the parameters associated with the masks, even if they were not selected.

    Methods
    -------
    __init__(self, *container_selection)
        Initialize the masks.
    __eq__(self, other)
        Checks if the given ParamMask is equal to itself.
    load_mask_container_params(self, container, sel)
        Load the one mask for the active parameters from the container.
    load_masks_objective_params(self)
        Load the masks for objective parameters.
    load_params(self, params)
        Assign given parameters of the objective to the appropriate containers.
    derivative_index(self, container, index) : {int, None}
        Convert the index of the objective parameters to the index of the given container.
        `None` is returned if the selected parameter is considered constant (frozen).

    """

    def __init__(self, *container_selection):
        """Initialize the masks.

        Parameters
        ----------
        container_selection : 2-tuple
            Container of parameters and subset that will be used in the objective.
            First element is a `ParamContainer` instance (e.g. wavefunction or Hamiltonian) that
            contains the parameters.
            Second element is a numpy index array (boolean or indices) that will select the
            parameters from the container that will be used in the objective.
            If the second element is `None`, then all parameters of the container will be active.

        Notes
        -----
        The order in which `container_selection` is loaded will affect the ordering of the objective
        parameters.

        """
        self._masks_container_params = collections.OrderedDict()
        for container, sel in container_selection:
            self.load_mask_container_params(container, sel)

        self._masks_objective_params = collections.OrderedDict()
        self.load_masks_objective_params()

    def __eq__(self, other):
        """Check if the given ParamMask is equal to itself.

        Parameters
        ----------
        other : ParamMask
            Other ParamMask.

        Raises
        ------
        TypeError
            If `other` is not a ParamMask

        """
        if not isinstance(other, ParamMask):
            raise TypeError(
                "Cannot compare ParamMask instance with something that is not " "ParamMask."
            )

        #  pylint: disable=W0212
        if list(self._masks_container_params.keys()) == list(other._masks_container_params.keys()):
            return all(
                np.array_equal(
                    self._masks_container_params[container],
                    other._masks_container_params[container],
                )
                and np.array_equal(
                    self._masks_objective_params[container],
                    other._masks_objective_params[container],
                )
                for container in self._masks_container_params.keys()
            )
        return False

    def load_mask_container_params(self, container, sel):
        """Load one mask for the active parameters from the container.

        Parameters
        ----------
        container : ParamContainer
            Container with parameters on which the objective depends.
        sel : {bool, np.ndarray, None}
            Index array (boolean or indices) that selects the parameters from the given container to
            be used in the objective.
            If `None`, then all parameters of the container will be active.

        Raises
        ------
        TypeError
            If `container` is not a `ParamContainer` instance.
            If `sel` is not a numpy array.
            If `sel` does not have a data type of int or bool.
        ValueError
            If `sel` as integer indices have entries that are either less than 0 or greater than the
            number of parameters.
            If `sel` as boolean indices does not have the same number of entries as there are
            parameters.

        Notes
        -----
        The indicing is converted to integers (rather than the boolean) at the very end of this
        method. It was boolean originally, but the method `derivative_index` requires the specific
        position of each active parameter within the corresponding container, meaning that some form
        of sorting/search is needed (e.g. `np.where`). Since this method should be executed quite
        frequently during the course of the optimization, the indices are converted to integers
        beforehand. Other use-cases of these indices are not time limiting: both methods
        `active_params` and `load_params` are executed seldomly in comparison. Also, the performance
        of integer and boolean indices should not be too different.

        """
        if not isinstance(container, ParamContainer):
            raise TypeError("The provided container must be a `ParamContainer` instance.")
        nparams = container.nparams

        if sel is None:
            sel = np.ones(nparams, dtype=bool)
        elif isinstance(sel, bool):
            sel = np.array(sel)
        elif not isinstance(sel, np.ndarray):
            raise TypeError("The provided selection must be a numpy array.")
        # check index types
        if sel.dtype not in [int, bool]:
            raise TypeError("The provided selection must have dtype of bool or int.")
        if sel.dtype == int:
            if not np.all(np.logical_and(sel >= 0, sel < nparams)):
                raise ValueError(
                    "The integer indices for selecting the parameters must be greater "
                    "than or equal to 0 and less than the number of parameters."
                )
            bool_sel = np.zeros(nparams, dtype=bool)
            bool_sel[sel] = True
            sel = bool_sel
        elif sel.size != nparams:
            raise ValueError(
                "The provided boolean selection must have the same number of entries "
                "as there are parameters in the provided container."
            )
        # convert to integer indicing
        self._masks_container_params[container] = np.where(sel)[0]

    def load_masks_objective_params(self):
        """Load the masks for objective parameters.

        Though this method can simply be a property, this mask is stored within the instance because
        it will be accessed rather frquently and its not cheap enough (I think) to construct on the
        fly.

        """
        nparams_objective = sum(sel.size for sel in self._masks_container_params.values())
        nparams_cumsum = 0
        masks_objective_params = collections.OrderedDict()
        for container, sel in self._masks_container_params.items():
            nparams_sel = sel.size
            objective_sel = np.zeros(nparams_objective, dtype=bool)
            objective_sel[nparams_cumsum : nparams_cumsum + nparams_sel] = True
            masks_objective_params[container] = objective_sel
            nparams_cumsum += nparams_sel
        self._masks_objective_params = masks_objective_params

    @property
    def all_params(self):
        """Return all of the parameters associated with the mask, even if they were not selected.

        Returns
        -------
        params : np.ndarray
            All of the parameters associated with the mask.
            Parameters are first ordered by the ordering of each container, then they are ordered by
            the order in which they appear in the container.

        """
        return np.hstack([obj.params.ravel() for obj in self._masks_container_params.keys()])

    @property
    def active_params(self):
        """Return the parameters that are active in the objective.

        Returns
        -------
        params : np.ndarray
            Parameters that are selected for optimization.
            Parameters are first ordered by the ordering of each container, then they are ordered by
            the order in which they appear in the container.

        Examples
        --------
        Suppose you have two containers `container1` and `container2` with parameters `[1, 2, 3]`,
        `[4, 5, 6, 7]`, respectively.

        >>> x = ParamMask((container1, [True, False, True]), (container2, [3, 1]))
        >>> x.active_params
        np.ndarray([1, 3, 5, 7])

        Note that the order of the indices during initialization has no affect on the ordering of
        the parameters.

        """
        return np.hstack(
            [obj.params.ravel()[sel] for obj, sel in self._masks_container_params.items()]
        )

    def load_params(self, params):
        """Assign given parameters of the objective to the appropriate containers.

        Parameters
        ----------
        params : np.ndarray
            Parameters of the objective that will need to be updated onto the stored containers.

        Raises
        ------
        TypeError
            If the parameter is not a numpy array.
            If the parameter is not one dimensional.
        ValueError
            If the number of parameters does not match the selection.

        """
        if not isinstance(params, np.ndarray):
            raise TypeError("Given parameter must be a numpy array.")
        if len(params.shape) != 1:
            raise TypeError("Given parameter must be one-dimensional.")

        num_sel = sum(sel.size for sel in self._masks_container_params.values())
        if num_sel != params.size:
            raise ValueError("Number of given parameters does not match up with the selection.")

        for container, sel in self._masks_container_params.items():
            new_params = container.params.ravel()
            new_params[sel] = params[self._masks_objective_params[container]]
            container.assign_params(new_params)
            if hasattr(container, "_cache_fns"):
                container.clear_cache()

    def derivative_index(self, container, index):
        """Convert the index of the objective parameters to the index of the given container.

        Parameters
        ----------
        container : ParamContainer
            Container with parameters on which the objective depends.
        index : {int, None}
            Index of the objective parameter.

        Returns
        -------
        index : {int, None}
            Index of the selected parameter within the given container.
            If the selected parameter is not part of the given container, then `None` is returned.

        """
        if not isinstance(container, ParamContainer):
            raise TypeError("Given container must be a ParamContainer instance.")

        try:
            is_active = self._masks_objective_params[container][index]
        except KeyError:
            # NOTE: This will be useful when the given object is not included in the ParamMask
            #       instead of including it and freezing all of the parameters. For example, if the
            #       objective is independent of the Hamiltonian, then we can still "derivatize" wrt
            #       it without including it in the mask
            return None

        if is_active:
            # index from the list of active parameters in the container
            ind_active_container = np.sum(self._masks_objective_params[container][:index])
            # ASSUMES: indices in self._masks_container_params[container] are ordered from smallest
            #          to largest
            return self._masks_container_params[container][ind_active_container]
        return None
