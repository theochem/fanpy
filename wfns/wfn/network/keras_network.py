"""Wavefunction using Keras NN."""
import keras
import numpy as np
from wfns.backend.sd_list import sd_list
import wfns.backend.slater as slater
from wfns.wfn.base import BaseWavefunction


class KerasNetwork(BaseWavefunction):
    r"""Base wavefunction class.

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
    dtype
        Data type of the wavefunction.

    Methods
    -------
    __init__(self, nelec, nspin, memory=None)
        Initialize the wavefunction.
    assign_nelec(self, nelec)
        Assign the number of electrons.
    assign_nspin(self, nspin)
        Assign the number of spin orbitals.
    assign_memory(self, memory=None)
        Assign memory available for the wavefunction.
    assign_params(self, params)
        Assign parameters of the wavefunction.
    load_cache(self)
        Load the functions whose values will be cached.
    clear_cache(self)
        Clear the cache.
    get_overlap(self, sd, deriv=None) : float
        Return the overlap of the wavefunction with a Slater determinant.

    """

    # pylint: disable=W0223
    def __init__(self, nelec, nspin, model=None, params=None, memory=None, num_layers=2):
        """Initialize the wavefunction.

        Parameters
        ----------
        nelec : int
            Number of electrons.
        nspin : int
            Number of spin orbitals.
        mode : {keras.Model, None}
            Model instance from keras.
            Default is 2 layers.
        memory : {float, int, str, None}
            Memory available for the wavefunction.
            Default does not limit memory usage (i.e. infinite).

        """
        super().__init__(nelec, nspin, memory=memory)
        self.num_layers = num_layers
        self.assign_model(model=model)
        self._default_params = None
        self.assign_params(params=params)
        keras.backend.common._FLOATX = "float64"  # pylint: disable=W0212

    def assign_model(self, model=None):
        """Assign the Keras model used to represent the neural network.

        Parameters
        ----------
        model : {keras.engine.training.Model, None}
            Keras Model instance.
            Default is a neural network with two hidden layers with ReLU activations. The number of
            hidden units in each layer is the number of spin orbitals.

        Raises
        ------
        TypeError
            If the given model is not an instance of keras.engine.training.Model.
        ValueError
            If the number of "types" of input variables is not one.
            If the number of input variables is not the number of spin orbitals.
            If the number of "types" of output variables is not one.
            If the number of output variables is not one.

        """
        if model is None:
            model = keras.engine.sequential.Sequential()
            for _ in range(self.num_layers):
                model.add(
                    keras.layers.core.Dense(
                        self.nspin,
                        activation=keras.activations.relu,
                        input_dim=self.nspin,
                        use_bias=False,
                    )
                )
            model.add(
                keras.layers.core.Dense(
                    1, activation=keras.activations.relu, input_dim=self.nspin, use_bias=False
                )
            )

        #if not isinstance(model, keras.engine.training.Model):
        #    raise TypeError("Given model must be an instance of keras.engine.network.Network.")
        if len(model.inputs) != 1:
            raise ValueError(
                "Given model must only have one set of inputs (for the occupations of "
                "the Slater determinant)."
            )
        if model.inputs[0].shape[1] != self.nspin:
            raise ValueError(
                "Given model must have exactly the same number of input nodes as the "
                "number of spin orbitals."
            )
        if len(model.outputs) != 1:
            raise ValueError(
                "Given model must only have one set of outputs (for the overlap of "
                "the Slater determinant)."
            )
        if model.outputs[0].shape[1] != 1:
            raise ValueError("Given model must have exactly one output.")

        # compile model because some of the methods/attributes do not exist until compilation
        # (e.g. get_gradient)
        # NOTE: Keras has a built in method for gradient but it includes the loss function. To make
        # things easier, we modify the loss function so that it does not do anything (i.e. identify
        # function)
        def loss(y_true, y_pred):
            """Loss function used to hack in objective into Keras."""
            return keras.backend.sum(y_true - y_pred)

        model.compile(loss=loss, optimizer="sgd")

        self.model = model

    @property
    def nparams(self):
        """Return the number of wavefunction parameters.

        Returns
        -------
        nparams : int
            Number of parameters.

        """
        return self.model.count_params()

    @property
    def params_shape(self):
        """Return the shape of the wavefunction parameters.

        Returns
        -------
        params_shape : tuple of int
            Shape of the parameters.

        Notes
        -----
        Instance must have attribut `model`.

        """
        return (self.nparams,)

    # FIXME: not a very robust way of building an initial guess. It is not very good and requires
    # specific network structures.
    def assign_default_params(self):
        r"""Assign the intial guess for the HF ground state wavefunction.

        Since the default parameters are calculated/approximated, they are computed and stored away
        rather than generating each one on the fly.

        Raises
        ------
        ValueError
            If any of the layers of the model has more than one type of weights. For example, bias
            is not allowed.
            If the number of units `K` in the final hidden layer is greater than
            :math:`1 + (K-N)N + \binom{K-N}{2} \binom{N}{2}`.

        Notes
        -----
        The default parameters can only be created for networks without bias and sufficiently large
        final hidden layer. Additionally, the produced parameters may not be a good initial guess
        for the HF ground state.

        """
        params = []
        for layer in self.model.layers[:-1]:
            # NOTE: it was ASSUMED that there is only one variable for weights
            if len(layer.weights[:-1]) > 1:
                raise ValueError(
                    "Cannot generate initial guess for Keras networks that have layers with more "
                    "than one variable for weights."
                )
            params += np.eye(*layer.weights[0].shape).flatten().tolist()
        # solve for last layer
        num_hidden_orbs = int(self.model.layers[-1].weights[0].shape[0])
        hidden_sds = [
            slater.occ_indices(sd)
            for sd in sd_list(self.nelec, num_hidden_orbs // 2, exc_orders=[1, 2])
        ]
        if len(hidden_sds) < num_hidden_orbs:
            raise ValueError(
                "Cannot generate initial guess for Keras network because the final "
                "hidden layer does not have enough units for the number of electrons."
            )
        hidden_units = np.zeros((len(hidden_sds), num_hidden_orbs))
        for i, hidden_sd in enumerate(hidden_sds):
            hidden_units[i, hidden_sd] = 1
        output = np.eye(1, len(hidden_sds))[0]
        # TODO: weights are not normalized

        params += np.linalg.lstsq(hidden_units, output)[0].tolist()
        self._default_params = np.array(params)

    def assign_params(self, params=None, add_noise=False):
        """Assign the parameters of the wavefunction.

        Parameters
        ----------
        params : {np.ndarray, None}
            Parameters of the wavefunction.
            Default tries to approximate the ground state HF wavefunction.
        add_noise : {bool, False}
            Option to add noise to the given parameters.
            Default is False.

        """
        if params is None:
            if self._default_params is None:
                self.assign_default_params()
            params = self._default_params

        # store parameters
        super().assign_params(params=params, add_noise=add_noise)

        # divide parameters into a list of two dimensional numpy arrays
        weights = []
        counter = 0
        for var_weights in self.model.weights:
            next_counter = counter + np.prod(var_weights.shape)
            var_params = self.params[counter:next_counter].reshape(var_weights.shape)
            weights.append(var_params)
            counter = next_counter
        # change weights of model
        self.model.set_weights(weights)

    def get_overlap(self, sd, deriv=None):
        r"""Return the overlap of the wavefunction with a Slater determinant.

        .. math::

            \left< \mathbf{m} \middle| \Psi \right>

        Parameters
        ----------
        sd : {int, mpz}
            Slater Determinant against which the overlap is taken.
        deriv : int
            Index of the parameter to derivatize.
            Default does not derivatize.

        Returns
        -------
        overlap : float
            Overlap of the wavefunction.

        Raises
        ------
        TypeError
            If given Slater determinant is not compatible with the format used internally.

        Notes
        -----
        Overlaps and their derivatives are not cached.

        """
        sd = slater.internal_sd(sd)
        occ_vector = np.zeros(self.nspin)
        occ_vector[np.array(slater.occ_indices(sd))] = 1
        # NOTE: overlaps are not being cached

        # if no derivatization
        if deriv is None:
            return self.model.predict(np.array([occ_vector]))[0, 0]
        # if derivatization
        if not isinstance(deriv, (int, np.int64)):
            raise TypeError("Index for derivatization must be provided as an integer.")

        if deriv >= self.nparams:
            return 0.0

        grad = keras.backend.function(
            self.model.inputs,
            self.model.optimizer.get_gradients(self.model.output, self.model.weights),
        )
        # FIXME: this is pretty bad... gradient is computed for the complete set of parameters and
        # only one is selected.
        grad_val = np.hstack([val.flatten() for val in grad([np.array([occ_vector])])])
        return grad_val[deriv]
