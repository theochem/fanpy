"""Parent class of the objective equations.

The objective is used to optimize the wavefunction and/or Hamiltonian.

"""
import abc
import numpy as np
from wfns.wrapper.docstring import docstring_class
from wfns.wavefunction.base_wavefunction import BaseWavefunction
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

    """
    def __init__(self, wfn, ham, tmpfile=''):
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

        Raises
        ------
        TypeError
            If wavefunction is not an instance (or instance of a child) of BaseWavefunction.
            If Hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian.
            If save_file is not a string.
        ValueError
            If wavefunction and Hamiltonian do not have the same data type.
            If wavefunction and Hamiltonian do not have the same number of spin orbitals.

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

    @abc.abstractmethod
    def gradient(self, params):
        """Return the gradient of the objective for the given parameters

        Parameters
        ----------
        params : np.ndarray
            Parameters that are input to the objective.

        Returns
        -------
        objective_gradient : np.ndarray
            Gradient of the objective for the given parameters.

        """
        pass

    def save_params(self, params):
        """Save the given parameters in the temporary file.

        Parameters
        ----------
        params : np.ndarray
            Parameters used by the objective method.

        """
        if self.tmpfile != '':
            np.save(self.tmpfile, params)
