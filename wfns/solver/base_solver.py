"""Base solver for wavefunctions."""
from __future__ import absolute_import, division, print_function
from ..wavefunction.base_wavefunction import BaseWavefunction
from ..hamiltonian.chemical_hamiltonian import ChemicalHamiltonian


class BaseSolver:
    """Solver for the wavefunction within the given Hamiltonian.

    Attributes
    ----------
    wavefunction : BaseWavefunction
        Wavefunction that defines the state of the system (number of electrons and excited
        state)
    hamiltonian : ChemicalHamiltonian
        Hamiltonian that defines the system under study
    """
    def __init__(self, wavefunction, hamiltonian):
        """Initialize solver.

        Parameters
        ----------
        wavefunction : BaseWavefunction
            Wavefunction that defines the state of the system (number of electrons and excited
            state)
        hamiltonian : ChemicalHamiltonian
            Hamiltonian that defines the system under study
        """
        self.assign_wavefunction(wavefunction)
        self.assign_hamiltonian(hamiltonian)

    def __call__(self):
        """Solve the wavefunction for the given Hamiltonian.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError('Base class cannot actually solve wavefunctions.')

    def assign_wavefunction(self, wavefunction):
        """Assign the wavefunction.

        Parameters
        ----------
        wavefunction : BaseWavefunction
            Wavefunction that defines the state of the system (number of electrons and excited
            state)

        Raises
        ------
        TypeError
                If wavefunction is not an instance (or instance of a child) of BaseWavefunction
        """
        if not isinstance(wavefunction, BaseWavefunction):
            raise TypeError('Given wavefunction is not an instance of BaseWavefunction (or its '
                            'child).')
        self.wavefunction = wavefunction

    def assign_hamiltonian(self, hamiltonian):
        """Assign the Hamiltonian

        Parameters
        ----------
        hamiltonian : ChemicalHamiltonian
            Hamiltonian that defines the system under study

        Raises
        ------
        TypeError
                If hamiltonian is not an instance (or instance of a child) of ChemicalHamiltonian
        """
        if not isinstance(hamiltonian, ChemicalHamiltonian):
            raise TypeError('Given Hamiltonian is not an instance of BaseWavefunction (or its '
                            'child).')
        self.hamiltonian = hamiltonian
