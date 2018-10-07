=================
 FANCI |version|
=================

FANCI is a free and open source Python module for constructing and solving the Schrödinger
equation. It is designed to be a research tool for developing new methods that address different
aspects of the Schrödinger equation: the Hamiltonian, the wavefunction, the objective that
corresponds to the Schrödinger equation, and the algorithm with which the objective is solved.
We aim to provide an accessible tool that can facilitate the research between multiple aspects of
the electronic structure problem. (FIXME) To do so, we make the following design choices:


* Easy to install
    * Standard dependencies
        - The FANCI module depends only on modules that are easily available to most operating
          systems and that have been stable for the last few years. Non-standard modules are
          supported optionally, such that the FANCI module is still useable without this dependency.

    * Pure Python
        - Compilation is often not trivial for some operating systems. To support as many systems
          as possible, the FANCI module will be pure Python (with an *optional* compilation).

* Modular
    * Extensible
        - A new wavefunction or Hamiltonian should not result in a refactor due to its
          incompatibility with the rest of the module. All wavefunctions, Hamiltonians, and
          objectives have a minimal (abstract) base structure to ensure compatibility with the rest
          of the module.

    * Localized bottleneck
        - Except for the `slater` module, which handles the Slater determinants, all of the
          submodules in the FANCI module are independent of one another. Independent structures will
          make it easier to identify and remedy performance issues.

* Fully Documented
    * Docstring
        - The docstring of a structure (class, function, etc.) defines the API of the code. A user
          should be able to figure out the *exact* behaviour of the code by only reading the
          docstring. For a quick understanding of the code, it is essential that each structure is
          **fully** documented. All of the attributes, parameters, returned/yielded values, raised
          errors, etc. must be explicitly described. See NumPy docstring format for details.

    * Human-readable naming schemes
        - Human-readable names can be useful for quickly understanding the contents of a variable
          or the behaviour of a method without having to skip to its docstring. Even though PEP8
          encourages shorter names, longer names are used to improve readability.

    * Comments
        - In the cases where a part of the code needs further documentation, they should be
          commented. If possible (especially if the code is repeated multiple times), the code
          is abstracted out as a method/class.

* Well Tested
    * Unit Tests and coverage
        - To "quickly" identify bugs after modifying the code, the FANCI module is extensively unit
          tested. Each structure of the code is tested independently from the other structures so
          that a failed test can be used to the locate the bug. To ensure that all parts of the code
          is tested, code coverage is measured in the development process.

    * Look Before You Leap (LBYL) preferred
        - Since the FANCI module revolves around the numerical optimization of the objective, small
          bugs can be unnoticeable except in the final result. To ensure that the code functions
          **exactly** as it is intended, the LBYL approach is preferred over the EAFP ("easier to
          ask for forgiveness than permission") approach, especially when checking the input of a
          class or method. Though this preference is not absolute in the FANCI module, the behaviour
          of the code should be explicit and deviations should result in an error.

.. toctree::
  :maxdepth: 2

  installation
  overview
  hamiltonian
  wavefunction
  objective
  solver
  tutorial_index
  example_index
  tech_api
