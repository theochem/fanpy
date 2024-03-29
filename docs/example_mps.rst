=============================
 MPS Wavefunction Example
=============================

The following scripts are generated by using :code:`wfns_make_script.py` and by tweaking the
generated script. For more information on using :code:`wfns_make_script.py`, go to
:ref:`Using a script that makes a script <tutorial_calc_make_script>` for the tutorial and
:ref:`wfns_make_script.py <script_make_script>` for the API. For more information on customizing the
script, go to :ref:`How to run a calculation by making a script <tutorial_calc_code>`.

For more information, see :py:class:`MPS <wfns.wfn.geminal.mps.MPS>`.

Default MPS Configuration
----------------------------
.. code:: bash

   wfns_make_script.py --nelec 4 --nspin 8 --one_int_file oneint.npy \
                       --two_int_file twoint.npy --wfn_type mps \
                       --solver cma --objective least_squares --optimize_orbs \
                       --filename mps.py

Wavefunction
   MPS
Hamiltonian
   Restricted Chemical Hamiltonian
Optimized Parameters
   Orbitals are optimized
   MPS parameters are optimized
Projection Space
   HF ground state and its first and second order excitations
Objective
   Squared sum of the Projected Schrodinger equation
Optimizer
   CMA solver

.. code:: python

    import numpy as np
    import os
    from wfns.wfn.network.mps import MatrixProductState
    from wfns.ham.restricted_chemical import RestrictedChemicalHamiltonian
    from wfns.backend.sd_list import sd_list
    from wfns.objective.schrodinger.least_squares import LeastSquaresEquations
    from wfns.solver.equation import cma


    # Number of electrons
    nelec = 4
    print('Number of Electrons: {}'.format(nelec))

    # Number of spin orbitals
    nspin = 8
    print('Number of Spin Orbitals: {}'.format(nspin))

    # One-electron integrals
    one_int_file = 'oneint.npy'
    one_int = np.load(one_int_file)
    print('One-Electron Integrals: {}'.format(os.path.abspath(one_int_file)))

    # Two-electron integrals
    two_int_file = 'twoint.npy'
    two_int = np.load(two_int_file)
    print('Two-Electron Integrals: {}'.format(os.path.abspath(two_int_file)))

    # Nuclear-nuclear repulsion
    nuc_nuc = 0.0
    print('Nuclear-nuclear repulsion: {}'.format(nuc_nuc))

    # Initialize wavefunction
    wfn = MatrixProductState(nelec, nspin, params=None, memory=None, dimension=None)
    print('Wavefunction: MatrixProductState')

    # Initialize Hamiltonian
    ham = RestrictedChemicalHamiltonian(one_int, two_int, energy_nuc_nuc=nuc_nuc, params=None)
    print('Hamiltonian: RestrictedChemicalHamiltonian')

    # Projection space
    pspace = sd_list(nelec, nspin//2, num_limit=None, exc_orders=[1, 2], spin=None,
                    seniority=wfn.seniority)
    print('Projection space (orders of excitations): [1, 2]')

    # Select parameters that will be optimized
    param_selection = [(wfn, np.ones(wfn.nparams, dtype=bool)), (ham, np.ones(ham.nparams, dtype=bool))]

    # Initialize objective
    objective = LeastSquaresEquations(wfn, ham, param_selection=param_selection, tmpfile='',
                                      pspace=pspace, refwfn=None, energy_type='compute', energy=None,
                                      constraints=None, eqn_weights=None)

    # Solve
    print('Optimizing wavefunction: cma solver')
    results = cma(objective, save_file='', sigma0=0.01, options={'ftarget': None, 'timeout': np.inf,
                  'tolfun': 1e-11, 'verb_filenameprefix': 'outcmaes', 'verb_log': 0})

    # Results
    if results['success']:
        print('Optimization was successful')
    else:
        print('Optimization was not successful: {}'.format(results['message']))
    print('Final Energy: {}'.format(results['energy']))

MPS with different dimensions
-----------------------------
The current implementation of the MPS wavefunction uses a row vector for the occupation of the first
orbital, a column vector for the occupation of the last orbital, and square matrices for the
occupation of the remaining orbitals. The default MPS wavefunction uses matrices of shape
:math:`(1, 2K)` for the occupations of the first orbital, matrices of shape :math:`(2K, 1)` for the
occupations of the last orbital, and matrices of shape :math:`(2K, 2K)` for the occupations of the
remaining orbitals, where :math:`2K` is the number of spin orbitals. To change the dimension of
these matrices, modify the :code:`dimension` parameter in the initialization. For example,

.. code:: python

    wfn = MatrixProductState(nelec, nspin, params=None, memory=None, dimension=20)

would result in matrices of shapes :math:`(1, 20)`, :math:`(20, 20)`, and :math:`(20, 1)` for a
given set of occupations.

At the moment, the shapes of the matrices cannot be modified beyond this modification. All matrices
that correspond to the occupations of all non terminal orbitals are constrained to be square with
the number of rows (and columns) constrained to the value of :math:`dimension`.
