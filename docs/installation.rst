.. installation:

============
Installation
============

Dependencies
============

The FANCI module has been developed using the following dependencies:

* Python >= 3.5: http://www.python.org/
* SciPy >= 0.18.0: http://www.scipy.org/
* NumPy >= 1.11.0: http://www.numpy.org/
* gmpy2 >= 2.0.8:
* pytest >= 4.0.0: https://docs.pytest.org/en/latest/

The latest versions of these modules can be accessed with conda or pip.

Conda
-----

  .. code-block:: bash

     conda install numpy scipy gmpy2 pytest

Pip
---

  .. code-block:: bash

     pip install numpy scipy gmpy2 pytest


Optional Dependencies
=====================

The FANCI module supports different solvers from modules `cma` (CMA-ES solver) and `skopt`
(FIXME). These modules can be used to help solve the Schrödinger equation when the solvers from
`scipy` have difficulty converging.

* cma >= 2.3.1:
* skopt >= :

The one and two-electron integrals are not generated within FanCI and must be obtained from some
external sources. The `horton` module can be used to generate integrals from Gaussian fchk file and
from its HF code. The `pyscf` module can be used to generate integrals from its HF code. Scripts for
generating the integrals are provided in the `scripts` directory.

* HORTON == 2.1.0: http://theochem.github.io/horton/2.0.1/index.html
* PySCF == 1.4.1: https://github.com/sunqm/pyscf

Please note that these modules are not compatible with Python 3.5, which means that they must be
installed on a different version of Python (2.7). It is recommended that you use some form of
virtual environment to isolate the two versions of Python.

  .. code-block:: bash

     conda create --name integrals python=2.7
     conda install -n integrals -c theochem horton
     conda install -n integrals -c pyqc libxc
     conda install -n integrals -c pyscf pyscf
     source activate integrals

The Python interpreters for both HORTON and PySCF must be specified in the environment variables
`HORTONPYTHON` and `PYSCFPYTHON`, respectively. To find the location of the Python interpreter,
either use `which python` in the terminal or `import sys; print(sys.executable)`

  .. code-block:: bash

     export HORTONPYTHON=/abspath/to/hortonpython
     export PYSCFPYTHON=/abspath/to/pyscfpython

Documentation Building
======================
* sphinx
