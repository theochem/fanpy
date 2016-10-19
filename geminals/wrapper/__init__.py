from __future__ import division, print_function
from __future__ import absolute_import

horton_mia = False
try:
    from . import horton
except ImportError:
    print('WARNING: Error in loading horton wrapper. Check that you have HORTON installed'
            ' and that PYTHONPATH is set properly')
    horton_mia = True

pyscf_mia = False
try:
    from . import pyscf
except ImportError:
    print('WARNING: Error in loading pyscf wrapper. Check that you have PySCF installed'
            ' and that PYTHONPATH is set properly')
    pyscf_mia = True

if horton_mia and pyscf_mia:
    raise ImportError('One of HORTON and PySCF must be installed.')
