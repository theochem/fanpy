"""Test wfns.ham.restricted_base."""
import numpy as np
from wfns.ham.restricted_base import BaseRestrictedHamiltonian


class TestBaseRestrictedHamiltonian(BaseRestrictedHamiltonian):
    """BaseRestrictedHamiltonian class that bypasses the abstract methods."""
    def integrate_wfn_sd(self, wfn, sd, deriv=None):
        """Abstract method."""
        pass

    def integrate_sd_sd(self, sd1, sd2, deriv=None):
        """Abstract method."""
        pass


def test_nspin():
    """Test BaseGeneralizedHamiltonian.nspin."""
    one_int = np.arange(1, 5, dtype=float).reshape(2, 2)
    two_int = np.arange(5, 21, dtype=float).reshape(2, 2, 2, 2)
    test = TestBaseRestrictedHamiltonian(one_int, two_int)
    assert test.nspin == 4
