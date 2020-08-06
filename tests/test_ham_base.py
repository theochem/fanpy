"""Test wfns.ham.base."""
import numpy as np
import pytest
from wfns.ham.base import BaseHamiltonian


class Empty:
    """Empty container class."""

    pass


def test_assign_integrals():
    """Test BaseHamiltonian.assign_integrals."""
    test = Empty()
    with pytest.raises(NotImplementedError):
        BaseHamiltonian.assign_integrals(test, None, None)
