"""Full Configuration Interaction wavefunction."""
from __future__ import absolute_import, division, print_function
from wfns.wavefunction.ci.ci_wavefunction import CIWavefunction
from pydocstring.wrapper import docstring_class

__all__ = []


@docstring_class(indent_level=1)
class FCI(CIWavefunction):
    r""" Full Configuration Interaction Wavefunction

    Given :math:`2K` spin orbitals (or :math:`K` spatial orbitals) and :math:`N` electrons, there
    are a total of :math:`\binom{2K}{N}` different Slater determinants. While many of these are
    equivalent via certain symmetry operations (e.g. flipping the spin) under some conditions (e.g.
    alpha and beta orbitals are the same), no such considerations are taken in the class.

    .. math::

        \ket{\Psi_{\mathrm{FCI}}} =
        \sum_{\mathbf{m} \in S_{\mathrm{FCI}}} c_{\mathbf{m}} \ket{\mathbf{m}}

    """
    def assign_seniority(self, seniority=None):
        r"""

        All seniority must be allowed for a FCI wavefunction.

        Raises
        ------
        ValueError
            If the seniority is not `None` (default value)

        """
        super().assign_seniority(seniority)
        if self.seniority is not None:
            raise ValueError('Only seniority of `None` is supported with the FCI wavefunction. '
                             'i.e. All seniorities must be enabled.')

    def assign_sd_vec(self, sd_vec=None):
        """

        Ignores user input and uses the Slater determinants for the FCI wavefunction (within the
        given spin)

        Raises
        ------
        ValueError
            If the sd_vec is not `None` (default value)

        """
        if sd_vec is None:
            super().assign_sd_vec(sd_vec)
        else:
            raise ValueError('Only the default list of Slater determinants is allowed. i.e. sd_vec '
                             'is `None`. If you would like to customize your CI wavefunction, use '
                             'CIWavefunction instead.')
