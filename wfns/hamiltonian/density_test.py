import wfns.backend.slater as slater


def integrate_wfn_dm_sd(wfn, sd, indices, deriv=None):
        r"""Integrate the density operator against the wavefunction and the Slater determinant.

        .. math::

            \braket{\Phi | \hat{\Gamma} | \Psi}
            &= \braket{\Phi | \hat{\Gamma} | \Psi}

        where :math:`\Psi` is the wavefunction, :math:`\hat{\Gamma}` is the density matrix operator,
        and :math:`\Phi` is the Slater determinant.

        Parameters
        ----------
        wfn : Wavefunction
            Wavefunction against which the operator is integrated.
            Needs to have the following in `__dict__`: `get_overlap`.
        indices : tuple of int
            Spin orbital indices that describe the creators and annihilators used in the density
            matrix operators.
            List of indices will divided in two, where the first half corresponds to the creators
            and the second half corresponds to the annihilators.
        sd : int
            Slater Determinant against which the operator is integrated.
        deriv : {int, None}
            Index of the wavefunction parameter against which the integral is derivatized.
            Default is no derivatization.

        Returns
        -------
        density_element : float
            Density matrix element.

        Raises
        ------
        ValueError
            If number of indices is not even.
        TypeError
            If an index is not an integer.

        """
        if len(indices) % 2 != 0:
            raise ValueError('There must be even number of indices.')
        elif not all(isinstance(i, int) for i in indices):
            raise TypeError('All indices must be integers.')
        creators = indices[:len(indices)]
        annihilators = indices[len(indices):]

        sign = 1
        for creator, annihilator in zip(creators, annihilators):
            # since Slater determinant is on the left, the creator annihilates and annihilator
            # creates on to the Slater determinant
            sign *= (-1)**slater.find_num_trans_swap(sd, creator, annihilator)
            sd = slater.excite(sd, creator, annihilator)
            if sd is None:
                return 0.0
        return sign * wfn.get_overlap(sd, deriv=deriv)
