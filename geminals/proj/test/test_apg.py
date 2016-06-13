from __future__ import absolute_import, division, print_function
from nose.tools import assert_raises

from geminals.proj.apg import APG, ind_from_orbs_to_gem, ind_from_gem_to_orbs

def test_ind_converter():
    nspin = 100
    counter = 0
    for i in range(nspin):
        # if C_{p;ij} \neq C_{p;ji}
        for j in range(nspin):
        # # if C_{p;ij} = C_{p;ji}
        # for j in range(i):
            if i == j:
                assert_raises(ValueError, lambda: ind_from_orbs_to_gem(i, j, nspin))
            else:
                gem_ind = ind_from_orbs_to_gem(i, j, nspin)
                assert gem_ind == counter
                assert ind_from_gem_to_orbs(gem_ind, nspin) == (i,j)
                counter += 1
