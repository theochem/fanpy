from . import slater
from itertools import combinations, product

"""
Functions
---------
ci_sd_list
    Generates a list of Slater determinants that corresponds to the given order of excitations
doci_sd_list
    Generates a list of doubly occupied Slater determinants that corresponds to the given order of
    (double) excitations

"""


def ci_sd_list(self, num_limit, exc_orders=[]):
    """ Generates Slater determinants

    Number of Slater determinants is limited by num_limit. First Slater determinant is the ground
    state, next are the first excitations from exc_orders, then second excitation from
    exc_orders, etc

    Parameters
    ----------
    self : instance of Wavefunction
        Any instance that contains nspatial and nelec
    num_limit : int
        Maximum number of Slater determinants to be generated
    exc_orders : list of int
        Orders of excitations that is to be included (with respect to ground state
        Slater determinant)
        Slater determinants of the first excitation will be included first, second excitations
        included second, etc

    Returns
    -------
    civec : list of ints
        Integer that describes the occupation of a Slater determinant as a bitstring

    Note
    ----
    Crashes if order of excitation is less than 0
    """
    nspatial = self.nspatial
    nelec = self.nelec
    civec = []
    if exc_orders == []:
        exc_orders = range(1, nelec + 1)
    # assert all(i>0 for i in exc_orders), 'Excitation orders must be greater than 0'

    # ASSUME: certain structure for civec
    # spin orbitals are ordered by energy
    ground = slater.ground(nelec, 2 * nspatial)
    civec.append(ground)

    occ_indices = slater.occ_indices(ground)
    vir_indices = slater.vir_indices(ground, 2 * nspatial)
    # order by energy
    occ_indices = sorted(occ_indices, key=lambda x: x - nspatial if x >= nspatial else x, reverse=True)
    vir_indices = sorted(vir_indices, key=lambda x: x - nspatial if x >= nspatial else x)

    count = 1
    for nexc in exc_orders:
        occ_combinations = combinations(occ_indices, nexc)
        vir_combinations = combinations(vir_indices, nexc)
        for occ, vir in product(occ_combinations, vir_combinations):
            sd = slater.annihilate(ground, *occ)
            sd = slater.create(sd, *vir)
            civec.append(sd)
            count += 1
            if count >= num_limit:
                return civec[:num_limit]
    else:
        return civec[:num_limit]


def doci_sd_list(self, num_limit, exc_orders=[]):
    """ Generates doubly occupied (DOCI) Slater determinants

    Number of Slater determinants is limited by num_limit. First Slater determinant is the ground
    state, next are the first excitations from exc_orders, then second excitation from
    exc_orders, etc


    DOCI Slater determinants are all pairwise (alpha beta) excitations of the ground state
    singlet Slater determinant

    Parameters
    ----------
    self : instance of Wavefunction
        Any instance that contains nspatial, nelec, and npair
    num_limit : int
        Maximum number of Slater determinants to be generated
    exc_orders : list of int
        Orders of excitations that is to be included (with respect to ground state
        Slater determinant)
        Slater determinants of the first excitation will be included first, second excitations
        included second, etc

    Returns
    -------
    civec : list of ints
        Integer that describes the occupation of a Slater determinant as a bitstring

    Note
    ----
    Crashes if order of excitation is less than 0
    """
    nspatial = self.nspatial
    nelec = self.nelec
    npair = self.npair
    civec = []
    if exc_orders == []:
        exc_orders = range(1, npair + 1)
    # assert all(i>0 for i in exc_orders), 'Excitation orders must be greater than 0'
    # ASSUME: certain structure for civec
    # spin orbitals are ordered by energy
    ground = slater.ground(nelec, 2 * nspatial)
    civec.append(ground)

    count = 1
    for nexc in exc_orders:
        # this part assumes that the ground state has the first N electrons occupied
        occ_combinations = combinations(reversed(range(npair)), nexc)
        vir_combinations = combinations(range(npair, nspatial), nexc)
        for occ, vir in product(occ_combinations, vir_combinations):
            occ = [i for i in occ] + [i + nspatial for i in occ]
            vir = [a for a in vir] + [a + nspatial for a in vir]
            sd = slater.annihilate(ground, *occ)
            sd = slater.create(sd, *vir)
            civec.append(sd)
            count += 1
            if count >= num_limit:
                return civec[:num_limit]
    else:
        return civec[:num_limit]

def apig_doubles_sd_list(self):
    """ Generates doubly excited (DOCI) Slater determinants

    DOCI Slater determinants are all pairwise (alpha beta) excitations of the ground state
    singlet Slater determinant

    Parameters
    ----------
    self : instance of Wavefunction
        Any instance that contains nspatial, nelec, and npair

    Returns
    -------
    civec : list of ints
        Integer that describes the occupation of a Slater determinant as a bitstring

    """
    
    nspatial = self.nspatial
    nelec = self.nelec
    npair = self.npair
    civec = []
    
    # ASSUME: certain structure for civec
    # spin orbitals are ordered by energy
    ground = slater.ground(nelec, 2 * nspatial)
    civec.append(ground)

    # this part assumes that the ground state has the first N electrons occupied
    occ_combinations = combinations(reversed(range(npair)), 1)
    vir_combinations = combinations(range(npair, nspatial), 1)
    
    for occ, vir in product(occ_combinations, vir_combinations):
        occ = [i for i in occ] + [i + nspatial for i in occ]
        vir = [a for a in vir] + [a + nspatial for a in vir]
        sd = slater.annihilate(ground, *occ)
        sd = slater.create(sd, *vir)
        civec.append(sd)
    
    return civec

def apsetg_doubles_sd_list(self):
    """ Generates doubly excited (fullCI) Slater determinants with closed shell
    occupied orbitals and open shell vitual orbitals within the disjoint set
    approximation 

    Parameters
    ----------
    self : instance of Wavefunction
        Any instance that contains nspatial and nelec

    Returns
    -------
    civec : list of ints
        Integer that describes the occupation of a Slater determinant as a bitstring

    """
    
    nspatial = self.nspatial
    nelec = self.nelec
    civec = []

    # ASSUME: certain structure for civec
    # spin orbitals are ordered by energy
    ground = slater.ground(nelec, 2 * nspatial)
    civec.append(ground)

    occ_indices = slater.occ_indices(ground)
    vir_indices = slater.vir_indices(ground, 2 * nspatial)
    
    # order by energy
    occ_indices = sorted(occ_indices, key=lambda x: x - nspatial if x >= nspatial else x, reverse=True)
    vir_indices = sorted(vir_indices, key=lambda x: x - nspatial if x >= nspatial else x)
    
    # Spit virtuals into alpha and beta
    vir_alpha_indices = vir_indices[:len(vir_indices) // 2]
    vir_beta_indices = vir_indices[len(vir_indices) // 2:]

    occ_combinations = combinations(occ_indices, 1)
    vir_alpha_combinations = combinations(vir_alpha_indices, 1)
    vir_beta_combinations = combinations(vir_beta_indices, 1)
    vir_combinations = zip(vir_alpha_combinations, vir_beta_combinations)

    for occ, vir in product(occ_combinations, vir_combinations):
        occ = [i for i in occ] + [i + nspatial for i in occ]
        sd = slater.annihilate(ground, *occ)
        sd = slater.create(sd, *vir[0])
        sd = slater.create(sd, *vir[1])
        # Place holder because we make garbage determinants
        if sd is None:
            continue
        civec.append(sd)

    return civec

def apg_doubles_sd_list(self):
    """ Generates doubly excited (fullCI) Slater determinants with closed shell
    occupied orbitals and open shell vitual orbitals 

    Parameters
    ----------
    self : instance of Wavefunction
        Any instance that contains nspatial and nelec

    Returns
    -------
    civec : list of ints
        Integer that describes the occupation of a Slater determinant as a bitstring

    """
    
    nspatial = self.nspatial
    nelec = self.nelec
    civec = []

    # ASSUME: certain structure for civec
    # spin orbitals are ordered by energy
    ground = slater.ground(nelec, 2 * nspatial)
    civec.append(ground)

    occ_indices = slater.occ_indices(ground)
    vir_indices = slater.vir_indices(ground, 2 * nspatial)
   
   # order by energy
    occ_indices = sorted(occ_indices, key=lambda x: x - nspatial if x >= nspatial else x, reverse=True)
    vir_indices = sorted(vir_indices, key=lambda x: x - nspatial if x >= nspatial else x)

    occ_combinations = combinations(occ_indices, 1)
    vir_combinations = combinations(vir_indices, 2)

    for occ, vir in product(occ_combinations, vir_combinations):
        occ = [i for i in occ] + [i + nspatial for i in occ]
        sd = slater.annihilate(ground, *occ)
        sd = slater.create(sd, *vir)
        # Place holder because we make garbage determinants
        if sd is None:
            continue
        civec.append(sd)

    return civec
