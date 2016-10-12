# Silence the elephant... for today. -------------------------------------------

import horton
horton.log.set_level(0)

# Write our demo wavefunction! -------------------------------------------------

import numpy as np

from geminals.proj import doci_hamiltonian, ProjectionWavefunction
from geminals.sd_list import doci_sd_list
from geminals import slater


class DemoWfn(ProjectionWavefunction):

    @property
    def template_coeffs(self):
        return np.eye(self.npair, self.nspatial)

    def compute_pspace(self, nsd):
        return doci_sd_list(self, nsd)

    def compute_hamiltonian(self, sd, deriv=None):
        return sum(doci_hamiltonian(self, sd, self.orb_type, deriv=deriv))

    def compute_overlap(self, sd, deriv=None):
        e_deriv = self.params.size - 1
        alpha, beta = slater.split_spin(sd, self.nspatial)
        alpha_occ = slater.occ_indices(alpha)
        beta_occ = slater.occ_indices(beta)
        coeffs = self.get_coeffs()
        if deriv is None or deriv == e_deriv:
            coeffs = coeffs[:, alpha_occ]
            return np.linalg.det(coeffs)
        elif isinstance(deriv, int):
            row_remove = deriv // self.nspatial
            col_remove = deriv % self.nspatial
            if col_remove in alpha_occ:
                rows = list(range(self.npair))
                rows.remove(row_remove)
                cols = list(alpha_occ)
                cols.remove(col_remove)
                deriv_coeffs = coeffs[rows, :][:, cols]
                if (not rows) or (not cols):
                    return 0.
                return np.linalg.det(deriv_coeffs)
            else:
                return 0.
        else:
            raise Exception

    def normalize(self):
        pass


# Run some calculations. -------------------------------------------------------

from geminals.hort import hartreefock


nelec = 4
hf_dict = hartreefock(fn="h4.xyz", basis="6-31g**", nelec=nelec)
E_hf = hf_dict["energy"]
H = hf_dict["H"]
G = hf_dict["G"]
nn = hf_dict["nuc_nuc"]

wfn = DemoWfn(nelec, H, G)
wfn.assign_nuc_nuc(nn)
wfn(method="root")
print("energy\n", wfn.compute_energy())
print("coeffs\n", wfn.params)
