# Set up modules ---------------------------------------------------------------

# Silence the elephant... for today
import horton
horton.log.set_level(0)

# All math is simple NumPy
import numpy as np

# Import assets for building wavefunctions
from geminals.proj import doci_hamiltonian, ProjectionWavefunction
from geminals.sd_list import doci_sd_list

# Import Slater determinant "bitstring" module
from geminals import slater

# Write our demo wavefunction! -------------------------------------------------

class DemoWfn(ProjectionWavefunction):

    # Required; describe the structure of the wfn coefficients
    @property
    def template_coeffs(self):
        template = np.eye(self.npair, self.nspatial) 
        template[:, self.npair:] = 2. * (np.random.rand(self.npair, self.nspatial - self.npair) - 0.5)
        return template

    # Required; use the DOCI projection space
    def compute_pspace(self, nsd):
        return doci_sd_list(self, nsd)

    # Required; use the DOCI Hamiltonian
    def compute_hamiltonian(self, sd, deriv=None):
        return sum(doci_hamiltonian(self, sd, self.orb_type, deriv=deriv))

    # Required; compute the determinant of the matrix with included columns
    # corresponding to doubly-occupied spatial orbitals
    def compute_overlap(self, sd, deriv=None):

        # Which `deriv` is the energy?  The last one.
        e_deriv = self.params.size - 1

        # Get occupied indices, check for double occupations only
        alpha, beta = slater.split_spin(sd, self.nspatial)
        alpha_occ = slater.occ_indices(alpha)
        beta_occ = slater.occ_indices(beta)
        assert alpha_occ == beta_occ

        # Compute the overlap
        coeffs = self.get_coeffs()
        if deriv is None or deriv == e_deriv:
            coeffs = coeffs[:, alpha_occ]
            return np.linalg.det(coeffs)

        # Compute the derivative of the overlap wrt a matrix element
        elif isinstance(deriv, int):
            row_remove = deriv // self.nspatial
            col_remove = deriv % self.nspatial
            if col_remove in alpha_occ:
                rows = list(range(self.npair))
                rows.remove(row_remove)
                cols = list(alpha_occ)
                cols.remove(col_remove)
                deriv_coeffs = coeffs[rows, :][:, cols]
                # Emptied the matrix; return zero
                if (not rows) or (not cols):
                    return 0.
                return np.linalg.det(deriv_coeffs)
                # Derivatizing wrt a constant; return zero
            else:
                return 0.

        else:
            raise Exception("Something bad happened :(")

    # Required; normalizes the wavefunction. To disable normalization, just have
    # the function "pass".
    def normalize(self):
        pass

# Run some calculations. -------------------------------------------------------

# Get our HORTON Hartree-Fock wrapper
from geminals.hort import hartreefock

# Run RHF to get MO Hamiltonians
nelec = 4
hf_dict = hartreefock(fn="h4.xyz", basis="3-21g", nelec=nelec)
H = hf_dict["H"]
G = hf_dict["G"]
nuc_nuc = hf_dict["nuc_nuc"]

# Instantiate an instance of our wfn
wfn = DemoWfn(nelec, H, G, nuc_nuc=nuc_nuc)

# Optimize it
wfn()

# Print stuff
energy = wfn.compute_energy(include_nuc=1)
coeffs = wfn.get_coeffs()
print("demo_wfn energy", energy)
print("demo_wfn coeffs", coeffs)

# ------------------------------------------------------------------------------
