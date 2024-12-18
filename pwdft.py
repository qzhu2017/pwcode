import numpy as np
from scipy.special import erf, gamma
from scipy import linalg
from scipy.fft import fftn, ifftn

class Structure:
    def __init__(self, lattice, positions):
        self.lattice = lattice
        self.rec_lattice = 2 * np.pi * np.linalg.inv(lattice)
        self.positions = positions
        self.volume = np.abs(np.linalg.det(lattice))

    def __str__(self):
        strs = "System Setup\n"
        strs += f"Model volume: {self.volume:.4f}\n"
        strs += "Model lattice (Bohr):\n"
        for r in self.lattice:
            strs += " ".join(f"{x:12.6f}" for x in r) + "\n"
        strs += "Model rec.lat (Bohr^-1):\n"
        for r in self.rec_lattice:
            strs += " ".join(f"{x:12.6f}" for x in r) + "\n"
        strs += "Model coordinates:\n"
        for r in self.positions:
            strs += " ".join(f"{x:12.6f}" for x in r) + "\n"
        return strs

    
class PlaneWaveBasis:
    def __init__(self, model, Ecut, kpoints, kweights):
        """
        Parameters:
            model: The structure class instance.
            Ecut: float, The energy cutoff for the plane-wave basis.
            kpoints: array, The k-points used in the calculation.
            kweights: array, The weights of the k-points.
        """
        # System input
        self.Ecut = Ecut
        self.model = model

        # FFT and G_vector setup
        Gmax = 2 * np.sqrt(2 * self.Ecut)
        inv_lat_t = np.linalg.inv(self.model.rec_lattice.T)
        norm = np.ceil(np.linalg.norm(inv_lat_t, axis=1) * Gmax)
        self.fx = int(norm[0])
        self.fy = int(norm[1])
        self.fz = int(norm[2])
        grid_point = np.array([self.fx, self.fy, self.fz], dtype=int)
        self.grid_point = 2 * grid_point + 1
        self.num_grids = np.prod(self.grid_point)
        self.get_g_vectors()
        
        # Kpoints
        self.kpoints = kpoints
        self.kweights = kweights

    def __str__(self):
        strs = "Planewave Setup\n"
        strs += f"Cutoff Energy (Ry):       {self.Ecut}\n"
        strs += f"FFT Grid Size:            {self.grid_point}\n"
        strs += f"Number of used g vectors: {len(self.gs)}\n"
        return strs

    def get_g_vectors(self):
        """
        Compute g-vectors and 3D masks for the FFT grid.
        """
        Gs = []
        masks = np.zeros(self.grid_point)

        for h in range(-self.fx, self.fx):
            for k in range(-self.fy, self.fy):
                for l in range(-self.fz, self.fz):
                    g = np.array([h, k, l]) @ self.model.rec_lattice
                    if np.sum(g**2) <= 2*self.Ecut:
                        Gs.append(g)
                        masks[h+self.fx, k+self.fy, l+self.fz] = 1 
        self.gs = np.array(Gs)
        self.masks = masks.astype(bool)

    def orthonormalize(self, psi):
        """
        Make the wavefunction be orthonormal
        """
        psi_sqrt = linalg.sqrtm(np.conj(psi) @ psi.T)
        return linalg.inv(psi_sqrt).T @ psi

    def random_guess(self, num_states):
        """
        Random guess of wavefunction given the number of occupied states
        """
        num_gs = len(self.gs)
        real_part = np.random.rand(num_states, num_gs) 
        imag_part = np.random.rand(num_states, num_gs) 
        psi = real_part + 1j * imag_part
        psi = self.orthonormalize(psi)

        [fx, fy, fz] = self.grid_point
        psi_3d = np.zeros([num_states, fx, fy, fz], dtype=complex)
        for i in range(num_states):
            psi_3d[i, self.masks] += psi[i] 

        return psi_3d

    def compute_density(self, psi_iks, occs):
        """
        Compute electron density from wavefunction and occupations

        Args:
            psi_iks [num_kpts, fx, fy, fz]
            occs [num_kpts]
        """
        occs = np.array(occs)
        num_gs = len(self.gs)
        vol = self.model.volume
        rho = np.zeros(self.grid_point) 

        for kid, kw in enumerate(self.kweights):
            for i, occ in enumerate(occs):
                psi_r = np.fft.ifftn(psi_iks[kid][i])
                psi_r *= np.sqrt(self.num_grids / vol)
                rho_r = np.real(psi_r * np.conj(psi_r))
                rho_r /= np.sum(rho_r)
                rho_r *= 2 * occ * kw * self.num_grids / vol
                rho += rho_r

        return rho

class PspHgh:
    """
    A class to represent a pseudopotential in the HGH form.
    Modification of the original code from DFTK.jl:
    https://github.com/JuliaMolSim/DFTK.jl/blob/master/src/pseudo/PspHgh.jl
    Equations are taken from the original paper:
    Hartwigsen, Goedecker and Hutter. Phys. Rev. B, 58, 3641, 1998
    
    Attributes
    ----------
        Z : float, The ionic charge.
        rloc : float, The local pseudopotential radius.
        cloc : array-like, The coefficients for the local pseudopotential polynomial.
        rp : array-like, The projector radii.
        h : array-like, The projector coefficients.

    Methods
    -------
        v_local_r(r): Evaluates V_loc in real space.
        v_nonlocal_r(r): Evaluates V_nonloc in real space.
        v_local_g(g): Evaluates V_loc in reciprocal space.
        v_nonlocal_g(g): Evaluates V_nonloc in reciprocal space.
    """

    def __init__(self, Z, rloc, cloc, rp, h):
        self.Z = Z
        self.rloc = rloc
        if len(cloc) < 4:
            self.cloc = np.pad(cloc, (0, 4 - len(cloc)), 'constant')
        else:
            self.cloc = cloc
        self.lmax = len(h) - 1
        self.rp = rp
        self.h = h

    def eval_v_local_r(self, r):
        """
        Evaluate the local pseudopotential in real space, eq. (12.5)
        """
        cloc = self.cloc
        rr = r / self.rloc
        return (-self.Z / r * erf(rr / np.sqrt(2))
                + np.exp(-rr**2 / 2) * (cloc[0] + 
                cloc[1] * rr**2 + cloc[2] * rr**4 + cloc[3] * rr**6))
    
    def eval_v_local_g(self, g2):
        """
        Compute the local pseudopotential polynomial, eq. (12.2)
        """
        if g2 == 0:
            return 0
        rloc = self.rloc
        gr_loc2 = g2 * (r_loc ** 2)
        Z = self.Z
        exp = np.exp(-0.5 * gr_loc2)
        term1 = -4 * np.pi * Z / g2 * exp

        P = (self.cloc[0]
             + self.cloc[1] * (3 - t**2)
             + self.cloc[2] * (15 - 10 * t**2 + t**4)
             + self.cloc[3] * (105 - 105 * t**2 + 21 * t**4 - t**6))
        term2 = np.sqrt(8 * np.pi **3) * rloc ** 3 * exp * P

        return term1 + term2
 
    def eval_v_nonlocal_g(self, i, l):
        """
        Compute the projector polynomial, eq. (12.4)

        Args:
        i : int, The projector index.
        l : int, The angular momentum.
        """

        assert 0 <= l <= len(self.rp) - 1
        assert i > 0
        rp = self.rp[l]
        common = 4 * np.pi**(5 / 4) * np.sqrt(2**(l + 1) * rp**3)

        if l == 0 and i == 1:
            return common
        if l == 0 and i == 2:
            return common * 2 / np.sqrt(15) * (3 - t**2)
        if l == 0 and i == 3:
            return common * 4 / 3 / np.sqrt(105) * (15 - 10 * t**2 + t**4)
        if l == 1 and i == 1:
            return common * t / np.sqrt(3)
        if l == 1 and i == 2:
            return common * 2 / np.sqrt(105) * t * (5 - t**2)
        if l == 1 and i == 3:
            return common * 4 / 3 / np.sqrt(1155) * t * (35 - 14 * t**2 + t**4)
        if l == 2 and i == 1:
            return common * t**2 / np.sqrt(15)
        if l == 2 and i == 2:
            return common * 2 / 3 / np.sqrt(105) * t**2 * (7 - t**2)
        if l == 3 and i == 1:
            return common * t**3 / np.sqrt(105)

    def get_v_loc_r(self, pw):
        grid_point = pw.grid_point
        vol = pw.model.volume
        g_vector_mask = pw.g_vector_mask
        v_loc_r = np.zeros(grid_point, dtype=np.complex128)
        v_loc_g = np.zeros(grid_point, dtype=np.complex128)

        # Sum over all grid point
        for i in range(grid_point[0]):
            for j in range(grid_point[1]):
                for k in range(grid_point[2]):
                    if pw.masks[i, j, k]:
                        g2 = np.sum(pw.g_vector[i, j, k] ** 2)
                        v_loc_g[i, j, k] = self.eval_v_loc_g(g2)
        # v_loc_g * sf
        v_loc_g *= sf / vol
        # FFT from g to r
        v_loc_r = np.fft.ifftn(v_loc_g) * np.prod(grid_point)
        return v_loc_r

    def get_v_nloc(self, psi_g, pw):
        #grid_point = pw.grid_point

        #for psi in psi_g:
        #    v = np.zeros(grid_point, dtype=np.complex128)
        #    for l in range(0, self.lmax):
        #        for m in range(-l, l+1):
        #            for iprj in range():
        #                for jprj in range():
        #                    v += self. * beta_nlm[i,l,m+1,iprj] * 
        pass

    def get_vnloc_g(self, pw):
        #grid_point = pw.grid_point
        #vol = pw.model.volume
        #coords = pw.model.coordinates

        #for i in range(len(coord)):
        #    for l in range(self.lmax):
        #        for m in range(-l, l+1):
        #            for proj in self.:
        #                for ix in range(grid_point[0]):
        #                    for iy in range(grid_point[1]):
        #                        for iz in range(grid_point[2]):
        #                            if pw.masks[ix, iy, iz]:
        #                                g = pw.gs[ix, iy, iz]
        #                                gm = np.linalg.norm(g)
        #                                gx = np.sum(coords[i] * g)
        #                                sf = np.exp(1j*gx)
        #                                term[ix, iy, iz] = ylm_real(l, m, g) * \ 
        #                                eval_proj_g(ps, l, iprj, gm, vol) * sf
        pass
                        

if __name__ == "__main__":

    np.random.seed(42)
    lattice = 5.23 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    positions = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    model = Structure(lattice, positions)
    print(model)

    pw = PlaneWaveBasis(model, 
                        Ecut=15.0, 
                        kpoints=np.array([0, 0, 0]), 
                        kweights=np.array([1.0]))
    print(pw)
    psi_g_3d = pw.random_guess(5)
    rho = pw.compute_density([psi_g_3d], [1, 1, 1, 1, 0]).sum()
    rho *= pw.model.volume/pw.num_grids
    print(f"Initial wavefunction:     {psi_g_3d.shape}")
    print(f"Number of electrons:      {rho}")

    print("\nPseudopotential")
    psp = PspHgh(Z=4, rloc=np.array([0.44000000]), 
                 cloc=np.array([-7.33610297, 0, 0, 0]), 
                 rp=np.array([0.42273813, 0.48427842]), 
                 h=np.array([[[5.90692831, -1.26189397],
                              [0.00000000, 3.25819622]],
                             [[2.72701346, 0.00000000],
                              [0.00000000, 0.00000000]]]))
    print("v_local    real", psp.v_local_r(1))
    print("v_nonlocal real", psp.v_nonlocal_r(1, 1, 1))


#def Hamiltonian:
#    """
#    Construct the Hamiltonian
#    """
#    def __init__():
#        self.V
#
#    def update(self, psi):
#        V_xc_r = 
#        V_loc_
#
#    
#    def V_external(self):
#        
#    def V_XC(self):
#        return -1. 
#
#    def V_hartree(self)
#
#    V_ext = 
#    T = 
#    V_hartree =
#    V_xc = 
#    return energies, ham
#
#    def operator():
#        H + 
#    def diag(self, N_):
#        
#
#
#class Density:
#    def __init__(self, basis):
#        """
#        Initialize the Density class with the given basis.
#        
#        Parameters:
#            basis: Object containing plane wave basis details.
#        """
#        self.basis = basis
#
#    def random_density(self, n_electrons):
#        rho_total = np.random.rand(*self.basis.fft_size)
#
#    def random_density(self, n_electrons):
#        """
#        Generate a random charge density normalized to the given number of electrons.
#        
#        Parameters:
#            n_electrons: Number of electrons to normalize the density.
#        
#        Returns:
#            rho_total: The generated random charge density.
#        """
#        # Generate random total density
#        rho_total = np.random.rand(*self.basis.fft_size)
#        
#        # Normalize to integrate to n_electrons
#        rho_total *= n_electrons / (np.sum(rho_total) * self.basis.dv)
#
#        return rho_total
#
#    def compute_density(self, psi, occupation):
#        """
#        Compute the electron density in a plane wave basis.
#        
#        Parameters:
#            psi: List of wavefunctions for each k-point.
#            occupation: List of occupation numbers for each wavefunction.
#        
#        Returns:
#            rho: Computed charge density as a 3D array.
#        """
#        rho = np.zeros(self.basis.fft_size)
#
#        for ik, psi_k in enumerate(psi):
#            kpt = self.basis.kpoints[ik]
#            # Find the maximum index of non-zero elements in occupation
#            nmax = np.max(np.nonzero(occupation[ik]))
#            # Perform inverse FFT to compute real-space wavefunction
#            psi_real = ifftn(psi[ik][:, nmax].reshape(
#                self.basis.fft_size), norm="backward")
#
#            # Accumulate density contributions
#            norm_factor = (self.basis.ifft_normalization) ** 2
#            rho += (
#                occupation[ik][nmax]
#                * self.basis.kweights[ik]
#                * norm_factor
#                * np.abs(psi_real) ** 2
#            )
#
#        return rho
#
#    def compute_kinetic_energy(self, psi, occupation):
#        """
#        Compute the kinetic energy density in a plane wave basis.
#
#        Parameters:
#            psi: List of wavefunctions for each k-point.
#            occupation: List of occupation numbers for each wavefunction.
#
#        Returns:
#            tau: Computed kinetic energy density as a 3D array.
#        """
#        tau = np.zeros((*self.basis.fft_size, self.basis.model.n_spin_components))
#
#        for ik, kpt in enumerate(self.basis.kpoints):
#            # Retrieve G + k vectors for the current k-point
#            G_plus_k = [
#                np.array([p[alpha] for p in self.basis.Gplusk_vectors_cart(kpt)])
#                for alpha in range(3)
#            ]
#
#            for n in range(psi[ik].shape[1]):  # Loop over wavefunctions
#                for alpha in range(3):  # Loop over Cartesian components
#                    # Apply the gradient operator in reciprocal space and perform inverse FFT
#                    d_alpha_psi_real = ifftn(
#                        1j * G_plus_k[alpha] * psi[ik][:, n].reshape(self.basis.fft_size),
#                        norm="backward"
#                    )
#
#                    # Update kinetic energy density
#                    tau[..., kpt.spin] += (
#                        occupation[ik][n]
#                        * self.basis.kweights[ik]
#                        / 2
#                        * np.abs(d_alpha_psi_real) ** 2
#                    )
#
#        return tau
#
#def SCF(basis, rho, maxiter=100, tol=1e-6, damping=0.8):
#    
#    for it in range(max_iter):
#        ψ, occupation, eigenvalues, εF, n_iter, converged, timedout = info
#        n_iter += 1
#
#        H = energy_hamiltonian(basis, ψ, occupation, ρ=ρin, eigenvalues=eigenvalues, εF=εF)
#
#        nextstate = next_density(ham, nbandsalg, fermialg, eigensolver=eigensolver, ψ=ψ, eigenvalues=eigenvalues,
#                                 occupation=occupation, miniter=1, tol=determine_diagtol(diagtolalg, info))
#        ψ, eigenvalues, occupation, εF, ρout = nextstate
#        Δρ = ρout - ρin
#        n_matvec = info['n_matvec'] + nextstate['n_matvec']
#
#        if compute_consistent_energies:
#            energies = energy(basis, ψ, occupation, ρ=ρout, eigenvalues=eigenvalues, εF=εF)
#        
#        rhonext = rhoin + damping * mix_density(mixing, basis, drho)
#
#        converged = n_iter >= miniter and is_converged(info_next)
#        converged = MPI.COMM_WORLD.bcast(converged, root=0)
#        info_next.update({'converged': converged})
#
#        timedout = MPI.COMM_WORLD.bcast(datetime.now() >= timeout_date, root=0)
#        info_next.update({'timedout': timedout})
#
#        callback(info_next)
#
#        return ρnext, info_next
#
#    _, info = solver(fixpoint_map, ρ, info_init, maxiter=maxiter)
#    energies, ham = energy_hamiltonian(basis, phi, occupation, ρ=ρout, eigenvalues=eigenvalues, εF=εF)
#
#    return scfres
#
#
#def V_xc(term, basis, ψ, occupation, ρ, τ=None):
#    return E, potential, Vτ
#
#def V_hartree
#    return V_H

   
# Example
    #print("\nTest density")
    #density = Density(pw)
    #rho0 = density.random_density(8)
    #print("random_density", rho0.shape)
    ##rho1 = density.compute_density([rho0], [1])
    ##print("\ncompute_density", rho1)
    ##tau = density.compute_kinetic_energy([rho0], [1])
    ##print("\ncompute_kinetic_energy", tau) 
