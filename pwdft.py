import numpy as np
from scipy.special import erf, gamma
from scipy import linalg
from scipy.fft import fftn, ifftn

class Structure:
    def __init__(self, lattice, positions):
        self.lattice = lattice
        self.rec_lattice = 2 * np.pi * np.linalg.inv(lattice)
        self.frac_positions = positions
        self.cart_positions = positions @ lattice
        self.volume = np.abs(np.linalg.det(lattice))

    def __str__(self):
        strs = "\nSystem Setup\n"
        strs += f"Model volume: {self.volume:.4f}\n"
        strs += "Model lattice (Bohr):\n"
        for r in self.lattice:
            strs += " ".join(f"{x:12.6f}" for x in r) + "\n"
        strs += "Model rec.lat (Bohr^-1):\n"
        for r in self.rec_lattice:
            strs += " ".join(f"{x:12.6f}" for x in r) + "\n"
        strs += "Model coordinates (frac):\n"
        for r in self.frac_positions:
            strs += " ".join(f"{x:12.6f}" for x in r) + "\n"
        strs += "Model coordinates (cart):\n"
        for r in self.cart_positions:
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
        self.Ecutwfc = Ecut
        self.Ecutrho = 4 * Ecut
        self.model = model

        # FFT and G_vector setup
        Gmax = 2 * np.sqrt(2 * self.Ecutwfc)
        inv_lat_t = np.linalg.inv(self.model.rec_lattice.T)
        norm = np.ceil(np.linalg.norm(inv_lat_t, axis=1) * Gmax)
        self.fx = int(norm[0])
        self.fy = int(norm[1])
        self.fz = int(norm[2])
        grid_point = np.array([self.fx, self.fy, self.fz], dtype=int)
        self.grid_point = 2 * grid_point + 1
        self.num_grids = np.prod(self.grid_point)
        self.get_g_vectors()
        self.max_g2 = (self.g_rhos ** 2).sum(axis=1).max()
        self.max_g2w = (self.g_wfcs ** 2).sum(axis=1).max()
        
        # Kpoints
        self.kpoints = kpoints @ self.model.rec_lattice
        self.kweights = kweights

    def __str__(self):
        strs = "\nPlanewave Setup\n"
        strs += f"Cutoff Energy in planewave (Ry): {self.Ecutwfc}\n"
        strs += f"Cutoff Energy in density (Ry):   {self.Ecutrho}\n"
        strs += f"FFT Grid Size:                   {self.grid_point}\n"
        strs += f"Num g vectors in density:        {len(self.g_wfcs)}\n"
        strs += f"Max g2 vectors in density:       {self.max_g2}\n"
        strs += f"Num g vectors in planewave:      {len(self.g_rhos)}\n"
        strs += f"Max g2 vectors in planwwave:     {self.max_g2w}\n"
        strs += f"Num kpoints used:                {len(self.kpoints)}\n"
        for kpt, kw in zip(self.kpoints, self.kweights):
            strs += " ".join(f"{x:12.6f}" for x in kpt) 
            strs += f"   weight: {kw:12.6f}\n"
 
        return strs

    def get_g_vectors(self):
        """
        Compute g-vectors and 3D masks for the FFT grid.
        Note we have two cutoff values for electron density 
        and planewave expansions
        """
        g_rhos = [] 
        g_wfcs = []
        g_masks_r = np.zeros(self.grid_point, dtype=int)
        g_masks_w = np.zeros(self.grid_point, dtype=int)

        #for h in range(-self.fx, self.fx):
        #    for k in range(-self.fy, self.fy):
        #        for l in range(-self.fz, self.fz):
        for i in range(self.grid_point[0]):
            ii = i-self.grid_point[0]  if i > self.grid_point[0] // 2 else i
            for j in range(self.grid_point[1]):
                jj = j-self.grid_point[1] if j > self.grid_point[1] // 2 else j
                for k in range(self.grid_point[2]):
                    kk = k-self.grid_point[0]  if k > self.grid_point[2] // 2 else k
                    g = np.array([ii, jj, kk]) @ self.model.rec_lattice
                    g2 = np.sum(g**2)
                    if g2 <= 2*self.Ecutrho:
                        g_rhos.append(g)
                        g_masks_r[i, j, k] = 1 
                        if g2 <= 2*self.Ecutwfc:
                            g_wfcs.append(g)
                            g_masks_w[i, j, k] = 1

        self.g_rhos = np.array(g_rhos)
        self.g_wfcs = np.array(g_wfcs)
        self.g_masks_r = g_masks_r.astype(bool)
        self.g_masks_w = g_masks_w.astype(bool)
        #print(self.g_rhos[0], self.g_masks_r[0, 0, 0])
        #print(self.g_rhos[-1],self.g_masks_r[-1, -1, -1])
        #print("g_masks", g_masks_r.shape, self.g_masks_r.flatten()[:10]); import sys; sys.exit()

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
        num_gs = len(self.g_wfcs)
        num_ks = len(self.kweights)
        real_part = np.random.rand(num_states, num_gs) 
        imag_part = np.random.rand(num_states, num_gs) 
        psi = real_part + 1j * imag_part
        psi = self.orthonormalize(psi)

        [fx, fy, fz] = self.grid_point
        psi_3d = np.zeros([num_ks, num_states, fx, fy, fz], 
                          dtype=complex)
        for k in range(num_ks):
            for i in range(num_states):
                psi_3d[k, i, self.g_masks_w] += psi[i] 

        return psi_3d

    def compute_density(self, psi_iks, occs):
        """
        Compute electron density from wavefunction and occupations

        Args:
            psi_iks [num_kpts, fx, fy, fz]
            occs [num_kpts]
        """
        occs = np.array(occs)
        num_gs = len(self.g_wfcs)
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
        self.name = "Si"
        self.Z = Z
        self.rloc = rloc
        if len(cloc) < 4:
            self.cloc = np.pad(cloc, (0, 4 - len(cloc)), 'constant')
        else:
            self.cloc = cloc
        self.lmax = len(h) - 1
        self.proj = ['s', 'p', 'd', 'f'][:len(h)]
        self.rp = rp
        self.h = h

    def __str__(self):
        strs = "\nPseudopotential Setup\n"
        strs += f"Element:                  {self.name}\n"
        strs += f"Number of electrons:      {self.Z}\n"
        strs += f"local radius:             {self.rloc:.6f}\n"
        strs += f"local coefficients:   "
        for cloc in self.cloc:
            if abs(cloc) > 0:
                strs += f"{cloc:12.6f}"
            else:
                strs += "\n"
                break
        for l in range(self.lmax+1):
            proj, rp, h = self.proj[l], self.rp[l], self.h[l]
            strs += f"Nonlocal Projector {proj}: {rp:12.6f}\n"
            strs += f"Coupling matrix\n"
            for c in h:
                strs += "                      "
                strs += " ".join(f"{x:12.6f}" for x in c) + "\n"
        return strs

    def eval_v_local_r(self, r):
        """
        Evaluate the local pseudopotential in real space, eq. (12.3)
        """
        cloc = self.cloc
        rr = r / self.rloc
        return (-self.Z / r * erf(rr / np.sqrt(2))
                + np.exp(-rr**2 / 2) * (cloc[0] + 
                cloc[1] * rr**2 + cloc[2] * rr**4 + cloc[3] * rr**6))
    
    def eval_v_local_g(self, g):
        """
        Compute the local pseudopotential polynomial, eq. (12.6)
        """
        g = np.array(g)
        g2 = (g ** 2).sum()
        if g2 == 0:
            return 0

        rloc = self.rloc
        x2 = g2 * (rloc ** 2)
        Z = self.Z

        exp = np.exp(-0.5 * x2)
        term1 = -4 * np.pi * Z / g2 * exp

        P = (self.cloc[0]
             + self.cloc[1] * (3. - x2)
             + self.cloc[2] * (15. - 10. * x2 + x2**2)
             + self.cloc[3] * (105. - 105. * x2 + 21. * x2**2 - x2**3))
        term2 = np.sqrt(8 * np.pi **3) * rloc ** 3 * exp * P

        return term1 + term2

    def get_v_loc_r(self, pw):
        """
        Compute the local V_ps for the given structure

        Args:
            pw: planewave instance

        """
        grid_point = pw.grid_point
        vol = pw.model.volume
        g_masks = pw.g_masks_r
        g_vectors = pw.g_rhos
        v_loc_g = np.zeros(grid_point, dtype=complex)

        # Get the sum of v_loc_g
        pos = (pw.model.cart_positions).T
        sf = np.exp(1j*g_vectors @ pos).sum(axis=1)
        v_loc_g_1D = np.zeros(len(g_vectors), dtype=complex)
        for i, g in enumerate(g_vectors):
            v_loc_g_1D[i] = self.eval_v_local_g(g)
        v_loc_g[g_masks] = v_loc_g_1D#; print(v_loc_g.flatten()[:5])
        v_loc_g_1D *= sf / vol

        # convert to 3D
        v_loc_g[g_masks] = v_loc_g_1D
        #print(v_loc_g.flatten()[:5], g_masks.flatten()[:5])
        v_loc_r = np.zeros(grid_point, dtype=complex)
        v_loc_r = np.fft.ifftn(v_loc_g) * np.prod(grid_point)
        return v_loc_r



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
    lattice = 5.13155 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    positions = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    model = Structure(lattice, positions)
    print(model)

    pw = PlaneWaveBasis(model, 
                        Ecut=15.0, 
                        kpoints=np.array([[0, 0, 0],
                                          [0.25, 0.25, 0.25],
                                          [0, 0.5, 0],
                                          [0.5, 0, 0.5],
                                          ]), 
                        kweights=np.array([1., 6., 8., 12.])/27.
                        )
    print(pw)
    psi_g_3d = pw.random_guess(5)
    rho = pw.compute_density(psi_g_3d, [1, 1, 1, 1, 0]).sum()
    rho *= pw.model.volume/pw.num_grids
    print(f"Initial wavefunction:     {psi_g_3d.shape}")
    print(f"Number of electrons:      {rho}")

    psp = PspHgh(Z=4, rloc=0.44000000, 
                 cloc=np.array([-7.33610297, 0, 0, 0]), 
                 rp=np.array([0.42273813, 0.48427842]), 
                 h=np.array([[[5.90692831, -1.26189397],
                              [0.00000000, 3.25819622]],
                             [[2.72701346, 0.00000000],
                              [0.00000000, 0.00000000]]]))
    print(psp)
    print("v_local_g", psp.eval_v_local_r(1))
    v_loc_r = psp.get_v_loc_r(pw)
    print("v_local_r", v_loc_r.flatten()[:10])
    #g = [1, 0, 0]
    #print(f"v_nonlocal real (g={g})", psp.eval_v_local_g(g))


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
