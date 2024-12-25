import numpy as np
from scipy.special import erf, gamma, sph_harm

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
    def __init__(self, model, Ecut, kpoints, kweights, occs):
        """
        Parameters:
            model: The structure class instance.
            Ecut: float, The energy cutoff for the plane-wave basis.
            kpoints: array, The k-points used in the calculation.
            kweights: array, The weights of the k-points.
            occs: array, The list of occupations
        """
        # System input
        self.Ecutwfc = Ecut
        self.Ecutrho = 4 * Ecut
        self.model = model
        self.occs = occs

        # Kpoints
        self.kpoints = kpoints @ self.model.rec_lattice
        #self.kpoints = np.array([[0., 0., 0.],
        #                         [-0.0000000000,   0.4081408351,  -0.0000000000],
        #                         [-0.2040704176,   0.2040704176,  -0.2040704176],
        #                         [ 0.4081408351,  -0.0000000000,   0.4081408351]])
 
        self.kweights = kweights
        self.n_kpts = len(self.kpoints)

        # FFT and G_vector setup
        Gmax = 2 * np.sqrt(2 * self.Ecutwfc)
        inv_lat_t = np.linalg.inv(self.model.rec_lattice.T)
        norm = np.ceil(np.linalg.norm(inv_lat_t, axis=1) * Gmax)
        self.fx = int(norm[0])
        self.fy = int(norm[1])
        self.fz = int(norm[2])
        grids = np.array([self.fx, self.fy, self.fz], dtype=int)
        self.grids = 2 * grids + 1
        self.num_grids = np.prod(self.grids)
        self.get_g_vectors()
        self.num_gs = len(self.g_rhos)
        self.num_gws = [len(g) for g in self.g_wfcs]
        self.max_g2 = (self.g_rhos**2).sum(axis=1).max()
        self.max_g2w = [(g**2).sum(axis=1).max() for g in self.g_wfcs]
        

    def __str__(self):
        strs = "\nPlanewave Setup\n"
        strs += f"Cutoff Energy in planewave (Ry): {self.Ecutwfc}\n"
        strs += f"Cutoff Energy in density (Ry):   {self.Ecutrho}\n"
        strs += f"FFT Grid Size:                   {self.grids}\n"
        strs += f"Num g vectors in density:        {self.num_gs}\n"
        strs += f"Num g vectors in planewave:      {self.num_gws}\n"
        strs += f"Max g2 vectors in density:     "
        strs += f"{self.max_g2:10.4f}" 
        strs += f"\nMax g2 vectors in planwwave:  "
        strs += " ".join(f"{x:10.4f}" for x in self.max_g2w) 
        strs += "\n"
        strs += f"Num kpoints used:                {self.n_kpts}\n"
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
        n_kpts = self.n_kpts
        [gx, gy, gz] = self.grids

        g_rhos = [] 
        g_wfcs = [[] for _ in range(n_kpts)]
        g_masks_r = np.zeros([gx, gy, gz], dtype=int)
        g_masks_w = np.zeros([n_kpts, gx, gy, gz], dtype=int)

        # a bit funny sorting (0, 1, 2, .., N, N-1, N-2, .... -1)
        for i in range(gx):
            ii = i - gx if i > gx // 2 else i
            for j in range(gy):
                jj = j - gy if j > gy // 2 else j
                for k in range(gz):
                    kk = k - gz if k > gz // 2 else k
                    g = np.array([ii, jj, kk]) 
                    g = g @ self.model.rec_lattice
                    if np.sum(g**2) <= 2 * self.Ecutrho:
                        g_rhos.append(g)
                        g_masks_r[i, j, k] = 1 
                        for l in range(n_kpts):
                            g1 = g + self.kpoints[l]
                            if np.sum(g1**2) <= 2 * self.Ecutwfc:
                                g_wfcs[l].append(g1)
                                g_masks_w[l, i, j, k] = 1
        self.g_wfcs = [np.array(g_wfc) for g_wfc in g_wfcs]
        self.g_rhos = np.array(g_rhos)
        self.g_masks_r = g_masks_r.astype(bool)
        self.g_masks_w = g_masks_w.astype(bool)

    def orthonormalize(self, psi):
        """
        Make the wavefunction be orthonormal
        """
        psi_sqrt = linalg.sqrtm(np.conj(psi) @ psi.T)
        return linalg.inv(psi_sqrt).T @ psi

    def random_guess(self):
        """
        Random wavefunction from the number of occupied states
        """
        n_states = len(self.occs)
        n_kpts = self.n_kpts
        psi_1d = [[] for _ in range(n_kpts)]

        for ik in range(n_kpts):
            n_gs = len(self.g_wfcs[ik])
            real_part = np.random.rand(n_states, n_gs) 
            imag_part = np.random.rand(n_states, n_gs) 
            psi = real_part + 1j * imag_part
            psi = self.orthonormalize(psi)
            psi_1d[ik] = psi
            print("create wavefunctions", ik, psi_1d[ik].shape)
        self.psi_1d = psi_1d
        self.get_psi_3d()
        self.get_rho_r()

    def get_psi_3d(self):
        """
        get psi in 3d grids format
        """
        [gx, gy, gz] = self.grids
        n_kpts = self.n_kpts
        psi_3d = [[] for _ in range(n_kpts)]
        for ik in range(n_kpts):
            n_states = len(self.psi_1d[ik])
            mask = self.g_masks_w[ik]
            psi_3d[ik] = np.zeros([n_states, gx, gy, gz],
                                   dtype=complex)
            for i in range(n_states):
                psi_3d[ik][i][mask] += self.psi_1d[ik][i] 
        self.psi_3d = psi_3d

    def get_rho_r(self):
        """
        Get electron density in real space
        """
        num_gs = len(self.g_wfcs)
        vol = self.model.volume
        rho = np.zeros(self.grids) 

        for ik, kw in enumerate(self.kweights):
            for i, occ in enumerate(self.occs):
                psi_r = np.fft.ifftn(self.psi_3d[ik][i])
                psi_r *= np.sqrt(self.num_grids / vol)
                rho_r = np.real(psi_r * np.conj(psi_r))
                rho_r /= np.sum(rho_r)
                rho_r *= 2 * occ * kw * self.num_grids / vol
                rho += rho_r
        self.rho_r = rho

class PspHgh:
    """
    A class to represent a pseudopotential in the HGH form.
    Modification of the original code from DFTK.jl:
    https://github.com/JuliaMolSim/DFTK.jl
    Equations are taken from the original paper:
    Hartwigsen, Goedecker and Hutter. Phys. Rev. B, 58, 3641, 1998
    
    Parameters:
        Z : float, The ionic charge.
        rloc : float, The local pseudopotential radius.
        cloc : array-like, The coefficients for the local pp
        rp : array-like, The projector radii.
        h : array-like, The projector coefficients.
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
        self.ilm_indices = [(1, 0, 0),
                            (2, 0, 0),
                            (1, 1, -1),
                            (1, 1, 0),
                            (1, 1, 1)]
        self.num_ilms = len(self.ilm_indices)

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
        g2 = (g ** 2).sum(axis=1)
        V = np.zeros(len(g2), dtype=complex)

        # only deal with non-zero g vectors
        ids = g2 > 1e-8
        ids0 = g2 <= 1e-8
        g2 = g2[ids]

        rloc = self.rloc
        x2 = g2 * (rloc ** 2)
        Z = self.Z

        exp = np.exp(-0.5 * x2)

        term1 = -4 * np.pi * Z / g2 * exp
        
        P = (self.cloc[0]
             + self.cloc[1] * (3. - x2)
             + self.cloc[2] * (15. - 10. * x2 + x2**2)
             + self.cloc[3] * (105. - 105. * x2 + 21. * x2**2 - x2**3))
        term2 = np.sqrt(8.0 * np.pi **3) * rloc ** 3 * exp * P
        V[ids] = term1 + term2

        term0 =  (2 * np.pi)**1.5 * rloc**3 * (self.cloc[0] +
                 3.0 * self.cloc[1] + 
                 15.0 * self.cloc[2] + 
                 105. * self.cloc[3])
        V[ids0] = 2 * np.pi * Z * rloc**2 + term0

        return V 

    def eval_proj_g(self, g, i, l, vol):
        """
        Compute the projector polynomial, eq. (12.8)-(12.10)

        Args:
        i : int, The projector index.
        l : int, The angular momentum.
        """
        g1 = np.linalg.norm(g, axis=1)
        rp = self.rp[l]
        x2 = (g1 * rp)**2 
        exp = np.exp(-0.5 * x2)
        prefactor = 4 * np.pi**(5 / 4) * np.sqrt(2**(l + 1) * rp**(2 * l + 3) / vol)
        common = exp * prefactor

        if [i, l] == [1, 0]:
            return common
        if [i, l] == [1, 1]: 
            return common * g1 / np.sqrt(3)
        if [i, l] == [2, 0]:
            return common * 2. / np.sqrt(15.) * (3 - x2)

    def get_ylm_real(self, g, l, m):
        # compute spherical harmonics
        ylms = np.zeros(len(g))
        gm = np.linalg.norm(g, axis=1) + 1e-7
        theta = np.arccos(g[:, 2] / gm)
        phi = np.arctan2(g[:, 1], g[:, 0])
        ylm = sph_harm(m, l, phi, theta)
        if m > 0:
            return np.sqrt(2) * (-1)**m * np.real(ylm)
        elif m < 0:
            return np.sqrt(2) * (-1)**m * np.imag(ylm)
        else:
            return np.real(ylm)


    def get_beta_nonlocal(self, pw):
        """
        Get beta_ilm for the given structure
        [N_kpt][N_gx, N_atom]
        """
        vol = pw.model.volume
        pos = pw.model.cart_positions
        beta_ilms = [[] for _ in range(pw.n_kpts)]

        for ik in range(pw.n_kpts):
            gs = pw.g_wfcs[ik] + pw.kpoints[ik]
            sf = np.exp(1j* (gs @ pos.T))    # (ng, 3) (3, 2)
            beta_ilm = np.zeros([self.num_ilms, len(gs), len(pos)], 
                                 dtype=complex)

            for id, ilm in enumerate(self.ilm_indices): 
                (i, l, m) = ilm
                proj = self.eval_proj_g(gs, i, l, vol) 
                ylms = self.get_ylm_real(gs, l, m)     
                proj *= ylms
                beta_ilm[id] = np.einsum('i,ij->ij', proj, sf) 
                beta_ilm[id] *= (-1j)**l 
            beta_ilms[ik] = beta_ilm
            print("initialize beta", ik, beta_ilms[ik].shape)
        self.beta_nl = beta_ilms

    def get_v_loc_r(self, pw):
        """
        Compute the local V_ps for the given structure

        Args:
            pw: planewave instance
        """
        grids = pw.grids
        vol = pw.model.volume
        g_masks = pw.g_masks_r
        g_vectors = pw.g_rhos
        v_loc_g = np.zeros(grids, dtype=complex)

        # get v_loc_g
        pos = (pw.model.cart_positions).T
        sf = np.exp(1j*g_vectors @ pos).sum(axis=1)
        v_loc_g_1D = self.eval_v_local_g(g_vectors)
        v_loc_g_1D *= sf / vol

        # convert to 3D
        v_loc_g[g_masks] = v_loc_g_1D

        # fft to real space
        v_loc_r = np.zeros(grids, dtype=complex)
        v_loc_r = np.fft.ifftn(v_loc_g) * np.prod(grids)
        self.v_loc_r = v_loc_r

    def get_v_nloc(self, pw):

        V = [[] for _ in range(pw.n_kpts)]

        for ik in range(pw.n_kpts):
            # h[i,j] * beta * < beta^* | psi>
            n_ilms = len(self.ilm_indices)
            psi = pw.psi_1d[ik] # (N_states, Ngx)
            beta = self.beta_nl[ik] # (N_ilm, Ngx, Natoms)
            # <beta|psi> => (N_ilm, Nat, Nst)
            # (N_ilm, Ngx, Nat) (Nst, Ngx) 
            out = np.einsum('ijk,lj->ikl', beta, psi).conj() 
            #out = np.einsum('ijk,lj->ikl', beta, psi)

            V[ik] = np.zeros(psi.shape, dtype=complex)
            for idx1 in range(n_ilms):
                (i1, l1, m1) = self.ilm_indices[idx1]
                for idx2 in range(n_ilms):
                    (i2, l2, m2) = self.ilm_indices[idx2]
                    if [l1, m1] == [l2, m2]:
                        coef = self.h[l1][i1-1, i2-1]
                        tmp2 = np.einsum('ij,jk->ki', beta[idx1], out[idx2])
                        V[ik] += coef * tmp2.conj() 
        #for i in range(len(V)): print("V_nloc", V[i, 0], psi[i, 0])
        return V

    def get_E_loc(self, pw):
        """
        Compute the local E_ps for the given structure

        Args:
            pw: planewave instance
        """
        dvol = pw.model.volume / pw.num_grids
        E_loc = (self.v_loc_r.real * pw.rho_r).sum() * dvol
        self.E_loc = E_loc

    def get_E_nloc(self, pw):

        E = 0
        occs = pw.occs
        for ik in range(pw.n_kpts):
            n_ilms = len(self.ilm_indices)
            psi = pw.psi_1d[ik] # (N_states, Ngx)
            beta = self.beta_nl[ik] # (N_ilm, Ngx, Natoms)
            out = np.einsum('ijk,lj->ikl', beta, psi)

            for idx1 in range(n_ilms):
                (i1, l1, m1) = self.ilm_indices[idx1]
                for idx2 in range(n_ilms):
                    (i2, l2, m2) = self.ilm_indices[idx2]
                    if [l1, m1] == [l2, m2]:
                        coef = self.h[l1][i1-1, i2-1]
                        beta2 = (out[idx1] * out[idx2].conj()).real
                        beta2 = beta2 * occs[None, None, :]
                        E += coef * np.sum(beta2)
        return E


class Hamiltionian:
    """
    A class to compute the hamiltonian
    
    Parameters:
        pw: float, The ionic charge.
        V_ps_loc: array-like
        ps_beta_nl:
    """

    def __init__(self, pw, psp):
        # planewaves and pseudopotential
        self.pw = pw
        psp.get_v_loc_r(pw)
        psp.get_E_loc(pw)
        psp.get_beta_nonlocal(pw)
        self.psp = psp

        # potentials
        self.V_ps_loc = psp.v_loc_r
        self.V_Hartree = np.zeros(pw.grids)
        self.V_XC = np.zeros(pw.grids)
        self.V_total = np.zeros(pw.grids)

        # energies
        self.E_Kinetic = 0
        self.E_ps_loc = psp.E_loc
        self.E_XC = 0
        self.E_Hartree = 0
        self.E_ps_nloc = 0
        self.E_total = 0
        self.E_NN = -8.3979274
    
    def __str__(self):
        strs = "\nHamiltonian"
        strs += f"\nE_Kinetic:   {self.E_Kinetic:12.6f}"
        strs += f"\nE_ps_local:  {self.E_ps_loc:12.6f}"
        strs += f"\nE_ps_nloc:   {self.E_ps_nloc:12.6f}"
        strs += f"\nE_Hartree:   {self.E_Hartree:12.6f}"
        strs += f"\nE_XC:        {self.E_XC:12.6f}"
        strs += f"\nE_NN:        {self.E_NN:12.6f}"
        strs += f"\nE_total:     {self.E_total:12.6f}"
        return strs

    def get_E_total(self):
        self.E_XC = self.get_E_XC()
        self.E_Hartree = self.get_E_Hartree()
        self.E_ps_nloc = self.get_E_ps_nloc()
        self.E_Kinetic = self.get_E_Kinetic()
        self.E_total = self.E_ps_loc + self.E_XC + self.E_Hartree
        self.E_total += self.E_ps_nloc + self.E_Kinetic + self.E_NN

    def get_H_op(self, verbose=False):

        ns = len(self.pw.psi_3d[0])
        # Update kinetic operator
        self.get_K_op()

        # Update local potential
        self.get_V_XC()
        self.get_V_Hartree()
        V = self.V_ps_loc.real + self.V_XC + self.V_Hartree
        self.V_total = V

        # Update nonlocal potential
        self.V_ps_nloc = self.psp.get_v_nloc(self.pw)

        # Get the H
        self.H = [[] for _ in range(self.pw.n_kpts)]
        for ik in range(self.pw.n_kpts):
            # local: V(r) => V(g)
            mask = self.pw.g_masks_w[ik]
            num_gs = len(self.pw.psi_1d[ik][0])
            Vg = np.zeros([ns, num_gs], dtype=complex)
            for i, psi in enumerate(self.pw.psi_3d[ik]):
                psi_r = np.fft.ifftn(psi)
                Vg[i] += np.fft.fftn(V * psi_r)[mask]

            self.H[ik] = self.T[ik] + Vg + self.V_ps_nloc[ik] 

        if verbose:
            print("debug Kinetic", self.T[0][0].real.flatten()[:5])
            print("debug V_XC", self.V_XC.flatten()[:5], self.V_XC.sum())
            print("debug V_Hartree", self.V_Hartree.flatten()[:5], self.V_Hartree.sum())
            print("debug V_ps_loc", self.V_ps_loc.flatten().real[:5], self.V_ps_loc.sum())
            print("debug V_total", self.V_total.flatten().real[:5], V.sum())
            print("debug V_ps_nloc", self.V_ps_nloc[0][0].flatten()[:5])
            print('op_H', self.H[0][0].flatten()[:5].real)


    def get_K_op(self):
        """
        get kinetic operator in 1D g space
        """
        T = [[] for _ in range(self.pw.n_kpts)]
        for ik in range(self.pw.n_kpts):
            k = self.pw.kpoints[ik]
            gs = self.pw.g_wfcs[ik]
            g2s = np.sum((gs + k)**2, axis=1)
            T[ik] = 0.5 * (g2s * self.pw.psi_1d[ik])
        self.T = T

    def get_V_XC(self):
        """
        get V_xc in 3d real space
        """
        self.V_XC = -0.5 * np.cbrt(3. * self.pw.rho_r / np.pi)
    
    def get_V_Hartree(self):
        """
        get V_hartree in 3D real space from electron density
        """
        mask = self.pw.g_masks_r
        gs = self.pw.g_rhos
        g2 = (gs**2).sum(axis=1)  + 1e-12
        rho_g = np.fft.fftn(self.pw.rho_r) * 4 * np.pi
        V_g = np.zeros(self.pw.grids, dtype=complex)
        V_g[mask] = rho_g[mask] / g2

        # Reset the gamma to 0
        V_g[0, 0, 0] = 0
        self.V_Hartree = np.real(np.fft.ifftn(V_g))

    def diag_davidson(self):
        
        ns = len(self.pw.psi_3d[0])
        self.get_op_H()
        for ik in range(self.pw.n_kpts):
            psi, H = self.pw.psi_1d[ik], self.H[ik]

            # Initial eigenvalues
            eigval0 = (H * psi.conj()).sum(axis=1).real
            print('eigval', ik, eigval0)

            # residuals
            R = eigval0[:, None] * psi - H  # (ns, ngs)
            residual = np.sqrt(np.einsum('ij,ij->i', R, R.conj()).real)
            print('res', residual)

            for i in range(50):
                res_norm = 1.0 /residual
                R *= res_norm[:, None] 
                R /= (self.pw.psi_1d + 1.0)

                HR = self.get_op_H(R)

                H1 = np.zeros([ns*2, ns*2], dtype=complex)
                S1 = np.zeros([ns*2, ns*2], dtype=complex)

                # Build H
                if i == 0:
                    H1[:ns, :ns] = psi @ H.T
                else:
                    H1.diag = eigval0

                H1[:ns, ns:] = psi @ HR.T
                H1[ns:, ns:] = psi.conj() @ HR.T
                H1[ns:, :ns] = HR[:ns, ns:].T.conj()

                # Build S
                S1[:ns, :ns] = np.diag(ns)
                S1[:ns, ns:] = psi.conj() @ R.T
                S1[ns:, ns:] = R.conj @ R.T
                S1[ns:, :ns] = S[:ns, ns:].T.conf()

                H1 = 0.5 * (H1 + H1.conj())
                S1 = 0.5 * (S1 + S1.conj())

                H11 = np.triu(H1) + np.conj()

                lr, Xr = np.linalg.eigh(H11, S11)

                # update eigvalue and psi
                eigval = lr.real
                psi = None

                d_eigval = ((eigval - eigval0)**2).sum()
                if d_eigval < 1e-10:
                    break

        return eigval0[:ns], psi[:ns]

    def scf(self, max_iter=50):
        pass
        #eigval0, psi0 = self.diag()
        #E0 = self.get_E_total()

        #for i in range(max_iter):
        #    # update wavefunctions and electron density
        #    self.pw.psi_1d = **
        #    self.pw.get_psi_3d()
        #    self.pw.get_rho_r()
        #    
        #    # mixing
        #    rho_new = rho_new * 0.8 + rho_new1 * 0.2
        #    d_rho = np.sum(np.abs(rho_new - rho_new1))

        #    E1 = self.get_E_total()
        #    dE = E1 - E0

        #    print(f"scf{:3d} dE: {:12.8f} E_total:{:12.6f} drho:{12.4f}")

        #    if dE < 1e-10 and d_rho < 1e-3 and d_eigval < 1e-6:
        #        print("\nSCF is Converged")
        #        break

    def get_E_Kinetic(self):
        E = 0.0
        occs = self.pw.occs
        for ik in range(self.pw.n_kpts):
            k = self.pw.kpoints[ik]
            gs = self.pw.g_wfcs[ik]
            g2s = np.sum((gs + k)**2, axis=1)

            psi = self.pw.psi_1d[ik]
            factor = occs[:, None] * g2s[None, :]
            E += ((psi.conj() * psi).real * factor).sum()

        return E

    def get_E_XC(self):
        dvol = self.pw.model.volume / self.pw.num_grids
        E = 0.5 * (self.V_XC * self.pw.rho_r).sum() * dvol
        return E

    def get_E_Hartree(self):
        dvol = self.pw.model.volume / self.pw.num_grids
        E = 0.5 * (self.V_Hartree * self.pw.rho_r).sum() * dvol
        return E

    def get_E_ps_nloc(self):
        return self.psp.get_E_nloc(self.pw)
 
    def get_E_nn(self):
        """
        Ewald summation of the nuclear-nuclear energy
        """
        pass



if __name__ == "__main__":

    np.random.seed(42)

    # System
    lattice = 5.13155 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    #lattice = 5.13155 * np.array([[-1, 0, -1], [0, 1, 1], [1, 1, 0]])
    positions = np.array([[0, 0, 0], [0.25, 0.25, 0.25]])
    model = Structure(lattice, positions)
    print(model)

    # Planewave
    pw = PlaneWaveBasis(model, 
                        Ecut=15.0, 
                        kpoints=np.array([[0, 0, 0],
                                          #[0.25, 0.25, 0.25],
                                          #[0, 0.5, 0],
                                          #[0.5, 0, 0.5],
                                          ]), 
                        kweights=np.array([1.]),#, 6., 8., 12.])/27.,
                        occs = np.array([1, 1, 1, 1, 0, 0]),
                        )
    print(pw)
    pw.random_guess()
    #print(pw.rho_r.flatten()[:10])
    rho = pw.rho_r.sum() * pw.model.volume/pw.num_grids
    print(f"Total Number of electrons:      {rho}")

    # Pseudopotential
    psp = PspHgh(Z=4, rloc=0.44000000, 
                 cloc=np.array([-7.33610297, 0, 0, 0]), 
                 rp=np.array([0.42273813, 0.48427842]), 
                 h=np.array([[[5.90692831, -1.26189397],
                              [-1.26189397, 3.25819622]],
                             [[2.72701346, 0.00000000],
                              [0.00000000, 0.00000000]]]))
    print(psp)
    #print("v_local_r\n", psp.v_loc_r.flatten()[:10].real)
    #for n in range(pw.n_kpts):
    #    print("beta_nl\n", n, psp.beta_nl[n].real[:, :5, 0].T)

    #V = psp.get_v_nloc(pw)[0]
    #print("V_nloc", V.shape)
    #for i in range(len(V)): print(V[i, 0], pw.psi_1d[0][i, 0])

    # Hamiltonian
    ham = Hamiltionian(pw, psp)
    ham.get_H_op()
    ham.get_E_total()
    print(ham)
    #ham.diag_davidson()
