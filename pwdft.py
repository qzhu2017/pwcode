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
        self.fx = int(norm[0]) #+ 2
        self.fy = int(norm[1]) #+ 2
        self.fz = int(norm[2]) #+ 2
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
        n_kpts = self.n_kpts
        psi_3d = [[] for _ in range(n_kpts)]
        for ik in range(n_kpts):
            psi_3d[ik] = self.get_psi_3d_single(self.psi_1d[ik], ik)
        self.psi_3d = psi_3d
        return psi_3d

    def get_psi_3d_single(self, psi_1d, ik):
        [gx, gy, gz] = self.grids
        mask = self.g_masks_w[ik]
        ns = len(psi_1d)

        psi_3d_k = np.zeros([ns, gx, gy, gz], dtype=complex)
        for i in range(ns):
            psi_3d_k[i][mask] += psi_1d[i] 
        return psi_3d_k

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

    def precondition_KE(self, psi, ik):
        """
        Simple preconditioner based on kinetic energy
        """
        ns = len(psi[0])
        gs = self.g_wfcs[ik][:ns] + self.kpoints[ik]
        g2s = (gs**2).sum(axis=1)
        psi /= (1.0 + g2s)
        return psi

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
        ids = g2 > 1e-2
        ids0 = g2 <= 1e-2
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
        
        if True: #False:
            term0 = (2. * np.pi)**1.5 * rloc**3 * (self.cloc[0] +
                     3. * self.cloc[1] + 
                     15. * self.cloc[2] + 
                     105. * self.cloc[3])
            V[ids0] = 2 * np.pi * Z * rloc**2 + term0
        #print("Debug", len(V[ids0])); import sys; sys.exit()

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
        n_grids = pw.num_grids
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
        v_loc_g[g_masks] = v_loc_g_1D.conj()

        # fft to real space
        self.v_loc_g = v_loc_g #_1D
        self.v_loc_r = np.fft.ifftn(v_loc_g).real * n_grids

    def get_v_nloc(self, pw, psi=None, ik=0):

        if psi is None:
            psi = pw.psi_1d[ik] # (N_states, Ngx)

        V = np.zeros(psi.shape, dtype=complex)

        # h[i,j] * beta * < beta^* | psi>
        n_ilms = len(self.ilm_indices)
        beta = self.beta_nl[ik] # (N_ilm, Ngx, Natoms)
        # <beta|psi> => (N_ilm, Nat, Nst)
        # (N_ilm, Ngx, Nat) (Nst, Ngx) 
        out = np.einsum('ijk,lj->ikl', beta, psi).conj() 
        #out = np.einsum('ijk,lj->ikl', beta, psi)

        for id1 in range(n_ilms):
            (i1, l1, m1) = self.ilm_indices[id1]
            for id2 in range(n_ilms):
                (i2, l2, m2) = self.ilm_indices[id2]
                if [l1, m1] == [l2, m2]:
                    coef = self.h[l1][i1-1, i2-1]
                    tmp2 = np.einsum('ij,jk->ki', beta[id1], out[id2])
                    V += coef * tmp2.conj() 
        return V

    def get_E_loc(self, pw):
        """
        Compute the local E_ps for the given structure

        Args:
            pw: planewave instance
        """
        dvol = pw.model.volume / pw.num_grids
        E_loc = (self.v_loc_r * pw.rho_r).sum() * dvol
        return E_loc

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
        return 2*E


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
        self.V_total = None #np.zeros(pw.grids)

        # energies
        self.E_Kinetic = 0
        self.E_ps_loc = 0 #psp.E_loc
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
        self.E_ps_loc = self.get_E_ps_loc()
        self.E_ps_nloc = self.get_E_ps_nloc()
        self.E_Kinetic = self.get_E_Kinetic()
        self.E_total = self.E_ps_loc + self.E_XC + self.E_Hartree
        self.E_total += self.E_ps_nloc + self.E_Kinetic + self.E_NN
        return self.E_total

    def get_H_op(self, ik=0, psi=None, verbose=False):
        #self.V_ps_loc = np.load("source/V_loc_r.npy")
        if psi is None: psi = self.pw.psi_1d[ik]

        ns = len(psi)
        ngs = len(psi[0])
        mask = self.pw.g_masks_w[ik]

        # Update local potential
        if self.V_total is None:
            V_XC = self.get_V_XC()
            V_H = self.get_V_Hartree()
            self.V_XC = V_XC
            self.V_Hartree = V_H
            self.V_total = self.V_ps_loc.real + V_XC + V_H
        V = self.V_total

        # local: V(r) => V(g)
        Vg = np.zeros([ns, ngs], dtype=complex)
        psi_3d = self.pw.get_psi_3d_single(psi, ik)
        for i, _psi in enumerate(psi_3d):
            psi_r = np.fft.ifftn(_psi)
            Vg[i] += np.fft.fftn(V * psi_r)[mask]

        # Update kinetic operator
        T = self.get_K_op(psi, ik)

        # Update nonlocal potential
        V_ps_nloc = self.psp.get_v_nloc(self.pw, psi, ik)

        # Get the H
        H = T + Vg + V_ps_nloc

        if verbose:
            print("debug psi", psi[0][:5], psi.real.flatten().sum())
            print("debug V_Kin", T[0,:5], T.sum())
            print("debug V_XC", self.V_XC.flatten()[:5], self.V_XC.sum())
            print("debug V_Hartree", self.V_Hartree.flatten()[:5], self.V_Hartree.sum())
            print("debug V_ps_loc", self.V_ps_loc.flatten().real[:5], self.V_ps_loc.sum())
            print("debug V_ps_loc", self.V_ps_loc.flatten().real[-5:])
            print("debug V_total", V.flatten().real[:5], V.sum())
            print("debug V_ps_nloc", V_ps_nloc[0].flatten()[:5], V_ps_nloc[0].flatten().sum())
            for i in range(ns):
                print('op_H', H[i].flatten()[:5])
            print("debug rho", self.pw.rho_r.sum())

        return H


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

    def get_K_op(self, psi_1d, ik):
        """
        get kinetic operator in 1D g space
        """
        k = self.pw.kpoints[ik]
        gs = self.pw.g_wfcs[ik]
        g2s = np.sum((gs + k)**2, axis=1)
        T = 0.5 * (g2s * psi_1d)
        return T

    def get_V_XC(self):
        """
        Get XC from libxc via python wrapper
        """
        import pylibxc

        rho = self.pw.rho_r
        func = pylibxc.LibXCFunctional("lda_x", "unpolarized")
        results = func.compute({"rho": rho})  
        V_X = results["zk"]
        func = pylibxc.LibXCFunctional("lda_c_pw", "unpolarized")
        results = func.compute({"rho": rho})
        V_C = results["zk"]
        return (V_X + V_C).reshape(self.pw.grids)

    def get_V_XC_PW(self):
        """
        Get V_XC (Exchange-Correlation potential) in 3D real space using LDA.
        Uses the Perdew-Zunger (PZ81) parametrization for correlation.
        """
        # Constants
        pi = np.pi
    
        # Electron density
        rho = np.array(self.pw.rho_r, dtype=np.float64)  # Ensure double precision
        rho = np.maximum(rho, 1e-10)  # Avoid division by zero
    
        # 1. Exchange Potential (V_X)
        V_X = -0.75 * (3.0 * rho / pi) ** (1 / 3)
    
        # 2. Correlation Potential (V_C) using PZ81
        # Compute Wigner-Seitz radius (rs)
        rs = (3 / (4 * pi * rho)) ** (1 / 3)
        rs = np.minimum(rs, 1e6)  # Avoid excessively large values

    
        # PZ81 Parameters
        A, B, C, D = 0.0311, -0.048, 0.002, -0.0116  # High-density (rs < 1)
        A_low, B_low, C_low = 0.0311, -0.01342, 0.00435  # Low-density (rs >= 1)
    
        # Correlation energy per particle, eps_C(rs)
        eps_C = np.zeros_like(rs)
        V_C = np.zeros_like(rs)
    
        # High-density region (rs < 1)
        mask_high = rs < 1
        rs_high = rs[mask_high]
    
        eps_C[mask_high] = -A + B * rs_high * np.log(rs_high) + C * rs_high
        V_C[mask_high] = -A + (B * (np.log(rs_high) + 1)) + C * rs_high
    
        # Low-density region (rs >= 1)
        mask_low = ~mask_high
        rs_low = rs[mask_low]
    
        # Corrected formula for correlation energy and potential
        eps_C[mask_low] = -A_low / (1 + B_low * np.sqrt(rs_low) + C_low * rs_low)
        V_C[mask_low] = eps_C[mask_low] - (1 / 6) * (B_low / np.sqrt(rs_low) + C_low)

        # Add derivative correction for low-density region
        #V_C[mask_low] = eps_C[mask_low] * (1 + 
        #                              (B_low / np.sqrt(rs_low) + C_low) /
        #                              (1 + B_low * np.sqrt(rs_low) + C_low * rs_low))
        # 3. Total XC potential
        V_XC = V_X + V_C
    
        return V_XC


    
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
        #self.V_Hartree = np.real(np.fft.ifftn(V_g))
        return np.real(np.fft.ifftn(V_g))

    def diag(self, psi, ik=0, debug=False):
        """
        Davidson dialgonalization
        """
        ns = len(psi)
        HX = self.get_H_op(ik, psi, verbose=True)
        #HX = np.load('source/a.npy')
        #if debug: import sys; sys.exit()

        # Initial guess eigenvalues
        eigval0 = (psi.conj() * HX).sum(axis=1).real  # HV
        print('Initial eigval', ik, eigval0, HX.shape, psi.shape)
        import sys; sys.exit()

        # residuals R = eig*X - HX 
        R = eigval0[:, None] * psi - HX  # (ns, ngs)
        residual = np.sqrt((R * R.conj()).sum(axis=1).real)

        for i in range(50):
            res_norm = 1.0 /residual
            R *= res_norm[:, None] 
            R = self.pw.precondition_KE(R, ik)
            #R = R / (1 + (self.pw.g_wfcs[ik]**2).sum(axis=1))
            #print('debug R', R[0, :5])
            #print('debug res_norm', res_norm)

            # H of R
            HR = self.get_H_op(ik, R)
            #HR = np.load(f'source/{i}.npy')
            #print('debug HR', HR[0, :5])

            # H to be updated
            H1 = np.zeros([ns*2, ns*2], dtype=complex)
            S1 = np.zeros([ns*2, ns*2], dtype=complex)

            # Build H from
            if i == 0:
                H1[:ns, :ns] = psi.conj() @ HX.T
            else:
                np.fill_diagonal(H1, eigval0)

            H1[:ns, ns:] = psi.conj() @ HR.T
            H1[ns:, ns:] = R.conj() @ HR.T
            H1[ns:, :ns] = H1[:ns, ns:].conj().T

            # Build S
            S1[:ns, :ns] = np.diag([1.+ 0.j] * ns)
            S1[:ns, ns:] = psi.conj() @ R.T
            S1[ns:, ns:] = R.conj() @ R.T
            S1[ns:, :ns] = S1[:ns, ns:].conj().T

            # Average
            H1 = 0.5 * (H1 + H1.T.conj())
            S1 = 0.5 * (S1 + S1.T.conj())
            hd = np.diag(np.diag(H1))
            sd = np.diag(np.diag(S1))
            H1 = np.triu(H1) + np.conj(np.triu(H1)-hd).T  - 1.j*hd.imag
            S1 = np.triu(S1) + np.conj(np.triu(S1)-sd).T  - 1.j*sd.imag

            lam_red, psi_red = linalg.eigh(H1, S1)

            # update eigvalue and psi
            eigval1 = lam_red[:ns].real
            psi = psi_red[:ns, :ns].T @ psi + psi_red[ns:, :ns].T @ R
            HX = psi_red[:ns, :ns].T @ HX + psi_red[ns:, :ns].T @ HR
            HX *= -1
            psi *= -1

            # get residual
            R = eigval1[:, None] * psi - HX  # (ns, ngs)
            residual = np.sqrt(np.einsum('ij,ij->i', R, R.conj()).real)

            # Check convergence
            d_eigval = np.abs(eigval1[:ns] - eigval0[:ns]).sum() 
            #print(i, 'eigval', ik, eigval1[:4], d_eigval, residual[0])
            if d_eigval < 1e-6:
                #print("Converged", d_eigval)
                break

            eigval0 = eigval1

        return eigval1, psi

    def scf(self, max_iter=50):

        E0 = self.get_E_total()
        rho_0 = self.pw.rho_r

        for i in range(max_iter):
            # update eigenwavefunctions
            for ik in range(self.pw.n_kpts):
                psi = self.pw.psi_1d[ik]
                if i>0: 
                    debug = True
                else:
                    debug = False
                eigval, psi = self.diag(psi, ik, debug)
                import sys; sys.exit()
                self.pw.psi_1d[ik] = self.pw.orthonormalize(psi)

            self.pw.get_psi_3d()
            self.pw.get_rho_r()
          
            # mixing
            rho = self.pw.rho_r
            rho = rho * 0.8 + rho_0 * 0.2
            self.pw.rho_r = rho
            d_rho = np.sum(np.abs(rho - rho_0))

            # Update H
            V_XC = self.get_V_XC()
            V_H = self.get_V_Hartree()
            self.V_XC = V_XC
            self.V_Hartree = V_H
            self.V_total = self.V_ps_loc.real + V_XC + V_H
            E1 = self.get_E_total()
            dE = E1 - E0

            print(f"SCF{i:3d} dE:{dE:12.8f} E_total:{E1:12.6f} drho:{d_rho:12.4f} {rho.sum()}")
            print(self)

            if dE < 1e-10 and d_rho < 1e-3:
                print("\nSCF is Converged")
                break

            E0 = E1
            rho_0 = rho
        print("Final eigval", eigval)

    def get_E_XC(self):
        dvol = self.pw.model.volume / self.pw.num_grids
        E = (self.V_XC * self.pw.rho_r).sum() * dvol
        return E

    def get_E_Hartree(self):
        dvol = self.pw.model.volume / self.pw.num_grids
        E = 0.5 * (self.V_Hartree * self.pw.rho_r).sum() * dvol
        print(self.pw.rho_r.sum()*dvol, self.pw.rho_r.sum()*self.pw.model.volume / self.pw.num_grids)
        return E

    def get_E_ps_nloc(self):
        return self.psp.get_E_nloc(self.pw)
 
    def get_E_ps_loc(self):
        return self.psp.get_E_loc(self.pw)
 
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
                        kweights=np.array([1]),#., 6., 8., 12.])/27.,
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
    #ham = Hamiltionian(pw, psp)
    #ham.get_H_op()
    #ham.get_E_total()
    #print(ham)
    #ham.diag()
    #ham.scf(50)
    
    # Load DFTK results
    #import h5py
    ## Open the HDF5 file
    #with h5py.File('wavefunction_f.h5', 'r') as f:
    #    # Load wavefunctions
    #    print(list(f.keys()))
    #    psi = [f[key][:] for key in f.keys()]  # Each ψ[i] is a complex 2D array
    #print("ψ[0] shape:", psi[0].shape)

    # load and resort
    psi = np.load('psiks.npy').T
    ids = np.load('g.npy') - 1
    ids2 = np.load('gr.npy') - 1
    ids_sorted = ids.argsort()
    ids2_sorted = ids2.argsort()
    pw.psi_1d[0] = pw.orthonormalize(psi[:6])
    pw.psi_1d[0] = pw.psi_1d[0][:, ids_sorted]
    ids = ids[ids_sorted]
    ids2 = ids2[ids2_sorted]

    [gx, gy, gz] = pw.grids
    g_wfcs_3D = []
    mask1 = np.zeros([gx, gy, gz], dtype=int)
    mask2 = np.zeros([gx, gy, gz], dtype=int)
    count = 0
    for i in range(gx):
        ii = i - gx if i > gx // 2 else i
        for j in range(gy):
            jj = j - gy if j > gy // 2 else j
            for k in range(gz):
                kk = k - gz if k > gz // 2 else k
                g = np.array([ii, jj, kk]) 
                g = g @ pw.model.rec_lattice
                g_wfcs_3D.append(g)
                if count in ids: mask1[i, j, k] = 1
                if count in ids2: mask2[i, j, k] = 1
                count += 1
    g_wfcs_3D = np.array(g_wfcs_3D)
    pw.g_wfcs[0] = g_wfcs_3D[ids]
    pw.g_rhos = g_wfcs_3D[ids2]
    pw.g_masks_w[0] = mask1.astype(bool)
    pw.g_masks_r = mask2.astype(bool)
    pw.get_psi_3d()
    pw.get_rho_r()

    ham = Hamiltionian(pw, psp)
    ham.get_H_op(verbose=True)
    ham.get_E_total()
    print(ham)
    ham.scf(1)
    #psp.get_v_loc_r(pw)
    #p = psp.v_loc_r.flatten()
    #Ps_loc = np.load("Ps_loc.npy")
    #print(np.abs(p-Ps_loc).sum())
    #rho = np.load("rho.npy")[:, 0]
    #r = pw.rho_r.flatten()
    #print("density 10", r[:10], r.shape)
    #print("density 10", rho[:10], rho.shape)
    #print("density diff", np.abs(r - rho).max())
    #dvol = pw.model.volume / pw.num_grids
    #E_loc1 = (p * r).sum() * dvol
    #E_loc2 = (p * rho).sum() * dvol
    #print("PS local energy", p.shape, r.shape, E_loc1, p.shape, rho.shape, E_loc2)

    #p = np.load("source/V_loc_r.npy").real; print(p[0].sum(), p[0,0,:20])
    """
    V_loc_r sum is wrong
    """

