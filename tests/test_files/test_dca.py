# Test file
# Goal : to test the dynamical cluster approximation (pyqcm.cdmft.DCA, model_instance(dca=True)) on the 2x2 plaquette of the Hubbard model,
# with one bath (classic DCA) and with two sub-baths that each couple to two cluster momenta.
# The CDMFT host computed in C++ is compared with the same formula written out in numpy:
#   Sigma_K = [U Sigma_s U^T]_KK from the system s whose bath couples to K
#   Gbar_K = < (w - eps(K+k) - Sigma_K)^-1 >_k  over the patch k in [-pi/2, pi/2)^2 (grid of kgrid_side^2 points)
#   host = U^T diag(Gbar_K^-1 + Sigma_K + [U t_c U^T]_KK - w) U
# The periodization 'G' must give (w - eps(k) - Sigma_K(k))^-1: the DCA lattice Green function is in the periodic gauge
# of CPT. At U = 0 (Sigma = 0) the DCA lattice is exact, so its kinetic energy must equal CPT's. set_DCA_channels() and
# DCA_momenta() are checked on the way. 'loop': DCA(K=...) runs the CDMFT loop on DCA instances with those channels.
# 'mixed': a bath that does not couple to definite K points makes DCA warn.
#--------------------------------------------------------------------------------
import subprocess
import sys
import numpy as np
import pyqcm
from pyqcm import qcm
from pyqcm.cdmft import DCA, frequency_grid

no = 4
# rows: cluster momenta K = (0,0), (pi,pi), (0,pi), (pi,0) for the sites (0,0), (1,0), (0,1), (1,1)
U_mat = np.array([[1, 1, 1, 1], [1, -1, -1, 1], [1, 1, -1, -1], [1, -1, 1, -1]], dtype=float) / 2
K_pts = np.array([[0, 0], [np.pi, np.pi], [0, np.pi], [np.pi, 0]])
nk = 16
pyqcm.set_global_parameter('kgrid_side', nk)

def build(systems, nb, coupling=U_mat):
    """2x2 plaquette; systems = list of lists of K indices; nb bath orbitals per K, coupled to site i with coupling[K, i]
    (by default the K channel only)"""
    CMs = []
    for s, Ks in enumerate(systems):
        nbath = nb * len(Ks)
        nt = no + nbath
        CM = pyqcm.cluster_model(no, nbath, name=f'clus_{s}')
        for c, K in enumerate(Ks):
            for j in range(1, nb + 1):
                x = c * nb + j
                CM.new_operator(f'eb{x}', 'one-body', [(no + x, no + x, 1.0), (no + x + nt, no + x + nt, 1.0)])
                elem = []
                for i in range(no):
                    elem += [(i + 1, no + x, coupling[K, i]), (i + 1 + nt, no + x + nt, coupling[K, i])]
                CM.new_operator(f'tb{x}', 'one-body', elem)
        elem = []
        for i, j in ((1, 2), (1, 3), (2, 4), (3, 4)):
            elem += [(i, j, -1.0), (i + nt, j + nt, -1.0)]
        CM.new_operator('dt', 'one-body', elem)
        CMs.append(CM)
    clus = pyqcm.cluster(CMs, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)))
    model = pyqcm.lattice_model(f'DCA_2x2_{len(systems)}sys', clus, ((2, 0, 0), (0, 2, 0)))
    model.interaction_operator('U')
    model.hopping_operator('t', (1, 0, 0), -1)
    model.hopping_operator('t', (0, 1, 0), -1)
    model.set_target_sectors([f'R0:N{no + nb * len(Ks)}:S0' for Ks in systems])
    P = 'U=4\nmu=0.5*U\nt=1\n'
    rng = np.random.default_rng(1)
    for s, Ks in enumerate(systems):
        P += f'dt_{s + 1}={4 / np.pi - 1}\n'
        for c, K in enumerate(Ks):
            for j in range(1, nb + 1):
                x = c * nb + j
                P += f'eb{x}_{s + 1}={rng.uniform(-1.5, 1.5):.4f}\ntb{x}_{s + 1}={rng.uniform(0.3, 0.8):.4f}\n'
    model.set_parameters(P)
    return model


def reference_host(I, systems, freqs):
    """numpy version of the DCA host, on the same k grid as the C++ code (i/nk folded into [-1/2,1/2))"""
    P = I.model.parameters()
    k1 = np.pi * (np.arange(nk) / nk - 0.5)
    kx, ky = np.meshgrid(k1, k1, indexing='ij')
    eps = np.array([-2 * P['t'] * (np.cos(Kx + kx) + np.cos(Ky + ky)).ravel() - P['mu'] for Kx, Ky in K_pts])
    tcK = np.diag(U_mat @ np.asarray(I.cluster_hopping_matrix(0)) @ U_mat.T)
    nsys = I.model.nsys
    H = []
    for f in freqs:
        w = 1j * f
        sig = np.zeros(4, dtype=complex)
        for s, Ks in enumerate(systems):
            S = U_mat @ np.asarray(qcm.self_energyC(w, False, I.label * nsys + s)) @ U_mat.T
            for K in Ks:
                sig[K] = S[K, K]
        gbar = np.mean(1 / (w - eps - sig[:, None]), axis=1)
        H.append(U_mat.T @ np.diag(1 / gbar + sig + tcK - w) @ U_mat)
    return np.array(H), sig, eps


LAYOUTS = {'single': ([[0, 1, 2, 3]], 1), 'pairs': ([[0, 1], [2, 3]], 2), 'free': None, 'loop': None, 'mixed': None}
if len(sys.argv) == 1:  # one lattice model per process: run each layout in its own
    for name in LAYOUTS:
        out = subprocess.run([sys.executable, __file__, name], check=True, capture_output=True, text=True).stdout
        print('\n'.join(l for l in out.splitlines() if l.startswith((name + ':', 'DCA WARNING'))))
        assert ('DCA WARNING' in out) == (name == 'mixed'), f'{name}: DCA warning expected only for the mixed bath'
    sys.exit(0)

if sys.argv[1] == 'mixed':  # a bath orbital coupled to (K=(0,0) + K=(0,pi))/sqrt(2): Sigma is not K-diagonal, DCA warns
    C = U_mat.copy()
    C[0] = (U_mat[0] + U_mat[2]) / np.sqrt(2)
    model = build([[0, 1, 2, 3]], 1, coupling=C)
    I = pyqcm.model_instance(model, dca=True)
    qcm.CDMFT_host(np.array([0.5]), np.ones(1), I.label)
    print('mixed: host computed')
    sys.exit(0)

if sys.argv[1] == 'free':  # U = 0: DCA and CPT are both exact; kinetic energy and density must agree
    model = build([[0, 1, 2, 3]], 1)
    model.set_parameter('U', 0.0)
    ave = {}
    for dca in (0, 1):
        ave[dca] = pyqcm.model_instance(model, dca=bool(dca)).averages(['t', 'mu'])
    print(f"free: <t> CPT {ave[0]['t']:.6f} DCA {ave[1]['t']:.6f};  <n> CPT {ave[0]['mu']:.6f} DCA {ave[1]['mu']:.6f}")
    assert abs(ave[0]['t'] - ave[1]['t']) < 1e-4 and abs(ave[0]['mu'] - ave[1]['mu']) < 1e-4
    sys.exit(0)

freqs = np.array([0.1, 0.5, 2.0])
if sys.argv[1] == 'loop':  # DCA class with K: the loop runs on DCA instances with those channels
    systems = [[0, 1], [2, 3]]
    model = build(systems, 2)
    varia = [f'{p}{x}_{s}' for s in (1, 2) for p in ('eb', 'tb') for x in range(1, 5)]
    try:
        DCA(model, K=[[(0.5, 0)]], **dict(varia=varia, grid=frequency_grid('legendre', (1, 10, 5, 10, 5))))
        raise AssertionError('a wavevector that is not a cluster momentum was accepted')
    except ValueError as e:
        assert 'not all cluster momenta' in str(e), e
    opts = dict(varia=varia, grid=frequency_grid('legendre', (1, 10, 5, 10, 5)), maxiter=3, method='trf', iteration='fixed_point')
    made = []  # the instances the loop works with
    class Recording(DCA):
        def new_instance(self):
            made.append(super().new_instance())
            return made[-1]
    try:  # 3 iterations, not converged
        Recording(model, K=[[(0, 0), (1, 1)], [(0, 1), (1, 0)]], **opts)
    except (pyqcm.TooManyIterationsError, pyqcm.SolverError):
        pass
    assert len(made) >= 3 and all(I.dca for I in made), [I.dca for I in made]
    I = made[-1]  # with the channels set from K; its host was built by the loop, on the grid
    err = np.abs(np.asarray(qcm.get_CDMFT_host(0, False, I.label)) - reference_host(I, systems, opts['grid'].wr)[0]).max()
    assert err < 1e-10, err
    print(f'loop: {len(made)} DCA instances; host of the last = DCA reference ({err:.1e})')
    sys.exit(0)

for systems, nb in [LAYOUTS[sys.argv[1]]]:
    model = build(systems, nb)
    I = pyqcm.model_instance(model, dca=True)
    qcm.CDMFT_host(freqs, np.ones(len(freqs)), I.label)
    H = qcm.get_CDMFT_host(0, False, I.label)
    H_ref, sig, eps = reference_host(I, systems, freqs)
    err = np.abs(H - H_ref).max()
    print(f'{len(systems)} system(s): max |host - reference| = {err:.2e}')
    assert err < 1e-10, err

    # cluster self-energy is K-diagonal and carries Sigma_K
    S = U_mat @ np.asarray(I.cluster_self_energy(2j)) @ U_mat.T
    assert np.abs(S - np.diag(np.diag(S))).max() < 1e-12

    # periodization 'G': G(k) = (w - eps(k) - Sigma_K(k))^-1, at w = 2i (the last frequency of reference_host)
    P = model.parameters()
    for K, (Kx, Ky) in enumerate(K_pts):
        k = np.array([[Kx / (2 * np.pi) + 0.05, Ky / (2 * np.pi) - 0.1, 0.0]])  # units of 2 pi, inside the patch of K
        g = np.asarray(I.periodized_Green_function(2j, k)).ravel()[0]
        e = -2 * P['t'] * (np.cos(2 * np.pi * k[0, 0]) + np.cos(2 * np.pi * k[0, 1])) - P['mu']
        assert abs(g - 1 / (2j - e - sig[K])) < 1e-12, (K, g)

    # cluster momenta and transform; perm[r] = index in DCA_momenta() of row r of U_mat
    Kp = np.asarray(qcm.DCA_momenta(I.label))[:, :2]
    same = lambda a, b: np.allclose((a - b + 0.5) % 1 - 0.5, 0)
    perm = [[a for a in range(4) if same(Kp[a], K_pts[r] / (2 * np.pi))][0] for r in range(4)]
    assert sorted(perm) == [0, 1, 2, 3]

    # explicit channels: the same layout gives the same host; a cluster momentum without a carrier is an error
    I2 = pyqcm.model_instance(model, dca=True)
    for s_, Ks in enumerate(systems):
        qcm.set_DCA_channels(s_, [perm[K] for K in Ks], I2.label)
    qcm.CDMFT_host(freqs, np.ones(len(freqs)), I2.label)
    H2 = qcm.get_CDMFT_host(0, False, I2.label)  # a new instance solves the ED again: 1e-12 noise
    assert np.abs(H2 - H).max() < 1e-9
    try:
        I3 = pyqcm.model_instance(model, dca=True)
        qcm.set_DCA_channels(0, [0], I3.label)
        qcm.CDMFT_host(freqs, np.ones(len(freqs)), I3.label)
        raise AssertionError('a cluster momentum without a carrier was accepted')
    except RuntimeError as e:
        assert 'no system' in str(e), e
print(f'{sys.argv[1]}: DCA host and periodization agree with the numpy reference')
