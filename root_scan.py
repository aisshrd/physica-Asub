"""
Uniqueness of the rotational steady state of the heated gas (Sec. 5.1 (ii)).

For every parameter set of the two monotonicity scans (240 + 384 = 624 sets) and every sampled translational
temperature, the rotational balance  dT_r/dt = 0  is examined as a function of theta = T_r/T_t on a grid
(10^-8 <= theta <= 10^6, 30 points per decade) and the number of sign changes is counted.  A single sign change
for every (set, T_t) means that the steady states form one branch theta*(T_t) and that no disconnected branch
exists in the scanned window.  The collision integrals are evaluated by a vectorised quadrature on a
logarithmic grid in y (checked against adaptive quadrature: python3 root_scan.py check).

Output: root_scan.pkl.   Runtime ~25 min on 2 cores.
`python3 root_scan.py outside` repeats the count for six temperatures outside the window (10^-7..10^-5 and
10^4..10^6 T_S) with theta up to 10^12 (output root_scan_outside.pkl, ~4 min).
"""
import numpy as np, itertools, pickle, sys
from multiprocessing import Pool
from scipy.special import erf
from theory import Model

Y = np.geomspace(1e-7, 14.0, 2400); LY = np.log(Y); W = np.gradient(LY) * Y     # trapezoid weights in ln y
W[0] *= 0.5; W[-1] *= 0.5
THETA = np.geomspace(1e-8, 1e6, 421)

def F12_vec(M, Tt, theta):
    """F1, F2 for an array of theta at fixed T_t (Eqs. 13-15 of the paper)."""
    u = 1.0 + theta[:, None] / M.q
    sv = np.sqrt(2.0 * Tt * u / M.m)
    lam = M.c * M.mu(sv * Y[None, :]) / np.sqrt(u)
    z = Y[None, :] / lam
    g1 = np.sqrt(np.pi / 2) * (lam / Y[None, :]) * erf(z / np.sqrt(2))
    g2 = 2.0 * (lam / Y[None, :]) ** 2 * (-np.expm1(-0.5 * z ** 2))
    wgt = (Y ** 3 * np.exp(-0.5 * Y ** 2) * W)[None, :]
    return (g1 * wgt).sum(1), (g2 * wgt).sum(1)

def balance(M, Tt, theta):
    """Rotational bracket of Eq. (17) divided by T_t:  -2 eta0 F1 theta + 2 eta0^2 F2 (1 + theta/q)."""
    F1, F2 = F12_vec(M, Tt, theta)
    return -2 * M.eta0 * F1 * theta + 2 * M.eta0 ** 2 * F2 * (1 + theta / M.q), F1, F2

# temperatures outside the window of the scans, with a wider theta window (the crossover of the friction law then
# occurs at very large theta for small T_t)
TTS_OUT = np.array([1e-7, 1e-6, 1e-5, 1e4, 1e5, 1e6]); THETA_OUT = np.geomspace(1e-8, 1e12, 601)
def one_out(par): return one(par, TTS_OUT, THETA_OUT)

GRID1 = [(e, 0.0, 0.4, muk, mus, p) for e, muk, mus, p in itertools.product(
    [0.90, 0.95, 0.98, 0.995], [0.003, 0.01, 0.03], [0.05, 0.1, 0.2, 0.4, 0.8], [1, 2, 4, 8])]
GRID2 = list(itertools.product([0.7, 0.85, 0.95, 0.99], [-0.5, 0.5, 1.0], [0.25, 0.4, 0.5, 2 / 3],
                               [0.003, 0.03], [0.2, 0.8], [2, 8]))
TTS = np.logspace(-4, 3, 71)

def one(par, TTS=None, THETA=None):
    TTS = globals()['TTS'] if TTS is None else TTS; THETA = globals()['THETA'] if THETA is None else THETA
    e, b0, q, muk, mus, p = par
    M = Model(e=e, b0=b0, q=q, mu_k=muk, mu_s=mus, vS=1., p=p)
    nroots = []; roots = []; mono = True; peak = 0.0
    for Tt in TTS:
        b, F1, F2 = balance(M, Tt, THETA)
        sgn = np.sign(b); ch = np.nonzero(sgn[1:] * sgn[:-1] < 0)[0]
        nroots.append(len(ch))
        r = 2 * M.eta0 ** 2 * F2 * (1 + THETA / M.q) / (2 * M.eta0 * F1 * THETA)      # source / sink
        d = np.diff(r)
        if (d >= 0).any():                      # source/sink ratio not strictly decreasing in theta
            mono = False
            if len(ch):                         # largest value the ratio returns to beyond the root (1 = second root)
                k = ch[0] + 1; up = np.nonzero(d[k:] >= 0)[0]
                if len(up): peak = max(peak, r[k + up[0]:].max())
        roots.append(np.sqrt(THETA[ch[0]] * THETA[ch[0] + 1]) if len(ch) else np.nan)
    return par, np.array(nroots), mono, peak, np.array(roots)

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'check':          # accuracy of the vectorised quadrature
        from scipy.integrate import quad
        def F12_ref(M, Tt, th):          # adaptive quadrature with tight tolerances and a break point at g = v_S
            u = 1 + th / M.q; sv = np.sqrt(2 * Tt * u / M.m)
            lam = lambda y: M.c * M.mu(sv * y) / np.sqrt(u)
            g1 = lambda y: np.sqrt(np.pi / 2) * y ** 2 * np.exp(-y * y / 2) * lam(y) * erf(y / (np.sqrt(2) * lam(y)))
            g2 = lambda y: 2 * y * np.exp(-y * y / 2) * lam(y) ** 2 * (-np.expm1(-y * y / (2 * lam(y) ** 2)))
            br = sorted({min(M.vS / sv, 13.9), min(0.5 * M.vS / sv, 13.9), min(2 * M.vS / sv, 13.9)})
            kw = dict(epsabs=0, epsrel=1e-11, limit=400, points=br)
            return quad(g1, 0, 14, **kw)[0], quad(g2, 0, 14, **kw)[0]
        worst = 0
        for par in [(0.98, 0, 0.4, 0.01, 0.2, 2), (0.995, 0, 0.4, 0.003, 0.8, 8), (0.7, 1.0, 2 / 3, 0.03, 0.8, 8),
                    (0.9, -0.5, 0.25, 0.003, 0.2, 2)]:
            M = Model(e=par[0], b0=par[1], q=par[2], mu_k=par[3], mu_s=par[4], vS=1., p=par[5])
            for Tt in [1e-4, 1e-2, 0.3, 1, 30, 1e3]:
                th = np.array([1e-6, 1e-2, 0.3, 5., 300., 1e5])
                F1, F2 = F12_vec(M, Tt, th)
                for k, t in enumerate(th):
                    f1, f2 = F12_ref(M, Tt, t)
                    worst = max(worst, abs(F1[k] / f1 - 1), abs(F2[k] / f2 - 1))
        print('largest relative deviation of the vectorised quadrature from adaptive quadrature: %.1e' % worst)
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == 'outside':        # six temperatures outside the window, all 624 sets
        with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else None) as pool:
            res = pool.map(one_out, GRID1 + GRID2, chunksize=8)
        nr = np.array([r[1] for r in res])
        print('outside the temperature window: T_t =', TTS_OUT, ' theta window: %g .. %g' % (THETA_OUT[0], THETA_OUT[-1]))
        print('(set, T_t) pairs with exactly one root:', int((nr == 1).sum()), 'of', nr.size,
              '  source/sink strictly decreasing in', sum(r[2] for r in res), 'of', len(res), 'sets')
        pickle.dump(dict(res=res, TTS=TTS_OUT, THETA=THETA_OUT), open('root_scan_outside.pkl', 'wb'))
        sys.exit()
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else None) as pool:
        res = pool.map(one, GRID1 + GRID2, chunksize=4)
    nr = np.array([r[1] for r in res])
    print('parameter sets:', len(res), ' temperatures per set:', len(TTS), ' theta window: %g .. %g' % (THETA[0], THETA[-1]))
    print('(set, T_t) pairs with exactly one root:', int((nr == 1).sum()), 'of', nr.size)
    print('pairs with no root:', int((nr == 0).sum()), '  pairs with more than one root:', int((nr > 1).sum()))
    print('sets in which source/sink is strictly decreasing in theta at every T_t:', sum(r[2] for r in res), 'of', len(res))
    print('largest value to which source/sink returns beyond the root (1 would be a second root): %.3f' % max(r[3] for r in res))
    for r in res:
        if (r[1] != 1).any(): print('  not unique:', r[0], r[1])
    pickle.dump(dict(res=res, TTS=TTS, THETA=THETA), open('root_scan.pkl', 'wb'))
