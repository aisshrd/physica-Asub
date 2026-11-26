"""
Heated gas, direct test for coexisting steady states without the closure (Sec. 5.1 (iii)).
For three friction laws (the one of Fig. 4 and the two parameter sets with the flattest heating curves of the
scans) the DSMC is started from four widely different initial conditions,
   hot / cold translational temperature   x   slow / fast rotation (contact speeds below / far above v_S),
and the steady translational temperature, temperature ratio and rotational kurtosis are compared.  The fast-rotation starts relax slowly (rate set by mu_k), hence the long runs.
Usage: python3 run_driven2.py fig4|flat|all.   Output: driven2_<arg>.pkl.   Runtime ~35 min (fig4), ~15 min (flat), N = 5000.
"""
import numpy as np, pickle, time
from theory import Model
from sliding_check import steady
from scipy.optimize import brentq

def dsmc_driven(M, Tt0, Tr0, h, t_end, N=5000, seed=0, nout=200):
    """Same algorithm as dsmc2.dsmc with the white-noise thermostat; returns t, T_t, T_r, kurtosis of omega_x."""
    rng = np.random.default_rng(seed); m = M.m; s = 1.; q = M.q; I = q * m * s ** 2 / 4
    v = rng.normal(0, np.sqrt(Tt0 / m), (N, 3)); v -= v.mean(0); w = rng.normal(0, np.sqrt(Tr0 / I), (N, 3))
    t = 0.; cmax = 6 * np.sqrt(2 * Tt0 / m); touts = np.linspace(0, t_end, nout + 1); k = 0; out = []
    while k < len(touts):
        if t >= touts[k]:
            out.append((t, m * (v ** 2).sum(1).mean() / 3, I * (w ** 2).sum(1).mean() / 3,
                        (w[:, 0] ** 4).mean() / (w[:, 0] ** 2).mean() ** 2)); k += 1; continue
        p = rng.permutation(N); i, j = p[:N // 2], p[N // 2:2 * (N // 2)]
        n = rng.normal(size=(N // 2, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
        v12 = v[i] - v[j]; un = np.einsum('ij,ij->i', v12, n); cmax = max(cmax, un.max() * 1.05)
        acc = rng.random(N // 2) * cmax < un; i, j, n, v12, un = i[acc], j[acc], n[acc], v12[acc], un[acc]
        Om = w[i] + w[j]; g = v12 - (s / 2) * np.cross(n, Om); gt = g - un[:, None] * n; gtn = np.linalg.norm(gt, axis=1)
        tau = gtn / un; tc = M.c * M.mu(gtn); onepb = (1 + M.b0) * np.minimum(1, tc / np.maximum(tau, 1e-300))
        J = (-(m / 2) * (1 + M.e) * un)[:, None] * n - (m / 2) * M.kt * onepb[:, None] * gt
        v[i] += J / m; v[j] -= J / m; dw = (s / (2 * I)) * np.cross(n, J); w[i] += dw; w[j] += dw
        dt = 1 / (4 * np.pi * cmax)
        v += rng.normal(0, np.sqrt(h * dt / m), (N, 3)); v -= v.mean(0)
        t += dt
        if rng.random() < 0.02: cmax = max(6 * np.sqrt(2 * (v ** 2).sum(1).mean() / 3), 1e-12)
    return np.array(out)

SETS = {'Fig. 4':   (dict(e=0.98, b0=0.0, q=0.4, mu_k=0.01, mu_s=0.2, p=2),  [(3e-4, 20000.), (1e-3, 15000.), (3e-3, 10000.), (1e-2, 6000.)]),
        'flat 1':   (dict(e=0.995, b0=0.0, q=0.4, mu_k=0.003, mu_s=0.8, p=8), [(0.024, 4000.)]),
        'flat 2':   (dict(e=0.99, b0=1.0, q=0.5, mu_k=0.003, mu_s=0.2, p=8),  [(0.018, 4000.)])}
STARTS = [('hot, slow rotation', 5.0, 0.5), ('cold, slow rotation', 0.002, 0.0002),
          ('cold, fast rotation', 0.002, 2.0), ('hot, fast rotation', 5.0, 20.0)]

if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'          # 'fig4', 'flat' or 'all'
    sel = {'fig4': ['Fig. 4'], 'flat': ['flat 1', 'flat 2'], 'all': list(SETS)}[which]
    res = []
    for name in sel:
        par, hl = SETS[name]
        M = Model(vS=1., **par)
        for h, t_end in hl:
            Tc = brentq(lambda lT: np.log(steady(M, np.exp(lT))[0] / h), np.log(1e-4), np.log(1e2)); Tc = np.exp(Tc)
            thc = steady(M, Tc)[1]; rows = []
            for sd, (lab, Tt0, Tr0) in enumerate(STARTS):
                t0 = time.time(); o = dsmc_driven(M, Tt0, Tr0, h, t_end, seed=sd)
                n = len(o); a = o[int(0.8 * n):]; b = o[int(0.6 * n):int(0.8 * n)]
                rows.append(dict(start=lab, Tt=a[:, 1].mean(), theta=(a[:, 2] / a[:, 1]).mean(), K=a[:, 3].mean(),
                                 Tt_prev=b[:, 1].mean(), theta_prev=(b[:, 2] / b[:, 1]).mean(), out=o))
                print(f"{name:7s} h={h:<7g} {lab:20s} T_t={rows[-1]['Tt']:.5f} (prev. window {rows[-1]['Tt_prev']:.5f})  "
                      f"theta={rows[-1]['theta']:.4f} ({rows[-1]['theta_prev']:.4f})  K={rows[-1]['K']:.2f}  [{time.time()-t0:.0f}s]", flush=True)
            T = np.array([r['Tt'] for r in rows]); th = np.array([r['theta'] for r in rows])
            print(f"   -> spread between the four starts: T_t {100*(T.max()/T.min()-1):.1f}%, theta {100*(th.max()/th.min()-1):.1f}%;"
                  f"  closure: T_t={Tc:.5f}, theta*={thc:.4f}", flush=True)
            res.append(dict(set=name, par=par, h=h, t_end=t_end, rows=rows, Tt_closure=Tc, theta_closure=thc))
            pickle.dump(res, open(f'driven2_{which}.pkl', 'wb'))
