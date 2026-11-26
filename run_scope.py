"""
Scope of the spin-selective freezing result (Sec. 5.4): dependence of the frozen population on
(1) the system size N, (2) the observation time, (3) the initial temperature ratio theta_0,
(4) controls with constant friction mu = mu_s (2.5e6 particles) and (5) with mu = 0.02 (parameters of Fig. 3a);
the two counting criteria are kept apart:

  n_f      particles with contact speed  c = sigma|omega|/2 > v_S           (threshold count, Table 5)
  n_Delta  particles with                c > Delta(t), Eq. (36)              (dynamical decoupling criterion)
  n_und    particles with 6 sqrt(T_t) < c < Delta(t): outside the Gaussian bulk but not yet certainly
           decoupled ("undecided"); n_Delta <= n_infinity <= n_Delta + n_und brackets the final number.

Friction law of run A (e = 0.98, beta_0 = 0, q = 0.4, mu_k = 0.01, mu_s = 0.12, p = 2, v_S = 1), T_t(0) = 100 T_S.
Homogeneous DSMC, same algorithm as dsmc2.dsmc.   Usage:  python3 run_scope.py size|theta|control [seed ...]|control2 [seed ...]|all
Output: scope_size.pkl, scope_theta.pkl, scope_control_<seed>.pkl.   Runtime: size ~15 min, theta ~60 min, control ~50 min (one core each).
"""
import numpy as np, pickle, sys, time
from theory import Model

M = Model(e=0.98, b0=0.0, q=0.4, mu_k=0.01, mu_s=0.12, vS=1.0, p=2)
ZETA = 0.272                                                  # cooling rate of the rough bulk (Sec. 5.2)
COEF = 6 * np.sqrt(np.pi) * (M.mu_s * (1 + M.e) / 2) / (M.q * ZETA)   # Delta(t) = COEF sqrt(T_t/m), Eq. (36)
TSNAP = [3e2, 1e3, 3e3, 1e4, 3e4, 1e5]

def dsmc_c(Tt0, Tr0, N, seed, tsnap=TSNAP, M=M):
    """Returns for each snapshot time a dict with T_t, T_r and the contact speeds c_i = sigma|omega_i|/2."""
    rng = np.random.default_rng(seed); m = M.m; s = 1.; q = M.q; I = q * m * s ** 2 / 4
    v = rng.normal(0, np.sqrt(Tt0 / m), (N, 3)); v -= v.mean(0); w = rng.normal(0, np.sqrt(Tr0 / I), (N, 3))
    t = 0.; cmax = 6 * np.sqrt(2 * Tt0 / m); k = 0; snaps = []
    while k < len(tsnap):
        if t >= tsnap[k]:
            snaps.append(dict(t=t, Tt=m * (v ** 2).sum(1).mean() / 3, Tr=I * (w ** 2).sum(1).mean() / 3,
                              c=0.5 * s * np.linalg.norm(w, axis=1))); k += 1; continue
        p = rng.permutation(N); i, j = p[:N // 2], p[N // 2:2 * (N // 2)]
        n = rng.normal(size=(N // 2, 3)); n /= np.linalg.norm(n, axis=1)[:, None]
        v12 = v[i] - v[j]; un = np.einsum('ij,ij->i', v12, n); cmax = max(cmax, un.max() * 1.05)
        acc = rng.random(N // 2) * cmax < un; i, j, n, v12, un = i[acc], j[acc], n[acc], v12[acc], un[acc]
        Om = w[i] + w[j]; g = v12 - (s / 2) * np.cross(n, Om); gt = g - un[:, None] * n; gtn = np.linalg.norm(gt, axis=1)
        tau = gtn / un; tc = M.c * M.mu(gtn); onepb = (1 + M.b0) * np.minimum(1, tc / np.maximum(tau, 1e-300))
        J = (-(m / 2) * (1 + M.e) * un)[:, None] * n - (m / 2) * M.kt * onepb[:, None] * gt
        v[i] += J / m; v[j] -= J / m; dw = (s / (2 * I)) * np.cross(n, J); w[i] += dw; w[j] += dw
        t += 1 / (4 * np.pi * cmax)
        if rng.random() < 0.02: cmax = max(6 * np.sqrt(2 * (v ** 2).sum(1).mean() / 3), 1e-12)
    return snaps

def analyse(snaps, N, vS=M.vS, coef=COEF, und_lo=6.0):
    # vS = inf for the constant-friction control, which has no Stribeck speed: then n_Delta = #(c > Delta(t)) at all times
    """Counts by the two criteria, energy shares, state of the bulk, identity of the counted particles."""
    rows = []; I = M.q / 4
    for sn in snaps:
        c = sn['c']; Tt = sn['Tt']; D = coef * np.sqrt(Tt); E = c ** 2
        f = c > vS; d = c > min(D, vS); und = (c > und_lo * np.sqrt(Tt)) & ~d; bulk = ~d
        cb = c[bulk]
        rows.append(dict(t=sn['t'], Tt=Tt, Tr=sn['Tr'], Delta=D, n_f=int(f.sum()), n_D=int(d.sum()), n_und=int(und.sum()),
                         share_f=E[f].sum() / E.sum(), share_D=E[d].sum() / E.sum(),
                         theta_bulk=I * 4 * (cb ** 2).mean() / 3 / Tt, a2r_bulk=0.6 * (cb ** 4).mean() / (cb ** 2).mean() ** 2 - 1,
                         idx_f=np.nonzero(f)[0].astype(np.int32) if f.sum() < 50000 else None,
                         idx_D=np.nonzero(d)[0].astype(np.int32) if d.sum() < 50000 else None,
                         c_D=np.sort(c[d]).astype(np.float32) if d.sum() < 50000 else None))
    return rows

def run(label, theta0, N, seed, store, model=M, tsnap=TSNAP, vS=M.vS, coef=COEF, und_lo=6.0):
    t0 = time.time(); rows = analyse(dsmc_c(100., theta0 * 100., N, seed, tsnap=tsnap, M=model), N, vS=vS, coef=coef, und_lo=und_lo)
    store.append(dict(label=label, theta0=theta0, N=N, seed=seed, rows=rows))
    print(f'{label:>8s} theta0={theta0:<5g} N={N:<8d} seed={seed:<3d} [{time.time()-t0:5.0f}s]  ' + '  '.join(
        f"t={r['t']:.0e}: n_f={r['n_f']} n_D={r['n_D']} und={r['n_und']}" for r in rows), flush=True)

def haff_state(Mc):
    """Rough Haff state of the closure for constant friction: temperature ratio theta*, cooling rate zeta,
    coefficient of Delta(t) in Eq. (36) and root-mean-square contact speed in units of sqrt(T_t/m)."""
    from scipy.optimize import brentq
    Tt = 1.0
    f = lambda x: (lambda A, B: B - x * A)(*Mc.rhs_brackets(Tt, x * Tt))      # d(theta)/dt up to a positive factor
    g = np.geomspace(1e-4, 50., 200); v = np.array([f(x) for x in g]); k = np.nonzero((v[:-1] > 0) & (v[1:] < 0))[0][0]
    th = brentq(f, g[k], g[k + 1])                                            # the stable (rough) fixed point
    A, B = Mc.rhs_brackets(Tt, th * Tt); zeta = -A / Tt
    return th, zeta, 6 * np.sqrt(np.pi) * (Mc.mu_s * (1 + Mc.e) / 2) / (Mc.q * zeta), np.sqrt(3 * th / Mc.q)

if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('size', 'all'):          # (1)+(2): run A at different sizes, six observation times
        S = []
        for N, seeds in [(5000, range(100, 116)), (20000, range(200, 208)), (80000, range(300, 302)), (320000, [400])]:
            for sd in seeds: run('A', 0.1, N, sd, S)
            pickle.dump(S, open('scope_size.pkl', 'wb'))
    if which in ('theta', 'all'):         # (3): initial temperature ratio, N large enough to resolve small fractions
        S = []
        for th0, N, seeds in [(1.0, 20000, [500, 501]), (0.5, 20000, [510, 511]), (0.3, 20000, [520, 521]),
                              (0.2, 50000, [530, 531]), (0.07, 200000, [540]), (0.05, 200000, [550]),
                              (0.03, 400000, [560]), (0.02, 1000000, [570]), (0.01, 1000000, [580])]:
            for sd in seeds: run('theta', th0, N, sd, S)
            pickle.dump(S, open('scope_theta.pkl', 'wb'))
    if which in ('control', 'all'):       # (4): constant friction mu = mu_s = 0.12, same initial state as theta_0 = 0.02
        Mc = Model(e=M.e, b0=M.b0, q=M.q, mu_k=M.mu_s, mu_s=M.mu_s, vS=1.0, p=2)
        TC = [30., 100.] + TSNAP             # the constant-friction gas is scale free: counts from t = 30 on
        seeds = [int(a) for a in sys.argv[2:]] or [590, 591, 592]
        for sd in seeds:                     # 2.5 x 10^6 particles in total; one file per realisation
            S = []; run('const', 0.02, 500000 if sd == 590 else 1000000, sd, S, model=Mc, tsnap=TC, vS=np.inf)
            pickle.dump(S, open(f'scope_control_{sd}.pkl', 'wb'))
    if which in ('control2', 'all'):      # (5): second constant-friction rough state, parameters of Fig. 3(a): mu = 0.02
        Mc = Model(e=0.98, b0=0.0, q=0.4, mu_k=0.02, mu_s=0.02, vS=1.0, p=2); th, zeta, coef, rms = haff_state(Mc)
        print(f'control2: theta* = {th:.4f}, zeta = {zeta:.4f}, Delta = {coef:.2f} sqrt(T_t/m), rms contact speed = {rms:.3f} sqrt(T_t/m)', flush=True)
        TC = [30., 100.] + TSNAP
        seeds = [int(a) for a in sys.argv[2:]] or [595, 596]
        for sd in seeds:
            S = []; run('const2', 0.1, 400000, sd, S, model=Mc, tsnap=TC, vS=np.inf, coef=coef, und_lo=4.6 * rms)
            S[0].update(theta_star=th, zeta=zeta, coef=coef, rms=rms)
            pickle.dump(S, open(f'scope_control2_{sd}.pkl', 'wb'))
