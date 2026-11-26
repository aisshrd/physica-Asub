"""
Event-driven molecular dynamics (EDMD) of rough hard spheres with the Walton / Stribeck collision rule.
Independent check of the homogeneous DSMC: real positions, real collision sequence, no molecular chaos assumed.

Units: m = sigma = k_B = 1, I = q m sigma^2/4.  Periodic cubic box, number density n.
Time is reported in the DSMC units (n sigma^2 = 1) after the Enskog rescaling t* = chi(phi) n t_MD,
together with the mean number of collisions per particle.

Requires numba (pip install numba).
"""
import numpy as np
from numba import njit


def chi_cs(phi):
    """Carnahan-Starling pair correlation at contact."""
    return (1 - phi / 2) / (1 - phi) ** 3


@njit(cache=True)
def _next_event(i, tt, r, v, tl, L, N):
    """Earliest collision of particle i with any other particle, starting from time tt."""
    best, jbest = np.inf, -1
    xi = r[i, 0] + v[i, 0] * (tt - tl[i]); yi = r[i, 1] + v[i, 1] * (tt - tl[i]); zi = r[i, 2] + v[i, 2] * (tt - tl[i])
    for j in range(N):
        if j == i:
            continue
        dtj = tt - tl[j]
        dx = xi - (r[j, 0] + v[j, 0] * dtj); dy = yi - (r[j, 1] + v[j, 1] * dtj); dz = zi - (r[j, 2] + v[j, 2] * dtj)
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
        dvx = v[i, 0] - v[j, 0]; dvy = v[i, 1] - v[j, 1]; dvz = v[i, 2] - v[j, 2]
        b = dx * dvx + dy * dvy + dz * dvz
        if b >= 0.0:
            continue
        dv2 = dvx * dvx + dvy * dvy + dvz * dvz
        dr2 = dx * dx + dy * dy + dz * dz
        if dr2 < 1.0:                    # tiny numerical overlap, approaching: collide now
            tc = 0.0
        else:
            disc = b * b - dv2 * (dr2 - 1.0)
            if disc <= 0.0:
                continue
            tc = (-b - np.sqrt(disc)) / dv2
        if tc < best:
            best, jbest = tc, j
    return tt + best, jbest


@njit(cache=True)
def _run(r, v, w, L, q, e, b0, mu_k, mu_s, vS, p, touts, scale, is_snap, wsnap):
    N = r.shape[0]
    I = q / 4.0
    kt = q / (1.0 + q)
    cc = (1.0 + e) * (1.0 + q) / (q * (1.0 + b0))
    tl = np.zeros(N)
    cnt = np.zeros(N, np.int64)
    tnext = np.empty(N); partner = np.empty(N, np.int64); pcnt = np.empty(N, np.int64)
    for i in range(N):
        tnext[i], partner[i] = _next_event(i, 0.0, r, v, tl, L, N)
        pcnt[i] = cnt[partner[i]] if partner[i] >= 0 else 0
    nout = touts.shape[0]
    out = np.zeros((nout, 5))
    k = 0; ncol = 0; t = 0.0; ks = 0
    nc = 4
    while k < nout:
        i = np.argmin(tnext)
        te = tnext[i]
        if te >= touts[k]:
            t = touts[k]
            Tt = 0.0; Tr = 0.0
            for a in range(N):
                Tt += v[a, 0] ** 2 + v[a, 1] ** 2 + v[a, 2] ** 2
                Tr += w[a, 0] ** 2 + w[a, 1] ** 2 + w[a, 2] ** 2
            # homogeneity: kinetic energy carried by the coarse-grained velocity field (4^3 cells)
            vs = np.zeros((nc * nc * nc, 3)); cn = np.zeros(nc * nc * nc)
            for a in range(N):
                ix = int(((r[a, 0] + v[a, 0] * (t - tl[a])) % L) / (L / nc)) % nc
                iy = int(((r[a, 1] + v[a, 1] * (t - tl[a])) % L) / (L / nc)) % nc
                iz = int(((r[a, 2] + v[a, 2] * (t - tl[a])) % L) / (L / nc)) % nc
                c = (ix * nc + iy) * nc + iz
                vs[c, 0] += v[a, 0]; vs[c, 1] += v[a, 1]; vs[c, 2] += v[a, 2]; cn[c] += 1.0
            ef = 0.0
            for c in range(nc * nc * nc):
                if cn[c] > 0:
                    ef += (vs[c, 0] ** 2 + vs[c, 1] ** 2 + vs[c, 2] ** 2) / cn[c]
            out[k, 0] = t * scale; out[k, 1] = Tt / (3.0 * N); out[k, 2] = I * Tr / (3.0 * N)
            out[k, 3] = 2.0 * ncol / N; out[k, 4] = ef / Tt
            if is_snap[k]:                    # store the angular velocities at this output time
                for a in range(N):
                    wsnap[ks, a, 0] = w[a, 0]; wsnap[ks, a, 1] = w[a, 1]; wsnap[ks, a, 2] = w[a, 2]
                ks += 1
            k += 1
            continue
        j = partner[i]
        if j < 0:
            break
        if cnt[j] != pcnt[i]:            # partner has collided meanwhile: reschedule i
            tnext[i], partner[i] = _next_event(i, t, r, v, tl, L, N)
            pcnt[i] = cnt[partner[i]] if partner[i] >= 0 else 0
            continue
        t = te
        for a in (i, j):
            for d in range(3):
                r[a, d] = (r[a, d] + v[a, d] * (t - tl[a])) % L
            tl[a] = t
        dx = r[i, 0] - r[j, 0]; dy = r[i, 1] - r[j, 1]; dz = r[i, 2] - r[j, 2]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
        d = np.sqrt(dx * dx + dy * dy + dz * dz)
        n = np.array([-dx / d, -dy / d, -dz / d])            # unit vector from i to j
        v12 = v[i] - v[j]
        gn = v12[0] * n[0] + v12[1] * n[1] + v12[2] * n[2]
        if gn > 0.0:
            Om = w[i] + w[j]
            cr = np.array([n[1] * Om[2] - n[2] * Om[1], n[2] * Om[0] - n[0] * Om[2], n[0] * Om[1] - n[1] * Om[0]])
            g = v12 - 0.5 * cr
            gt = g - gn * n
            gtn = np.sqrt(gt[0] ** 2 + gt[1] ** 2 + gt[2] ** 2)
            mu = mu_k + (mu_s - mu_k) * np.exp(-(gtn / vS) ** (2.0 * p))
            tau = gtn / gn
            onepb = (1.0 + b0) * min(1.0, cc * mu / max(tau, 1e-300))
            J = -0.5 * (1.0 + e) * gn * n - 0.5 * kt * onepb * gt
            for dd in range(3):
                v[i, dd] += J[dd]; v[j, dd] -= J[dd]
            nj = np.array([n[1] * J[2] - n[2] * J[1], n[2] * J[0] - n[0] * J[2], n[0] * J[1] - n[1] * J[0]])
            for dd in range(3):
                w[i, dd] += 0.5 / I * nj[dd]; w[j, dd] += 0.5 / I * nj[dd]
            ncol += 1
        cnt[i] += 1; cnt[j] += 1
        tnext[i], partner[i] = _next_event(i, t, r, v, tl, L, N)
        pcnt[i] = cnt[partner[i]] if partner[i] >= 0 else 0
        tnext[j], partner[j] = _next_event(j, t, r, v, tl, L, N)
        pcnt[j] = cnt[partner[j]] if partner[j] >= 0 else 0
    return out


def run_md(M, Tt0, Tr0, n=0.05, nside=16, t_end=200.0, seed=0, nout=60, snap_times=None):
    """M: theory.Model.  Returns (out, v, w, info); out columns: t*, T_t, T_r, collisions/particle, flow-energy fraction.
    snap_times: optional list of times t* at which the angular velocities are stored; they are returned as
    info['snaps'] = [(t*, T_t, w), ...] and the corresponding rows are added to out."""
    rng = np.random.default_rng(seed)
    N = nside ** 3
    L = (N / n) ** (1 / 3)
    q = M.q; I = q / 4.0
    phi = np.pi * n / 6; chi = chi_cs(phi); scale = chi * n
    g = (np.arange(nside) + 0.5) * L / nside
    r = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T.copy()
    r += rng.uniform(-0.2, 0.2, r.shape)
    v = rng.normal(0, np.sqrt(Tt0), (N, 3)); v -= v.mean(0)
    w = rng.normal(0, np.sqrt(Tr0 / I), (N, 3))
    tstd = np.concatenate([[0.0], np.logspace(-2, np.log10(t_end), nout)])
    tsn = np.array(sorted(snap_times)) if snap_times is not None else np.zeros(0)
    tall = np.concatenate([tstd, tsn]); order = np.argsort(tall, kind='stable')
    touts = tall[order] / scale; is_snap = (order >= len(tstd))
    wsnap = np.zeros((len(tsn), N, 3))
    out = _run(r, v, w, L, q, M.e, M.b0, M.mu_k, M.mu_s, M.vS, float(M.p), touts, scale, is_snap, wsnap)
    info = dict(N=N, L=L, n=n, phi=phi, chi=chi)
    if snap_times is not None:
        info['snaps'] = [(out[k, 0], out[k, 1], wsnap[i]) for i, k in enumerate(np.nonzero(is_snap)[0])]
    return out, v, w, info


if __name__ == "__main__":
    import pickle, time, sys
    from theory import Model
    which = sys.argv[1] if len(sys.argv) > 1 else "abcsB"
    cases = {
        # constant friction (Fig. 3)
        "a": dict(M=Model(e=0.98, b0=0.0, q=0.4, mu_k=0.02, mu_s=0.02), Tt0=1.0, Tr0=0.1, t_end=3000.0, nside=16, seeds=2),
        "b": dict(M=Model(e=0.98, b0=0.0, q=0.4, mu_k=0.02, mu_s=0.02), Tt0=1.0, Tr0=10.0, t_end=3000.0, nside=16, seeds=2),
        "c": dict(M=Model(e=0.90, b0=0.0, q=0.4, mu_k=0.02, mu_s=0.02), Tt0=1.0, Tr0=0.1, t_end=3000.0, nside=16, seeds=2),
        # Stribeck friction: run A (small frozen fraction) and run B (large frozen fraction) of Table 3
        "s": dict(M=Model(e=0.98, b0=0.0, q=0.4, mu_k=0.01, mu_s=0.12, vS=1.0, p=2), Tt0=100.0, Tr0=10.0, t_end=3000.0, nside=16, seeds=3),
        "B": dict(M=Model(e=0.98, b0=0.0, q=0.4, mu_k=0.01, mu_s=0.12, vS=1.0, p=2), Tt0=100.0, Tr0=100.0, t_end=3000.0, nside=16, seeds=1),
    }
    res = {}
    for key in which:
        c = cases[key]; runs = []
        for seed in range(c["seeds"]):
            t0 = time.time()
            out, v, w, info = run_md(c["M"], c["Tt0"], c["Tr0"], nside=c["nside"], t_end=c["t_end"], seed=seed)
            runs.append(dict(out=out, v=v, w=w))
            print(key, seed, f"{time.time()-t0:.0f}s", "final", out[-1], info, flush=True)
        res[key] = dict(runs=runs, info=info, pars=dict(e=c["M"].e, mu_k=c["M"].mu_k, mu_s=c["M"].mu_s, Tt0=c["Tt0"], Tr0=c["Tr0"]))
        pickle.dump(res, open(f"md_{which}.pkl", "wb"))
