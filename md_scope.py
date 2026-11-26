"""
Event-driven molecular dynamics of run A (Stribeck friction) at three system sizes and, for N = 4096, up to
t = 3 x 10^4: system-size and observation-time dependence of the frozen population (Sec. 5.4), with the
threshold count n_f and the dynamical count n_Delta of Eq. (36) kept apart (see run_scope.py).
Usage: python3 md_scope.py small|long|large [seed]     Output: md_scope_<case>.pkl
Runtime (one core): small ~10 min, long ~20 min, large ~50 min per realisation.
"""
import numpy as np, pickle, sys, time
from md_event import run_md
from run_scope import M, COEF

def analyse(snaps):
    rows = []; I = M.q / 4
    for t, Tt, w in snaps:
        c = 0.5 * np.linalg.norm(w, axis=1); D = COEF * np.sqrt(Tt); E = c ** 2
        f = c > M.vS; d = c > min(D, M.vS); und = (c > 6 * np.sqrt(Tt)) & ~d; cb = c[~d]
        rows.append(dict(t=t, Tt=Tt, Tr=I * 4 * E.mean() / 3, Delta=D, n_f=int(f.sum()), n_D=int(d.sum()), n_und=int(und.sum()),
                         share_f=E[f].sum() / E.sum(), share_D=E[d].sum() / E.sum(),
                         theta_bulk=I * 4 * (cb ** 2).mean() / 3 / Tt, a2r_bulk=0.6 * (cb ** 4).mean() / (cb ** 2).mean() ** 2 - 1,
                         idx_f=np.nonzero(f)[0], idx_D=np.nonzero(d)[0]))
    return rows

def collect(t=3000.):
    """Counts (n_f, n_Delta) at time t for all molecular-dynamics realisations of run A, by system size:
    the runs of this script and the three N = 4096 realisations of md_event.py (md_sB.pkl), which end at t = 3000."""
    import glob, os
    md = {}
    for f in sorted(glob.glob('md_scope_*.pkl')):
        for r in pickle.load(open(f, 'rb')):
            k = int(np.argmin([abs(np.log(x['t'] / t)) for x in r['rows']]))
            if abs(np.log(r['rows'][k]['t'] / t)) < 0.05: md.setdefault(r['N'], []).append((r['rows'][k]['n_f'], r['rows'][k]['n_D']))
    if os.path.exists('md_sB.pkl') and abs(t - 3000.) < 1:
        for run in pickle.load(open('md_sB.pkl', 'rb'))['s']['runs']:
            c = 0.5 * np.linalg.norm(run['w'], axis=1); D = COEF * np.sqrt(run['out'][-1, 1])
            md.setdefault(len(c), []).append((int((c > M.vS).sum()), int((c > D).sum())))
    return md

CASES = {'small': dict(nside=12, t_end=3000., seeds=range(20, 32), snaps=[300., 1000., 3000.]),
         'long':  dict(nside=16, t_end=30000., seeds=range(40, 43), snaps=[300., 1000., 3000., 10000., 30000.]),
         'large': dict(nside=24, t_end=3000., seeds=range(60, 62), snaps=[300., 1000., 3000.])}

if __name__ == '__main__':
    case = sys.argv[1]; c = CASES[case]
    seeds = [int(sys.argv[2])] if len(sys.argv) > 2 else list(c['seeds'])
    tag = case + (('_%d' % seeds[0]) if len(sys.argv) > 2 else ''); S = []
    for sd in seeds:
        t0 = time.time()
        out, v, w, info = run_md(M, 100., 10., nside=c['nside'], t_end=c['t_end'], seed=sd, snap_times=c['snaps'])
        rows = analyse(info['snaps'])
        S.append(dict(N=info['N'], L=info['L'], seed=sd, out=out, rows=rows))
        print(f"{case} N={info['N']} seed={sd} [{time.time()-t0:.0f}s] col/part={out[-1,3]:.0f} flow={out[:,4].max():.3f}  " + '  '.join(
            f"t={r['t']:.0f}: n_f={r['n_f']} n_D={r['n_D']} und={r['n_und']} thb={r['theta_bulk']:.3f}" for r in rows), flush=True)
        pickle.dump(S, open(f'md_scope_{tag}.pkl', 'wb'))
