"""Collects the numbers quoted in Sec. 5.4 (scope of spin-selective freezing) and in Sec. 6.3 (homogeneity of the molecular-dynamics runs) from run_scope.py and md_scope.py."""
import numpy as np, pickle, glob, os
pc = lambda n, N: f'{100*n/N:.3g}%'

def pooled(runs, k, key):
    n = sum(r['rows'][k][key] for r in runs); N = sum(r['N'] for r in runs); return n, N

def time_index(rows, t):  return int(np.argmin([abs(np.log(r['t'] / t)) for r in rows]))

print('=== (1) DSMC, run A: system size (pooled over realisations; +- = Poisson error of the pooled count) ===')
S = pickle.load(open('scope_size.pkl', 'rb'))
for N in sorted({r['N'] for r in S}):
    runs = [r for r in S if r['N'] == N]; line = f'N={N:<7d} realisations={len(runs):<3d}'
    for t in [3e3, 1e5]:
        k = time_index(runs[0]['rows'], t)
        nf, Nt = pooled(runs, k, 'n_f'); nd, _ = pooled(runs, k, 'n_D')
        line += f' | t={t:.0e}: n_f/N={100*nf/Nt:.3f}+-{100*np.sqrt(nf)/Nt:.3f}%  n_D/N={100*nd/Nt:.3f}+-{100*np.sqrt(nd)/Nt:.3f}%'
    k = time_index(runs[0]['rows'], 1e5)
    fr = np.array([r['rows'][k]['n_f'] / r['N'] for r in runs])
    line += f' | seed scatter of n_f/N: {100*fr.min():.3f}-{100*fr.max():.3f}%'
    line += f" | share_f={np.mean([r['rows'][k]['share_f'] for r in runs]):.3f} share_D={np.mean([r['rows'][k]['share_D'] for r in runs]):.4f}"
    line += f" | theta_bulk={np.mean([r['rows'][k]['theta_bulk'] for r in runs]):.3f} a2r_bulk={np.mean([r['rows'][k]['a2r_bulk'] for r in runs]):.3f}"
    print(line)
allr = S; k = time_index(S[0]['rows'], 1e5)
nf, Nt = pooled(allr, k, 'n_f'); nd, _ = pooled(allr, k, 'n_D')
print(f'all sizes pooled ({Nt} particles): n_f/N={100*nf/Nt:.4f}+-{100*np.sqrt(nf)/Nt:.4f}%  n_D/N={100*nd/Nt:.4f}+-{100*np.sqrt(nd)/Nt:.4f}%  n_D/n_f={nd/nf:.2f}')

print('\n=== (2) DSMC, run A: observation time ===')
big = max(S, key=lambda r: r['N'])
print(f"N={big['N']}:")
ref_f = set(big['rows'][-1]['idx_f']); ref_D = set(big['rows'][-1]['idx_D'])
for r in big['rows']:
    f = set(r['idx_f']); d = set(r['idx_D'])
    print(f"  t={r['t']:9.0f}  T_t={r['Tt']:.2e}  Delta={r['Delta']:.2e}  n_f={r['n_f']}  n_D={r['n_D']}  undecided={r['n_und']}  "
          f"T_r={r['Tr']:.4e}  share_f={r['share_f']:.3f}  share_D={r['share_D']:.4f}  theta_bulk={r['theta_bulk']:.3f}  a2r_bulk={r['a2r_bulk']:.3f}  "
          f"final n_f set contained: {len(ref_f & f)}/{len(ref_f)}; counted now but not at the end: n_f {len(f - ref_f)}, n_D {len(d - ref_D)}")
print('all realisations pooled:')
for k in range(len(S[0]['rows'])):
    nf, Nt = pooled(S, k, 'n_f'); nd, _ = pooled(S, k, 'n_D'); nu, _ = pooled(S, k, 'n_und')
    print(f"  t={S[0]['rows'][k]['t']:9.0f}  n_f={nf} ({100*nf/Nt:.4f}%)  n_D={nd} ({100*nd/Nt:.4f}%)  undecided={nu}")

print('reliability of the dynamical criterion (particles satisfying it at time t that no longer satisfy it at the last time):')
for k in range(len(S[0]['rows']) - 1):
    lost = sum(len(set(r['rows'][k]['idx_D']) - set(r['rows'][-1]['idx_D'])) for r in S); tot = sum(r['rows'][k]['n_D'] for r in S)
    lostf = sum(len(set(r['rows'][k]['idx_f']) - set(r['rows'][-1]['idx_f'])) for r in S); totf = sum(r['rows'][k]['n_f'] for r in S)
    print(f"  t={S[0]['rows'][k]['t']:9.0f}: n_D {lost} of {tot};   threshold count n_f: {lostf} of {totf}")
print('\n=== (3) DSMC: initial temperature ratio (T_t(0)=100 T_S), counts at the last time ===')
T = pickle.load(open('scope_theta.pkl', 'rb')); T = T + [dict(theta0=0.1, N=r['N'], rows=r['rows']) for r in S]
for th in sorted({r['theta0'] for r in T}, reverse=True):
    runs = [r for r in T if r['theta0'] == th]; line = f'theta0={th:<5g} particles={sum(r["N"] for r in runs):<8d}'
    for t in [3e3, 1e5]:
        k = time_index(runs[0]['rows'], t); nf, Nt = pooled(runs, k, 'n_f'); nd, _ = pooled(runs, k, 'n_D'); nu, _ = pooled(runs, k, 'n_und')
        line += f' | t={t:.0e}: n_f={nf} ({nf/Nt:.2e})  n_D={nd} ({nd/Nt:.2e})  und={nu}'
    k = time_index(runs[0]['rows'], 1e5)
    line += f" | T_r={np.mean([r['rows'][k]['Tr'] for r in runs]):.2e} theta_bulk={np.mean([r['rows'][k]['theta_bulk'] for r in runs]):.3f} a2r_bulk={np.mean([r['rows'][k]['a2r_bulk'] for r in runs]):.2f}"
    cD = np.concatenate([r['rows'][k]['c_D'] for r in runs if r['rows'][k].get('c_D') is not None] or [np.zeros(0)])
    if 0 < len(cD) < 40: line += ' | contact speeds of the decoupled particles: ' + ' '.join(f'{x:.2g}' for x in cD)
    print(line)

print('\n=== (4) Event-driven MD, run A ===')
for f in sorted(glob.glob('md_scope_*.pkl')):
    R = pickle.load(open(f, 'rb')); N = R[0]['N']; line = f"{f}: N={N} L={R[0]['L']:.1f} realisations={len(R)} max flow fraction={max(r['out'][:,4].max() for r in R):.3f} (64/N={64/N:.3f})"
    print(line)
    for k in range(len(R[0]['rows'])):
        nf = sum(r['rows'][k]['n_f'] for r in R); nd = sum(r['rows'][k]['n_D'] for r in R); nu = sum(r['rows'][k]['n_und'] for r in R); Nt = N * len(R)
        lost = sum(len(set(r['rows'][k]['idx_D']) - set(r['rows'][-1]['idx_D'])) for r in R)
        print(f"   t={R[0]['rows'][k]['t']:7.0f}: n_f={nf} ({100*nf/Nt:.3f}+-{100*np.sqrt(nf)/Nt:.3f}%)  n_D={nd} ({100*nd/Nt:.3f}+-{100*np.sqrt(nd)/Nt:.3f}%)  undecided={nu}  "
              f"per realisation n_f: {[r['rows'][k]['n_f'] for r in R]}  n_D: {[r['rows'][k]['n_D'] for r in R]}  "
              f"T_r={np.mean([r['rows'][k]['Tr'] for r in R]):.3e} ({min(r['rows'][k]['Tr'] for r in R):.2e}-{max(r['rows'][k]['Tr'] for r in R):.2e})  theta_bulk={np.mean([r['rows'][k]['theta_bulk'] for r in R]):.3f}  a2r_bulk={np.mean([r['rows'][k]['a2r_bulk'] for r in R]):.2f}  lost later (n_D): {lost}")

print('\n=== (5) DSMC, slow late decoupling: small theta_0 (Stribeck) and constant-friction control ===')
def late(label, runs):
    Nt = sum(r['N'] for r in runs); ts = [x['t'] for x in runs[0]['rows']]
    nD = [sum(r['rows'][k]['n_D'] for r in runs) for k in range(len(ts))]; nu = [sum(r['rows'][k]['n_und'] for r in runs) for k in range(len(ts))]
    k0 = int(np.argmin([abs(np.log(t / 300.)) for t in ts])); new = nD[-1] - nD[k0]; ef = np.log(ts[-1] / ts[k0])
    lost = sum(len(set(r['rows'][k]['idx_D']) - set(r['rows'][-1]['idx_D'])) for r in runs for k in range(len(ts) - 1))
    print(f"{label}: {Nt} particles; n_D(t) at t={[float('%.0e' % t) for t in ts]}: {nD}; undecided: {nu}")
    print(f"     newly counted between t=300 and t=1e5: {new}  ->  rate {new/Nt/ef:.2e} (+-{np.sqrt(max(new,1))/Nt/ef:.1e}) per particle per e-fold of time; counted earlier but not at the end: {lost}")
    for r in runs:
        x = r['rows'][-1]; print(f"     N={r['N']} seed={r.get('seed','-')}: T_r/T_t at the last time {x['Tr']/x['Tt']:.3g}, theta_bulk {x['theta_bulk']:.3f}, a2r_bulk {x['a2r_bulk']:.2f}, contact speeds of the counted particles: " + ' '.join('%.2g' % c for c in x['c_D'][:25]))
late('Stribeck, theta_0 = 0.01 and 0.02', [r for r in pickle.load(open('scope_theta.pkl', 'rb')) if r['theta0'] <= 0.02])
def rate_all(label, runs):
    Nt = sum(r['N'] for r in runs); ts = [x['t'] for x in runs[0]['rows']]
    nD = [sum(r['rows'][k]['n_D'] for r in runs) for k in range(len(ts))]; new = nD[-1] - nD[0]; ef = np.log(ts[-1] / ts[0])
    print(f"     {label}: from t={ts[0]:.0f} to t={ts[-1]:.0f}: {new} newly counted among {Nt} -> rate {new/Nt/ef:.2e} +- {np.sqrt(max(new,1))/Nt/ef:.1e} per particle per e-fold; fraction counted at the end {nD[-1]/Nt:.1e}")
    for r in runs:
        tt = [x['Tr'] / x['Tt'] for x in r['rows']]; tb = [x['theta_bulk'] for x in r['rows']]; ab = [x['a2r_bulk'] for x in r['rows']]
        print(f"        N={r['N']} seed={r['seed']}: T_r/T_t of the whole gas: " + ' '.join('%.3g' % v for v in tt) + f"   bulk theta {min(tb):.4f}-{max(tb):.4f}, bulk a2r {min(ab):.2f}-{max(ab):.2f}")
ctl = [r for f in sorted(glob.glob('scope_control_*.pkl')) for r in pickle.load(open(f, 'rb'))]
if ctl: late('constant friction mu = 0.12 (control)', ctl); rate_all('constant friction mu = 0.12', ctl)
ctl2 = [r for f in sorted(glob.glob('scope_control2_*.pkl')) for r in pickle.load(open(f, 'rb'))]
if ctl2:
    print(f"constant friction mu = 0.02 (rough state of Fig. 3a): closure theta* = {ctl2[0]['theta_star']:.4f}, zeta = {ctl2[0]['zeta']:.4f}, Delta = {ctl2[0]['coef']:.2f} sqrt(T_t), rms contact speed {ctl2[0]['rms']:.3f} sqrt(T_t)")
    late('constant friction mu = 0.02 (control 2)', ctl2); rate_all('constant friction mu = 0.02', ctl2)

from md_scope import collect
print('\nmolecular dynamics, all realisations at t=3000 by system size (including the three N=4096 runs of md_event.py):')
for N, v in sorted(collect(3000.).items()):
    v = np.array(v); Nt = N * len(v); nf, nd = v.sum(0)
    print(f'  N={N:<6d} realisations={len(v):<3d} n_f={nf} ({100*nf/Nt:.3f}+-{100*np.sqrt(nf)/Nt:.3f}%)   n_D={nd} ({100*nd/Nt:.3f}+-{100*np.sqrt(nd)/Nt:.3f}%)   n_D/n_f={nd/nf:.2f}')

print('\n=== (6) Event-driven MD: homogeneity (Sec. 6.3) ===')
print('fraction of the translational kinetic energy carried by the velocity field coarse-grained on 4^3 cells;')
print('for uncorrelated velocities it equals 64/N')
def flow_stats(label, runs, N):
    allv = np.concatenate([r['out'][:, 4] for r in runs]); late_ = np.concatenate([r['out'][r['out'][:, 0] >= 300, 4] for r in runs])
    e = late_.std(ddof=1) / np.sqrt(len(late_))
    print(f'  {label:<34s} runs={len(runs):<3d} range {allv.min():.4f}-{allv.max():.4f}   64/N={64/N:.4f}   mean over t>=300 divided by 64/N: {late_.mean()/(64/N):.3f}+-{e/(64/N):.3f}')
old = []
for f in ['md_abc.pkl', 'md_abc_t300.pkl', 'md_sB.pkl']:
    if os.path.exists(f):
        O = pickle.load(open(f, 'rb'))
        for k in O: old += O[k]['runs']
groups = {}
for f in sorted(glob.glob('md_scope_*.pkl')):
    for r in pickle.load(open(f, 'rb')): groups.setdefault(r['N'], []).append(r)
if old: flow_stats('N=4096, runs of Fig. 8 (all cases)', old + groups.get(4096, []), 4096)
runA = list(groups.get(4096, []))
if os.path.exists('md_sB.pkl'): runA += pickle.load(open('md_sB.pkl', 'rb'))['s']['runs']
for N in sorted(groups):
    flow_stats(f'N={N}, run A', runA if N == 4096 else groups[N], N)
def Tt_at(runs, t):
    v = []
    for r in runs:
        o = r['out']; m = o[:, 0] > 0; v.append(np.exp(np.interp(np.log(t), np.log(o[m, 0]), np.log(o[m, 1]))))
    return np.mean(v)
if runA:
    for N in sorted(groups):
        if N != 4096:
            dev = max(abs(Tt_at(groups[N], t) / Tt_at(runA, t) - 1) for t in np.logspace(0, np.log10(3000.), 40))
            print(f'  T_t(N={N})/T_t(N=4096) - 1, maximum over 1 <= t <= 3000: {100*dev:.1f}%')
