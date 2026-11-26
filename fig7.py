"""Fig. 7: history dependence of the frozen population and scope of the evidence (initial ratio, system size)."""
import numpy as np, pickle, glob, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3})
R = pickle.load(open('freeze.pkl', 'rb'))
fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.9)); ax = axs.ravel()
sel = [('A: Tt0=100, th0=0.1', 'A', 'C0'), ('B: Tt0=100, th0=1', 'B', 'C3'), ('C: Tt0=100, th0=0.01', 'C', 'C2'), ('D: Tt0=1e4, th0=0.1', 'D', 'C1'),
       ('F: Tt0=100, th0=0.1, mu_k=0.003', 'F', 'C4'), ('G: Tt0=100, th0=0.1, mu_k=0.03', 'G', 'C5'), ('E: Tt0=1, th0=0.1 (start at v_S)', 'E', 'C6')]
# (a) rotational temperature versus time
for lab, short, c in sel:
    o = R[lab]['out']; ax[0].loglog(o[1:, 0], o[1:, 2], '-', c=c, lw=1.2, label=short)
ax[0].loglog(R['A: Tt0=100, th0=0.1']['out'][1:, 0], R['A: Tt0=100, th0=0.1']['out'][1:, 1], 'k--', lw=0.8, label='$T_t$ (run A)')
ax[0].set_xlabel('$t$'); ax[0].set_ylabel('$T_r$'); ax[0].legend(fontsize=6, ncol=2); ax[0].set_ylim(1e-7, 3e2)
# (b) final rotational temperature versus threshold count
fr = [100 * R[l]['nthr'] / R[l]['N'] for l, _, _ in sel]; Tr = [R[l]['Tr_inf'] for l, _, _ in sel]
ax[1].scatter(np.maximum(fr, 3e-3), np.maximum(Tr, 1e-7), c=[c for _, _, c in sel], s=40, zorder=3)
for f, T, (l, s, c) in zip(fr, Tr, sel): ax[1].annotate(s, (max(f, 3e-3), max(T, 1e-7)), textcoords='offset points', xytext=(5, 3), fontsize=8)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlim(2e-3, 300); ax[1].set_xlabel(r'$n_f/N$ [%] at $t=3000$'); ax[1].set_ylabel(r'$T_r(t=3000)$')

def pooled(runs, key, t=1e5):
    k = int(np.argmin([abs(np.log(r['t'] / t)) for r in runs[0]['rows']]))
    return sum(r['rows'][k][key] for r in runs), sum(r['N'] for r in runs)
def errbar(a, x, n, N, **kw):
    """Poisson error bar; for n = 0 an upper limit 3/N (95% confidence) with a downward arrow."""
    if n > 0:
        lo = min(np.sqrt(n), 0.999 * n)
        a.errorbar([x], [n / N], yerr=[[lo / N], [np.sqrt(n) / N]], capsize=2, **kw)
    else:
        kw2 = dict(kw); kw2.pop('label', None)
        a.errorbar([x], [3 / N], yerr=[[1.8 / N], [0]], uplims=True, capsize=2, **kw2)

# (c) dependence on the initial temperature ratio
S = pickle.load(open('scope_size.pkl', 'rb')); T = pickle.load(open('scope_theta.pkl', 'rb'))
T = T + [dict(theta0=0.1, N=r['N'], rows=r['rows']) for r in S]
first = True
for th in sorted({r['theta0'] for r in T}):
    runs = [r for r in T if r['theta0'] == th]
    nf, N = pooled(runs, 'n_f'); nd, _ = pooled(runs, 'n_D')
    errbar(ax[2], th, nf, N, fmt='o', c='C0', ms=4)
    if th <= 0.3: errbar(ax[2], th * 1.06, nd, N, fmt='s', c='C3', ms=4, mfc='none')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ctl = [r for g in sorted(glob.glob('scope_control_*.pkl')) for r in pickle.load(open(g, 'rb'))]     # constant friction, mu = 0.12
nc = sum(r['rows'][-1]['n_D'] for r in ctl); Nc = sum(r['N'] for r in ctl)
ax[2].axhspan((nc - np.sqrt(nc)) / Nc, (nc + np.sqrt(nc)) / Nc, color='0.75', alpha=0.6, lw=0, zorder=0)
ax[2].set_xscale('log'); ax[2].set_yscale('log'); ax[2].set_xlabel(r'initial temperature ratio $\theta_0$'); ax[2].set_ylabel('frozen fraction at $t=10^5$')
ax[2].legend(handles=[Line2D([], [], marker='o', c='C0', ls='', ms=4, label=r'$n_f/N$ (above $v_S$)'),
                      Line2D([], [], marker='s', c='C3', ls='', ms=4, mfc='none', label=r'$n_\Delta/N$ (Eq. 36)'),
                      Patch(color='0.75', alpha=0.6, label=r'$n_\Delta/N$, constant friction')], fontsize=6.5, loc='lower right')
ax[2].set_ylim(5e-7, 2)
# (d) dependence on the system size, DSMC and molecular dynamics
first = True
for N in sorted({r['N'] for r in S}):
    runs = [r for r in S if r['N'] == N]; nf, Nt = pooled(runs, 'n_f'); nd, _ = pooled(runs, 'n_D')
    ax[3].errorbar([N], [100 * nf / Nt], yerr=[100 * np.sqrt(nf) / Nt], fmt='o', c='C0', ms=4, capsize=2, label='DSMC, $n_f/N$' if first else None)
    ax[3].errorbar([N], [100 * nd / Nt], yerr=[100 * np.sqrt(nd) / Nt], fmt='s', c='C3', ms=4, mfc='none', capsize=2, label=r'DSMC, $n_\Delta/N$' if first else None)
    first = False
nf, Nt = pooled(S, 'n_f'); nd, _ = pooled(S, 'n_D')
ax[3].axhline(100 * nf / Nt, c='C0', lw=0.6, ls=':'); ax[3].axhline(100 * nd / Nt, c='C3', lw=0.6, ls=':')
first = True
from md_scope import collect
md = collect(3000.)
for N, v in sorted(md.items()):
    v = np.array(v); Nt = N * len(v); nf, nd = v.sum(0)
    ax[3].errorbar([N * 1.07], [100 * nf / Nt], yerr=[100 * np.sqrt(nf) / Nt], fmt='^', c='k', ms=5, capsize=2, label='MD, $n_f/N$' if first else None)
    ax[3].errorbar([N * 1.07], [100 * nd / Nt], yerr=[100 * np.sqrt(nd) / Nt], fmt='v', c='k', ms=5, mfc='none', capsize=2, label=r'MD, $n_\Delta/N$' if first else None)
    first = False
ax[3].set_xscale('log'); ax[3].set_xlabel('number of particles $N$'); ax[3].set_ylabel('frozen fraction [%] (run A)'); ax[3].set_ylim(0, 0.5)
ax[3].legend(fontsize=6.5, ncol=2, loc='upper right')
for a, l in zip(ax, 'abcd'): a.text(0.03, 0.93, f'({l})', transform=a.transAxes, fontsize=9, va='center')
plt.tight_layout(); plt.savefig('fig7_history.pdf'); print('ok')
