"""Writes the LaTeX body of Table 6 (scope of the evidence for run A: observation time and system size)."""
import numpy as np, pickle
from md_scope import collect
S = pickle.load(open('scope_size.pkl', 'rb'))
L = pickle.load(open('md_scope_long.pkl', 'rb'))
def sci(x):
    e = int(np.floor(np.log10(x))); m = float('%.2g' % (x / 10 ** e))
    if m >= 10: m /= 10; e += 1
    return (r'$10^{%d}$' % e) if m == 1 else (r'$%g\times10^{%d}$' % (m, e))
Ntot = sum(r['N'] for r in S)
A = [r'\noindent\emph{(a) Observation time.} DSMC: $%s$ particles in 27 runs with $N=5\times10^3$ to $3.2\times10^5$; MD: three runs with $N=4096$.\\[3pt]' % sci(Ntot).strip('$'),
     r'\begin{tabular}{ccccccccc}\toprule',
     r' & & & \multicolumn{3}{c}{DSMC} & & \multicolumn{2}{c}{MD}\\\cmidrule{4-6}\cmidrule{8-9}',
     r'$t$ & $\Tt/T_S$ & $\Delta/v_S$ & $n_f$ & $n_\Delta$ & $n_u$ & & $n_f$ & $n_\Delta$\\\midrule']
big = max(S, key=lambda r: r['N'])
for k in range(len(S[0]['rows'])):
    nf = sum(r['rows'][k]['n_f'] for r in S); nd = sum(r['rows'][k]['n_D'] for r in S); nu = sum(r['rows'][k]['n_und'] for r in S)
    t = [3e2, 1e3, 3e3, 1e4, 3e4, 1e5][k]; b = big['rows'][k]
    if k < len(L[0]['rows']):
        mf = sum(r['rows'][k]['n_f'] for r in L); md = sum(r['rows'][k]['n_D'] for r in L); mdtxt = f'{mf} & {md}'
    else: mdtxt = '-- & --'
    A.append(f"{sci(t)} & {sci(b['Tt'])} & {sci(b['Delta'])} & {nf} & {nd} & {nu} & & {mdtxt}\\\\")
A.append(r'\bottomrule\end{tabular}\\[8pt]')
B = [r'\noindent\emph{(b) System size.} Counts at $t=3000$; $\pm$: Poisson error of the pooled count.\\[3pt]',
     r'\begin{tabular}{lcccc}\toprule',
     r'method & $N$ & runs & $n_f/N$ [\%] & $n_\Delta/N$ [\%]\\\midrule']
k3 = 2
for N in sorted({r['N'] for r in S}):
    runs = [r for r in S if r['N'] == N]; Nt = N * len(runs)
    nf = sum(r['rows'][k3]['n_f'] for r in runs); nd = sum(r['rows'][k3]['n_D'] for r in runs)
    B.append(f"DSMC & {sci(N)} & {len(runs)} & ${100*nf/Nt:.3f}\\pm{100*np.sqrt(nf)/Nt:.3f}$ & ${100*nd/Nt:.3f}\\pm{100*np.sqrt(nd)/Nt:.3f}$\\\\")
for N, v in sorted(collect(3000.).items()):
    v = np.array(v); Nt = N * len(v); nf, nd = v.sum(0); Lbox = (N / 0.05) ** (1 / 3)
    B.append(f"MD ($L={Lbox:.1f}\\,\\sigma$) & ${N}$ & {len(v)} & ${100*nf/Nt:.3f}\\pm{100*np.sqrt(nf)/Nt:.3f}$ & ${100*nd/Nt:.3f}\\pm{100*np.sqrt(nd)/Nt:.3f}$\\\\")
B.append(r'\bottomrule\end{tabular}')
rows = A + B
open('tab_scope.tex', 'w').write('\n'.join(rows) + '\n'); print('\n'.join(rows))

ru = '\n'.join(rows)
for a, b in [(r'\emph{(a) Observation time.} DSMC:', r'\emph{(a) Время наблюдения.} DSMC:'), ('particles in 27 runs with', 'частиц в 27 прогонах с'),
             (r'$N=5\times10^3$ to $3.2\times10^5$; MD: three runs with $N=4096$.', r'$N=5\times10^3$--$3.2\times10^5$; МД: три прогона с $N=4096$.'),
             (r'\multicolumn{2}{c}{MD}', r'\multicolumn{2}{c}{МД}'),
             (r'\emph{(b) System size.} Counts at $t=3000$; $\pm$: Poisson error of the pooled count.', r'\emph{(b) Размер системы.} Счёт при $t=3000$; $\pm$ --- пуассоновская ошибка объединённого счёта.'),
             ('method &', 'метод &'), ('& runs &', '& прогонов &'), ('MD ($L', 'МД ($L')]:
    assert a in ru, a; ru = ru.replace(a, b)
open('tab_scope_ru.tex', 'w', encoding='utf8').write(ru + '\n')
