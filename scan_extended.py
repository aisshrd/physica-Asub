"""
Extended monotonicity scan of the heating curve D(T_t) (Sec. 5.1 (ii)), beyond beta_0 = 0, q = 0.4:
384 parameter sets with e in {0.7,0.85,0.95,0.99}, beta_0 in {-0.5,0.5,1}, q in {1/4,0.4,1/2,2/3},
mu_k in {0.003,0.03}, mu_s in {0.2,0.8}, p in {2,8}; full collision integrals, sticking included.
Output: scan_ext.pkl.  Runtime ~10 min on 2 cores.
"""
import numpy as np, itertools, pickle
from multiprocessing import Pool
from theory import Model
from sliding_check import steady

Tts = np.logspace(-4, 3, 71)
GRID = list(itertools.product([0.7, 0.85, 0.95, 0.99], [-0.5, 0.5, 1.0], [0.25, 0.4, 0.5, 2/3],
                              [0.003, 0.03], [0.2, 0.8], [2, 8]))

def one(par):
    e, b0, q, muk, mus, p = par
    M = Model(e=e, b0=b0, q=q, mu_k=muk, mu_s=mus, vS=1., p=p)
    hs = np.array([steady(M, T)[0] for T in Tts])
    return par + (float((np.diff(np.log(hs)) / np.diff(np.log(Tts))).min()),)

if __name__ == '__main__':
    with Pool() as pool:
        res = np.array(pool.map(one, GRID))
    i = np.argmin(res[:, 6])
    print('parameter sets:', len(res), ' non-monotonic:', int((res[:, 6] <= 0).sum()),
          ' smallest slope: %.3f at e=%g beta0=%g q=%.3g mu_k=%g mu_s=%g p=%g' % (res[i, 6], *res[i, :6]))
    for col, name in enumerate(['e', 'beta0', 'q', 'mu_k', 'mu_s', 'p']):
        print(name, {round(v, 3): round(res[res[:, col] == v, 6].min(), 3) for v in np.unique(res[:, col])})
    pickle.dump(dict(scan=res, Tts=Tts, grid=GRID), open('scan_ext.pkl', 'wb'))
