"""
Refined temperature grid for the flattest heating curves (Sec. 5.1 (ii)): the 12 parameter sets with the smallest
logarithmic slope d ln D / d ln T_t in the two scans are recomputed on 701 temperatures (100 per decade instead
of 10), to check that no narrow non-monotonic interval is hidden between the sampled temperatures.
Output: scan_fine.pkl.   Runtime ~8 min on 2 cores.
"""
import numpy as np, pickle
from multiprocessing import Pool
from theory import Model
from sliding_check import steady

TF = np.logspace(-4, 3, 701)

def one(par):
    e, b0, q, muk, mus, p, coarse = par
    M = Model(e=e, b0=b0, q=q, mu_k=muk, mu_s=mus, vS=1., p=p)
    D = np.array([steady(M, T)[0] for T in TF])
    sl = np.diff(np.log(D)) / np.diff(np.log(TF))
    return par[:6] + (coarse, float(sl.min()), int((sl <= 0).sum()))

if __name__ == '__main__':
    a = pickle.load(open('sliding.pkl', 'rb'))['scan']            # columns e, mu_k, mu_s, p, smallest slope
    b = pickle.load(open('scan_ext.pkl', 'rb'))['scan']           # columns e, b0, q, mu_k, mu_s, p, smallest slope
    sets = [(r[0], 0.0, 0.4, r[1], r[2], r[3], r[4]) for r in a] + [tuple(r) for r in b]
    sets = sorted(sets, key=lambda r: r[6])[:12]
    import sys
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else None) as pool: res = pool.map(one, sets)
    for r in res:
        print('e=%g b0=%g q=%.3g mu_k=%g mu_s=%g p=%g:  smallest slope  71 points %.3f   701 points %.3f   intervals with dD/dT<=0: %d' % r)
    pickle.dump(dict(res=res, TF=TF), open('scan_fine.pkl', 'wb'))
