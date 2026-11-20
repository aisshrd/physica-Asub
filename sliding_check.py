"""
Referee comment 5: (i) size of the sticking contribution neglected in the sliding approximation,
(ii) monotonicity of the exact heating curve h(T_t) with sticking included, over a parameter scan.
"""
import numpy as np, pickle, itertools
from scipy.integrate import quad
from scipy.optimize import brentq
from theory import Model

def mu_eff(M, s):
    return np.sqrt(2/np.pi)*quad(lambda y: y*y*np.exp(-y*y/2)*M.mu(s*y), 0, np.inf, limit=200)[0]

def stick_fraction(M, Tt, Tr):
    """fraction of collisions that stick: P(y <= lambda(y) x) for independent Rayleigh x, y."""
    return quad(lambda y: y*np.exp(-y*y/2)*np.exp(-(y/M.lam(y,Tt,Tr))**2/2), 0, np.inf, limit=200)[0]

def steady(M, Tt):
    th=brentq(lambda th: M.rhs_brackets(Tt, th*Tt)[1], 1e-8, 200., xtol=1e-12)
    A,B=M.rhs_brackets(Tt, th*Tt)
    return -M.Gamma(Tt)*A, th

if __name__=='__main__':
    # (i) sliding approximation vs full quadrature along the steady branch (parameters of the heating figure)
    print('mu_s   T_t      theta*   lam_max  P_stick  F1_full   (pi/2)lam_bar  rel.err   bound')
    tab=[]
    for mus in [0.05,0.1,0.2]:
        M=Model(e=0.98,b0=0.,q=0.4,mu_k=0.01,mu_s=mus,vS=1.,p=2)
        for Tt in [1e-3,1e-2,1e-1,1,10,100]:
            h,th=steady(M,Tt); Tr=th*Tt
            s=np.sqrt(2*(Tt+Tr/M.q)); F1,F2=M.F12(Tt,Tr)
            lbar=M.c*mu_eff(M,s)/np.sqrt(1+th/M.q); lmax=M.c*mus/np.sqrt(1+th/M.q); lmin=M.c*M.mu_k/np.sqrt(1+th/M.q)
            F1s=np.pi/2*lbar; bound=(np.pi/2)*lmax**4/(1+lmax**2)**1.5
            tab.append((mus,Tt,th,lmax,stick_fraction(M,Tt,Tr),F1,F1s,(F1s-F1)/F1,bound/F1))
            print(f'{mus:<5g} {Tt:<8g} {th:.4f}   {lmax:.3f}   {tab[-1][4]:.4f}   {F1:.5f}  {F1s:.5f}       {tab[-1][7]:+.4f}   {tab[-1][8]:.4f}')
    # (ii) monotonicity scan of the exact heating curve
    Tts=np.logspace(-4,3,71); res=[]
    grid=list(itertools.product([0.90,0.95,0.98,0.995],[0.003,0.01,0.03],[0.05,0.1,0.2,0.4,0.8],[1,2,4,8]))
    for e,muk,mus,p in grid:
        M=Model(e=e,b0=0.,q=0.4,mu_k=muk,mu_s=mus,vS=1.,p=p)
        hs=np.array([steady(M,T)[0] for T in Tts])
        dl=np.diff(np.log(hs))/np.diff(np.log(Tts))
        res.append((e,muk,mus,p,dl.min()))
        print(f'e={e} mu_k={muk} mu_s={mus} p={p}: min dln h/dln T = {dl.min():.3f}',flush=True)
    res=np.array(res); print('parameter sets:',len(res),' non-monotonic:',(res[:,4]<=0).sum(),' smallest slope:',res[:,4].min(), 'at', res[np.argmin(res[:,4])])
    pickle.dump(dict(tab=tab,scan=res,Tts=Tts),open('sliding.pkl','wb'))
