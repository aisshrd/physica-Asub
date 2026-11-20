"""Seed robustness of the 'no freezing' runs C, E, G of Table 3."""
import numpy as np, pickle
from theory import Model
from dsmc2 import dsmc
runs={'C':(100.,0.01,0.01),'E':(1.,0.1,0.01),'G':(100.,0.1,0.03),'A':(100.,0.1,0.01)}
res={}
for lab,(Tt0,th0,muk) in runs.items():
    M=Model(e=0.98,b0=0.,q=0.4,mu_k=muk,mu_s=0.12,vS=1.,p=2)
    for seed in range(4):
        out,v,w=dsmc(M,Tt0,th0*Tt0,3000.,N=20000,seed=seed,return_state=True)
        wn=np.linalg.norm(w,axis=1); cs=0.5*wn; Tt=out[-1,1]; Tr=out[-1,2]
        E=wn**2; top=np.sort(E)[::-1]
        thr=12*np.sqrt(Tt)
        res[(lab,seed)]=dict(Tt=Tt,Tr=Tr,K=(w[:,0]**4).mean()/(w[:,0]**2).mean()**2,nf=(cs>1).sum(),n_dec=(cs>thr).sum(),csmax=cs.max(),top1=top[0]/E.sum())
        r=res[(lab,seed)]; print(f"run {lab} seed {seed}: Tt={Tt:.2e} Tr={Tr:.2e} theta={Tr/Tt:.3g} K={r['K']:.1f} n(cs>vS)={r['nf']} n(cs>12sqrtTt)={r['n_dec']} max contact speed={r['csmax']:.3f} top-1 share of E_rot={r['top1']:.2f}",flush=True)
pickle.dump(res,open('check_CEG.pkl','wb'))
