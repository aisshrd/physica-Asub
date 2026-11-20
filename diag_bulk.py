"""Two-population check: closure rates evaluated with the rotational temperature of the bulk (non-frozen) particles."""
import numpy as np, pickle
from theory import Model
from diag_nongauss import dsmc_snap, measured_rates, cumulants
const=dict(b0=0.0,q=0.4)
M=Model(e=0.98,mu_k=0.01,mu_s=0.12,vS=1.,p=2,**const)
out={}
for lab,Tr0 in [('A',10.),('C',1.)]:
    rows=[]
    for t,v,w in dsmc_snap(M,100.,Tr0,[30,100,300,1000,3000]):
        Tt=(v**2).sum(1).mean()/3; I=M.q/4; wn=np.linalg.norm(w,axis=1)
        froz=0.5*wn>M.vS; Tr=I*(w**2).sum(1).mean()/3; Trb=I*(w[~froz]**2).sum(1).mean()/3
        rt,rr,nu=measured_rates(M,v,w); G=M.Gamma(Tt)
        A,B=M.rhs_brackets(Tt,Tr); Ab,Bb=M.rhs_brackets(Tt,Trb)
        c=cumulants(v[~froz],w[~froz])
        rows.append(dict(t=t,Tt=Tt,Tr=Tr,Trb=Trb,f=froz.mean(),zeta_meas=-rt/(G*Tt),zeta_closure=-A/Tt,zeta_bulk=-Ab/Tt,a2r_all=cumulants(v,w)['a2r'],a2r_bulk=c['a2r']))
        r=rows[-1]; print(f"run {lab} t={t:7.1f} Tt={Tt:.2e} Tr={Tr:.2e} Tr_bulk={Trb:.2e} f={100*r['f']:.3f}% zeta: measured {r['zeta_meas']:.4f} closure(Tr) {r['zeta_closure']:.4f} closure(Tr_bulk) {r['zeta_bulk']:.4f} a2r all {r['a2r_all']:.1f} bulk {r['a2r_bulk']:.2f}",flush=True)
    out[lab]=rows
pickle.dump(out,open('bulk.pkl','wb'))
