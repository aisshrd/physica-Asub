import numpy as np, pickle, time
from theory import Model
from dsmc2 import dsmc
# Stribeck cooling: hot gas (mu ~ mu_k) cools through vS into sticky regime (mu ~ mu_s)
M=Model(e=0.98,b0=0.0,q=0.4,mu_k=0.01,mu_s=0.12,vS=1.0,p=2)
Tt0,Tr0=100.,10.
t0=time.time(); d,v,w=dsmc(M,Tt0,Tr0,3000.,N=20000,return_state=True); print('dsmc',time.time()-t0)
pickle.dump(dict(w=w,v=v,d=d),open('state_stribeck.pkl','wb'))          # final state, used for the spin distribution (Fig. 6)
Mc=Model(e=0.98,b0=0.0,q=0.4,mu_k=0.12,mu_s=0.12)
d2,v2,w2=dsmc(Mc,1.,0.1,3000.,N=20000,return_state=True)
pickle.dump(dict(w=w2,v=v2,d=d2),open('state_const.pkl','wb'))
t,y=M.cool(Tt0,Tr0,3000.)
# mean-field version: mu evaluated at characteristic slip speed vbar=sqrt(pi(Tt+Tr/q)/m)
class MF(Model):
    def F12(s,Tt,Tr):
        from theory import F1c,F2c
        vbar=np.sqrt(np.pi*(Tt+Tr/s.q)/s.m); mu=Model.mu(s,vbar)
        l=s.c*mu/np.sqrt(1+Tr/(Tt*s.q)); return F1c(l),F2c(l)
MFm=MF(e=0.98,b0=0.0,q=0.4,mu_k=0.01,mu_s=0.12,vS=1.0,p=2)
tm,ym=MFm.cool(Tt0,Tr0,3000.)
# constant-mu references
Mk=Model(e=0.98,mu_k=0.01,mu_s=0.01); Ms=Model(e=0.98,mu_k=0.12,mu_s=0.12)
tk,yk=Mk.cool(Tt0,Tr0,3000.); ts,ys=Ms.cool(Tt0,Tr0,3000.)
pickle.dump(dict(d=d,t=t,y=y,tm=tm,ym=ym,tk=tk,yk=yk,ts=ts,ys=ys),open('cool.pkl','wb'))
print('final DSMC',d[-1],'exact',y[:,-1],'MF',ym[:,-1])
