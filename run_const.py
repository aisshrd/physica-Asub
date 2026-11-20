"""Constant-friction DSMC runs of Fig. 3 (writes res.pkl used by figs.py)."""
import numpy as np, pickle
from walton_dsmc_check import dsmc, ode, nroots
q=0.4; b0=0.; mu=0.02
print('zeros of Phi at e=0.98, mu=0.02:', nroots(0.98,mu,b0,q))
cases={'A: e=0.98, theta0=0.1 (below separatrix)':(1,0.1,0.98),
       'B: e=0.98, theta0=10 (above separatrix)':(1,10,0.98),
       'C: e=0.90, theta0=0.1 (R>R_c)':(1,0.1,0.90)}
res={}
for name,(Tt0,Tr0,e) in cases.items():
    d=dsmc(Tt0,Tr0,e,mu,b0,q,N=20000,tmax=3000.); to=ode(Tt0,Tr0,e,mu,b0,q,3000.)
    res[name]=(d,to); print(name,'final DSMC',d[-1],'ODE',to[1][:,-1])
pickle.dump(res,open('res.pkl','wb'))
