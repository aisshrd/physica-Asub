import numpy as np, pickle
from theory import Model
from dsmc2 import dsmc
from driven import steady_curve
M=Model(e=0.98,b0=0.0,q=0.4,mu_k=0.01,mu_s=0.2,vS=1.0,p=2)
out=[]
for h in [3e-4,1e-3,3e-3,1e-2]:
    for T0,lab in [(5.0,'hot'),(0.002,'cold')]:
        touts=np.concatenate([[0],np.linspace(1,800,40)])
        d=dsmc(M,T0,0.1*T0,800.,N=10000,touts=touts,h_of_t=lambda t:h)
        Tss=d[-10:,1].mean(); out.append((h,lab,Tss))
    hE,thE=steady_curve(M,np.array([np.sqrt(out[-1][2]*out[-2][2])]))
    print(f'h={h}: DSMC steady T (hot start)={out[-2][2]:.4f}, (cold start)={out[-1][2]:.4f}; exact theory h at that T = {hE[0]:.2e}')
pickle.dump(out,open('driven_dsmc.pkl','wb'))
