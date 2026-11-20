import numpy as np, pickle
from scipy.optimize import brentq
from theory import Model
def steady_curve(M,Tts):
    hs=[];ths=[]
    for Tt in Tts:
        f=lambda th: M.rhs_brackets(Tt,th*Tt)[1]
        th=brentq(f,1e-6,50.,xtol=1e-10)
        A,B=M.rhs_brackets(Tt,th*Tt); hs.append(-M.Gamma(Tt)*A); ths.append(th)
    return np.array(hs),np.array(ths)
if __name__=='__main__':
    Tts=np.logspace(-3,2,120); res={}
    for lab,pars in {'mu_s=0.2':dict(mu_k=0.01,mu_s=0.2),'mu_s=0.1':dict(mu_k=0.01,mu_s=0.1),'mu_s=0.05':dict(mu_k=0.01,mu_s=0.05),'const 0.01':dict(mu_k=0.01,mu_s=0.01)}.items():
        M=Model(e=0.98,b0=0.0,q=0.4,vS=1.0,p=2,**pars); h,th=steady_curve(M,Tts); res[lab]=(h,th)
        dh=np.diff(np.log(h)); print(lab,'non-monotonic:',np.any(dh<0),' h range of S-branch:',(h[1:][dh<0].min(),h[1:][dh<0].max()) if np.any(dh<0) else '-')
    pickle.dump(dict(Tts=Tts,res=res),open('driven.pkl','wb'))
