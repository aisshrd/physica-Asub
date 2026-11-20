import numpy as np, pickle
from theory import F1c,F2c
def Phi(th,e,mu,b0,q):
    tc=mu*(1+e)*(1+q)/(q*(1+b0)); lam=tc/np.sqrt(1+th/q); d=1-e**2; eta=q*(1+b0)/(2*(1+q))
    return d*th+2*eta*F1c(lam)*th*(1-1/q)+2*eta**2*F2c(lam)*(1+th/q)*(1/q-th)
def roots(e,mu,b0,q):
    th=np.logspace(-8,9,40000); p=Phi(th,e,mu,b0,q); ix=np.where(np.sign(p[1:])!=np.sign(p[:-1]))[0]
    return th[ix]
def mu1(e,b0,q):
    lo,hi=1e-5,3.0
    for _ in range(60):
        mid=np.sqrt(lo*hi)
        if len(roots(e,mid,b0,q))>0: hi=mid
        else: lo=mid
    return hi
if __name__=='__main__':
    q=0.4; es=np.linspace(0.3,0.995,40); out={}
    for b0 in [0.0,0.5]:
        out[b0]=np.array([mu1(e,b0,q) for e in es])
    mu2=np.sqrt((1-es)/(1+es)); mu2hhz=np.sqrt((1-es)/(2*(1+es)))
    # small-mu asymptote of mu1 from R_c formula: 1-e = (pi(1-q)/(2q)) mu R_c(mu),  R_c=1-sqrt(8 m_mu/(pi q(1-q))), m_mu=mu(1+e)/2
    mus=np.logspace(-3,-1,50); e_as=[]
    for mu in mus:
        e=0.99
        for _ in range(50):
            mm=mu*(1+e)/2; Rc=1-np.sqrt(8*mm/(np.pi*q*(1-q))); e=1-np.pi*(1-q)*mu*Rc/(2*q)
        e_as.append(e)
    pickle.dump(dict(es=es,mu1=out,mu2=mu2,mu2hhz=mu2hhz,mus=mus,e_as=np.array(e_as)),open('phase.pkl','wb'))
    print('mu1(e=0.9,b0=0.5)=',mu1(0.9,0.5,q),' mu1(0.97,-0.99)=',mu1(0.97,-0.99,q))
    print('roots e=0.98 mu=0.02:',roots(0.98,0.02,0.,q))
    # T_r(inf) formula check table vs ODE for constant mu
    from theory import Model
    for th0 in [3,10,30,100]:
        M=Model(e=0.98,mu_k=0.02,mu_s=0.02); t,y=M.cool(1.,th0,5000.); mm=0.02*1.98/2; d=1-0.98**2
        pred=(np.sqrt(th0)-np.pi*mm/(np.sqrt(q)*d))**2
        print(f'theta0={th0}: T_r(inf) ODE={y[1,-1]:.4f}  asymptotic formula={pred:.4f}')
