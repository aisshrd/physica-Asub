import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import erf
# ---------- collision model: Walton + velocity-dependent (Stribeck) friction -------------
# mu(g) = mu_k + (mu_s-mu_k) * exp(-(g/vS)^(2p)),  g = |g_t| slip speed at contact
class Model:
    def __init__(s, e=0.95, b0=0.0, q=0.4, mu_k=0.02, mu_s=0.02, vS=1.0, p=2, m=1.0):
        s.e,s.b0,s.q,s.mu_k,s.mu_s,s.vS,s.p,s.m=e,b0,q,mu_k,mu_s,vS,p,m
        s.kt=q/(1+q); s.eta0=q*(1+b0)/(2*(1+q)); s.delta=1-e**2
        s.c=(1+e)*(1+q)/(q*(1+b0))       # tau_c = c*mu
    def mu(s,g):
        return s.mu_k+(s.mu_s-s.mu_k)*np.exp(-(g/s.vS)**(2*s.p))
    def lam(s,y,Tt,Tr):
        # lambda(y) = tau_c(mu(|g_t|)) * a/s  with |g_t| = s*y
        sv=np.sqrt(2*(Tt+Tr/s.q)/s.m); a=np.sqrt(2*Tt/s.m)
        return s.c*s.mu(sv*y)*a/sv
    # inner x-integrals (analytic) then outer y-integral (numeric)
    def F12(s,Tt,Tr):
        def G(y):
            l=s.lam(y,Tt,Tr)
            if l<=0: return 0.0,0.0
            xc=y/l; E=np.exp(-xc**2/2)
            g1=E+(l/y)*(np.sqrt(np.pi/2)*erf(xc/np.sqrt(2))-xc*E)
            g2=E+(l/y)**2*(2-(xc**2+2)*E)
            return g1,g2
        f1=quad(lambda y:y**3*np.exp(-y**2/2)*G(y)[0],0,np.inf,limit=200)[0]
        f2=quad(lambda y:y**3*np.exp(-y**2/2)*G(y)[1],0,np.inf,limit=200)[0]
        return f1,f2
    def rhs_brackets(s,Tt,Tr):
        F1,F2=s.F12(Tt,Tr); S=Tt+Tr/s.q
        A=-(s.delta+2*s.eta0*F1)*Tt+2*s.eta0**2*F2*S
        B=(-2*s.eta0*F1*Tr+2*s.eta0**2*F2*S)/s.q
        return A,B
    def Gamma(s,Tt,n_sig2=1.0):   # Gamma = nu/3, nu=4 n sigma^2 sqrt(pi Tt/m)
        return 4*n_sig2*np.sqrt(np.pi*Tt/s.m)/3
    def cool(s,Tt0,Tr0,tmax,h=0.0,teval=None):
        def rhs(t,y):
            Tt,Tr=max(y[0],1e-300),max(y[1],0.)
            A,B=s.rhs_brackets(Tt,Tr); G=s.Gamma(Tt)
            return [G*A+h,G*B]
        if teval is None: teval=np.logspace(-2,np.log10(tmax),300); teval[-1]=tmax
        r=solve_ivp(rhs,(0,tmax),[Tt0,Tr0],t_eval=teval,rtol=1e-8,atol=1e-14,method='LSODA')
        return r.t,r.y
def F1c(l): return l**2/(1+l**2)+l*(np.pi/2-np.arctan(l))
def F2c(l): return 2*l**2/(1+l**2)
# ---------- Monte-Carlo of the collision rule (independent check) ----------
def mc_brackets(M,Tt,Tr,N=3_000_000,seed=0):
    rng=np.random.default_rng(seed); m=M.m;q=M.q
    n=np.zeros((N,3)); n[:,2]=1
    a=np.sqrt(2*Tt/m); un=a*np.sqrt(-2*np.log(rng.random(N)))
    v12=np.column_stack([rng.normal(0,a,N),rng.normal(0,a,N),un])
    W=rng.normal(0,np.sqrt(2*Tr/(q*m)),(N,3))     # W = (sigma/2) Omega
    g=v12-np.cross(n,W); gt=g-un[:,None]*n; gtn=np.linalg.norm(gt,axis=1)
    tau=gtn/un; tc=M.c*M.mu(gtn)
    onepb=(1+M.b0)*np.minimum(1,tc/np.maximum(tau,1e-300))
    J=(-(m/2)*(1+M.e)*un)[:,None]*n-(m/2)*M.kt*onepb[:,None]*gt
    Jt=J-np.einsum('ij,ij->i',J,n)[:,None]*n
    dEtr=np.einsum('ij,ij->i',J,v12)+np.einsum('ij,ij->i',J,J)/m
    dErot=np.einsum('ij,ij->i',np.cross(n,J),W)+np.einsum('ij,ij->i',Jt,Jt)/(q*m)
    return dEtr.mean(),dErot.mean(), dEtr.std()/np.sqrt(N), dErot.std()/np.sqrt(N)
if __name__=='__main__':
    # 1) constant-mu limit reproduces closed forms
    M=Model(e=0.9,mu_k=0.05,mu_s=0.05)
    for Tt,Tr in [(1,1),(1,5),(0.01,3)]:
        th=Tr/Tt; l=M.c*0.05/np.sqrt(1+th/M.q)
        print('const-mu  F12 integral',M.F12(Tt,Tr),' closed',F1c(l),F2c(l))
    # 2) velocity dependent friction: integral vs MC
    print('\n--- Stribeck friction: 1D integrals vs Monte Carlo of the collision rule ---')
    rows=[]
    for (e,b0,mk,ms,p,Tt,Tr) in [(0.95,0.0,0.02,0.3,2,1,0.5),(0.95,0.0,0.02,0.3,2,0.05,0.2),(0.9,0.4,0.05,0.5,1,2,0.3),(0.8,0.0,0.01,0.2,3,0.3,3.0),(0.99,0.5,0.1,0.1,2,1,1)]:
        M=Model(e=e,b0=b0,mu_k=mk,mu_s=ms,p=p)
        A,B=M.rhs_brackets(Tt,Tr); a,b,sa,sb=mc_brackets(M,Tt,Tr)
        rows.append((e,b0,mk,ms,p,Tt,Tr,A,B,a,b,sa,sb))
        print(f'e={e} b0={b0} mu_k={mk} mu_s={ms} p={p} Tt={Tt} Tr={Tr}: theory {A:.5f} {B:.5f} | MC {a:.5f}±{sa:.5f} {b:.5f}±{sb:.5f}')
    import pickle; pickle.dump(rows,open('mc_table.pkl','wb'))
