# Проверочный код: (1) Монте-Карло проверка интегралов столкновений §2, (2) бифуркация по точной Phi, (3) корректный DSMC модели Уолтона
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
# Единицы: m=sigma=n*sigma^2=1, k_B=1. I = q m sigma^2/4.
import numpy as np
rng=np.random.default_rng(1)
def collide(v12,W,n,e,mu,b0,q,m=1.0):
    # W = Omega sum * sigma/2 ; g = v12 - n x W
    un=np.einsum('ij,ij->i',v12,n)
    g=v12-np.cross(n,W)
    gn=np.einsum('ij,ij->i',g,n)
    gt=g-gn[:,None]*n
    tau=np.linalg.norm(gt,axis=1)/gn
    kt=q/(1+q)
    tc=mu*(1+e)*(1+q)/(q*(1+b0))
    onepb=(1+b0)*np.minimum(1,tc/np.maximum(tau,1e-300))
    Jn=-(m/2)*(1+e)*gn
    Jt=-(m/2)*kt*onepb[:,None]*gt
    J=Jn[:,None]*n+Jt
    dEtr=np.einsum('ij,ij->i',J,v12)+np.einsum('ij,ij->i',J,J)/m
    # rotational: sigma/2 (n x J).Omega + sigma^2/(4I) Jt^2 ; sigma/2*Omega=W ; sigma^2/(4I)=1/(q m)
    dErot=np.einsum('ij,ij->i',np.cross(n,J),W)+np.einsum('ij,ij->i',Jt,Jt)/(q*m)
    return dEtr,dErot
def rates(Tt,Tr,e,mu,b0,q,N=4_000_000,m=1.0):
    n=np.zeros((N,3));n[:,2]=1
    # flux-weighted normal component: Rayleigh with a^2=2Tt/m ; tangential gaussian
    a=np.sqrt(2*Tt/m)
    un=a*np.sqrt(-2*np.log(rng.random(N)))
    v12=np.column_stack([rng.normal(0,a,N),rng.normal(0,a,N),un])
    b=np.sqrt(2*Tr/(q*m))  # per component of W= sigma/2 Omega_sum:(sigma/2)^2*2Tr/I=2Tr/(qm)
    W=rng.normal(0,b,(N,3))
    dEtr,dErot=collide(v12,W,n,e,mu,b0,q)
    return dEtr.mean(),dErot.mean()
def F1(l): return l**2/(1+l**2)+l*(np.pi/2-np.arctan(l))
def F2(l): return 2*l**2/(1+l**2)
def theory(Tt,Tr,e,mu,b0,q):
    th=Tr/Tt; tc=mu*(1+e)*(1+q)/(q*(1+b0)); lam=tc/np.sqrt(1+th/q)
    d=1-e**2; eta=q*(1+b0)/(2*(1+q))
    t=-(d+2*eta*F1(lam))*Tt+2*eta**2*F2(lam)*(Tt+Tr/q)
    r=(-2*eta*F1(lam)*Tr+2*eta**2*F2(lam)*(Tt+Tr/q))/q
    return t,r
from scipy.optimize import brentq
def Phi(th,e,mu,b0,q):
    tc=mu*(1+e)*(1+q)/(q*(1+b0)); lam=tc/np.sqrt(1+th/q)
    d=1-e**2; eta=q*(1+b0)/(2*(1+q))
    return d*th+2*eta*F1(lam)*th*(1-1/q)+2*eta**2*F2(lam)*(1+th/q)*(1/q-th)
def nroots(e,mu,b0,q):
    th=np.logspace(-8,8,20000); p=Phi(th,e,mu,b0,q)
    return np.sum(np.sign(p[1:])!=np.sign(p[:-1])), th[np.where(np.sign(p[1:])!=np.sign(p[:-1]))]
def mu1(e,b0,q):
    # smallest mu with >=1 root
    lo,hi=1e-5,2.0
    for _ in range(60):
        mid=np.sqrt(lo*hi)
        if nroots(e,mid,b0,q)[0]>0: hi=mid
        else: lo=mid
    return hi
if __name__=='__main__':
  q=0.4
  print('HHZ check e=0.9,b0=0.5: our mu1=',mu1(0.9,0.5,q),' our mu2=',np.sqrt(0.1/1.9))
  print('e=0.97,b0=-0.99 our mu1', mu1(0.97,-0.99,q))
def dsmc(Tt0,Tr0,e,mu,b0,q,N=20000,tmax=200.,seed=0,nout=60):
    rng=np.random.default_rng(seed); m=1.;s=1.;I=q*m*s**2/4
    v=rng.normal(0,np.sqrt(Tt0/m),(N,3)); v-=v.mean(0)
    w=rng.normal(0,np.sqrt(Tr0/I),(N,3))
    tc=mu*(1+e)*(1+q)/(q*(1+b0)); kt=q/(1+q)
    t=0.; cmax=6*np.sqrt(2*Tt0/m)
    touts=np.concatenate([[0],np.logspace(-2,np.log10(tmax),nout)]); k=0; out=[]
    while k<len(touts):
        if t>=touts[k]:
            out.append((t,m*(v**2).sum(1).mean()/3,I*(w**2).sum(1).mean()/3)); k+=1; continue
        p=rng.permutation(N); i,j=p[:N//2],p[N//2:2*(N//2)]
        n=rng.normal(size=(N//2,3)); n/=np.linalg.norm(n,axis=1)[:,None]
        v12=v[i]-v[j]; un=np.einsum('ij,ij->i',v12,n)
        cmax=max(cmax,un.max()*1.05)
        acc=rng.random(N//2)*cmax<un
        i,j,n,v12,un=i[acc],j[acc],n[acc],v12[acc],un[acc]
        Om=w[i]+w[j]
        g=v12-(s/2)*np.cross(n,Om); gt=g-un[:,None]*n
        tau=np.linalg.norm(gt,axis=1)/un
        onepb=(1+b0)*np.minimum(1,tc/np.maximum(tau,1e-300))
        J=(-(m/2)*(1+e)*un)[:,None]*n-(m/2)*kt*onepb[:,None]*gt
        v[i]+=J/m; v[j]-=J/m
        dw=(s/(2*I))*np.cross(n,J); w[i]+=dw; w[j]+=dw
        t+=1/(4*np.pi*cmax)        # N/2 candidates * 2/(N*4pi*cmax), n*sigma^2=1
        # shrink cmax slowly as gas cools
        if rng.random()<0.01: cmax=max(np.sqrt(2*m*(v**2).sum(1).mean()/3)*6, 1e-12)
    return np.array(out)
def ode(Tt0,Tr0,e,mu,b0,q,tmax):
    d=1-e**2; eta=q*(1+b0)/(2*(1+q)); tc=mu*(1+e)*(1+q)/(q*(1+b0))
    def rhs(t,y):
        Tt,Tr=max(y[0],1e-300),max(y[1],0)
        lam=tc/np.sqrt(1+Tr/(q*Tt)); G=4*np.sqrt(np.pi*Tt)/3
        a=-(d+2*eta*F1(lam))*Tt+2*eta**2*F2(lam)*(Tt+Tr/q)
        b=(-2*eta*F1(lam)*Tr+2*eta**2*F2(lam)*(Tt+Tr/q))/q
        return [G*a,G*b]
    te=np.logspace(-2,np.log10(tmax),300); te[-1]=tmax
    r=solve_ivp(rhs,(0,tmax),[Tt0,Tr0],t_eval=te,rtol=1e-9,atol=1e-14,method='LSODA')
    return r.t,r.y
from scipy.integrate import solve_ivp
