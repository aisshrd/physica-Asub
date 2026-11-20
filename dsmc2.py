import numpy as np
from theory import Model
def dsmc(M,Tt0,Tr0,tmax,N=20000,seed=0,touts=None,h_of_t=None,dtmax=None,return_state=False):
    """Homogeneous DSMC (Bird acceptance-rejection, hard-sphere rate ~ g_n). n*sigma^2 = 1.
       h_of_t(t): stochastic thermostat, translational-temperature injection rate dT_t/dt=h."""
    rng=np.random.default_rng(seed); m=M.m;s=1.;q=M.q;I=q*m*s**2/4
    v=rng.normal(0,np.sqrt(Tt0/m),(N,3)); v-=v.mean(0)
    w=rng.normal(0,np.sqrt(Tr0/I),(N,3))
    t=0.; cmax=6*np.sqrt(2*Tt0/m)
    if touts is None: touts=np.concatenate([[0],np.logspace(-2,np.log10(tmax),80)])
    k=0; out=[]
    while k<len(touts):
        if t>=touts[k]:
            out.append((t,m*(v**2).sum(1).mean()/3,I*(w**2).sum(1).mean()/3)); k+=1; continue
        p=rng.permutation(N); i,j=p[:N//2],p[N//2:2*(N//2)]
        n=rng.normal(size=(N//2,3)); n/=np.linalg.norm(n,axis=1)[:,None]
        v12=v[i]-v[j]; un=np.einsum('ij,ij->i',v12,n)
        cmax=max(cmax,un.max()*1.05)
        acc=rng.random(N//2)*cmax<un
        i,j,n,v12,un=i[acc],j[acc],n[acc],v12[acc],un[acc]
        Om=w[i]+w[j]; g=v12-(s/2)*np.cross(n,Om); gt=g-un[:,None]*n; gtn=np.linalg.norm(gt,axis=1)
        tau=gtn/un; tc=M.c*M.mu(gtn)
        onepb=(1+M.b0)*np.minimum(1,tc/np.maximum(tau,1e-300))
        J=(-(m/2)*(1+M.e)*un)[:,None]*n-(m/2)*M.kt*onepb[:,None]*gt
        v[i]+=J/m; v[j]-=J/m
        dw=(s/(2*I))*np.cross(n,J); w[i]+=dw; w[j]+=dw
        dt=1/(4*np.pi*cmax)
        if h_of_t is not None:
            h=h_of_t(t)
            if h>0:
                v+=rng.normal(0,np.sqrt(h*dt/m),(N,3)); v-=v.mean(0)
        t+=dt
        if rng.random()<0.02: cmax=max(6*np.sqrt(2*m*(v**2).sum(1).mean()/3/m),1e-12)
    return (np.array(out),v,w) if return_state else np.array(out)
