"""
Non-Gaussian diagnostics (referee comments 1 and 6).
For DSMC snapshots we measure
  a2t = (3/5)<v^4>/<v^2>^2 - 1        translational fourth cumulant (0 for a Gaussian)
  a2r = (3/5)<w^4>/<w^2>^2 - 1        rotational fourth cumulant
  b   = <v^2 w^2>/(<v^2><w^2>) - 1    translation-rotation energy correlation
  c   = 3<(v.w)^2>/(<v^2 w^2>) - 1    orientational correlation of spin and velocity
and the collisional production rates of T_t and T_r evaluated on the ACTUAL particle
distribution (pair sampling), divided by the Gaussian-closure prediction at the same (T_t, T_r).
"""
import numpy as np, pickle, sys
from theory import Model

def cumulants(v, w):
    v2=(v**2).sum(1); w2=(w**2).sum(1); vw=(v*w).sum(1)
    return dict(a2t=0.6*(v2**2).mean()/v2.mean()**2-1, a2r=0.6*(w2**2).mean()/w2.mean()**2-1,
                b=(v2*w2).mean()/(v2.mean()*w2.mean())-1, c=3*(vw**2).mean()/(v2*w2).mean()-1)

def measured_rates(M, v, w, npairs=4_000_000, seed=1):
    """dT_t/dt and dT_r/dt from the actual distribution (n sigma^2 = 1, molecular chaos for pairs)."""
    rng=np.random.default_rng(seed); N=len(v); m=M.m; q=M.q
    i=rng.integers(0,N,npairs); j=rng.integers(0,N,npairs); ok=i!=j; i,j=i[ok],j[ok]
    n=rng.normal(size=(len(i),3)); n/=np.linalg.norm(n,axis=1)[:,None]
    v12=v[i]-v[j]; gn=np.einsum('ij,ij->i',v12,n); pos=gn>0
    i,j,n,v12,gn=i[pos],j[pos],n[pos],v12[pos],gn[pos]
    W=0.5*(w[i]+w[j])                                   # (sigma/2) Omega
    g=v12-np.cross(n,W); gt=g-gn[:,None]*n; gtn=np.linalg.norm(gt,axis=1)
    onepb=(1+M.b0)*np.minimum(1,M.c*M.mu(gtn)/np.maximum(gtn/gn,1e-300))
    J=(-(m/2)*(1+M.e)*gn)[:,None]*n-(m/2)*M.kt*onepb[:,None]*gt
    Jt=J-np.einsum('ij,ij->i',J,n)[:,None]*n
    dEtr=np.einsum('ij,ij->i',J,v12)+np.einsum('ij,ij->i',J,J)/m
    dErot=np.einsum('ij,ij->i',np.cross(n,J),W)+np.einsum('ij,ij->i',Jt,Jt)/(q*m)
    norm=4*np.pi/3/ok.sum()
    return norm*(gn*dEtr).sum(), norm*(gn*dErot).sum(), 4*np.pi*gn.sum()/ok.sum()

def dsmc_snap(M, Tt0, Tr0, tsnaps, N=20000, seed=0, h=0.0):
    rng=np.random.default_rng(seed); m=M.m; s=1.; q=M.q; I=q*m*s**2/4
    v=rng.normal(0,np.sqrt(Tt0/m),(N,3)); v-=v.mean(0); w=rng.normal(0,np.sqrt(Tr0/I),(N,3))
    t=0.; cmax=6*np.sqrt(2*Tt0/m); k=0; snaps=[]
    while k<len(tsnaps):
        if t>=tsnaps[k]:
            snaps.append((t,v.copy(),w.copy())); k+=1; continue
        p=rng.permutation(N); i,j=p[:N//2],p[N//2:2*(N//2)]
        n=rng.normal(size=(N//2,3)); n/=np.linalg.norm(n,axis=1)[:,None]
        v12=v[i]-v[j]; un=np.einsum('ij,ij->i',v12,n); cmax=max(cmax,un.max()*1.05)
        acc=rng.random(N//2)*cmax<un; i,j,n,v12,un=i[acc],j[acc],n[acc],v12[acc],un[acc]
        g=v12-(s/2)*np.cross(n,w[i]+w[j]); gt=g-un[:,None]*n; gtn=np.linalg.norm(gt,axis=1)
        onepb=(1+M.b0)*np.minimum(1,M.c*M.mu(gtn)/np.maximum(gtn/un,1e-300))
        J=(-(m/2)*(1+M.e)*un)[:,None]*n-(m/2)*M.kt*onepb[:,None]*gt
        v[i]+=J/m; v[j]-=J/m; dw=(s/(2*I))*np.cross(n,J); w[i]+=dw; w[j]+=dw
        dt=1/(4*np.pi*cmax)
        if h>0: v+=rng.normal(0,np.sqrt(h*dt/m),(N,3)); v-=v.mean(0)
        t+=dt
        if rng.random()<0.02: cmax=max(6*np.sqrt(2*(v**2).sum(1).mean()/3),1e-12)
    return snaps

def analyse(M, snaps, label):
    rows=[]
    for t,v,w in snaps:
        Tt=M.m*(v**2).sum(1).mean()/3; Tr=M.q/4*(w**2).sum(1).mean()/3
        A,B=M.rhs_brackets(Tt,Tr); G=M.Gamma(Tt)
        rt,rr,nu=measured_rates(M,v,w)
        c=cumulants(v,w)
        F1,F2=M.F12(Tt,Tr); loss_t=G*(M.delta+2*M.eta0*F1)*Tt; loss_r=G*2*M.eta0*F1*Tr/M.q   # gross sink terms of Eq. (TT)
        rows.append(dict(t=t,Tt=Tt,Tr=Tr,theta=Tr/Tt,**c,ratio_t=rt/(G*A),ratio_r=rr/(G*B),ratio_nu=nu/(4*np.sqrt(np.pi*Tt)),
                         err_t=(rt-G*A)/loss_t,err_r=(rr-G*B)/loss_r))
        r=rows[-1]
        print(f"{label:22s} t={t:8.2f} theta={r['theta']:.3g} a2t={r['a2t']:+.3f} a2r={r['a2r']:+.3f} b={r['b']:+.3f} c={r['c']:+.3f} | rate_t/G={r['ratio_t']:.3f} rate_r/G={r['ratio_r']:.3f} nu/G={r['ratio_nu']:.3f} | err_t={r['err_t']:+.4f} err_r={r['err_r']:+.4f}",flush=True)
    return rows

if __name__=='__main__':
    res={}
    const=dict(b0=0.0,q=0.4)
    cases={
     '(a) rough, e=0.98, theta0=0.1':(Model(e=0.98,mu_k=0.02,mu_s=0.02,**const),1.,0.1,[1,10,100,1000],0.),
     '(b) freezing, e=0.98, theta0=10':(Model(e=0.98,mu_k=0.02,mu_s=0.02,**const),1.,10.,[1,10,30,100],0.),
     '(c) freezing, e=0.90':(Model(e=0.90,mu_k=0.02,mu_s=0.02,**const),1.,0.1,[1,3,10,30],0.),
     'Stribeck run A':(Model(e=0.98,mu_k=0.01,mu_s=0.12,vS=1.,p=2,**const),100.,10.,[0.3,3,10,30,100,1000],0.),
     'Stribeck run C (no freezing)':(Model(e=0.98,mu_k=0.01,mu_s=0.12,vS=1.,p=2,**const),100.,1.,[10,100,1000],0.),
     'driven h=1e-3, mu_s=0.2':(Model(e=0.98,mu_k=0.01,mu_s=0.2,vS=1.,p=2,**const),0.05,0.005,[600,800],1e-3),
     'driven h=1e-2, mu_s=0.2':(Model(e=0.98,mu_k=0.01,mu_s=0.2,vS=1.,p=2,**const),0.05,0.005,[300,400],1e-2),
     'driven h=1, mu_s=0.2':(Model(e=0.98,mu_k=0.01,mu_s=0.2,vS=1.,p=2,**const),5.,0.5,[40,60],1.0),
    }
    for lab,(M,Tt0,Tr0,ts,h) in cases.items():
        res[lab]=analyse(M,dsmc_snap(M,Tt0,Tr0,ts,h=h),lab)
        pickle.dump(res,open('nongauss.pkl','wb'))
