import numpy as np, pickle, time
from theory import Model
def dsmc_snap(M,Tt0,Tr0,tmax,N=20000,seed=0,snap_at_Tt=None):
    """as dsmc2.dsmc but stores particle states when T_t first drops below snap_at_Tt values."""
    rng=np.random.default_rng(seed); m=M.m;s=1.;q=M.q;I=q*m*s**2/4
    v=rng.normal(0,np.sqrt(Tt0/m),(N,3)); v-=v.mean(0); w=rng.normal(0,np.sqrt(Tr0/I),(N,3))
    t=0.; cmax=6*np.sqrt(2*Tt0/m); snaps={}; pending=sorted(snap_at_Tt or [],reverse=True)
    touts=np.concatenate([[0],np.logspace(-2,np.log10(tmax),80)]); k=0; out=[]
    while k<len(touts):
        Tt=m*(v**2).sum(1).mean()/3
        while pending and Tt<pending[0]:
            snaps[pending.pop(0)]=(t,w.copy(),v.copy())
        if t>=touts[k]:
            out.append((t,Tt,I*(w**2).sum(1).mean()/3)); k+=1; continue
        p=rng.permutation(N); i,j=p[:N//2],p[N//2:2*(N//2)]
        n=rng.normal(size=(N//2,3)); n/=np.linalg.norm(n,axis=1)[:,None]
        v12=v[i]-v[j]; un=np.einsum('ij,ij->i',v12,n); cmax=max(cmax,un.max()*1.05)
        acc=rng.random(N//2)*cmax<un; i,j,n,v12,un=i[acc],j[acc],n[acc],v12[acc],un[acc]
        Om=w[i]+w[j]; g=v12-(s/2)*np.cross(n,Om); gt=g-un[:,None]*n; gtn=np.linalg.norm(gt,axis=1)
        tau=gtn/un; tc=M.c*M.mu(gtn); onepb=(1+M.b0)*np.minimum(1,tc/np.maximum(tau,1e-300))
        J=(-(m/2)*(1+M.e)*un)[:,None]*n-(m/2)*M.kt*onepb[:,None]*gt
        v[i]+=J/m; v[j]-=J/m; dw=(s/(2*I))*np.cross(n,J); w[i]+=dw; w[j]+=dw
        t+=1/(4*np.pi*cmax)
        if rng.random()<0.02: cmax=max(6*np.sqrt(2*Tt),1e-12)
    return np.array(out),w,v,snaps
runs={ # label: (Tt0, theta0, mu_k, mu_s, vS, N, seed)
 'A: Tt0=100, th0=0.1':(100.,0.1,0.01,0.12,1.0,20000,0),
 'A2: same, N=40000':(100.,0.1,0.01,0.12,1.0,40000,3),
 'A3: same, seed 2':(100.,0.1,0.01,0.12,1.0,20000,7),
 'B: Tt0=100, th0=1':(100.,1.0,0.01,0.12,1.0,20000,0),
 'C: Tt0=100, th0=0.01':(100.,0.01,0.01,0.12,1.0,20000,0),
 'D: Tt0=1e4, th0=0.1':(1e4,0.1,0.01,0.12,1.0,20000,0),
 'E: Tt0=1, th0=0.1 (start at v_S)':(1.,0.1,0.01,0.12,1.0,20000,0),
 'F: Tt0=100, th0=0.1, mu_k=0.003':(100.,0.1,0.003,0.12,1.0,20000,0),
 'G: Tt0=100, th0=0.1, mu_k=0.03':(100.,0.1,0.03,0.12,1.0,20000,0),
}
res={}
for lab,(Tt0,th0,mk,ms,vS,N,seed) in runs.items():
    M=Model(e=0.98,b0=0.0,q=0.4,mu_k=mk,mu_s=ms,vS=vS,p=2); t0=time.time()
    out,w,v,snaps=dsmc_snap(M,Tt0,th0*Tt0,3000.,N=N,seed=seed,snap_at_Tt=[10.,1.,0.1])
    I=0.1; E=0.5*I*(w**2).sum(1); Es=np.sort(E)[::-1]; c=np.cumsum(Es)/E.sum()
    n99=np.searchsorted(c,0.99)+1; wn=np.linalg.norm(w,axis=1); nthr=(0.5*wn>vS).sum()
    frozen=set(np.argsort(-E)[:n99])
    ident={}
    for Tsnap,(ts,ws,vs) in snaps.items():
        above=set(np.where(0.5*np.linalg.norm(ws,axis=1)>vS)[0])
        ident[Tsnap]=(ts,len(above),len(above&frozen)/max(len(frozen),1),len(above&frozen)/max(len(above),1))
    res[lab]=dict(out=out,Tr_inf=out[-1,2],Tt_inf=out[-1,1],n99=n99,nthr=nthr,N=N,ident=ident,kurt=(w[:,0]**4).mean()/(w[:,0]**2).mean()**2,Etop=Es[:5])
    print(f'{lab}: T_r(inf)={out[-1,2]:.2e}  n(99%)={n99} ({100*n99/N:.2f}%)  n(sigma w/2>vS)={nthr}  kurt={res[lab]["kurt"]:.0f}  ident:',{k:(round(a,1),b,round(c,2),round(d,2)) for k,(a,b,c,d) in ident.items()},f'  [{time.time()-t0:.0f}s]')
pickle.dump(res,open('freeze.pkl','wb'))
