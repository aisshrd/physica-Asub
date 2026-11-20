"""Figure: event-driven MD versus DSMC (referee comment 3)."""
import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3})
R=pickle.load(open('res.pkl','rb')); A=pickle.load(open('md_abc.pkl','rb')); S=pickle.load(open('md_sB.pkl','rb')); C=pickle.load(open('cool.pkl','rb'))
fig,ax=plt.subplots(1,4,figsize=(7.6,2.5))
titles=[r'(a) $e=0.98,\ \theta_0=0.1$',r'(b) $e=0.98,\ \theta_0=10$',r'(c) $e=0.90,\ \theta_0=0.1$',r'(d) Stribeck, run A']
def interp(d,t,col):  # log-log interpolation of DSMC at MD times
    return np.exp(np.interp(np.log(t),np.log(d[1:,0]),np.log(d[1:,col])))
summary={}
for a,key,name,ti in zip(ax[:3],'abc',R.keys(),titles):
    d=R[name][0]
    a.loglog(d[1:,0],d[1:,1],'-',c='C0',lw=1,label='DSMC $T_t$'); a.loglog(d[1:,0],d[1:,2],'-',c='C3',lw=1,label='DSMC $T_r$')
    devs=[]
    for k,run in enumerate(A[key]['runs']):
        o=run['out'][1:]
        a.loglog(o[::2,0],o[::2,1],'o',ms=3,mfc='none',c='C0',label='MD $T_t$' if k==0 else None); a.loglog(o[::2,0],o[::2,2],'s',ms=3,mfc='none',c='C3',label='MD $T_r$' if k==0 else None)
        m=(o[:,0]>0.05)&(o[:,0]<d[-1,0])
        devs.append((np.abs(np.log(o[m,1]/interp(d,o[m,0],1))).max(),np.abs(np.log(o[m,2]/interp(d,o[m,0],2))).max(),o[-1,1],o[-1,2],o[-1,3],o[:,4].max()))
    summary[key]=(devs,d[-1])
    a.set_title(ti,fontsize=8); a.set_xlabel('$t$')
ax[0].legend(fontsize=5.5); ax[0].set_ylabel('$T$')
a=ax[3]; d=C['d']
a.loglog(d[1:,0],d[1:,1],'-',c='C0',lw=1); a.loglog(d[1:,0],d[1:,2],'-',c='C3',lw=1)
devs=[]
for k,run in enumerate(S['s']['runs']):
    o=run['out'][1:]; a.loglog(o[::2,0],o[::2,1],'o',ms=3,mfc='none',c='C0'); a.loglog(o[::2,0],o[::2,2],['s','^','v'][k],ms=3,mfc='none',c='C3')
    m=(o[:,0]>0.05)&(o[:,0]<d[-1,0]); devs.append((np.abs(np.log(o[m,1]/interp(d,o[m,0],1))).max(),np.abs(np.log(o[m,2]/interp(d,o[m,0],2))).max(),o[-1,1],o[-1,2],o[-1,3],o[:,4].max()))
summary['s']=(devs,d[-1])
a.set_title(titles[3],fontsize=8); a.set_xlabel('$t$'); a.set_ylim(1e-7,3e2)
plt.tight_layout(); plt.savefig('fig8_md.pdf')
for k,(devs,dl) in summary.items():
    print(k,'DSMC final',dl[:3])
    for dv in devs: print('   max|ln MD/DSMC| Tt %.3f Tr %.3f | MD final Tt %.4e Tr %.4e col/part %.0f flow_max %.3f'%dv)
