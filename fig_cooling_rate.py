"""Figure: cooling through the Stribeck crossover (referee comment 7).
(a) temperatures; (b) instantaneous dimensionless cooling rate zeta = -(dT_t/dt)/(k T_t^{3/2}) vs T_t
(independent of the time origin); (c) logarithmic slope -dlnT_t/dln(t+t0) for two choices of the origin."""
import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from theory import Model, F1c, F2c
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3})
k=4*np.sqrt(np.pi)/3
C=pickle.load(open('cool.pkl','rb')); d=C['d']
M=Model(e=0.98,b0=0.,q=0.4,mu_k=0.01,mu_s=0.12,vS=1.,p=2)
class MF(Model):
    def F12(s,Tt,Tr):
        vbar=np.sqrt(np.pi*(Tt+Tr/s.q)/s.m); mu=Model.mu(s,vbar); l=s.c*mu/np.sqrt(1+Tr/(Tt*s.q)); return F1c(l),F2c(l)
Mm=MF(e=0.98,b0=0.,q=0.4,mu_k=0.01,mu_s=0.12,vS=1.,p=2)
def zeta_theory(Mod,t,y):
    return np.array([-Mod.rhs_brackets(a,b)[0]/a for a,b in zip(y[0],y[1])])
def zeta_data(t,T):
    """exact inversion of dT/dt=-k zeta T^{3/2} on each sampling interval (no smoothing)."""
    z=2*(T[1:]**-0.5-T[:-1]**-0.5)/(k*np.diff(t)); return np.sqrt(T[1:]*T[:-1]), z
def slope(t,T,t0=0.):
    lt,lT=np.log(t+t0),np.log(T); return t[1:-1],-(lT[2:]-lT[:-2])/(lt[2:]-lt[:-2])
fig,ax=plt.subplots(1,3,figsize=(7.4,2.75))
a=ax[0]
a.loglog(d[1:,0],d[1:,1],'o',ms=2.5,c='C0',label='DSMC $T_t$'); a.loglog(d[1:,0],d[1:,2],'s',ms=2.5,c='C3',label='DSMC $T_r$')
a.loglog(C['t'],C['y'][0],'C0-',lw=1,label='theory $T_t$'); a.loglog(C['t'],C['y'][1],'C3-',lw=1,label='theory $T_r$')
a.loglog(C['tm'],C['ym'][0],'C0--',lw=1,label='mean-field $T_t$')
a.set_xlabel('$t$'); a.set_ylabel('$T$'); a.set_ylim(1e-7,2e2); a.legend(fontsize=5.5); a.set_title('(a)',fontsize=8,loc='left')
a=ax[1]
zt=zeta_theory(M,C['t'],C['y']); zm=zeta_theory(Mm,C['tm'],C['ym'])
sel=d[:,0]>0.05; Tm,zd=zeta_data(d[sel,0],d[sel,1])
a.loglog(C['y'][0],zt,'C0-',lw=1.2,label='theory (per-collision)'); a.loglog(C['ym'][0],zm,'C0--',lw=1,label='mean-field')
a.loglog(Tm,zd,'ko',ms=2.5,label='DSMC')
zk=zeta_theory(Model(e=0.98,mu_k=0.01,mu_s=0.01),C['tk'],C['yk']); zs=zeta_theory(Model(e=0.98,mu_k=0.12,mu_s=0.12),C['ts'],C['ys'])
a.axhline(zk[0],color='gray',ls=':',lw=0.8); a.axhline(zs[-1],color='gray',ls='-.',lw=0.8)
a.text(3e-6,zk[0]*1.07,r'const. $\mu_k$',fontsize=6,color='gray'); a.text(30,zs[-1]*1.07,r'const. $\mu_s$ (rough state)',fontsize=6,color='gray',ha='left')
a.set_xlim(2e2,5e-7); a.set_xlabel(r'$T_t/T_S$  (time $\rightarrow$)'); a.set_ylabel(r'$\zeta=-\dot T_t/(k\,T_t^{3/2})$'); a.legend(fontsize=5.5,loc='center left'); a.set_title('(b)',fontsize=8,loc='left')
a=ax[2]
tH=2/(k*zt[0]*np.sqrt(C['y'][0][0]))
tt,ex=slope(C['t'][5:],C['y'][0][5:]); a.semilogx(tt,ex,'C0-',lw=1.2,label='theory, origin $t_0=0$')
tt,ex=slope(C['t'][5:],C['y'][0][5:],tH); a.semilogx(tt,ex,'C2-',lw=1.0,label=r'theory, origin shifted by $t_H$')
tt,ex=slope(C['tm'][5:],C['ym'][0][5:]); a.semilogx(tt,ex,'C0--',lw=1,label='mean-field, $t_0=0$')
tt,ex=slope(d[1:,0],d[1:,1]); a.semilogx(tt,ex,'ko',ms=2.5,label='DSMC, $t_0=0$')
a.axhline(2,color='gray',lw=0.8); a.set_xlabel('$t$'); a.set_ylabel(r'$-\,d\ln T_t/d\ln(t+t_0)$'); a.set_ylim(0,5.6); a.legend(fontsize=5.5); a.set_title('(c)',fontsize=8,loc='left')
plt.tight_layout(); plt.savefig('fig4_stribeck.pdf')
print('t_H=',tH,'zeta initial',zt[0],'zeta final theory',zt[-1],'max zeta theory',zt.max(),'at Tt',C['y'][0][np.argmax(zt)],'MF max',zm.max(),'at',C['ym'][0][np.argmax(zm)])
print('const mu_k early zeta',zk[0],' const mu_s late zeta',zs[-1], 'delta',1-0.98**2)
i=np.argmax(zd); print('DSMC max zeta',zd[i],'at',Tm[i],' DSMC late zeta mean',zd[-8:].mean(), 'theory late',zt[-5:].mean())
tt,ex=slope(C['t'][5:],C['y'][0][5:]); print('max slope t0=0',ex.max(),'at t',tt[np.argmax(ex)]); tt,ex=slope(C['t'][5:],C['y'][0][5:],tH); print('max slope shifted',ex.max()); tt,ex=slope(C['t'][5:],C['y'][0][5:],10*tH); print('max slope shifted 10tH',ex.max())
