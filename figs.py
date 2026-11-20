import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from theory import Model,F1c,F2c,mc_brackets
from driven import steady_curve
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3})
q=0.4
# ---- Fig1: F1,F2 vs theta
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
th=np.logspace(-3,3,200)
for mu,c in [(0.02,'C0'),(0.1,'C1'),(0.5,'C2')]:
    l=(1.98*1.4/0.4)*mu/np.sqrt(1+th/q); ax[0].loglog(th,F1c(l),c,label=f'$\\mu={mu}$'); ax[1].loglog(th,F2c(l),c)
M=Model(e=0.98,mu_k=0.02,mu_s=0.5,vS=1.,p=2)
for Tt,ls in [(0.01,'--'),(1,':'),(100,'-.')]:
    f=np.array([M.F12(Tt,t*Tt) for t in th]); ax[0].loglog(th,f[:,0],'k'+ls,label=f'Stribeck, $T_t/T_S={Tt}$'); ax[1].loglog(th,f[:,1],'k'+ls)
ax[0].set_xlabel(r'$\theta=T_r/T_t$'); ax[1].set_xlabel(r'$\theta=T_r/T_t$'); ax[0].set_ylabel('$F_1$'); ax[1].set_ylabel('$F_2$'); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig('fig1_F.pdf')
# ---- Fig2: phase diagram
P=pickle.load(open('phase.pkl','rb'))
fig,ax=plt.subplots(figsize=(4.2,3.2))
ax.plot(P['es'],P['mu1'][0.0],'C0-',label=r'$\mu_1(e)$, $\beta_0=0$ (saddle-node)')
ax.plot(P['es'],P['mu1'][0.5],'C0--',label=r'$\mu_1(e)$, $\beta_0=0.5$')
ax.plot(P['es'],P['mu2'],'C3-',label=r'$\mu_2=\sqrt{(1-e)/(1+e)}$ (this work)')
ax.plot(P['es'],P['mu2hhz'],'C3:',label=r'$\mu_2$ of HHZ (2000)')
ax.plot(P['e_as'],P['mus'],'k-.',label=r'small-$\mu$ asymptote, Eq. (R_c)')
ax.plot([0.9],[0.08507],'kx',ms=6,label=r'$\mu_1$ of HHZ ($e=0.9,\beta_0=0.5$)')
ax.set_yscale('log'); ax.set_ylim(3e-3,1); ax.set_xlim(0.3,1); ax.set_xlabel('$e$'); ax.set_ylabel(r'$\mu$'); ax.legend(fontsize=6.5,loc='lower left')
ax.text(0.35,0.25,'rough (Haff, fixed $\\theta^*$)',fontsize=7.5); ax.text(0.74,0.012,'smooth\n(rotational freezing)',fontsize=7.5); ax.text(0.905,0.16,'bistable',fontsize=7.5)
plt.tight_layout(); plt.savefig('fig2_phase.pdf')
# ---- Fig3: constant-mu DSMC vs ODE
res=pickle.load(open('res.pkl','rb'))
fig,ax=plt.subplots(1,3,figsize=(7.2,2.6))
titles=[r'(a) $e=0.98,\ \theta_0=0.1<\theta_2^*$',r'(b) $e=0.98,\ \theta_0=10>\theta_2^*$',r'(c) $e=0.90$ ($R>R_c$)']
for a,(name,(d,(t,y))),ti in zip(ax,res.items(),titles):
    a.loglog(d[1:,0],d[1:,1],'o',ms=2.5,c='C0',label='DSMC $T_t$'); a.loglog(d[1:,0],d[1:,2],'s',ms=2.5,c='C3',label='DSMC $T_r$')
    a.loglog(t,y[0],'-',c='C0',lw=1,label='theory $T_t$'); a.loglog(t,y[1],'-',c='C3',lw=1,label='theory $T_r$')
    a.set_title(ti,fontsize=8); a.set_xlabel('$t$')
ax[0].legend(fontsize=6.5); ax[0].set_ylabel('$T$'); plt.tight_layout(); plt.savefig('fig3_lock.pdf')
# ---- Fig4: Stribeck cooling + local exponent
C=pickle.load(open('cool.pkl','rb')); d=C['d']
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
ax[0].loglog(d[1:,0],d[1:,1],'o',ms=2.5,c='C0',label='DSMC $T_t$'); ax[0].loglog(d[1:,0],d[1:,2],'s',ms=2.5,c='C3',label='DSMC $T_r$')
ax[0].loglog(C['t'],C['y'][0],'C0-',lw=1,label='theory (per-collision) $T_t$'); ax[0].loglog(C['t'],C['y'][1],'C3-',lw=1,label='theory $T_r$')
ax[0].loglog(C['tm'],C['ym'][0],'C0--',lw=1,label='mean-field $T_t$')
ax[0].loglog(C['tk'],C['yk'][0],'k:',lw=0.8,label=r'const $\mu=\mu_k$'); ax[0].loglog(C['ts'],C['ys'][0],'k-.',lw=0.8,label=r'const $\mu=\mu_s$')
ax[0].set_xlabel('$t$'); ax[0].set_ylabel('$T$'); ax[0].legend(fontsize=6); ax[0].set_ylim(1e-7,2e2)
def locexp(t,T):
    lt,lT=np.log(t),np.log(T); return t[1:-1],-(lT[2:]-lT[:-2])/(lt[2:]-lt[:-2])
tt,ex=locexp(C['t'][5:],C['y'][0][5:]); ax[1].semilogx(tt,ex,'C0-',label='theory (per-collision)')
tt,ex=locexp(C['tm'][5:],C['ym'][0][5:]); ax[1].semilogx(tt,ex,'C0--',label='mean-field')
td,Td=d[1:,0],d[1:,1]; tt,ex=locexp(td,Td); ax[1].semilogx(tt,ex,'ko',ms=2.5,label='DSMC')
ax[1].axhline(2,color='gray',lw=0.8); ax[1].set_xlabel('$t$'); ax[1].set_ylabel(r'$-\,d\ln T_t/d\ln t$'); ax[1].set_ylim(0,4.2); ax[1].legend(fontsize=6.5)
plt.tight_layout(); plt.savefig('fig4_stribeck.pdf')
# ---- Fig5: spin distribution
S=pickle.load(open('state_stribeck.pkl','rb')); S2=pickle.load(open('state_const.pkl','rb'))
fig,ax=plt.subplots(figsize=(4.2,2.8))
for st,lab,c in [(S,'Stribeck friction ($\\mu_k=0.01,\\ \\mu_s=0.12$)','C3'),(S2,'constant $\\mu=0.12$','C0')]:
    w=np.linalg.norm(st['w'],axis=1); w=w/np.sqrt((w**2).mean()) ; ax.hist(np.log10(w),bins=60,histtype='step',color=c,label=lab,density=True)
ax.set_xlabel(r'$\log_{10}\,(|\omega|/\omega_{\rm rms})$'); ax.set_ylabel('pdf'); ax.legend(fontsize=6.5); ax.set_yscale('log')
plt.tight_layout(); plt.savefig('fig5_spin.pdf')
# ---- Fig6: driven steady states exact vs mean field
D=pickle.load(open('driven.pkl','rb')); Tts=D['Tts']
class MF(Model):
    def F12(s,Tt,Tr):
        vbar=np.sqrt(np.pi*(Tt+Tr/s.q)/s.m); mu=Model.mu(s,vbar); l=s.c*mu/np.sqrt(1+Tr/(Tt*s.q)); return F1c(l),F2c(l)
fig,ax=plt.subplots(figsize=(4.2,3.0))
for lab,c in [('mu_s=0.2','C3'),('mu_s=0.1','C1'),('mu_s=0.05','C2'),('const 0.01','k')]:
    h,th=D['res'][lab]; ax.loglog(h,Tts,'-',c=c,label='exact, '+lab.replace('mu_s','$\\mu_s$').replace('const 0.01','const $\\mu=0.01$'))
for ms,c in [(0.2,'C3'),(0.1,'C1')]:
    Mm=MF(e=0.98,mu_k=0.01,mu_s=ms,vS=1.,p=2); h,th=steady_curve(Mm,Tts); ax.loglog(h,Tts,'--',c=c,lw=0.9,label=f'mean-field, $\\mu_s={ms}$')
ax.set_xlabel('heating rate $h$'); ax.set_ylabel('steady $T_t/T_S$'); ax.legend(fontsize=6); plt.tight_layout(); plt.savefig('fig6_driven.pdf')
pickle.dump(dict(Tts=Tts),open('tmp.pkl','wb'))
print('figs done')
