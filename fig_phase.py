import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3})
P=pickle.load(open('phase.pkl','rb')); q=0.4
def e_of_mu(mu,order):
    e=0.99
    for _ in range(200):
        m=mu*(1+e)/2; kap=8*m/(np.pi*q*(1-q)); Rc=1-np.sqrt(kap)+(3*kap/8+4*q*m/(np.pi*(1-q)) if order==2 else 0)
        e=0.5*e+0.5*(1-np.pi*(1-q)*mu*Rc/(2*q))
    return e
mus=np.logspace(-3,np.log10(0.12),60)
fig,ax=plt.subplots(figsize=(4.4,3.3))
ax.plot(P['es'],P['mu1'][0.0],'C0-',label=r'$\mu_1(e)$, $\beta_0=0$ (saddle-node)')
ax.plot(P['es'],P['mu1'][0.5],'C0--',label=r'$\mu_1(e)$, $\beta_0=0.5$')
ax.plot(P['es'],P['mu2'],'C3-',label=r'$\mu_2=\sqrt{(1-e)/(1+e)}$ (this work)')
ax.plot(P['es'],P['mu2hhz'],'C3:',label=r'$\mu_2$ of HHZ (2000)')
ax.plot([e_of_mu(m,1) for m in mus[mus<0.06]],mus[mus<0.06],'k-.',lw=0.9,label=r'leading asymptote $R_c^{(1)}$')
ax.plot([e_of_mu(m,2) for m in mus],mus,'k:',lw=1.3,label=r'second-order asymptote $R_c^{(2)}$')
ax.plot([0.9],[0.08507],'kx',ms=6,label=r'$\mu_1$ of HHZ ($e=0.9,\beta_0=0.5$)')
ax.set_yscale('log'); ax.set_ylim(3e-3,1); ax.set_xlim(0.3,1); ax.set_xlabel('$e$'); ax.set_ylabel(r'$\mu$'); ax.legend(fontsize=6,loc='lower left')
ax.text(0.35,0.25,'rough (Haff, fixed $\\theta^*$)',fontsize=7.5); ax.text(0.76,0.011,'smooth\n(rotational freezing)',fontsize=7.5); ax.text(0.905,0.16,'bistable',fontsize=7.5)
plt.tight_layout(); plt.savefig('fig2_phase.pdf')
