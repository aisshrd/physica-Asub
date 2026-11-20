"""Figure: heating curve of the stochastically driven gas (per-collision theory, mean field, DSMC)."""
import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from theory import Model, F1c, F2c
from driven import steady_curve
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3})
D=pickle.load(open('driven.pkl','rb')); Tts=D['Tts']
class MF(Model):
    def F12(s,Tt,Tr):
        vbar=np.sqrt(np.pi*(Tt+Tr/s.q)/s.m); mu=Model.mu(s,vbar); l=s.c*mu/np.sqrt(1+Tr/(Tt*s.q)); return F1c(l),F2c(l)
fig,ax=plt.subplots(figsize=(4.6,3.3))
for lab,c in [('mu_s=0.2','C3'),('mu_s=0.1','C1'),('mu_s=0.05','C2'),('const 0.01','k')]:
    h,th=D['res'][lab]; ax.loglog(h,Tts,'-',c=c,label='per-collision theory, '+lab.replace('mu_s=','$\\mu_s=$').replace('const 0.01','const. $\\mu=0.01$'))
for ms,c in [(0.2,'C3'),(0.1,'C1')]:
    Mm=MF(e=0.98,mu_k=0.01,mu_s=ms,vS=1.,p=2); h,th=steady_curve(Mm,Tts); ax.loglog(h,Tts,'--',c=c,lw=0.9,label=f'mean field, $\\mu_s={ms}$')
# DSMC steady states for mu_s = 0.2 (hot and cold initial conditions coincide)
hd=[3e-4,1e-3,3e-3,1e-2,1.0]; Td=[0.0049,0.0110,0.0233,0.0580,3.27]
ax.loglog(hd,Td,'o',ms=5,mfc='none',mec='C3',mew=1.2,label='DSMC, $\\mu_s=0.2$')
ax.set_xlabel('heating rate $h$  (injected power per particle)'); ax.set_ylabel('steady temperature $T_t/T_S$'); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig('fig6_driven.pdf')
