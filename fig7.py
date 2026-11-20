import numpy as np, pickle, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.grid':True,'grid.alpha':0.3})
R=pickle.load(open('freeze.pkl','rb'))
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0))
sel=[('A: Tt0=100, th0=0.1','A','C0'),('B: Tt0=100, th0=1','B','C3'),('C: Tt0=100, th0=0.01','C','C2'),('D: Tt0=1e4, th0=0.1','D','C1'),('F: Tt0=100, th0=0.1, mu_k=0.003','F','C4'),('G: Tt0=100, th0=0.1, mu_k=0.03','G','C5'),('E: Tt0=1, th0=0.1 (start at v_S)','E','C6')]
for lab,short,c in sel:
    o=R[lab]['out']; ax[0].loglog(o[1:,0],o[1:,2],'-',c=c,lw=1.2,label=short)
ax[0].loglog(R['A: Tt0=100, th0=0.1']['out'][1:,0],R['A: Tt0=100, th0=0.1']['out'][1:,1],'k--',lw=0.8,label='$T_t$ (run A)')
ax[0].set_xlabel('$t$'); ax[0].set_ylabel('$T_r$'); ax[0].legend(fontsize=6,ncol=2); ax[0].set_ylim(1e-7,3e2)
fr=[100*R[l]['nthr']/R[l]['N'] for l,_,_ in sel]; Tr=[R[l]['Tr_inf'] for l,_,_ in sel]
cols=[c for _,_,c in sel]
ax[1].scatter(np.maximum(fr,3e-3),np.maximum(Tr,1e-7),c=cols,s=40,zorder=3)
for f,T,(l,s,c) in zip(fr,Tr,sel): ax[1].annotate(s,(max(f,3e-3),max(T,1e-7)),textcoords='offset points',xytext=(5,3),fontsize=8)
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel(r'frozen fraction $n_f/N$ [%] ($\sigma|\omega|/2>v_S$ at $t=3000$)'); ax[1].set_ylabel(r'$T_r(t=3000)$')
plt.tight_layout(); plt.savefig('fig7_history.pdf'); print('ok')
