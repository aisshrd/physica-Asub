"""Accuracy of the asymptotic formulas of Sec. 4 (referee comment 4)."""
import numpy as np
from phase import Phi, roots
q=0.4; b0=0.0
def exact_ec(mu):
    lo,hi=0.3,0.999999
    for _ in range(60):
        mid=(lo+hi)/2
        if len(roots(mid,mu,b0,q))>0: hi=mid
        else: lo=mid
    return hi
def Rc1(m): return 1-np.sqrt(8*m/(np.pi*q*(1-q)))
def Rc2(m):
    kap=8*m/(np.pi*q*(1-q)); return 1-np.sqrt(kap)+3*kap/8+4*q*m/(np.pi*(1-q))
if __name__=='__main__':
    print('mu      m_mu     e_c      Rc_exact  Rc_1st   err1     Rc_2nd   err2    err1/m   err2/m^1.5')
    rows=[]
    for mu in [0.001,0.002,0.005,0.01,0.02,0.05,0.10]:
        ec=exact_ec(mu); m=mu*(1+ec)/2; R=2*q*(1-ec)/(np.pi*(1-q)*mu)
        rows.append((mu,m,ec,R,Rc1(m),Rc2(m)))
        print(f'{mu:<7g} {m:.5f}  {ec:.5f}  {R:.4f}   {Rc1(m):.4f}  {R-Rc1(m):+.4f}  {Rc2(m):.4f}  {R-Rc2(m):+.4f}  {(R-Rc1(m))/m:.2f}  {(R-Rc2(m))/m**1.5:.2f}')
    print('predicted O(m) coefficient: 3/(pi q(1-q)) + 4q/(pi(1-q)) =',3/(np.pi*q*(1-q))+4*q/(np.pi*(1-q)))
    print('\nroots: exact vs outer formulas (17) vs threshold formula (18)')
    print('mu    e      R      eps/sqrt(kappa)  th1_exact  th1_(17)   th1_(18)   th2_exact  th2_(17)  th2_(18)')
    for mu,e in [(0.005,0.999),(0.005,0.997),(0.005,0.995),(0.005,0.992),(0.02,0.995),(0.02,0.99),(0.02,0.98),(0.02,0.975)]:
        rt=roots(e,mu,b0,q); m=mu*(1+e)/2; R=2*q*(1-e)/(np.pi*(1-q)*mu); eps=1-R; kap=8*m/(np.pi*q*(1-q))
        if len(rt)<2: print(mu,e,'no pair of roots',rt); continue
        t1o=4*m/(np.pi*(1-q)*(1-R)); t2o=q*(R**-2-1)
        disc=eps**2-kap
        t1i=q*(eps-np.sqrt(disc)) if disc>0 else np.nan; t2i=q*(eps+np.sqrt(disc)) if disc>0 else np.nan
        print(f'{mu:<5g} {e:<6g} {R:.3f}  {eps/np.sqrt(kap):6.2f}          {rt[0]:.4f}     {t1o:.4f}     {t1i:.4f}     {rt[1]:.4f}     {t2o:.4f}    {t2i:.4f}')
