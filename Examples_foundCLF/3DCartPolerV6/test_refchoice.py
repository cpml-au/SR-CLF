#!/usr/bin/env python3
"""If the basin is thin because the TARGET sits on the PD cone edge, then
aim at a different, better-conditioned CLF. Q and R only select WHICH CLF the
ARE returns -- every one of them is valid by construction."""
import os,sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU","1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
HERE=Path(__file__).resolve().parent
for p in (HERE.parents[1],HERE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import numpy as np, scipy.linalg as sla, Evaluate, src.Functions
from src3DCartPoleV3.grid_fitness import gpu_pre_exact_mse_many
import src3DCartPoleV3.grid_fitness as GF
A=np.array([[-0.2,2,0],[0,0,1],[-0.1,6,0]],float); B=np.array([[0.2],[0],[0.1]])
T=(("x1","x1"),("x1","x2"),("x1","x3"),("x2","x2"),("x2","x3"),("x3","x3"))
pp=[f"mul(a, mul({u}, {v}))" for u,v in T]; e=pp[0]
for q in pp[1:]: e=f"add({e}, {q})"
IDX=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
def coefs(P): return np.array([P[i,j]*(1.0 if i==j else 2.0) for i,j in IDX])
H=Evaluate.BOX_HALF_WIDTHS
axes=[np.linspace(-h,h,21) for h in H]
d=src.Functions.Dataset("true_data",axes,None); d.mesh=list(np.meshgrid(*axes,indexing="ij"))
rng=np.random.default_rng(0)
def pd_loss(P,eps,n=400):
    L=0
    for _ in range(n):
        D=eps*np.abs(P)*rng.normal(size=P.shape); D=.5*(D+D.T)
        if np.linalg.eigvalsh(P+D).min()<=0: L+=1
    return 100*L/n
print(f"{'reference':<34}{'cond':>8}{'lam_min':>9}{'%PD lost 5%':>13}{'norm. lam_min':>15}")
print("-"*80)
cands=[("Q=I,      R=1e-4  (CURRENT)",np.eye(3),1e-4),
       ("Q=I,      R=1e-3",np.eye(3),1e-3),
       ("Q=(1,1,10), R=1e-2",np.diag([1.,1.,10.]),1e-2),
       ("Q=(1,10,100), R=1e-1",np.diag([1.,10.,100.]),1e-1),
       ("Q=(10,10,100), R=1e-1",np.diag([10.,10.,100.]),1e-1)]
store={}
for nm,Q,R in cands:
    P=sla.solve_continuous_are(A,B,Q,np.array([[R]])); P=.5*(P+P.T)
    ev=np.linalg.eigvalsh(P); store[nm]=P
    print(f"{nm:<34}{np.linalg.cond(P):>8.1f}{ev.min():>9.4f}{pd_loss(P,0.05):>12.0f}%"
          f"{ev.min()/ev.max():>15.5f}")
print("\nNow the FITNESS basin, with the shape term aimed at each reference:")
print(f"{'reference':<34}{'F(ref)':>9}{'F(+-2%)':>10}{'F(+-5%)':>10}{'F(+-10%)':>11}{'F(ones)':>10}")
print("-"*84)
for nm,P in store.items():
    c=coefs(P)
    def ref(X1,X2,X3,_P=P): return _P[0,0]*X1**2+2*_P[0,1]*X1*X2+2*_P[0,2]*X1*X3+_P[1,1]*X2**2+2*_P[1,2]*X2*X3+_P[2,2]*X3**2
    Evaluate._reference_quadratic=ref; Evaluate.GPU5_PD_REFERENCE_MATRIX=P
    GF._CONTEXTS.clear()
    r=np.random.default_rng(1)
    rows=[c]+[c*(1+eps*r.normal(size=6)) for eps in (0.02,)*5+(0.05,)*5+(0.10,)*5]+[np.ones(6)]
    v=np.asarray(gpu_pre_exact_mse_many([e]*len(rows),list(rows),d,Evaluate)[0],float)
    print(f"{nm:<34}{v[0]:>9.2f}{np.median(v[1:6]):>10.2f}{np.median(v[6:11]):>10.2f}"
          f"{np.median(v[11:16]):>11.2f}{v[16]:>10.2f}")
