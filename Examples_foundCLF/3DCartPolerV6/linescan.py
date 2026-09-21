import os,sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU","1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
HERE=Path(__file__).resolve().parent
for p in (HERE.parents[1],HERE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import numpy as np, Evaluate, src.Functions
from src3DCartPoleV3.grid_fitness import gpu_pre_exact_mse_many, refine_constants
H=Evaluate.BOX_HALF_WIDTHS
axes=[np.linspace(-h,h,21) for h in H]
d=src.Functions.Dataset("true_data",axes,None); d.mesh=list(np.meshgrid(*axes,indexing="ij"))
T=(("x1","x1"),("x1","x2"),("x1","x3"),("x2","x2"),("x2","x3"),("x3","x3"))
pp=[f"mul(a, mul({u}, {v}))" for u,v in T]; e=pp[0]
for q in pp[1:]: e=f"add({e}, {q})"
ARE=np.asarray(Evaluate._ARE_REFERENCE,float); n=6
def F(cs):
    v,_=gpu_pre_exact_mse_many([e]*len(cs),list(cs),d,Evaluate); return np.asarray(v,float)
ones=np.ones(n)
ts=np.linspace(0,1,21)
vals=F([(1-t)*ones+t*ARE for t in ts])
print("LINE SCAN  all-ones  ->  ARE   (is the basin connected?)")
for t,v in zip(ts,vals):
    bar="#"*int(min(60,v/12))
    print(f"  t={t:4.2f}  F={v:9.2f}  {bar}")
print(f"\n  max along the path = {vals.max():.2f} at t={ts[int(np.argmax(vals))]:.2f}")
print(f"  monotone decreasing toward ARE? {bool(np.all(np.diff(vals)<=1e-6))}")
print("\nMULTI-START refinement (what the GP population actually provides):")
rng=np.random.default_rng(7); best=(np.inf,None)
for k in range(8):
    x0=ARE*np.exp(rng.normal(scale=0.6,size=n))*np.where(rng.random(n)<0.15,-1,1) if k else ones.copy()
    v,x=refine_constants(e,x0,n,d,Evaluate,steps=50)
    if v<best[0]: best=(v,x)
    print(f"   start {k}: F {F([x0])[0]:9.2f} -> {v:9.3f}")
print(f"\n  best over 8 starts: {best[0]:.3f}   (ARE itself = 0.000)")
