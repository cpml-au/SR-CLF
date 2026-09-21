import os,sys,json
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU","1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
HERE=Path(__file__).resolve().parent
for p in (HERE.parents[1],HERE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import numpy as np, Evaluate, src.Functions
from src3DCartPoleV3.grid_fitness import gpu_pre_exact_mse_many
H=Evaluate.BOX_HALF_WIDTHS
axes=[np.linspace(-h,h,21) for h in H]
d=src.Functions.Dataset("true_data",axes,None); d.mesh=list(np.meshgrid(*axes,indexing="ij"))
T=(("x1","x1"),("x1","x2"),("x1","x3"),("x2","x2"),("x2","x3"),("x3","x3"))
pp=[f"mul(a, mul({u}, {v}))" for u,v in T]; e=pp[0]
for q in pp[1:]: e=f"add({e}, {q})"
ARE=np.asarray(Evaluate._ARE_REFERENCE,float)
def F(cs):
    v,_=gpu_pre_exact_mse_many([e]*len(cs),list(cs),d,Evaluate); return np.asarray(v,float)

print("CLAIM 1: the fitness is SCALE-INVARIANT, so the [-10,10] box is NOT binding")
ks=[1.0,0.5,0.25,1/2.6,0.1,4.0]
v=F([ARE*k for k in ks])
for k,val in zip(ks,v):
    print(f"   score(ARE x {k:6.4f})  max|c| = {np.abs(ARE*k).max():7.3f}   F = {val:.6f}")
inv = np.allclose(v, v[0], atol=1e-9)
print(f"   -> scale-invariant: {inv}    (if True, my 'ARE is outside the box' claim was WRONG)\n")

print("CLAIM 2: is the ARE the minimiser only because the SHAPE term names it?")
sw=Evaluate.V2_SHAPE_WEIGHT
rng=np.random.default_rng(0)
rand=[rng.uniform(-10,10,6) for _ in range(40)]
for w in (sw, 0.0):
    Evaluate.V2_SHAPE_WEIGHT=w
    fa=F([ARE])[0]; fr=F(rand)
    tag="ON " if w>0 else "OFF"
    print(f"   shape {tag}: F(ARE) = {fa:9.3f}   best of 40 random = {fr.min():9.3f}"
          f"   ARE is best: {fa<=fr.min()}")
Evaluate.V2_SHAPE_WEIGHT=sw
