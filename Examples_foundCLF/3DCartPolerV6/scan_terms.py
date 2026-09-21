import os,sys
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
ts=np.linspace(0,1,21); path=[(1-t)*np.ones(6)+t*ARE for t in ts]
def scan(): 
    v,_=gpu_pre_exact_mse_many([e]*len(path),path,d,Evaluate); return np.asarray(v,float)
base=scan()
print("MAX ALONG THE all-ones -> ARE PATH, with each term switched off")
print(f"{'term zeroed':<28}{'max on path':>13}{'at t':>7}{'  spike removed?':>18}")
print("-"*68)
print(f"{'(nothing -- baseline)':<28}{base.max():>13.2f}{ts[base.argmax()]:>7.2f}")
for k in ("V2_SHAPE_WEIGHT","V2_CERT_WEIGHT","V3_POSITIVITY_WEIGHT","CP3D_FLAT_GRAD_WEIGHT",
          "CP3D_ROA_WEIGHT","CP3D_SAT_WEIGHT","ORIGIN_PROBE_PENALTY","CP3D_ROA_NEG_WEIGHT"):
    if not hasattr(Evaluate,k): continue
    old=getattr(Evaluate,k); setattr(Evaluate,k,0.0)
    try: v=scan()
    except Exception: setattr(Evaluate,k,old); continue
    setattr(Evaluate,k,old)
    print(f"{k:<28}{v.max():>13.2f}{ts[v.argmax()]:>7.2f}"
          f"{('YES  %.0f -> %.0f'%(base.max(),v.max())) if v.max()<0.5*base.max() else '':>18}")
