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
rng=np.random.default_rng(0)
P5=[ARE*(1+0.05*rng.normal(size=6)) for _ in range(7)]
knobs=["V2_SHAPE_WEIGHT","V2_CERT_WEIGHT","V3_POSITIVITY_WEIGHT","CP3D_FLAT_GRAD_WEIGHT",
       "CP3D_ROA_WEIGHT","CP3D_SAT_WEIGHT","ORIGIN_PROBE_PENALTY","CP3D_ROA_NEG_WEIGHT",
       "MANIFOLD_VACUOUS_PENALTY","CP3D_MEAN_MARGIN_WEIGHT"]
base_vals=np.median(gpu_pre_exact_mse_many([e]*7,P5,d,Evaluate)[0])
print(f"F(ARE +- 5%) with everything on = {base_vals:.2f}\n")
print(f"{'zeroed knob':<28}{'F(ARE+-5%)':>13}{'drop':>10}")
print("-"*52)
for k in knobs:
    if not hasattr(Evaluate,k): print(f"{k:<28}{'(absent)':>13}"); continue
    old=getattr(Evaluate,k); setattr(Evaluate,k,0.0)
    try:
        v=float(np.median(gpu_pre_exact_mse_many([e]*7,P5,d,Evaluate)[0]))
    except Exception as ex: v=float('nan')
    setattr(Evaluate,k,old)
    print(f"{k:<28}{v:>13.2f}{base_vals-v:>10.2f}")
