import os,sys,json
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
J=HERE.parents[0]/"3DCartPoler"/"122467_best_per_generation.jsonl"
rows=[json.loads(l) for l in open(J) if l.strip()]
def g(k): return [r for r in rows if r["generation"]==k][0]
panel=[("ARE",e,ARE)]
for nm,k in (("g89",89),("g88",88)):
    r=g(k); panel.append((nm,r["expression"],np.asarray(r["constants"],float)))
ts=np.linspace(0,1,21)
print("V3_SOFTNESS sweep: keep DISCRIMINATION while smoothing the PATH")
print(f"{'softness':>9}{'ARE':>9}{'g89':>10}{'g88':>10}{'sep':>9}{'88>89':>7}"
      f"{'path max':>10}{'F(ones)':>10}{'refine best/8':>15}")
print("-"*90)
for soft in (1e9, 60., 30., 20., 12., 6.):
    Evaluate.V3_SOFTNESS=soft
    v=np.asarray(gpu_pre_exact_mse_many([p[1] for p in panel],[p[2] for p in panel],d,Evaluate)[0],float)
    path=np.asarray(gpu_pre_exact_mse_many([e]*21,[(1-t)*np.ones(n)+t*ARE for t in ts],d,Evaluate)[0],float)
    rng=np.random.default_rng(7); best=np.inf
    for k in range(6):
        x0=np.ones(n) if k==0 else ARE*np.exp(rng.normal(scale=0.6,size=n))*np.where(rng.random(n)<0.15,-1,1)
        bv,_=refine_constants(e,x0,n,d,Evaluate,steps=40); best=min(best,bv)
    sep=v[1:].min()-v[0]
    tag="hard" if soft>1e8 else f"{soft:.0f}"
    print(f"{tag:>9}{v[0]:>9.2f}{v[1]:>10.2f}{v[2]:>10.2f}{sep:>9.2f}"
          f"{('yes' if v[2]>v[1] else 'NO'):>7}{path.max():>10.1f}{path[0]:>10.2f}{best:>15.3f}")
