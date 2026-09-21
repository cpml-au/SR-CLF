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
rng=np.random.default_rng(0); rand=[rng.uniform(-10,10,6) for _ in range(12)]
def F(cs):
    v,_=gpu_pre_exact_mse_many([e]*len(cs),list(cs),d,Evaluate); return np.asarray(v,float)
print("Is the shape term alive? Sweep its weight and watch a fixed panel.")
print(f"{'V2_SHAPE_WEIGHT':>17}{'F(ARE)':>10}{'F(ones)':>10}{'median F(random)':>19}")
print("-"*58)
for w in (0.0, 300.0, 3000.0, 30000.0):
    Evaluate.V2_SHAPE_WEIGHT=w
    v=F([ARE,np.ones(6)]+rand)
    print(f"{w:>17.0f}{v[0]:>10.3f}{v[1]:>10.3f}{np.median(v[2:]):>19.3f}")
Evaluate.V2_SHAPE_WEIGHT=300.0
print("\nAnd the softness knob, same panel:")
print(f"{'V3_SOFTNESS':>17}{'F(ARE)':>10}{'F(ones)':>10}{'median F(random)':>19}")
print("-"*58)
for s in (1e9, 100.0, 30.0, 12.0):
    Evaluate.V3_SOFTNESS=s
    v=F([ARE,np.ones(6)]+rand)
    tag="hard" if s>1e8 else f"{s:.0f}"
    print(f"{tag:>17}{v[0]:>10.3f}{v[1]:>10.3f}{np.median(v[2:]):>19.3f}")
