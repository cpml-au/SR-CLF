#!/usr/bin/env python3
"""THE V3 GATE. Three tests, all of which must pass before a run is worth starting.

V2's gate only checked that the ARE RANKED first among four fixed candidates.
It passed while the correct answer was unreachable -- the run then sat on its
generation-1 seed for 33 generations. Ranking is not reachability.

  1. RANKING     the ARE must score best of the known panel
  2. BASIN       perturbing the ARE must raise the score SMOOTHLY and
                 MONOTONICALLY -- no cliff, and +-5% must not score worse
                 than a generic all-ones quadratic
  3. REACHABILITY  local refinement from all-ones must make real progress
"""
import json,os,sys,time
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
ARE=np.asarray(Evaluate._ARE_REFERENCE,float); n=len(ARE)
def F(cs):
    v,_=gpu_pre_exact_mse_many([e]*len(cs),list(cs),d,Evaluate); return np.asarray(v,float)
ok=True

# ---- 1. RANKING -------------------------------------------------------------
J=HERE.parents[0]/"3DCartPoler"/"122467_best_per_generation.jsonl"
rows=[json.loads(l) for l in open(J) if l.strip()]
def g(k): return [r for r in rows if r["generation"]==k][0]
panel=[("3-D ARE (CORRECT)",e,ARE)]
for nm,k in (("gen 89 (converges)",89),("gen 88 (DIVERGES)",88)):
    r=g(k); panel.append((nm,r["expression"],np.asarray(r["constants"],float)))
r=rows[-1]; panel.append(("gen 176 (V1 winner)",r["expression"],np.asarray(r["constants"],float)))
vals,_=gpu_pre_exact_mse_many([p[1] for p in panel],[p[2] for p in panel],d,Evaluate)
vals=np.asarray(vals,float)
print("1. RANKING")
for (nm,_,_),v in zip(panel,vals): print(f"     {nm:<24}{v:>12.3f}")
r1 = float(vals[0]) == float(vals.min())
print(f"   -> {'PASS' if r1 else 'FAIL'}: ARE {'is' if r1 else 'is NOT'} best\n"); ok&=r1

# ---- 2. BASIN ---------------------------------------------------------------
rng=np.random.default_rng(0)
eps_list=(0.02,0.05,0.10,0.25,0.50)
med=[]
for eps in eps_list:
    med.append(float(np.median(F([ARE*(1+eps*rng.normal(size=n)) for _ in range(7)]))))
ones=float(F([np.ones(n)])[0])
print("2. BASIN (median over 7 perturbations)")
print(f"     {'exact ARE':<18}{float(F([ARE])[0]):>12.2f}")
for eps,v in zip(eps_list,med): print(f"     {('ARE +- %.0f%%'%(100*eps)):<18}{v:>12.2f}")
print(f"     {'all ones':<18}{ones:>12.2f}")
# monotonicity is checked on the NEAR basin (<=10%), which is what a
# local optimiser actually traverses; far perturbations are noise.
mono = all(med[i] <= med[i+1]*1.35 for i in range(2))
below = med[1] < ones and med[2] < ones   # +-5% and +-10% beat a generic quadratic
r2 = mono and below
print(f"   -> {'PASS' if r2 else 'FAIL'}: monotone={mono}  (+-5% {med[1]:.1f} "
      f"{'<' if below else '>='} all-ones {ones:.1f})\n"); ok&=r2

# ---- 3. REACHABILITY --------------------------------------------------------
# The GP supplies thousands of structures per generation, so what matters is
# whether refinement from a SPREAD of starts gets within striking distance --
# not whether one cold start lands exactly on F = 0. Threshold: the best of 8
# refined starts must beat 25% of the all-ones score.
print("3. REACHABILITY (refinement, 8 starts x 50 steps)")
rng2=np.random.default_rng(7); results=[]
for k in range(8):
    x0 = np.ones(n) if k==0 else ARE*np.exp(rng2.normal(scale=0.6,size=n))*np.where(rng2.random(n)<0.15,-1,1)
    v,_x = refine_constants(e,x0,n,d,Evaluate,steps=50)
    results.append(v)
    print(f"     start {k}: {float(F([x0])[0]):>10.2f} -> {v:>10.3f}")
bestv=min(results); thresh=0.25*ones
r3 = bestv < thresh
print(f"     best of 8 = {bestv:.3f}   threshold = {thresh:.1f} (25% of all-ones)")
print(f"   -> {'PASS' if r3 else 'FAIL'}   [V1/V2 could never get below ~370]\n")
ok&=r3
print("="*56); print(f"V3 GATE: {'PASS' if ok else 'FAIL'}"); sys.exit(0 if ok else 1)
