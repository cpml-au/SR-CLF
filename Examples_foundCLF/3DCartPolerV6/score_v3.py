#!/usr/bin/env python3
"""ACCEPTANCE TEST for the V2 fitness.

The V1 fitness preferred the gen-176 needle (0.125) over the ARE quadratic
(0.175), because once everything passed, fitness WAS the length penalty and
the ARE form is 10 nodes longer. V2 must reverse that.
"""
import json,os,re,sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("JAX_PLATFORM_NAME","cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES",""); os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU","1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
HERE=Path(__file__).resolve().parent
for p in (HERE.parents[1],HERE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import numpy as np, Evaluate, src.Functions
from src3DCartPoleV3.grid_fitness import gpu_pre_exact_mse
from src3DCartPoleV3.fitness import _compute_exact_result,_attach_pd,_exact_result_penalty,_attach_normalized_margins

H=Evaluate.BOX_HALF_WIDTHS
axes=[np.linspace(-h,h,21) for h in H]
data=src.Functions.Dataset("true_data",axes,None)
data.mesh=list(np.meshgrid(*axes,indexing="ij"))

C=Evaluate._ARE_REFERENCE
T=(("x1","x1"),("x1","x2"),("x1","x3"),("x2","x2"),("x2","x3"),("x3","x3"))
pp=[f"mul(a, mul({u}, {v}))" for u,v in T]; e=pp[0]
for q in pp[1:]: e=f"add({e}, {q})"
ARE={"expression":e,"constants":list(map(float,C))}
J=HERE.parents[0]/"3DCartPoler"/"122467_best_per_generation.jsonl"
rows=[json.loads(l) for l in open(J) if l.strip()]
def g(n): return [r for r in rows if r["generation"]==n][0]
def nodes(x): return len(re.findall(r'[A-Za-z_]+\(',x))+len(re.findall(r'\b(x[0-9]+|a)\b',x))

print(f"V2 box half-widths = {H}\n")
print(f"{'candidate':<24}{'pre_exact':>11}{'exact':>9}{'length':>8}{'TOTAL':>10}{'  truth':>22}")
print("-"*86)
out=[]
for name,rec,truth in (("3-D ARE (CORRECT)",ARE,"valid by construction"),
                       ("gen 89",g(89),"converges 4/4"),
                       ("gen 176 (V1 winner)",rows[-1],"unknown"),
                       ("gen 88",g(88),"DIVERGES 4/4")):
    c=np.asarray(rec["constants"],float)
    pre,ent=gpu_pre_exact_mse(rec["expression"],c,data,Evaluate)
    res=_compute_exact_result(rec["expression"],c,Evaluate)
    res=_attach_pd(res,rec["expression"],c,Evaluate)
    res=_attach_normalized_margins(res,rec["expression"],c,Evaluate)
    ex=float(_exact_result_penalty(res,Evaluate))
    ln=0.005*nodes(rec["expression"])
    tot=pre+ex+ln
    out.append((name,tot))
    print(f"{name:<24}{pre:>11.3f}{ex:>9.3f}{ln:>8.3f}{tot:>10.3f}{truth:>22}")
print()
best=min(out,key=lambda r:r[1])
print(f"V2 ranks BEST: {best[0]}  ({best[1]:.3f})")
print(f"V1 ranked best: gen 176 (0.125), with the ARE at 0.175 -- WRONG WAY ROUND")
