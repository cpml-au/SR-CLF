#!/usr/bin/env python3
"""Break down the V2 pre_exact so the ARE form can be driven to ~0."""
import json,os,sys
from pathlib import Path
os.environ.setdefault("JAX_PLATFORMS","cpu"); os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
os.environ.setdefault("SYMCLF_GPU2_ALLOW_CPU","1"); os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
HERE=Path(__file__).resolve().parent
for p in (HERE.parents[1],HERE):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
import numpy as np, Evaluate, src.Functions
from src3DCartPoleV3.grid_fitness import gpu_pre_exact_mse
from src3DCartPoleV3.fitness import _pd_program, _cartpole_fields_np
from src3DCartPoleV3.cpu_polish import _dual2 as _cpu_dual2

H=Evaluate.BOX_HALF_WIDTHS
axes=[np.linspace(-h,h,21) for h in H]
data=src.Functions.Dataset("true_data",axes,None)
data.mesh=list(np.meshgrid(*axes,indexing="ij"))
X=np.stack([m.ravel() for m in np.meshgrid(*axes,indexing="ij")],axis=1)
P=Evaluate.GPU5_PD_REFERENCE_MATRIX
ref=np.einsum("ij,jk,ik->i",X,P,X); r2=np.sum(X*X,axis=1); nz=r2>0

C=Evaluate._ARE_REFERENCE
T=(("x1","x1"),("x1","x2"),("x1","x3"),("x2","x2"),("x2","x3"),("x3","x3"))
pp=[f"mul(a, mul({u}, {v}))" for u,v in T]; e=pp[0]
for q in pp[1:]: e=f"add({e}, {q})"
ARE={"expression":e,"constants":list(map(float,C))}
J=HERE.parents[0]/"3DCartPoler"/"122467_best_per_generation.jsonl"
rows=[json.loads(l) for l in open(J) if l.strip()]
def g(n): return [r for r in rows if r["generation"]==n][0]

def parts(rec):
    c=np.asarray(rec["constants"],float)
    pre,ent=gpu_pre_exact_mse(rec["expression"],c,data,Evaluate)
    op,opr,lit,nops,par=_pd_program(rec["expression"],c,0)
    o,p2=op.astype(np.int64),opr.astype(np.int64)
    V0=float(_cpu_dual2(o,p2,lit,nops,par,np.zeros(3))[0])
    W=np.empty(len(X)); A=np.empty(len(X)); B=np.empty(len(X))
    for i,x in enumerate(X):
        v,gr,_=_cpu_dual2(o,p2,lit,nops,par,x); gr=np.asarray(gr,float)
        d,gn=_cartpole_fields_np(x); W[i]=float(v)-V0; A[i]=gr@d; B[i]=gr@gn
    ok=nz&(ref>1e-12)&(W>0)
    lq=np.log(W[ok]/ref[ok]); shape=float(np.std(lq))
    badfrac=1.0-ok.sum()/max(nz.sum(),1)
    # violating mask exactly as the kernel builds it
    vv=(W<=1e-4*r2)&nz
    sat=nz&(A>float(Evaluate.CP3D_SAT_U_TARGET)*np.abs(B))
    vio=vv|sat
    minwv=W[vio].min() if vio.any() else np.inf
    return dict(pre=pre,ent=ent,shape=shape,badfrac=badfrac,
                minwv=minwv,nvv=int(vv.sum()),nsat=int(sat.sum()))

print(f"box {H}\n")
print(f"{'':<22}{'pre_exact':>11}{'c_max(rho)':>12}{'std(log q)':>11}{'W<=0 frac':>11}{'minW_viol':>12}{'nV<=0':>7}{'nSAT':>7}")
print("-"*95)
for n,rec in (("3-D ARE (CORRECT)",ARE),("gen 89",g(89)),("gen 88 (DIVERGES)",g(88))):
    d=parts(rec)
    print(f"{n:<22}{d['pre']:>11.2f}{float(d['ent']['c_max']):>12.5g}{d['shape']:>11.4f}"
          f"{d['badfrac']:>11.4f}{d['minwv']:>12.5g}{d['nvv']:>7}{d['nsat']:>7}")
print(f"\n  shape term  = 300 * (std(log q) + 4*W<=0 frac)")
print(f"  cert term   = 500 * clip((rho - minW_viol)/rho, 0, 1)")
