import os, sys, json, re
sys.path.insert(0, '/home/ali1377/symbolx/symclf-main_rnv')
EX = '/home/ali1377/symbolx/symclf-main_rnv/examples/4DCartPoler'
import numpy as np, sympy as sp
from sympy import symbols, sympify, diff, lambdify, Matrix
from scipy.optimize import minimize
from src.b_manifold_check_exact import check_b_manifold_exact
sys.path.insert(0, EX)
from SystemDynamicsSR import fSR, GSR

x1, x2, x3, x4 = symbols('x1 x2 x3 x4')
locs = {"sub": lambda a, b: a - b, "div": lambda a, b: a / b,
        "aq": lambda a, b: a / (1 + b**2)**sp.Rational(1, 2),
        "mul": lambda a, b: a * b, "add": lambda a, b: a + b,
        "neg": lambda a: -a, "pow": lambda a, b: a**b}

def rebuild(e, p):
    for i, m in enumerate(reversed(list(re.finditer(r"\ba\b", e)))):
        s, en = m.span(); e = e[:s] + str(p[-(i + 1)]) + e[en:]
    return sympify(e, locals=locs)

def probe_ratio(Vf):
    V0 = float(Vf(0, 0, 0, 0)); worst = 0.0
    for i in range(4):
        for d in (1e-4, 1e-2):
            for s in (d, -d):
                x = np.zeros(4); x[i] = s
                worst = max(worst, (float(Vf(*x)) - V0) / float(x @ x))
    return worst

# identify the candidate: match origin_probe_ratio 732.1502754197328
target = 732.1502754197328
found = None
for fn in ('68523_best_per_generation.jsonl', '68519_best_per_generation.jsonl'):
    recs = [json.loads(l) for l in open(os.path.join(EX, fn)) if l.strip()]
    # The job files can still be growing. Search every pinned generation rather
    # than assuming the target remains first/middle/last.
    for rec in reversed(recs):
        try:
            V = rebuild(rec['expression'], rec['constants'])
            Vf = lambdify((x1, x2, x3, x4), V, 'numpy')
            r = probe_ratio(Vf)
            if abs(r - target) < 1e-6:
                found = (fn, rec['generation'], V, Vf); break
        except Exception:
            pass
    if found: break

if not found:
    print("could not identify the candidate by probe signature in 68523+68519")
    sys.exit(0)

fn, gen, V, Vf = found
print(f"candidate identified: {fn} gen {gen}")

# Exact barrier margin used by both DE and SHGO:
#     a - rho*|b| + gamma1
Sx = -sp.sin(x3); Cx = -sp.cos(x3); D = 4 * (5 + (1 - Cx**2))
fv = Matrix([x2, (40*Cx*Sx + 4*(2*x4**2*Sx - x2))/D, x4, (-120*Sx - 2*Cx*(2*x4**2*Sx - x2))/D])
Gc = Matrix([0, 4/D, 0, -2*Cx/D])
g = Matrix([diff(V, s) for s in (x1, x2, x3, x4)])
a_e = (g.T*fv)[0]; b_e = (g.T*Gc)[0]
RHO, GAMMA1 = 800.0, 1e-6
ORIGIN_EXCLUDE_RADIUS = 1.1e-3
mf = lambdify((x1, x2, x3, x4),
              a_e - RHO*sp.Abs(b_e) + GAMMA1, 'numpy')
rng = np.random.default_rng(11)
P = rng.uniform(-0.25, 0.25, (4_000_000, 4))
P = P[np.linalg.norm(P, axis=1) > ORIGIN_EXCLUDE_RADIUS]
with np.errstate(all='ignore'):
    mv = np.asarray(mf(*P.T), float)
fin = np.isfinite(mv)
top = P[fin][np.argsort(mv[fin])[-50:]]
best = float(np.max(mv[fin])); bx = None

def _negative_margin_outside_origin(z):
    z = np.clip(z, -0.25, 0.25)
    if np.linalg.norm(z) <= ORIGIN_EXCLUDE_RADIUS:
        return 1e6 + (ORIGIN_EXCLUDE_RADIUS - np.linalg.norm(z))
    return -float(mf(*z))

for x0 in top:
    r = minimize(_negative_margin_outside_origin, x0,
                 method='Nelder-Mead', options=dict(maxiter=400))
    if -r.fun > best: best, bx = -r.fun, np.clip(r.x, -0.25, 0.25)
print(f"GROUND TRUTH sup(barrier a-800*|b|+gamma1) = {best:.4f}")
if bx is not None: print(f"  at x={np.round(bx,4)}")

# where is it? b and a there
af = lambdify((x1,x2,x3,x4), a_e, 'numpy'); bf = lambdify((x1,x2,x3,x4), b_e, 'numpy')
if bx is not None:
    print(f"  a={af(*bx):.4f}  b={bf(*bx):.2e}")

# does the all-axes exact manifold scan see a matching b=0 violation?
r = check_b_manifold_exact(V, fSR, GSR, bounds=[(-0.25, 0.25)]*4,
                           gamma1=GAMMA1, origin_tol=ORIGIN_EXCLUDE_RADIUS,
                           scan_axes=(0, 1, 2, 3), scan_points=1501,
                           grid_points_per_axis=11)
print(f"all-axes exact manifold: roots={r.n_roots} violations={r.n_violations} "
      f"margin_max={r.margin_max}")
