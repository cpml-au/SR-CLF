"""CPU checks (ported from symclf-main_rnv/examples/VerifierBenchmark, 2026-09).

conditions.py       the two DEFINITE CLF conditions + the DET3 condition, tolerances, predicates
symbolic.py         candidate string -> sympy V, a, b; numpy callables; sympy -> GP prefix form
exact_check_cpu.py  the reference exact Artstein manifold check + native NumPy PD check
det3_check_cpu.py   DET3: exponential bounded-input Artstein, deterministic lattice + polish
"""
