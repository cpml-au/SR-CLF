"""SR-CLF: a genetic-programming search for control Lyapunov functions of xdot = f(x) + G(x) u.

  src/CPU/          the pipeline, for n states and m inputs (Ray workers, one core each)
  src/cma_all.py    which trees get how many CMA evaluations: options (b) and (c)
  src/settings.py   config.yaml -> the environment variables the code reads
  src/SymFunctions.py, src/Functions.py, src/PredictScoreFuncs.py   helpers for flex
"""
