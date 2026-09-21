"""Execute a notebook's code cells in one namespace and (optionally) store the outputs back into the .ipynb.

    python run_notebook_cells.py 4DCartPolePrimeV9.ipynb              # test only: PASS/FAIL per cell
    python run_notebook_cells.py 4DCartPolePrimeV9.ipynb --write      # + stdout, results and figures stored

No jupyter/nbconvert needed (they are not in the flex environment): stdout becomes a `stream` output, the value
of a trailing expression an `execute_result`, every matplotlib figure a PNG `display_data`, and IPython's
display() is captured as well. Written 2026-09-17 to run the V9 notebook as a Slurm job.
"""
import argparse
import ast
import base64
import contextlib
import io
import json
import os
import sys
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("notebook")
ap.add_argument("--write", action="store_true", help="store the outputs back into the notebook file")
ap.add_argument("--stop-on-fail", action="store_true")
args = ap.parse_args()

nb = json.load(open(args.notebook, encoding="utf-8"))
current_outputs = []


def _display(*objs, **kwargs):
    """Stand-in for IPython.display.display: keep the richest text form of each object."""
    for obj in objs:
        for attr, mime in (("_repr_latex_", "text/latex"), ("_repr_html_", "text/html")):
            rep = getattr(obj, attr, None)
            if rep is not None:
                try:
                    text = rep()
                except Exception:
                    text = None
                if text:
                    current_outputs.append({"output_type": "display_data", "metadata": {},
                                            "data": {mime: text.splitlines(keepends=True),
                                                     "text/plain": [repr(obj)]}})
                    break
        else:
            current_outputs.append({"output_type": "display_data", "metadata": {},
                                    "data": {"text/plain": str(obj).splitlines(keepends=True)}})


import IPython.display as _ipd  # noqa: E402
_ipd.display = _display

ns = {"__name__": "__main__", "get_ipython": lambda: None, "display": _display}
fails = []
count = 0
t_all = time.time()
for index, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if not src.strip():
        cell["outputs"] = []
        cell["execution_count"] = None
        print(f"[{index:2d}] empty", flush=True)
        continue
    current_outputs = []
    buf = io.StringIO()
    t0 = time.time()
    ok = True
    try:
        tree = ast.parse(src)
        last = tree.body[-1] if tree.body else None
        with contextlib.redirect_stdout(buf):
            if isinstance(last, ast.Expr):
                exec(compile(ast.Module(body=tree.body[:-1], type_ignores=[]), f"<cell {index}>", "exec"), ns)
                value = eval(compile(ast.Expression(last.value), f"<cell {index}>", "eval"), ns)
                if value is not None:
                    current_outputs.append({"output_type": "execute_result", "execution_count": count + 1,
                                            "metadata": {}, "data": {"text/plain": repr(value).splitlines(keepends=True)}})
            else:
                exec(compile(tree, f"<cell {index}>", "exec"), ns)
    except Exception as exc:
        ok = False
        fails.append((index, f"{type(exc).__name__}: {exc}"))
        tb = traceback.format_exc()
        current_outputs.append({"output_type": "error", "ename": type(exc).__name__, "evalue": str(exc),
                                "traceback": tb.splitlines()})
    text = buf.getvalue()
    if text:
        current_outputs.insert(0, {"output_type": "stream", "name": "stdout",
                                   "text": text.splitlines(keepends=True)})
    for num in plt.get_fignums():                      # every figure the cell drew
        fig = plt.figure(num)
        png = io.BytesIO()
        fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
        current_outputs.append({"output_type": "display_data", "metadata": {},
                                "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii"),
                                         "text/plain": ["<Figure>"]}})
        plt.close(fig)
    count += 1
    cell["outputs"] = current_outputs
    cell["execution_count"] = count
    print(f"[{index:2d}] {'PASS' if ok else 'FAIL'}  {time.time() - t0:7.1f} s"
          + ("" if ok else f"  {fails[-1][1]}"), flush=True)
    if not ok and args.stop_on_fail:
        break

print(f"\n{len(fails)} failing cells in {time.time() - t_all:.0f} s: {[f[0] for f in fails]}")
for index, msg in fails:
    print(f"  [{index}] {msg[:200]}")

if args.write:
    with open(args.notebook, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print("outputs stored in", args.notebook, f"({os.path.getsize(args.notebook) / 1024:.0f} KB)")
