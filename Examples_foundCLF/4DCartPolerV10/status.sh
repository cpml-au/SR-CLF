#!/usr/bin/env bash
# status.sh [JOBID]  -- one-screen status of a V8 run (default: newest output_*.log)
cd "$(dirname "$0")"
J=${1:-$(ls -t output_*.log 2>/dev/null | head -1 | sed 's/output_\(.*\).log/\1/')}
echo "== job $J: $(squeue -j "$J" -h -o 'state %T  elapsed %M  node %N' 2>/dev/null || echo 'not in queue')  $(sacct -j "$J" -n -o State -X 2>/dev/null | head -1)"
echo "== milestones"; grep -E 'SMOKE|Head address|full exact warm-up|Evaluating initial|START OF EVOLUTION' output_$J.log 2>/dev/null | cut -c1-120 | tail -5
echo "== stage lines: $(grep -c 'V8 stages' output_$J.log 2>/dev/null)   (each = one mapper call of 2500; gen 0 = 5 x 500)"
grep 'V8 stages' output_$J.log 2>/dev/null | tail -3 | grep -o 'candidates=[0-9]*\|pre_s=[0-9.]*\|det4_unique=[0-9]*\|det4_checked=[0-9]*\|det4_cached=[0-9]*\|det4_dead=[0-9]*\|det4_certified=[0-9]*\|det4_s=[0-9.]*\|fitness_min=[^ ]*\|refine_improved=[0-9]*\|refine_s=[0-9.]*' | paste - - - - - - - - - - -
echo "== generations: $(grep -c '^GEN ' output_$J.log 2>/dev/null)"; grep '^GEN ' output_$J.log 2>/dev/null | tail -3 | cut -c1-330
echo "== errors"; grep -v 'FutureWarning\|warnings.warn\|INFO worker\|RuntimeWarning\|fsim\|det4_check_cpu.py:\|return -(\|_optimize.py\|invalid value\|hlo_lexer\|^\s*$' error_$J.log 2>/dev/null | tail -4 | cut -c1-200
