#!/bin/bash
# Every run behind the report, as many at once as the machine allows, then all
# figures and the Table III body, with per-run and total timings.
#   bash report/run_all.sh                         # from the repo root
#   PARALLEL=90 THREADS=4 bash report/run_all.sh   # PARALLEL*THREADS <= cores
#   FORCE=1 bash report/run_all.sh                 # ignore stamps, rerun all
#   ADOPT=1 bash report/run_all.sh                 # run nothing: stamp the jobs
#     earlier logs (report/logs/*/runs.txt, figures.txt) show as [ok] and whose
#     --out exists, i.e. declare those results current for the present code
#
# Skipping: a job whose code, data and command are unchanged since its last
# successful run gives the same numbers (the runs are deterministic), so it is
# skipped. Its fingerprint = sha1(command + hash of the code it depends on +
# data signature) is stored in report/.stamps/<job> after success; a job whose
# --out file (report/experiments/ or experiments/) is gone reruns. Rerun one
# job: rm report/.stamps/<job>.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- machine-specific: activate the Python environment ----------------------
# source "$HOME/venvs/interacting_maps/bin/activate"

PARALLEL=${PARALLEL:-80}
THREADS=${THREADS:-4}
export PYTHONPATH="$PWD" IM_MLFLOW=0 MPLBACKEND=Agg PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

export LOG="report/logs/$(date +%Y%m%d_%H%M%S)"
export STAMPS="report/.stamps" FORCE=${FORCE:-} ADOPT=${ADOPT:-}
mkdir -p "$LOG" "$STAMPS"
R="python report/run_experiments.py --no-frames"

code_hash() {   # code_hash PATHSPEC...: sha1 over names + working-tree contents
  git ls-files -co --exclude-standard -- "$@" | sort | xargs -d '\n' sha1sum | sha1sum | cut -c1-40
}
# What each group's results depend on. REPORT: all Python the report runs can
# import (plotting-only and diagnostic scripts excluded); DIAG: the diagnostic
# and the core it imports; FIGURES is set after the runs (code + run stamps).
export HASH_REPORT HASH_DIAG HASH_DATA HASH_FIGURES=
HASH_REPORT=$(code_hash '*.py' ':!tests/*' ':!diag_*.py' ':!demo.py' ':!debug_*.py' \
              ':!report/make_figures.py' ':!report/make_tables.py' ':!report/analyze_grid.py')
HASH_DIAG=$(code_hash 'interacting_maps/*.py' config.py diag_omega_ladder.py)
HASH_DATA=$(find data -type f -printf '%p %s %T@\n' 2>/dev/null | sort | sha1sum | cut -c1-40)

runs() {   # one line per independent run: "name|group|command"
  for s in boxes poster dynamic bicycle street; do
    for m in cook thesis thesis_imu thesis_cmax thesis_cmax_v2; do
      echo "main_${s}_${m}|REPORT|$R --what main --sequences $s --models $m --out main_base_${s}_${m}.csv"
    done
  done
  # 1.5 s, every model: the "network" column of tab:baseline and the
  # without-curl half of tab:curl; the with-curl half next to it.
  for s in boxes poster dynamic bicycle street; do
    for m in cook thesis thesis_imu thesis_cmax thesis_cmax_v2; do
      echo "dissoc_${s}_${m}|REPORT|$R --what main --duration 1.5 --sequences $s --models $m --out dissoc_1p5s_${s}_${m}.csv"
      echo "curl_${s}_${m}|REPORT|$R --what main --duration 1.5 --vary delta_curl=0.6 --sequences $s --models $m --out main_curl_${s}_${m}.csv"
    done
  done
  # tab:sweep: one point per job, around the starting values (dFR 0.5, anchor 0.3).
  # The base point is run once, as delta_FR=0.5; every block shares it.
  for s in boxes poster dynamic bicycle street; do
    for v in delta_FR=0.5 delta_IMU=0 delta_IMU=0.1 delta_IMU=0.5 \
             delta_FR=0.1 delta_FR=0.3 delta_FR=0.7 delta_FR=0.9 \
             n_iters=25 n_iters=50 n_iters=150 dt=0.01 dt=0.05; do
      echo "sweep_${s}_cmax_${v}|REPORT|$R --what sweep --name sweep --mode coord --model thesis_cmax --base delta_FR=0.5,delta_IMU=0.3 --sequences $s --vary $v --out sweep_${s}_cmax_${v}.csv"
    done
    for v in delta_IMU=0.1 delta_IMU=0.3 delta_IMU=0.5; do
      echo "sweep_${s}_imu_${v}|REPORT|$R --what sweep --name sweep --mode coord --model thesis_imu --base delta_FR=0.5,delta_IMU=0.3 --sequences $s --vary $v --out sweep_${s}_imu_${v}.csv"
    done
  done
  # tab:params: the (delta_FR, anchor) grid, 1.5 s, one delta_FR row per job
  for s in boxes poster dynamic bicycle street; do
    for fr in 0.02 0.05 0.10 0.30 0.50; do
      echo "grid_${s}_FR${fr}|REPORT|$R --what sweep --name grid --mode grid --model thesis_cmax --duration 1.5 --sequences $s --vary 'delta_FR=$fr;delta_IMU=0.1,0.3,0.5,0.7,0.9' --out grid_${s}_FR${fr}.csv"
    done
  done
  # tab:anchor: poster seg_A and seg_D (seg_C is the main table; thesis_cmax on
  # A and D is a beta run, so it is not repeated here)
  for g in seg_A seg_D; do
    for m in cook thesis thesis_imu thesis_cmax_v2; do
      echo "anchor_${g}_${m}|REPORT|python evaluation.py --exp 2 --dataset poster_rotation --segment $g --model $m --no-frames"
    done
  done
  echo "window|REPORT|$R --what window"
  for s in boxes poster dynamic bicycle; do
    for v in delta_shrinkI=0 delta_shrinkI=1 delta_map=0.03 delta_curl=0.6; do
      echo "ablation_${s}_${v}|REPORT|$R --what sweep --name ablation --mode coord --duration 1.5 --sequences $s --vary $v --out sweep_${s}_${v}.csv"
    done
  done
  # CMax and gyro alone, no network: Table tab:baseline and fig_cmax_baseline
  echo "baseline|REPORT|$R --what baseline --duration 1.5"
  # beta entries 4 and 16 (boxes seg_B, poster seg_C at dFR 0.1 / anchor 0.5) are
  # the Table III thesis_cmax runs above; running them twice would write one
  # run directory from two processes at once.
  for i in $(seq 0 20); do
    [ "$i" = 4 ] || [ "$i" = 16 ] || echo "beta_$i|REPORT|$R --what beta --index $i"
  done
  # OFCE step-instability diagnostic (not in the report): one job per variant.
  for v in $OFCE_VARIANTS; do
    echo "ofce_$v|DIAG|python diag_omega_ladder.py --mode ofce --variant $v"
  done
}
OFCE_VARIANTS=$(python -c "import diag_omega_ladder as d; print(*d.OFCE_VARIANTS)")

figures() {
  local F="python report/run_experiments.py"
  echo "converge_dynamic|FIGURES|$F --what converge --sequences dynamic --frame 45"
  echo "converge_bicycle|FIGURES|$F --what converge --sequences bicycle --frame 40"
  echo "recon|FIGURES|$F --what recon --sequences boxes,poster --duration 0.8 --no-frames"
  echo "dissoc|FIGURES|$F --what dissoc --sequences boxes,bicycle,street --duration 1.5"
  echo "regfig|FIGURES|$F --what regfig --sequences boxes,poster,dynamic,bicycle --duration 1.5"
  echo "make_figures|FIGURES|python report/make_figures.py"
  echo "make_tables|FIGURES|python report/make_tables.py > report/experiments/table_main.tex"
  echo "ofce_summary|FIGURES|python diag_omega_ladder.py --mode ofce_summary"
}

# make_tables merges main_dt20ms.csv and every main_base_*.csv; a piece no
# current run writes (renamed or dropped run) would override the new rows.
expected=$(runs | sed -n 's/.*--out \(main_base_[^ ]*\.csv\).*/report\/experiments\/\1/p' | sort)
stray=$(comm -23 <(ls report/experiments/main_dt20ms.csv report/experiments/main_base_*.csv 2>/dev/null | sort) \
                 <(echo "$expected"))
if [ -n "$stray" ]; then
  echo "Table III pieces no current run writes would be merged into the table. Move them aside first:"
  echo "  git rm $(echo $stray)"
  exit 1
fi

run_one() {   # "name|group|command" -> $LOG/name.log; skips if the stamp matches
  local name group cmd hv fp out t0=$SECONDS
  IFS='|' read -r name group cmd <<< "$1"
  hv="HASH_$group"
  fp=$(printf '%s\n%s\n%s\n' "$cmd" "${!hv}" "$HASH_DATA" | sha1sum | cut -c1-40)
  out=$(sed -n 's/.*--out \([^ ]*\).*/\1/p' <<< "$cmd")
  if [ -z "$FORCE" ] && [ "$(cat "$STAMPS/$name" 2>/dev/null)" = "$fp" ] \
     && { [ -z "$out" ] || [ -n "$(find report/experiments experiments -name "$out" -print -quit 2>/dev/null)" ]; }; then
    printf '[skip]      -    %s  (unchanged)\n' "$name"
    return 0
  fi
  if [ -n "$ADOPT" ]; then   # stamp jobs that earlier logs show succeeded; run nothing
    if cat report/logs/*/runs.txt report/logs/*/figures.txt 2>/dev/null \
         | awk -v n="$name" '$1 == "[ok]" && $NF == n {f = 1} END {exit !f}' \
       && { [ -z "$out" ] || [ -n "$(find report/experiments experiments -name "$out" -print -quit 2>/dev/null)" ]; }; then
      echo "$fp" > "$STAMPS/$name"
      printf '[adopt]     -    %s\n' "$name"
    else
      printf '[none]      -    %s  (no earlier success or output missing: will run)\n' "$name"
    fi
    return 0
  fi
  rm -f "$STAMPS/$name"
  if eval "$cmd" > "$LOG/$name.log" 2>&1; then
    echo "$fp" > "$STAMPS/$name"
    printf '[ok]   %6d s  %s\n' $((SECONDS - t0)) "$name"
  else
    printf '[FAIL] %6d s  %s  (%s)\n' $((SECONDS - t0)) "$name" "$LOG/$name.log"
    return 1
  fi
}
export -f run_one

stage() {   # stage NAME LISTFN: run every line of LISTFN in parallel, timed
  local t0=$SECONDS n
  n=$($2 | wc -l)
  echo "== $1: $n jobs, up to $PARALLEL at a time, $THREADS threads each"
  if ! $2 | xargs -P "$PARALLEL" -I{} bash -c 'run_one "$1"' _ {} | tee "$LOG/$1.txt"; then
    echo "== $1 had failures after $((SECONDS - t0)) s; stopping. Logs: $LOG"
    exit 1
  fi
  echo "== $1 done in $((SECONDS - t0)) s"
}

start=$SECONDS
stage runs runs
# Figures depend on every script and on what the runs produced: any rerun run
# (new stamp) or any code change reruns them.
HASH_FIGURES=$( { code_hash '*.py' ':!tests/*'; runs | cut -d'|' -f1 \
                  | sed "s|^|$STAMPS/|" | xargs cat 2>/dev/null || true; } | sha1sum | cut -c1-40)
stage figures figures
echo
echo "slowest runs:"; sort -k2 -n -r "$LOG/runs.txt" | head -5
echo "total: $((SECONDS - start)) s   logs: $LOG"
