#!/bin/bash
# Every run behind the report, as many at once as the machine allows, then all
# figures and the Table III body, with per-run and total timings.
#   bash report/run_all.sh                         # from the repo root
#   PARALLEL=90 THREADS=4 bash report/run_all.sh   # PARALLEL*THREADS <= cores
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# --- machine-specific: activate the Python environment ----------------------
# source "$HOME/venvs/interacting_maps/bin/activate"

PARALLEL=${PARALLEL:-80}
THREADS=${THREADS:-4}
export PYTHONPATH="$PWD" IM_MLFLOW=0 MPLBACKEND=Agg PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS

# make_tables merges main_dt20ms.csv and every main_base_*.csv; older pieces
# would override the new rows.
old=$(ls report/experiments/main_dt20ms.csv report/experiments/main_base_*.csv 2>/dev/null || true)
if [ -n "$old" ]; then
  echo "Older Table III pieces would be merged into the new table. Move them aside first:"
  echo "  git rm $(echo $old)"
  exit 1
fi

export LOG="report/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"
R="python report/run_experiments.py --no-frames"

runs() {   # one line per independent run: "name|command"
  for s in boxes poster dynamic bicycle street; do
    for m in cook thesis thesis_imu thesis_cmax thesis_cmax_v2; do
      echo "main_${s}_${m}|$R --what main --sequences $s --models $m --out main_base_${s}_${m}.csv"
    done
  done
  for s in boxes bicycle street; do
    for m in thesis thesis_cmax thesis_imu; do
      echo "dissoc_${s}_${m}|$R --what main --duration 1.5 --sequences $s --models $m --out dissoc_1p5s_${s}_${m}.csv"
    done
  done
  for s in boxes poster dynamic bicycle; do
    for v in delta_shrinkI=0 delta_shrinkI=1 delta_map=0.03 delta_curl=0.6; do
      echo "ablation_${s}_${v}|$R --what sweep --name ablation --mode coord --duration 1.5 --sequences $s --vary $v --out sweep_${s}_${v}.csv"
    done
  done
  # beta entries 4 and 16 (boxes seg_B, poster seg_C at dFR 0.1 / anchor 0.5) are
  # the Table III thesis_cmax runs above; running them twice would write one
  # run directory from two processes at once.
  for i in $(seq 0 20); do
    [ "$i" = 4 ] || [ "$i" = 16 ] || echo "beta_$i|$R --what beta --index $i"
  done
}

figures() {
  local F="python report/run_experiments.py"
  echo "converge_dynamic|$F --what converge --sequences dynamic --frame 45"
  echo "converge_bicycle|$F --what converge --sequences bicycle --frame 40"
  echo "recon|$F --what recon --sequences boxes,poster --duration 0.8 --no-frames"
  echo "dissoc|$F --what dissoc --sequences boxes,bicycle,street --duration 1.5"
  echo "regfig|$F --what regfig --sequences boxes,poster,dynamic,bicycle --duration 1.5"
  echo "make_figures|python report/make_figures.py"
  echo "make_tables|python report/make_tables.py > report/experiments/table_main.tex"
}

run_one() {   # "name|command" -> $LOG/name.log; prints status and seconds
  local name=${1%%|*} cmd=${1#*|} t0=$SECONDS
  if eval "$cmd" > "$LOG/$name.log" 2>&1; then
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
stage figures figures
echo
echo "slowest runs:"; sort -k2 -n -r "$LOG/runs.txt" | head -5
echo "total: $((SECONDS - start)) s   logs: $LOG"
