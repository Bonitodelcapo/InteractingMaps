#!/bin/bash
# Submit the report runs (array job), then the figure job once every run succeeded.
#   bash report/slurm/submit.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# make_tables merges main_dt20ms.csv and every main_base_*.csv; older pieces
# would override the new rows, so they must be moved out of the way first.
old=$(ls report/experiments/main_dt20ms.csv report/experiments/main_base_*.csv 2>/dev/null || true)
if [ -n "$old" ]; then
  echo "Older Table III pieces would be merged into the new table:"
  echo "$old"
  echo "Move them aside first, e.g.:  git rm $(echo $old)"
  exit 1
fi

mkdir -p report/slurm/logs
runs=$(sbatch --parsable report/slurm/runs.sbatch)
figs=$(sbatch --parsable --dependency=afterok:"$runs" report/slurm/figures.sbatch)
echo "runs:    $runs (13 array tasks)"
echo "figures: $figs (starts after every run task succeeds)"
echo "logs:    report/slurm/logs/"
