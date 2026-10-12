#!/bin/bash
#SBATCH --job-name=im-report-runs
#SBATCH --array=0-12
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=report/slurm/logs/%x_%A_%a.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Every run behind the report's tables and figures, one array task per entry of
# JOBS. Submit with report/slurm/submit.sh from the repo root; the figure job
# starts only after all tasks succeed.

set -euo pipefail
source report/slurm/env.sh

R="python report/run_experiments.py --no-frames"
VARY="'delta_shrinkI=0,1;delta_map=0.03;delta_curl=0.6'"
JOBS=(
  # Table III and fig_tracking: 5 models x 1 sequence, 3 s
  "$R --what main --sequences boxes   --out main_base_boxes.csv"
  "$R --what main --sequences poster  --out main_base_poster.csv"
  "$R --what main --sequences dynamic --out main_base_dynamic.csv"
  "$R --what main --sequences bicycle --out main_base_bicycle.csv"
  "$R --what main --sequences street  --out main_base_street.csv"
  # fig_dissociation: 1.5 s runs read by --what dissoc
  "$R --what main --duration 1.5 --sequences boxes   --models thesis,thesis_cmax,thesis_imu --out dissoc_1p5s_boxes.csv"
  "$R --what main --duration 1.5 --sequences bicycle --models thesis,thesis_cmax,thesis_imu --out dissoc_1p5s_bicycle.csv"
  "$R --what main --duration 1.5 --sequences street  --models thesis,thesis_cmax,thesis_imu --out dissoc_1p5s_street.csv"
  # fig_regularisers / Table regularisers: ablation, 1.5 s
  "$R --what sweep --name ablation --mode coord --duration 1.5 --sequences boxes   --vary $VARY --out sweep_boxes.csv"
  "$R --what sweep --name ablation --mode coord --duration 1.5 --sequences poster  --vary $VARY --out sweep_poster.csv"
  "$R --what sweep --name ablation --mode coord --duration 1.5 --sequences dynamic --vary $VARY --out sweep_dynamic.csv"
  "$R --what sweep --name ablation --mode coord --duration 1.5 --sequences bicycle --vary $VARY --out sweep_bicycle.csv"
  # fig_beta_speed: 7 real segments x 3 anchor settings, 3 s
  "$R --what beta"
)

echo "task ${SLURM_ARRAY_TASK_ID}: ${JOBS[$SLURM_ARRAY_TASK_ID]}"
eval "${JOBS[$SLURM_ARRAY_TASK_ID]}"
