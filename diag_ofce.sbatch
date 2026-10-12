#!/bin/bash
# OFCE step-instability test (bug 3): one SLURM array task per variant.
#
#   sbatch diag_ofce.sbatch                      # all 7 variants, city_short
#   SEGMENT=street_short sbatch diag_ofce.sbatch
#
# Each task writes diag_ofce/<variant>.csv + .png. When all tasks are done:
#   python diag_omega_ladder.py --mode plots     # -> diag_ofce/summary.csv + table
#
# Variants (see OFCE_VARIANTS in diag_omega_ladder.py):
#   0 E0_base       baseline (delta_VFG 0.12, max_grad 5, dt 20 ms)
#   1 E1_dVFG006    delta_VFG 0.06   bound ~4.0 px/frame
#   2 E1_dVFG024    delta_VFG 0.24   bound ~2.0 px/frame
#   3 E2_noclip     max_grad = inf
#   4 E3_normalize  step / (1 + |F|^2)   (pre-2026-05-27 form)
#   5 E4_dt10       dt 10 ms, same 1.5 s span
#   6 E4_dt05       dt  5 ms, same 1.5 s span (slowest task)
#
#SBATCH --job-name=diag_ofce
#SBATCH --array=0-6
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-ofce-%A_%a.out
# Adjust to your cluster if required:
##SBATCH --partition=<partition>
##SBATCH --account=<account>

set -euo pipefail

VARIANTS=(E0_base E1_dVFG006 E1_dVFG024 E2_noclip E3_normalize E4_dt10 E4_dt05)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
SEGMENT=${SEGMENT:-city_short}

cd "$SLURM_SUBMIT_DIR"                   # submit from the repo root (data/ is relative)
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-base}"
fi
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

echo "[$(date +%T)] task $SLURM_ARRAY_TASK_ID: $VARIANT on $SEGMENT ($(git rev-parse --short HEAD))"
python diag_omega_ladder.py --mode ofce --variant "$VARIANT" --segment "$SEGMENT"
echo "[$(date +%T)] done"
