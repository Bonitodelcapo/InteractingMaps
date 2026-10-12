#!/bin/bash
#SBATCH --job-name=im-report-figures
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=report/slurm/logs/%x_%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME
#
# Every report figure and the Table III body, from the runs of runs.sbatch.
# Writes report/figures/*.{pdf,png} and report/experiments/table_main.tex.

set -euo pipefail
source report/slurm/env.sh

R="python report/run_experiments.py"
$R --what converge --sequences dynamic --frame 45
$R --what converge --sequences bicycle --frame 40
$R --what recon    --sequences boxes,poster --duration 0.8 --no-frames
$R --what dissoc   --sequences boxes,bicycle,street --duration 1.5
$R --what regfig   --sequences boxes,poster,dynamic,bicycle --duration 1.5
python report/make_figures.py
python report/make_tables.py > report/experiments/table_main.tex
