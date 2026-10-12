# Sourced by every job. Edit the first block for your cluster.

# --- cluster-specific: load Python and activate the project environment ---
# module load python/3.12
# source "$HOME/venvs/interacting_maps/bin/activate"

export PYTHONPATH="$PWD"
export IM_MLFLOW=0                       # array tasks must not share the sqlite MLflow store
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
