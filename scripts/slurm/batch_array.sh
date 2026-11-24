#!/bin/bash
# One SLURM array job per pending TIFF of a mini2p batch.
#
#   mini2p-batch prepare --config <pipeline.toml>      # prints the sbatch line below
#   sbatch --array=1-<N> scripts/slurm/batch_array.sh <batch_tasks_<time>.json>
#
# Add %<k> to the array (--array=1-40%8) to run at most k at once. Each job runs one row of
# the task list (mini2p-batch task); rerunning the same array skips the rows this launch
# has already completed. Each job's log is slurm_logs/batch_<array id>_<task>.out, beside
# the pipeline's own preprocess.log in the run folder.
#
#SBATCH --job-name=mini2p-batch
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-08:00
#SBATCH --output=slurm_logs/batch_%A_%a.out
# To be emailed when jobs end or fail, add your address:
# #SBATCH --mail-type=END,FAIL
# #SBATCH --mail-user=<you>@ucl.ac.uk

set -euo pipefail

# The mini2p environment (made with uv sync in the repo); its bin/ has mini2p-batch.
ENV_DIR="${MINI2P_ENV:-/ceph/python_envs/myoung/mini2p}"

TASKS="${1:?give the batch_tasks_<time>.json that mini2p-batch prepare wrote}"
if [[ ! -x "$ENV_DIR/bin/mini2p-batch" ]]; then
    echo "No $ENV_DIR/bin/mini2p-batch: set MINI2P_ENV to the mini2p environment" >&2
    exit 1
fi

# Let numpy and Suite2p use the CPUs this job was given, and no more.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "$(date -Is) $(hostname) task $SLURM_ARRAY_TASK_ID of $TASKS"
"$ENV_DIR/bin/mini2p-batch" task --tasks "$TASKS" --index "$SLURM_ARRAY_TASK_ID"
