#!/bin/bash
# One SLURM array job per pending TIFF of a mini2p batch.
#
#   mini2p-batch prepare --config <pipeline.toml>      # prints the line below, filled in
#   mkdir -p slurm_logs && MINI2P_ENV=<env> sbatch --array=1-<N> scripts/slurm/batch_array.sh <tasks.json>
#
# Submit from the repo folder: each job's log is slurm_logs/batch_<array id>_<task>.out there
# (SLURM fails a job without a log if slurm_logs/ is missing); the pipeline's own log is
# preprocess.log in the run folder. Add %<k> to the array (--array=1-40%8) to run at most k
# at once. To retry failed tasks, resubmit only those (--array=3,5); a task whose row this
# launch already completed, or that another live job is running, is skipped.
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

# The mini2p environment: the folder whose bin/ holds mini2p-batch (prepare prints it).
ENV_DIR="${MINI2P_ENV:?set MINI2P_ENV to the mini2p environment, as mini2p-batch prepare prints}"
TASKS="${1:?give the batch_tasks_<time>.json that mini2p-batch prepare wrote}"
if [[ ! -x "$ENV_DIR/bin/mini2p-batch" ]]; then
    echo "No $ENV_DIR/bin/mini2p-batch: MINI2P_ENV is not the mini2p environment" >&2
    exit 1
fi

# Let numpy, numba and Suite2p use the CPUs this job was given, and no more.
CPUS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$CPUS" OPENBLAS_NUM_THREADS="$CPUS" MKL_NUM_THREADS="$CPUS"
export NUMEXPR_NUM_THREADS="$CPUS" NUMBA_NUM_THREADS="$CPUS"

echo "$(date -Is) $(hostname) task $SLURM_ARRAY_TASK_ID of $TASKS"
# exec, so SLURM's SIGTERM at the time limit reaches the task, which marks its run interrupted.
exec "$ENV_DIR/bin/mini2p-batch" task --tasks "$TASKS" --index "$SLURM_ARRAY_TASK_ID"
