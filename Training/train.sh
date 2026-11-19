#!/bin/bash
#SBATCH --account=b1229
#SBATCH --partition=b1229               # buy-in allocations usually use their own partition; check with: sinfo -p b1229
#SBATCH --time=08:00:00                 # few hrs expected; extra headroom. If it's killed, resubmitting resumes from the last checkpoint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25              # 24 self-play/eval workers + 1 main process (training, saving)
#SBATCH --mem=16G
#SBATCH --job-name=minichess-leela
#SBATCH --output=logs/%x_%j.out

# Train leela on Quest. Submit from the login node:
#     mkdir -p ~/minichess/logs
#     cd ~/minichess && sbatch Training/train.sh
#
# One-time setup (login node):
#     module load python/3.12.10
#     python -m venv ~/minichess/.venv
#     source ~/minichess/.venv/bin/activate
#     pip install numpy

PROJECT=$HOME/minichess
NUM_CYCLES=3501                         # cycle to stop at (not how many to run); training resumes from the latest checkpoint
WORKERS=$((SLURM_CPUS_PER_TASK - 1))

# one BLAS thread per process, so 24 workers don't oversubscribe the cores
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT"
module load python/3.12.10
source .venv/bin/activate

echo "job $SLURM_JOB_ID on $(hostname), $WORKERS workers, to cycle $NUM_CYCLES, started $(date)"
python -u -c "import main; main.train(main.leela, resume_training=True, num_cycles=$NUM_CYCLES, workers=$WORKERS)"
echo "finished $(date)"
