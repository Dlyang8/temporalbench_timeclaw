#!/bin/bash
#SBATCH --job-name=timecopilot
#SBATCH --account=bdem-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-2%3
#SBATCH --output=timecopilot-%j.out
#SBATCH --error=timecopilot-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
module load python/3.13.5-gcc13.3.1
source /u/dyang12/TemporalBench/.venv/bin/activate

set -a
source .env
set +a

if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "ERROR: DEEPSEEK_API_KEY is missing from .env"
    exit 1
fi

DATASETS=(
    "MIMIC"
    "M5"
    "causal_chambers"
)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

export CUDA_VISIBLE_DEVICES=""
export TOKENIZERS_PARALLELISM=false
export JOBLIB_START_METHOD=spawn
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Dataset: $DATASET"
echo "T1/T3 model: deepseek-v4-pro"
echo "T2/T4 model: deepseek:deepseek-v4-pro"


python -u run_timecopilot.py \
  --llm deepseek:deepseek-v4-pro \
  --qualitative-model deepseek-v4-pro \
  --subset "$DATASET" \
  --ratio 1.0  \
  --num-workers 1
