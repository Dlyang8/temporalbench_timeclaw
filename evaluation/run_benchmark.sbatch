#!/bin/bash
#SBATCH --job-name=temporalbench_eval
#SBATCH --account=bevu-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/nvme/bcqc/mweng/TemporalBench/evaluation
#SBATCH --output=/work/nvme/bcqc/mweng/TemporalBench/evaluation/logs/eval_%j.log
#SBATCH --error=/work/nvme/bcqc/mweng/TemporalBench/evaluation/logs/eval_%j.log

set -eo pipefail
mkdir -p /work/nvme/bcqc/mweng/TemporalBench/evaluation/logs

# Configure ENABLED_DATASETS / ENABLED_TIERS / ENABLED_MODELS at the top of
# evaluate_llm.py, then export the API key(s) for the selected model(s) before
# submitting, e.g.:
#   export GEMINI_API_KEY=...
#   sbatch run_benchmark.sbatch

PY=/projects/beei/mweng/conda/envs/agent_env/bin/python

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

"$PY" -u evaluate_llm.py
