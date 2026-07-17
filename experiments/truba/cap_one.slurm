#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --exclude=palamut3,palamut5,palamut6
set -euo pipefail
OUT=$1; MARGS=$2; BS=${3:-32}
module load apps/truba-ai/gpu-2024.0
source /arf/scratch/YOURUSER/gds_venv/bin/activate
export HF_HOME=/arf/scratch/YOURUSER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /arf/scratch/YOURUSER/goal_drift
nvidia-smi --query-gpu=name --format=csv,noheader
lm_eval --model hf --model_args ${MARGS},dtype=bfloat16 --tasks ifeval,gsm8k,mmlu --apply_chat_template --batch_size $BS --output_path results/capability/$OUT
echo "CAP_DONE $OUT"
