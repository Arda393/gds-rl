#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_smoke_agentic_qwen1.5b
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --exclude=palamut3

set -euo pipefail

module load apps/truba-ai/gpu-2024.0
source /arf/scratch/YOURUSER/gds_venv/bin/activate

export HF_HOME=/arf/scratch/YOURUSER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /arf/scratch/YOURUSER/goal_drift

nvidia-smi --query-gpu=name,memory.total --format=csv

python benchmark/agentic_eval.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --out results/smoke_agentic_qwen1.5b.json \
    --limit 20 \
    --max-steps 8 \
    --dtype bfloat16

echo "SMOKE_AGENTIC_DONE"
