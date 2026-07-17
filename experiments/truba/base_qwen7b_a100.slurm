#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_base_qwen7b_a100
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00

set -euo pipefail

module load apps/truba-ai/gpu-2024.0
source /arf/scratch/YOURUSER/gds_venv/bin/activate

export HF_HOME=/arf/scratch/YOURUSER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd /arf/scratch/YOURUSER/goal_drift

nvidia-smi --query-gpu=name,memory.total --format=csv

# A100 (compute 8.0) bf16 native; 7B tek A100'e rahat sigar.
python benchmark/evaluate.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --out results/baseline_qwen7b.json \
    --batch-size 8 \
    --dtype bfloat16

echo "BASELINE_DONE"
