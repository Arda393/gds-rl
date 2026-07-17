#!/bin/bash -l
#SBATCH -p akya-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_base_qwen7b_v100
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=20
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

# V100 (compute 7.0) bf16 destekler: Qwen2.5 native dtype'i bf16, fp16
# overflow sorunu yok. 2x V100 (~32GB) device_map=auto ile bolunur.
python benchmark/evaluate.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --out results/baseline_qwen7b.json \
    --batch-size 4 \
    --dtype bfloat16

echo "BASELINE_DONE"
