#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_train_qwen1.5b
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --exclude=palamut3

set -euo pipefail

module load apps/truba-ai/gpu-2024.0
source /arf/scratch/YOURUSER/gds_venv/bin/activate

export HF_HOME=/arf/scratch/YOURUSER/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

cd /arf/scratch/YOURUSER/goal_drift

nvidia-smi --query-gpu=name,memory.total --format=csv

# 1) GRPO ile egit (200 gorev, 3 epoch, GDS reward)
echo "=== EGITIM BASLIYOR ==="
python training/grpo_trainer.py --preset qwen1.5b

# 2) Egitilmis checkpoint'i agentic benchmark ile degerlendir ("with GDS-RL")
echo "=== EGITILMIS MODEL DEGERLENDIRILIYOR ==="
python benchmark/agentic_eval.py \
    --model results/checkpoints/qwen1.5b_grpo \
    --out results/agentic_qwen1.5b_grpo.json \
    --max-steps 8 --dtype bfloat16

echo "TRAIN_EVAL_DONE"
