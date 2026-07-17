#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_rig_qwen1.5b
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

# Egitim VE degerlendirme AYNI format (single-turn plan) -> confound yok.
# 1) BASELINE (egitilmemis) - TEST split (held-out), plan format, greedy
echo "=== 1/3 BASELINE TEST (plan) ==="
python benchmark/plan_eval.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --split test --out results/plan_test_baseline_qwen1.5b.json \
    --dtype bfloat16

# 2) GRPO EGITIM - TRAIN split (160 gorev), F1-tabanli reward, plan format
echo "=== 2/3 GRPO EGITIM (train split) ==="
python training/grpo_trainer.py --preset qwen1.5b

# 3) EGITILMIS model - TEST split (ayni held-out 40 gorev), plan format
echo "=== 3/3 EGITILMIS TEST (plan) ==="
python benchmark/plan_eval.py \
    --model results/checkpoints/qwen1.5b_grpo \
    --split test --out results/plan_test_grpo_qwen1.5b.json \
    --dtype bfloat16

echo "RIGOROUS_DONE"
