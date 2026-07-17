#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_abl_act_only
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
export HF_HOME=/arf/scratch/YOURUSER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd /arf/scratch/YOURUSER/goal_drift
echo "=== GRPO EGITIM ablation=act_only ==="
python training/grpo_trainer.py --preset qwen1.5b --ablation act_only
echo "=== EGITILMIS TEST (plan) ==="
python benchmark/plan_eval.py --model results/checkpoints/qwen1.5b_grpo_act_only --split test --out results/plan_test_grpo_qwen1.5b_act_only.json --dtype bfloat16
echo "RIGOROUS_DONE"
