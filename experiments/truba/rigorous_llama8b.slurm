#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_rig_llama8b
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --exclude=palamut3,palamut5
set -euo pipefail
module load apps/truba-ai/gpu-2024.0
source /arf/scratch/YOURUSER/gds_venv/bin/activate
export HF_HOME=/arf/scratch/YOURUSER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline
cd /arf/scratch/YOURUSER/goal_drift
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== 1/3 BASELINE TEST (plan) ==="
python benchmark/plan_eval.py --model meta-llama/Llama-3.1-8B-Instruct --split test --out results/plan_test_baseline_llama8b.json --dtype bfloat16
echo "=== 2/3 GRPO EGITIM ==="
python training/grpo_trainer.py --preset llama8b
echo "=== 3/3 EGITILMIS TEST (plan) ==="
python benchmark/plan_eval.py --model results/checkpoints/llama8b_grpo --split test --out results/plan_test_grpo_llama8b.json --dtype bfloat16
echo "RIGOROUS_DONE"
