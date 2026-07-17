#!/bin/bash -l
#SBATCH -p palamut-cuda
#SBATCH -A YOURACCOUNT
#SBATCH -J gds_capsmoke
#SBATCH -o /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.out
#SBATCH -e /arf/scratch/YOURUSER/goal_drift/results/logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --exclude=palamut3,palamut5,palamut6
set -euo pipefail
module load apps/truba-ai/gpu-2024.0
source /arf/scratch/YOURUSER/gds_venv/bin/activate
export HF_HOME=/arf/scratch/YOURUSER/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /arf/scratch/YOURUSER/goal_drift
echo "=== BASELINE (limit 50) ==="
lm_eval --model hf --model_args pretrained=Qwen/Qwen2.5-1.5B-Instruct,dtype=bfloat16 --tasks ifeval,gsm8k,mmlu --apply_chat_template --batch_size 32 --limit 50 --output_path results/capability/smoke_baseline
echo "=== GDS-RL (limit 50) ==="
lm_eval --model hf --model_args pretrained=results/checkpoints/qwen1.5b_grpo,dtype=bfloat16 --tasks ifeval,gsm8k,mmlu --apply_chat_template --batch_size 32 --limit 50 --output_path results/capability/smoke_grpo
echo "CAPSMOKE_DONE"
