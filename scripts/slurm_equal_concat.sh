#!/bin/bash
#SBATCH --job-name=concat_eq
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/concat_eq_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/concat_eq_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -e

echo "============================================"
echo "SLURM Job: E0 DirectConcat equal-parameter"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Date: $(date)"
echo "============================================"

module load miniforge3/24.11
source activate transpec

cd /data/home/sczc698/run/xxy/Trans-cross/code/

mkdir -p /data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/

echo "Starting E0 DirectConcat equal-parameter training..."
python -u scripts/train_smiles_ablation.py \
    --config configs/smiles_equal_param.yaml \
    --model concat_equal \
    --epochs 30 \
    --batch-size 32 \
    --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/equal_concat_seed42

echo "E0 training complete at $(date)"

# Evaluate
echo "Running evaluation..."
python -u scripts/evaluate_smiles_model.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --run-dir /data/home/sczc698/run/xxy/Trans-cross/runs/equal_concat_seed42 \
    --split valid \
    --save-predictions

python -u scripts/evaluate_smiles_model.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --run-dir /data/home/sczc698/run/xxy/Trans-cross/runs/equal_concat_seed42 \
    --split test \
    --save-predictions

echo "E0 evaluation complete at $(date)"
echo "Done."
