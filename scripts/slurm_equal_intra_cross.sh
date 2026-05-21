#!/bin/bash
#SBATCH --job-name=intracross_eq
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/intra_cross_eq_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/intra_cross_eq_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -e

echo "============================================"
echo "SLURM Job: E1 IntraCross equal-parameter"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Date: $(date)"
echo "============================================"

module load miniforge3/24.11
source activate transpec

cd /data/home/sczc698/run/xxy/Trans-cross/code/

mkdir -p /data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/

echo "Starting E1 IntraCross equal-parameter training..."
python -u scripts/train_smiles_ablation.py \
    --config configs/smiles_equal_param.yaml \
    --model intra_cross_equal \
    --epochs 30 \
    --batch-size 32 \
    --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/equal_intra_cross_seed42

echo "E1 training complete at $(date)"

# Evaluate
echo "Running evaluation..."
python -u scripts/evaluate_smiles_model.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --run-dir /data/home/sczc698/run/xxy/Trans-cross/runs/equal_intra_cross_seed42 \
    --split valid \
    --save-predictions

python -u scripts/evaluate_smiles_model.py \
    --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
    --run-dir /data/home/sczc698/run/xxy/Trans-cross/runs/equal_intra_cross_seed42 \
    --split test \
    --save-predictions

echo "E1 evaluation complete at $(date)"
echo "Done."
