#!/bin/bash
#SBATCH --job-name=spe1000-e1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe1000_e1_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe1000_e1_%j.err

set -e

echo "============================================"
echo "SPE-1000 E1 IntraCross Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================"

source /etc/profile
module load miniforge3/24.11
source activate transpec

cd /data/home/sczc698/run/xxy/Trans-cross/code/

echo "Python: $(which python)"
echo "Working dir: $(pwd)"

python -u scripts/train_smiles_ablation.py \
    --config configs/smiles_spe1000_ablation.yaml \
    --model intra_cross_equal \
    --epochs 50 \
    --batch-size 32 \
    --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/spe1000_intra_cross_seed42

echo "============================================"
echo "Training complete: $(date)"
echo "============================================"
