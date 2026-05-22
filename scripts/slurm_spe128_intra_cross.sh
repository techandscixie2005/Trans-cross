#!/bin/bash
#SBATCH --job-name=spe128-e1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe128_e1_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe128_e1_%j.err

set -e

echo "============================================"
echo "SPE-128 E1 IntraCross Training (v2, g0=-2)"
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
    --config configs/spe128_gate_active_v2.yaml \
    --model intra_cross_equal \
    --epochs 50 \
    --batch-size 32 \
    --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/spe128_intra_cross_seed42

echo "============================================"
echo "Training complete: $(date)"
echo "============================================"
