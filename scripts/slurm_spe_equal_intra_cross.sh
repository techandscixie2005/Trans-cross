#!/bin/bash
#SBATCH --job-name=spe-e1-intracross
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe_e1_intra_cross_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe_e1_intra_cross_%j.err

set -e

echo "============================================"
echo "SPE E1 IntraCross Training"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================"

module load miniforge3/24.11
source activate transpec

cd /data/home/sczc698/run/xxy/Trans-cross/code/

echo "Python: $(which python)"
echo "Working dir: $(pwd)"

python -u scripts/train_smiles_ablation.py \
    --config configs/smiles_spe_equal_param.yaml \
    --model intra_cross_equal \
    --epochs 30 \
    --batch-size 32 \
    --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_intra_cross_seed42

echo "============================================"
echo "Training complete: $(date)"
echo "============================================"
