#!/bin/bash
#SBATCH --job-name=spe128v3-e1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe128v3_e1_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe128v3_e1_%j.err

set -e
echo "SPE-128 E1-3Layer IntraCross Training (v3 diagnostic)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

source /etc/profile
module load miniforge3/24.11
source activate transpec
cd /data/home/sczc698/run/xxy/Trans-cross/code/

python -u scripts/train_smiles_ablation.py \
    --config configs/spe128_e1_3layer_v3.yaml \
    --model intra_cross_equal \
    --epochs 50 --batch-size 32 --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/spe128_v3_e1_3layer_seed42

echo "Done: $(date)"
