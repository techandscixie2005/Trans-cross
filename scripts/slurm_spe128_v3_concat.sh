#!/bin/bash
#SBATCH --job-name=spe128v3-e0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe128v3_e0_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe128v3_e0_%j.err

set -e
echo "SPE-128 E0 DirectConcat Training (v3 reference)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start: $(date)"

source /etc/profile
module load miniforge3/24.11
source activate transpec
cd /data/home/sczc698/run/xxy/Trans-cross/code/

python -u scripts/train_smiles_ablation.py \
    --config configs/spe128_e1_3layer_v3.yaml \
    --model concat_equal \
    --epochs 50 --batch-size 32 --seed 42 \
    --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/spe128_v3_concat_seed42

echo "Done: $(date)"
