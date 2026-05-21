#!/bin/bash
#SBATCH --job-name=smiles_intra_cross
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/smiles_intra_cross_seed42/slurm_%j.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/smiles_intra_cross_seed42/slurm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

mkdir -p /data/home/sczc698/run/xxy/Trans-cross/runs/smiles_intra_cross_seed42

module load miniforge3/24.11
source activate transpec

cd /data/home/sczc698/run/xxy/Trans-cross/code

python scripts/train_smiles_ablation.py \
  --processed-dir /data/home/sczc698/run/xxy/Trans-cross/data/processed \
  --model intra_cross \
  --epochs 30 \
  --batch-size 32 \
  --d-model 128 \
  --encoder-layers 2 \
  --decoder-layers 2 \
  --num-heads 4 \
  --patch-size 64 \
  --lr 1e-4 \
  --seed 42 \
  --max-smiles-len 160 \
  --out-dir /data/home/sczc698/run/xxy/Trans-cross/runs/smiles_intra_cross_seed42
