#!/bin/bash
#SBATCH --job-name=spe-grid
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe_grid_%A_%a.out
#SBATCH --error=/data/home/sczc698/run/xxy/Trans-cross/runs/slurm_logs/spe_grid_%A_%a.err

# SPE Multi-Seed Grid Ablation
# Job array maps:
#   0: SPE-256 E0 seed43
#   1: SPE-256 E1 seed43
#   2: SPE-256 E0 seed44
#   3: SPE-256 E1 seed44
#   4: SPE-512 E0 seed42
#   5: SPE-512 E1 seed42
#   6: SPE-512 E0 seed43
#   7: SPE-512 E1 seed43
#   8: SPE-512 E0 seed44
#   9: SPE-512 E1 seed44

set -e

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Define grid: task_id -> config,model,seed,out_dir
declare -A CONFIG=(
  [0]="configs/smiles_spe_equal_param.yaml"
  [1]="configs/smiles_spe_equal_param.yaml"
  [2]="configs/smiles_spe_equal_param.yaml"
  [3]="configs/smiles_spe_equal_param.yaml"
  [4]="configs/smiles_spe512_equal_param.yaml"
  [5]="configs/smiles_spe512_equal_param.yaml"
  [6]="configs/smiles_spe512_equal_param.yaml"
  [7]="configs/smiles_spe512_equal_param.yaml"
  [8]="configs/smiles_spe512_equal_param.yaml"
  [9]="configs/smiles_spe512_equal_param.yaml"
)

declare -A MODEL=(
  [0]="concat_equal"
  [1]="intra_cross_equal"
  [2]="concat_equal"
  [3]="intra_cross_equal"
  [4]="concat_equal"
  [5]="intra_cross_equal"
  [6]="concat_equal"
  [7]="intra_cross_equal"
  [8]="concat_equal"
  [9]="intra_cross_equal"
)

declare -A SEED=(
  [0]="43"
  [1]="43"
  [2]="44"
  [3]="44"
  [4]="42"
  [5]="42"
  [6]="43"
  [7]="43"
  [8]="44"
  [9]="44"
)

declare -A OUT_DIR=(
  [0]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_concat_seed43"
  [1]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_intra_cross_seed43"
  [2]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_concat_seed44"
  [3]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe_equal_intra_cross_seed44"
  [4]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe512_equal_concat_seed42"
  [5]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe512_equal_intra_cross_seed42"
  [6]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe512_equal_concat_seed43"
  [7]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe512_equal_intra_cross_seed43"
  [8]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe512_equal_concat_seed44"
  [9]="/data/home/sczc698/run/xxy/Trans-cross/runs/spe512_equal_intra_cross_seed44"
)

CFG="${CONFIG[$TASK_ID]}"
M="${MODEL[$TASK_ID]}"
S="${SEED[$TASK_ID]}"
OUT="${OUT_DIR[$TASK_ID]}"

echo "============================================"
echo "SPE Grid Ablation"
echo "Job ID: ${SLURM_JOB_ID}  Task: ${TASK_ID}"
echo "Config: ${CFG}"
echo "Model: ${M}"
echo "Seed: ${S}"
echo "Out dir: ${OUT}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "============================================"

module load miniforge3/24.11
eval "$(conda shell.bash hook)"
conda activate transpec

cd /data/home/sczc698/run/xxy/Trans-cross/code/

echo "Python: $(which python)"
echo "Working dir: $(pwd)"

python -u scripts/train_smiles_ablation.py \
    --config "${CFG}" \
    --model "${M}" \
    --epochs 30 \
    --batch-size 32 \
    --seed "${S}" \
    --out-dir "${OUT}"

echo "============================================"
echo "Training complete: $(date)"
echo "============================================"
