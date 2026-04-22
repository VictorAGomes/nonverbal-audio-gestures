#!/bin/bash
#SBATCH -J nonverbal-train
#SBATCH -p gpu-8-v100
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o slurm-%j.out

set -euo pipefail

module purge
module load singularity
module load compilers/nvidia/cuda/12.6

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
CONTAINER_PATH="${CONTAINER_PATH:-$PROJECT_DIR/cnn_env.sif}"

echo "Projeto: $PROJECT_DIR"
echo "Container: $CONTAINER_PATH"
echo "Execucao: python train.py"
echo "Resultados serao salvos no diretorio do projeto"

singularity exec --nv \
    --bind "$PROJECT_DIR:$PROJECT_DIR" \
    --pwd "$PROJECT_DIR" \
    "$CONTAINER_PATH" \
    python train.py
