#!/bin/bash
#SBATCH --job-name=parts-yolo-train
#SBATCH --account=YOUR_ACCOUNT_HERE          # edit: your HPC allocation/group
#SBATCH --partition=gpu_standard              # edit: your cluster's GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
nvidia-smi

# --- Environment setup (edit module names to match your cluster) ---
module purge
module load cuda/12.2
module load python/3.10

# Create/activate a venv once, reuse across jobs
VENV_DIR="$HOME/envs/parts-yolo"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r "$SLURM_SUBMIT_DIR/../requirements.txt"
else
    source "$VENV_DIR/bin/activate"
fi

cd "$SLURM_SUBMIT_DIR"

python train.py \
    --data ./data/data.yaml \
    --model yolov8s.pt \
    --epochs 150 \
    --imgsz 640 \
    --batch 32 \
    --device 0 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --project runs/detect \
    --name parts_v1

echo "Job finished at $(date)"
