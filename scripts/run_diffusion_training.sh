#!/bin/bash
#SBATCH --job-name=diffusion_train
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=15-00:00:00
#SBATCH --output=diffusion_training_%j.out
#SBATCH --chdir=/vast/palmer/scratch/lu_lu/ss5235/red-diffeq

# Load modules (adjust for your cluster)
module reset
module load miniconda
conda activate /gpfs/gibbs/project/lu_lu/ss5235/envs/red-diffeq

echo "============================================================"
echo "DIFFUSION MODEL TRAINING - BETA SCHEDULE COMPARISON"
echo "============================================================"
echo "Start time: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "============================================================"
echo ""

# Run training script
python scripts/train_diffusion_schedules.py

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "TRAINING FINISHED"
echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
