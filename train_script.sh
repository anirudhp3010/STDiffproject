#!/bin/bash
#SBATCH --job-name=stdiff_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=medium      # Ensure this partition has GPUs
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1            # This is the most important line!
#SBATCH --output=/csehome/pydah/STDiffProject/train_output.log
#SBATCH --error=/csehome/pydah/STDiffProject/train_output.log

# Load necessary modules (matching your cluster's environment)
module load gcc/11.4.0-gcc-12.3.0-73jjveq
module load cuda/11.8.0-gcc-12.3.0-4pg4hmh

# Verify GPU is allocated
nvidia-smi 

# Set project directory
ProjDir=/csehome/pydah/STDiffProject
cd $ProjDir

# Enable unbuffered output for immediate log flushing
export PYTHONUNBUFFERED=1

# Use your specific conda environment python
PYTHON=/home/anirudh/.conda/envs/stdiff/bin/python3

# Run training directly with Python (using -u flag for unbuffered output)
$PYTHON -u $ProjDir/stdiff/train_stdiff.py \
    --train_config $ProjDir/stdiff/configs/kitti_range_train_config.yaml