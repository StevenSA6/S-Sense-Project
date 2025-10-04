#!/bin/bash
#SBATCH --job-name=pp1-swallow 
#SBATCH --output=logs/slurm-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=SCT

# Initialize micromamba shell
eval "$(micromamba shell hook --shell=bash)"

# Activate environment
micromamba activate pp1-swallow

# Move to your project directory
cd ~/docs/s-sense/Sound

# Run your ML training
python src/main.py
