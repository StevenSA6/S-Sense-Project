#!/bin/bash
#SBATCH --job-name=pp1-swallow 
#SBATCH --output=logs/train-%j.out    # stdout (normal prints)
#SBATCH --error=logs/train-%j.err     # stderr (errors)
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=SCT

# --- Manually initialize micromamba (copied from .bashrc) ---
export MAMBA_EXE='/opt/home/s3976304/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/opt/home/s3976304/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback if shell hook fails
fi
unset __mamba_setup
# --- End micromamba setup ---

# Activate environment
micromamba activate pp1-swallow

# Move to your project directory
cd ~/docs/s-sense/Sound

# Run your ML training
python src/main.py
