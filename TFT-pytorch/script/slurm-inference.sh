#!/usr/bin/env bash
#SBATCH --job-name="total_early_stopped_target_cleaned_scaled"
#SBATCH --output=total_early_stopped_target_cleaned_scaled.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=ds6011-sp22-002
#SBATCH --mem=48GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn anaconda

conda deactivate
conda activate ml

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
python inference.py