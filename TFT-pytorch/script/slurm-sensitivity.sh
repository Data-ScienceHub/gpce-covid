#!/usr/bin/env bash
#SBATCH --job-name="sensitivity"
#SBATCH --output=outputs/sensitivity_version_1.out
#SBATCH --partition=gpu
#---SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=sds01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

python sensitivity_analysis.py