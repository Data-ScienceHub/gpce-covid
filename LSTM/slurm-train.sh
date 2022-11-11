#!/usr/bin/env bash
#SBATCH --job-name="lstm"
#SBATCH --output=total.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

python train.py