#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=train.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=ds6011-sp22-002
#SBATCH --mem=24GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda cudnn anaconda

conda deactivate
conda activate ml

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
python covid_case_and_death_overfitted.py