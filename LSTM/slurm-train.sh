#!/usr/bin/env bash
#SBATCH --job-name="lstm"
#SBATCH --output=total.out
#SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=jaguar02
#---SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load anaconda3

# conda deactivate
# conda activate ml
conda deactivate
conda activate ml

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/mi3se/anaconda3/envs/ml/lib
python train.py