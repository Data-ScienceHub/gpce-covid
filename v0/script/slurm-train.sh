#!/bin/bash -l
# --- this job will be run on any available node
#SBATCH --job-name="Train"
#SBATCH --partition=gpu
#SBATCH --time=15:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24GB
#SBATCH --gres=gpu:v100:1
#SBATCH --account=ds6011-sp22-002
#SBATCH --export=NONE

source /etc/profile.d/modules.sh
source ~/.bashrc
source ~/anaconda3/bin/activate

module load cuda cudnn anaconda

conda deactivate
conda activate ml

python train.py -c '../../reproduce/population_cut/config.json' -d '../../reproduce/population_cut/Population_cut.csv' -o '../../reproduce/population_cut/output' -p '../../reproduce/population_cut/output/checkpoints'
# python train.py -c '../../reproduce/rurality_cut/1/config.json' -d '../../reproduce/rurality_cut/1/Rurality_cut.csv' -o '../../reproduce/rurality_cut/1/output' -p '../../reproduce/rurality_cut/1/output/checkpoints'