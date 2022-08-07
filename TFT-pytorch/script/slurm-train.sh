#!/usr/bin/env bash
#SBATCH --job-name="total_target_cleaned_scaled"
#SBATCH --output=total_target_cleaned_scaled.out
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
python train.py

## The following is for UVA CS servers
## Note that there are some differences from the Rivanna
## It doesn't have same gpu models as Rivanna.
## cuda-toolkit and anaconda3 instead of cuda and anaconda
## doesn't require library path exporting

# #!/usr/bin/env bash
# #SBATCH --job-name="total_target_cleaned_scaled"
# #SBATCH --output=total_target_cleaned_scaled.out
# #SBATCH --partition=gpu
# #SBATCH --time=1:00:00
# #SBATCH --gres=gpu:1
# #---SBATCH --nodelist=lynx01
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# module load cuda-toolkit cudnn anaconda3

# conda deactivate
# conda activate ml

# python train.py