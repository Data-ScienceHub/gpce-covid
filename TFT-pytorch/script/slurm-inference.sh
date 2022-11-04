#!/usr/bin/env bash
#SBATCH --job-name="total_target_cleaned_scaled"
#SBATCH --output=total_target_cleaned_scaled.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

python inference.py

## The following is for UVA Rivanna servers
## Note that there are some differences from the CS servers
## It doesn't have same gpu models as Rivanna.
## cuda-toolkit and anaconda3 instead of cuda and anaconda
## doesn't require library path exporting

# #!/usr/bin/env bash
# #SBATCH --job-name="total_early_stopped_target_cleaned_scaled"
# #SBATCH --output=total_early_stopped_target_cleaned_scaled.out
# #SBATCH --partition=gpu
# #SBATCH --time=1:00:00
# #SBATCH --gres=gpu:v100:1
# #SBATCH --account=ds6011-sp22-002
# #SBATCH --mem=48GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# module load cuda cudnn anaconda

# conda deactivate
# conda activate ml

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/mi3se/.conda/envs/ml/lib
# python inference.py