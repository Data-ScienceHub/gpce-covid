#!/usr/bin/env bash
#SBATCH --job-name="sensitivity"
#SBATCH --output=outputs/sensitivity.out
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

## The above script is for UVA CS server
## Following script is for UVA Rivanna

# #!/usr/bin/env bash
# #SBATCH --job-name="sensitivity"
# #SBATCH --output=outputs/sensitivity.out
# #SBATCH --partition=gpu
# #SBATCH --time=1:00:00
# #SBATCH --gres=gpu:v100:1
# #SBATCH --account=ds--6013 # bii_dsc_community
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# # 1. when you are using singularity
# module load cuda cudnn singularity
# singularity run --nv ../tft_pytorch.sif python sensitivity_analysis.py

# # 2. when you have a working virtual env
# module load cuda cudnn anaconda
# conda deactivate
# conda activate ml
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/ml/lib
# python inference.py