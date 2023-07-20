#!/usr/bin/env bash
#SBATCH --job-name="lstm"
#SBATCH --output=lstm.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --account=bii_dsc_community
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

# 1. when you are using singularity
module load cuda cudnn singularity
singularity run --nv ../tft_pytorch.sif python train.py

# 2. when you have a working virtual env
# module load cuda cudnn anaconda
# conda deactivate
# conda activate ml
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/ml/lib
# python train.py

## The previous one is for UVA rivanna server
## The following is for UVA CS server. 

# #!/usr/bin/env bash
# #SBATCH --job-name="lstm"
# #SBATCH --output=outputs/lstm.out
# #SBATCH --partition=gpu
# #---SBATCH --time=1:00:00
# #SBATCH --gres=gpu:1
# #---SBATCH --nodelist=lynx01
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# # remove cuda and cudnn if you want to load the cuda installed in your virtual env
# module load cuda-toolkit cudnn anaconda3

# conda deactivate
# # make sure to create this virtual environment 
# # and install required libraries there
# conda activate ml

# # following line is needed if you are using anaconda venv
# # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#environment-setup
# export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# python train.py