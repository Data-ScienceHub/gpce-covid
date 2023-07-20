#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=outputs/train.out
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
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


## The above script is for UVA Rivanna servers
## The following is for UVA CS server
## Note that there are some differences from the script above
## It doesn't have same gpu models as Rivanna.
## cuda-toolkit and anaconda3 instead of cuda and anaconda
## doesn't require library path exporting

# #!/usr/bin/env bash
# #SBATCH --job-name="train"
# #SBATCH --output=outputs/train.out
# #SBATCH --partition=gpu
# #SBATCH --time=24:00:00
# #SBATCH --gres=gpu:1
# #---SBATCH --nodelist=lynx01
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# module load cuda-toolkit cudnn anaconda3

# conda deactivate
# conda activate ml

# python train.py