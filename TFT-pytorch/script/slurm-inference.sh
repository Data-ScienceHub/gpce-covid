#!/usr/bin/env bash
#SBATCH --job-name="inference"
#SBATCH --output=outputs/inference.out
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#---SBATCH --nodelist=lynx01
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

python inference.py

## The above script is for UVA CS server
## The following is for UVA Rivanna servers

# #!/usr/bin/env bash
# #SBATCH --job-name="inference"
# #SBATCH --output=outputs/inference.out
# #SBATCH --partition=gpu
# #SBATCH --time=2:00:00
# #SBATCH --gres=gpu:v100:1
# #SBATCH --account=ds--6013
# #SBATCH --mem=32GB

# source /etc/profile.d/modules.sh
# source ~/.bashrc

# # 1. when you are using singularity
# module load cuda cudnn singularity
# singularity run --nv ../tft_pytorch.sif python inference.py

# # 2. when you have a working virtual env
# module load cuda cudnn anaconda
# conda deactivate
# conda activate ml
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.conda/envs/ml/lib
# python inference.py