# Slurm

Remote servers use a `job scheduler` called [SLURM](https://en.wikipedia.org/wiki/Slurm_Workload_Manager). o allocate computational resources (servers) to users who submit “jobs” to a queue. The job scheduler looks at the requirements stated in the job's script and allocates to the job a server (or servers) which matches the requirements specified in the job script. 

Read more about how to use slurm for UVA servers from [Rivanna](https://www.rc.virginia.edu/userinfo/rivanna/slurm) and [CS site](https://www.cs.virginia.edu/wiki/doku.php?id=compute_slurm) sites.

Following shows two template slurm scripts that can be used to train TFT model in both servers. Note that I am running them in my python virtual environment (name `ml`) created using anaconda. If you have a different env, replace `conda activate ml` with `conda activate your_python_env`. Make sure you have the libraries installed in that env. A list of libraries needed for this pytorch version is given in the [requirements.txt](/requirements.txt) file. You can use python pip or anaconda to create the env. Details are in the project [README](/README.md).

## Rivanna

Rivanna [dashboard](https://rivanna-portal.hpc.virginia.edu/pun/sys/dashboard) is the UI that lets you easily manage and submit jobs. The account name is the group you are part of and will be charged for your resource usage. Your groups are managed [here](https://mygroups.virginia.edu/) and details about the allocations can be found [here](https://www.rc.virginia.edu/userinfo/rivanna/allocations).

```bash
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
python train.py
```

## [CS Server](https://www.cs.virginia.edu/wiki/doku.php)

The following is for UVA CS servers. You have to ssh into `portal.cs.virginia.edu`. Note that there are some differences from the Rivanna one. It doesn't have same gpu models as Rivanna. The list is available [here](https://www.cs.virginia.edu/wiki/doku.php?id=compute_resources). Also it has modules cuda-toolkit and anaconda3 instead of cuda and anaconda. Unlike Rivanna it didn't require library path exporting.

```bash
#!/usr/bin/env bash
#SBATCH --job-name="train"
#SBATCH --output=train.out
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

source /etc/profile.d/modules.sh
source ~/.bashrc

module load cuda-toolkit cudnn anaconda3

conda deactivate
conda activate ml

python train.py
```