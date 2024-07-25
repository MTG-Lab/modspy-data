#!/bin/bash
#SBATCH --account=def-mtarailo
#SBATCH --job-name=modspy-lightning
#SBATCH -e ./logs/%J-modspy-lightning.err
#SBATCH -o ./logs/%J-modspy-lightning.out

#SBATCH --time=00-23:59:00           # time (DD-HH:MM:SS)
#SBATCH --mem=20GB                   # memory; default unit is megabytes

#SBATCH --nodes=1
###SBATCH --exclusive
#SBATCH --tasks-per-node=1
####SBATCH --gres=gpu:v100:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-user=kmtahsinhassan.rahit@ucalgary.ca
#SBATCH --mail-type=ALL


# Load modules or your own conda environment here
# module load pytorch/v1.4.0-gpu
# conda activate ${CONDA_ENV}
module --force purge
module load StdEnv/2020
module load gcc/9.3.0 python/3.11.5 cuda/11.8.0 arrow/12.0.1 scipy-stack/2023b
source ./py311

export CUDA_LAUNCH_BLOCKING=1
export MAIN_NODE=$(hostname)
export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.

# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!
# srun python ./src/modspy_data/modspy-monarch.py
srun python ./src/modspy_data/models/modspy.py --n-samples=0
# srun python ./src/modspy_data/test_training.py