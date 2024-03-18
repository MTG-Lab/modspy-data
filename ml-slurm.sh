#!/bin/bash
#SBATCH --account=def-mtarailo
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1
#SBATCH --tasks-per-node=1   # This needs to match Trainer(devices=...)
### --ntasks-per-node=1   # This needs to match Trainer(devices=...)
#SBATCH --cpus-per-task=1
#SBATCH --time=00-00:59:00           # time (DD-HH:MM:SS)
#SBATCH --mem=16GB      # memory; default unit is megabytes
#SBATCH -J main-process
#SBATCH -e ./logs/%J-main-process.err
#SBATCH -o ./logs/%J-main-process.out
#SBATCH --mail-user=kmtahsinhassan.rahit@ucalgary.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2020
module load gcc/9.3 python/3.11.5 cuda/11.8.0 arrow/12.0.1 scipy-stack/2023b
# module load gcc/9.3.0 python/3.8.10 cuda/11.8.0 arrow/9.0.0 nodejs rust/1.70.0

# conda activate modspy
# source ~/jupyter_py3/bin/activate
source /home/rahit/projects/def-mtarailo/rahit/python_environments/py311/bin/activate

### export NCCL_BLOCKING_WAIT=1 #Pytorch Lightning uses the NCCL backend for inter-GPU communication by default. Set this variable to avoid timeout errors.


# PyTorch Lightning will query the environment to figure out if it is running inside a SLURM batch job
# If it is, it expects the user to have requested one task per GPU.
# If you do not ask for 1 task per GPU, and you do not run your script with "srun", your job will fail!
srun python ./src/modspy_data/modspy-monarch.py
# srun python ./src/modspy_data/test_training.py