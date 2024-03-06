#!/bin/bash
#SBATCH --account=def-mtarailo
#SBATCH --nodes=1             # This needs to match Trainer(num_nodes=...)
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --time=00-02:30:00           # time (DD-HH:MM:SS)
#SBATCH --mem=16GB      # memory; default unit is megabytes
#SBATCH -J main-process
#SBATCH -e ./logs/%J-main-process.err
#SBATCH -o ./logs/%J-main-process.out
#SBATCH --mail-user=kmtahsinhassan.rahit@ucalgary.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2020
module load gcc/11.3 python/3.11 cuda/11.8.0 scipy-stack/2023b
# module load gcc/9.3.0 python/3.8.10 cuda/11.8.0 arrow/9.0.0 nodejs rust/1.70.0

# conda activate modspy
# source ~/jupyter_py3/bin/activate
source /home/rahit/projects/def-mtarailo/rahit/python_environments/py311/bin/activate

srun python ./src/modspy_data/modspy-monarch.py