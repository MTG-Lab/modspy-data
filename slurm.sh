#!/bin/bash
#SBATCH --account=def-mtarailo
#SBATCH --cpus-per-task=1        # number of cores per MPI process
#####SBATCH --gres=gpu:a100:1
#SBATCH --mem-per-cpu=32GB      # memory; default unit is megabytes
#SBATCH --time=00-05:58:00           # time (DD-HH:MM:SS)
#SBATCH -J modspy_data_slurm_entrypoint
#SBATCH -e ./logs/modspy_data_slurm_entrypoint-%J.err
#SBATCH -o ./logs/modspy_data_slurm_entrypoint-%J.out
#SBATCH --mail-user=kmtahsinhassan.rahit@ucalgary.ca
#SBATCH --mail-type=ALL

module --force purge
module load StdEnv/2020
# module load gcc/9.3.0 python/3.11.5 cuda/11.8.0 arrow/12.0.1 scipy-stack/2023b
# source py311
module load gcc/9.3.0 python/3.8.10 cuda/11.7 arrow/9.0.0 nodejs
source /home/rahit/jupyter_py3/bin/activate


# time srun python src/modspy_data/test_metapath.py
time srun kedro run --runner="modspy_data.runner.SLURMRunner"
