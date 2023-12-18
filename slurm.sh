#!/bin/bash
#SBATCH --account=def-mtarailo_cpu
#SBATCH --cpus-per-task=4        # number of cores per MPI process
#SBATCH --mem-per-cpu=32GB      # memory; default unit is megabytes
#SBATCH --time=00-08:50           # time (DD-HH:MM)
#SBATCH -J dask-king
#SBATCH -e ./logs/dask-king-%J.err
#SBATCH -o ./logs/dask-king-%J.out
###SBATCH## --mail-user=kmtahsinhassan.rahit@ucalgary.ca
###SBATCH## --mail-type=ALL

module --force purge
module load StdEnv/2020
module load gcc/9.3.0 python/3.8.10 cuda/11.7 arrow/9.0.0 nodejs
source /home/rahit/jupyter_py3/bin/activate
# time srun python ./src/modspy_data/test_slurm.py
time srun kedro run --runner="modspy_data.runner.SLURMRunner"