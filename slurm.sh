#!/bin/bash
#SBATCH --account=def-mtarailo_cpu
#SBATCH --ntasks=1               # number of MPI processes
#SBATCH --mem-per-cpu=2G      # memory; default unit is megabytes
#SBATCH --time=00-00:30           # time (DD-HH:MM)
#SBATCH --mail-user=kmtahsinhassan.rahit@ucalgary.ca
#SBATCH --mail-type=ALL
##### SBATCH -J DASK_jobs_workers

module load gcc python/3.8.10 arrow/11 spark/3.3.0
# module load gcc/9.3.0 python nodejs perl mpi4py
source $HOME/kedro-environment/modspy_py38/bin/activate
# source $HOME/jupyter_py3/bin/activate

# python ./src/gemo/features/build_features.py data/raw/filenames.txt
# python ./src/gemo/data/gene_network.py
# python ./src/gemo/run.py hello
# mpirun -np 4 python ./src/gemo/test_mpi.py

time srun pip install -r src/requirements.txt