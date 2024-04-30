#!/bin/bash

module load StdEnv/2020
module load gcc/9.3 python/3.11.5 cuda/11.8.0 arrow/12.0.1 nodejs rust/1.70.0 scipy-stack/2023b

# virtualenv --no-download $SLURM_TMPDIR/ENV
# source $SLURM_TMPDIR/ENV/bin/activate
source ./py311

# pip install --upgrade pip --no-index

# pip install ray --no-index