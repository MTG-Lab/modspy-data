#!/bin/bash

# Reload environment modules and venv for The Alliance, Canada cluster

deactivate
module --force purge
module load StdEnv/2020
module load gcc/9.3.0 python/3.8.10 cuda/11.7 arrow/9.0.0
source /home/rahit/jupyter_py3/bin/activate
