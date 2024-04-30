#!/bin/bash

# Reload environment modules and venv for The Alliance, Canada cluster

deactivate
module --force purge
module load StdEnv/2020
module load gcc/9.3.0 python/3.11.5 cuda/11.8.0 arrow/12.0.1 scipy-stack/2023b
source py311
