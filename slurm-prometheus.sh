#!/bin/bash
#SBATCH --account=def-mtarailo_cpu
#SBATCH --cpus-per-task=1        # number of cores per MPI process
#SBATCH --mem-per-cpu=500      # memory; default unit is megabytes
#SBATCH --time=00-03:30:00           # time (DD-HH:MM:SS)
#SBATCH -J prometheus
#SBATCH -e ./logs/prometheus-%J.err
#SBATCH -o ./logs/prometheus-%J.out


export RAY_GRAFANA_HOST="http://${hostname}:3000"
export RAY_PROMETHEUS_HOST="http://${hostname}:9090"
# export RAY_PROMETHEUS_NAME="Prometheus"
# export RAY_GRAFANA_IFRAME_HOST="http://${hostname}:3000"


/home/rahit/projects/def-mtarailo/rahit/prometheus-2.51.1.linux-amd64/prometheus --config.file=/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/conf/prometheus.yml &
/project/6013424/rahit/grafana-v10.4.1/bin/grafana-server --config /home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/conf/grafana.ini web &

