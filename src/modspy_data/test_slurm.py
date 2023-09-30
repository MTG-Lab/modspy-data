from dask_jobqueue import SLURMCluster
from dask.distributed import Client
import dask.array as da


cluster = SLURMCluster(
    project='rahit-dask',
    cores=8,
    processes=1,
    memory="2GB",
    account="def-mtarailo_cpu",
    walltime="00:10:00",
    log_directory="../logs",
)


# Create a client
client = Client(cluster)

# Scale the cluster (adjust the number of jobs as needed)
cluster.scale(jobs=2)
print(cluster.job_script())

# Example computation
x = da.random.random((1000, 1000), chunks=(100, 100))
y = x + x.T
z = y[::2, 500:].mean(axis=1)
result = z.compute()

# Print the result
print(result)

# Shut down the cluster and client
client.close()
cluster.close()
