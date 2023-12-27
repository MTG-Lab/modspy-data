from loguru import logger
import string

# Data manipulation
import pandas as pd
import numpy as np
from modspy_data.helpers import KnowledgeGraphScores

# Distributed
import dask.dataframe as dd
from dask.distributed import Client, progress, performance_report
from dask_jobqueue import SLURMCluster
from dask import delayed
import dask

# Loading Interaction
ppi = catalog.load("string_interactions")
# Loading gene-protein map/alias file. Because ppi is indicated using stringdb ID.
ppi_alias = catalog.load("string_alias")
jvl = catalog.load("jvl_scored")
# Use all 8 cores
cluster = SLURMCluster(
    cores=8,
    processes=1,
    memory="4GB",
    account="def-mtarailo_cpu",
    walltime="00:10:00",
    log_directory="../logs",
)
cluster
print(cluster.job_script())
client = Client(cluster)
client
cluster.scale(2)
# apply lowercasing and removing punctuation to the textual column
jvl["QueryGene_norm"] = (
    jvl["QueryGene"].str.lower().str.replace("[{}]-".format(string.punctuation), "")
)
jvl["SuppressorGene_norm"] = (
    jvl["SuppressorGene"]
    .str.lower()
    .str.replace("[{}]-".format(string.punctuation), "")
)
ppi_alias["alias_norm"] = (
    ppi_alias["alias"].str.lower().str.replace("[{}]-".format(string.punctuation), "")
)
# Removing duplicates
jvl = jvl.drop_duplicates(subset=("QueryGene", "SuppressorGene"), keep="first")
ppi_alias = ppi_alias.drop_duplicates(subset="alias_norm", keep="first")
# Adding alias for query gene
_jvl = jvl.merge(ppi_alias, how="left", left_on="QueryGene_norm", right_on="alias_norm")
# Adding alias for suppressor gene
_jvl = _jvl.merge(
    ppi_alias, how="left", left_on="SuppressorGene_norm", right_on="alias_norm"
)
# AAdding StringDB interaction score
_jvl = _jvl.merge(
    ppi,
    how="left",
    left_on=["protein_x", "protein_y"],
    right_on=["protein1", "protein2"],
)
jvl_df = _jvl.compute()
catalog.save("jvl_features", jvl_df)
