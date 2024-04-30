"""
THis script is to run the code directly on SLURM as notebook was failing because of cuda memory problem
"""

from comet_ml import Experiment


import os
import time

from typing import List, Optional, Tuple, Union, cast
from traitlets import default
from rich import print

import numpy as np
import typer
import torch
import torchmetrics
from torch_geometric.datasets import FB15k_237
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
from torch_geometric.transforms import AddMetaPaths

import optuna
from pytorch_lightning.loggers import CometLogger

# import wandb
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data.lightning.datamodule import LightningNodeData
from torch_geometric.nn import ComplEx, DistMult, RotatE, TransE

from torch.nn.functional import cosine_similarity
from sklearn.metrics import accuracy_score

from ray import tune
from ray.tune.schedulers import ASHAScheduler


import ray.train.lightning
from ray.air.integrations.comet import CometLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)

graphdata_path = "./data/05_model_input/2024-02-monarch_heterodata_v1.pt"
#graphdata_path = './data/05_model_input/2024-04-monarch_graphdata_v2.pt'
monarch = torch.load(graphdata_path).to('cuda')

print(f""" 
        Total nodes: {monarch.num_nodes}
        Total node types: {len(monarch.node_types)}

        Total edges: {monarch.num_edges}
        Total edge types: {len(monarch.edge_types)}      
"""
)

# # Adding self loops to avoid 1. nodes without any edge, 2. consider intragenic modifier
# transform = AddSelfLoops()
# monarch = transform(monarch)
# Flip/Reverse source <-> target
def reverse_edges(data: HeteroData, edge_type: EdgeType):
    rev_edge_index = data.edge_index_dict[edge_type].flip([0])
    data[(edge_type[2], edge_type[1], edge_type[0])].edge_index = rev_edge_index
    del data.edge_index_dict[edge_type]
    return data

edge_types_to_reverse = [
    ('biolink:Gene', 'biolink:acts_upstream_of_or_within', 'biolink:BiologicalProcessOrActivity'),
    ('biolink:Gene', 'biolink:actively_involved_in', 'biolink:BiologicalProcessOrActivity'),
    ('biolink:Gene', 'biolink:participates_in', 'biolink:Pathway'),
    ('biolink:Disease', 'biolink:has_phenotype', 'biolink:PhenotypicFeature'),
    ('biolink:Gene', 'biolink:gene_associated_with_condition', 'biolink:Disease'),
    ('biolink:Gene', 'biolink:expressed_in', 'biolink:CellularComponent'),    
]
for edge_type in edge_types_to_reverse:
    monarch = reverse_edges(monarch, edge_type)
# Calculate the number of edges for each type
edge_counts = {edge_type: edge_index.size(1) for edge_type, edge_index in monarch.edge_index_dict.items()}

# Sort the edge types by the number of edges
sorted_edge_counts = sorted(edge_counts.items(), key=lambda item: item[1], reverse=True)

for edge_type, count in sorted_edge_counts[:20]:
    print(f"{edge_type}: {count}")

print(f""" 
        Total nodes: {monarch.num_nodes}
        Total node types: {len(monarch.node_types)}

        Total edges: {monarch.num_edges}
        Total edge types: {len(monarch.edge_types)}      
"""
)
metapaths = [
    [
        ("biolink:Gene", "biolink:orthologous_to", "biolink:Gene"),
        ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
    ],
    
    [
        ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
        ("biolink:Gene", "biolink:orthologous_to", "biolink:Gene"),
        ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
    ],
    
    # [
    #     ('biolink:Gene', 'biolink:has_phenotype', 'biolink:PhenotypicFeature'),
    #     ('biolink:PhenotypicFeature', 'biolink:subclass_of', 'biolink:PhenotypicFeature'),
    #     ('biolink:PhenotypicFeature', 'biolink:has_phenotype', 'biolink:Disease'),
    #     ('biolink:Disease', 'biolink:has_phenotype', 'biolink:PhenotypicFeature'),
    #     ('biolink:PhenotypicFeature', 'biolink:has_phenotype', 'biolink:Gene'),
    # ],
    
    # [
    #     ('biolink:Gene', 'biolink:enables', 'biolink:BiologicalProcessOrActivity'),
    #     ('biolink:BiologicalProcessOrActivity', 'biolink:subclass_of', 'biolink:BiologicalProcessOrActivity'),
    #     ('biolink:BiologicalProcessOrActivity', 'biolink:actively_involved_in', 'biolink:Gene'),
    # ],
    
    # # [
    # #     ('biolink:Gene', 'biolink:participates_in', 'biolink:Pathway'),
    # #     ('biolink:BiologicalProcessOrActivity', 'biolink:related_to', 'biolink:Pathway'),
    # #     ('biolink:Pathway', 'biolink:participates_in', 'biolink:Gene'),
    # # ]
]

monarch = AddMetaPaths(metapaths, drop_orig_edge_types=True, max_sample=5)(monarch)

print(len(monarch.edge_types))
print(monarch.metapath_dict)