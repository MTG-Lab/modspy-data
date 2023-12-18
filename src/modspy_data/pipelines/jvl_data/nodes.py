"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

import pandas as pd
from modspy_data.helpers import KnowledgeGraphScores
from kedro.io import *
from kedro.pipeline import node, pipeline
import dask.dataframe as dd
import logging
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict

# Remove wraping quotation sign ("") from dataframe
def remove_quotes(df):
    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
    cols = [c.strip('"') for c in df.columns]
    df.rename(columns=dict(zip(df.columns, cols)), inplace=True)
    return df


# Add info from GO annotations to the JVL data
def match_on_annotation(row, goa, col_name=['QueryGene']):
    for c in col_name:
        __goa_filt = goa[goa['DB_Object_Synonym'].str.contains(row[c])]
        __annos = __goa_filt['GO_ID'].unique()
        row[f"{c}_GO"] = __annos
    return row


def clean_jvl(df, goa):
    # io = DataCatalog()
    # goa = io.load("goa")
    # jvl = io.load('jvl')
    # jvl_pairs = df[['QueryGene', 'SuppressorGene']]
    ddf = dd.from_pandas(df[['QueryGene','SuppressorGene']], npartitions=9)
    ddf.drop_duplicates(subset=['QueryGene','SuppressorGene'])
    _goa = goa.dropna(subset=['DB_Object_Synonym'])
    __jvl = ddf.apply(match_on_annotation, axis=1, goa=_goa, col_name=['QueryGene', 'SuppressorGene'], meta={    
        'QueryGene': 'object', 'SuppressorGene': 'object', 'QueryGene_GO': 'object', 'SuppressorGene_GO': 'object'})
    __jvl = __jvl.compute()
    __jvl = __jvl[['QueryGene', 'SuppressorGene', 'QueryGene_GO', 'SuppressorGene_GO']]
    # io.save('jvl_annotated', __jvl)
    # __jvl.to_csv('jvl_go.csv', index=False)
    return __jvl


def annotate_olida(df, goa, use_dask=False):
    # keeping `Oligogenic Effect` so that we can identify type of relationship
    if use_dask:
        _df = dd.from_pandas(df[['gene_a','gene_b','Oligogenic Effect']], npartitions=1)
    _df = df.drop_duplicates(subset=['gene_a','gene_b','Oligogenic Effect'])
    
    _goa = goa.dropna(subset=['DB_Object_Synonym'])
    _goa['gene_symbol'] = _goa['DB_Object_Synonym'].str.split('|') # Alternative gene symbols are | seperated
    goa_genes = _goa.explode('gene_symbol')

    _df = _df.merge(goa_genes, how='left', left_on='gene_a', right_on='gene_symbol')
    _df = _df.groupby(['gene_a','gene_b', 'Oligogenic Effect']).agg({'GO_ID': list})
    _df.rename(columns={'GO_ID':'gene_a_GO'})

    _df = _df.merge(goa_genes, how='left', left_on='gene_b', right_on='gene_symbol')
    _df = _df.groupby(['gene_a','gene_b', 'Oligogenic Effect']).agg({'GO_ID': list})
    _df.rename(columns={'GO_ID':'gene_b_GO'})

    return _df.compute()



def add_annotations(df, kg):
    # io = DataCatalog()
    # go = io.load('go')
    # jvl = io.load('jvl_annotated')
    kg_scores = KnowledgeGraphScores({'go': (kg, df)}, 
                                     col_names=('QueryGene', 'SuppressorGene', 'QueryGene_GO', 'SuppressorGene_GO'))
    cols = kg_scores.score_names
    cols_dtype = {
        'QueryGene': 'object',
        'SuppressorGene': 'object'
    }
    for c in cols:
        cols_dtype[c] = 'float32'
    df_scored = df.apply(kg_scores.get_scores, axis=1) 
    # io.save('jvl_scored', jvl_s)
    return df_scored



def compute_similarity(df, kg, col_names=('gene_a', 'gene_b', 'target_GOs', 'modifier_GOs'), dask_params=None):
    # logger = logging.getLogger(__name__)
    
    # Deriving the name programattically. 
    # # Not reliable. Change to parameterized if time allows
    # Get the third element of the tuple, split it on '_', and get the second part
    kg_name = col_names[2].split('_')[1][:2]

    kg_scores = KnowledgeGraphScores({kg_name: (kg, [])}, 
                                     col_names=col_names)
    cols = kg_scores.score_names
    cols_dtype = {
        col_names[0]: 'object',
        col_names[1]: 'object'
    }
    for c in cols:
        cols_dtype[c] = 'float32'    
    dask_npartition = 2
    
    # logger.info(dask_params)
    if dask_params != None and 'n_workers' in dask_params:
        dask_npartition = int(dask_params['n_workers'])
        if 'slurm' in dask_params and 'cores' in dask_params['slurm']:
            dask_npartition = dask_npartition * int(dask_params['slurm']['cores'])
    else:
        if df.shape[0] > 1000:
            dask_npartition = 8
        elif df.shape[0] > 100:
            dask_npartition = 4
    ddf = dd.from_pandas(df, npartitions=dask_npartition)
    __df = ddf.apply(kg_scores.get_scores, axis=1, meta=cols_dtype) 
    return __df.compute()

# Define a function to apply to each row of the `edges` DataFrame
def process_row(row, id_to_category, node_mapping):
    subject_type = id_to_category[row['subject']]
    object_type = id_to_category[row['object']]
    edge_key = (subject_type, row['predicate'], object_type)
    return edge_key, (node_mapping[row['subject']], node_mapping[row['object']])

def kgx_to_pyg(nodes, edges):
    logger = logging.getLogger(__name__)    
        
    nodes = nodes.reset_index(drop=True)
    # Prepare node mapping and node types
    node_mapping = {node_id: i for i, node_id in enumerate(nodes['id'].unique().compute())}
    node_types = nodes['category'].unique().compute()

    # Initialize HeteroData for the heterogeneous graph
    data = HeteroData()

    logger.info(f"🔵⚫🔘 Processing Nodes")
    # Add nodes to the graph
    for node_type in node_types:
        # mask = nodes['category'].compute() == node_type
        type_nodes = nodes[nodes['category'] == node_type].compute()
        data[node_type].x = torch.tensor(type_nodes.index.to_list(), dtype=torch.long)

    logger.info(f"Node processing done 🔵⚫🔘 ")

    # Assuming `nodes` is a Dask DataFrame with 'id' and 'category' columns
    # and `node_mapping` is a dictionary mapping node ids to some values

    # Create a dictionary mapping node ids to categories
    id_to_category = nodes.set_index('id')['category'].compute().to_dict()

    logger.info(f"➡️ Creating edges")
    # Apply the function to each row of the `edges` DataFrame
    edges['result'] = edges.apply(lambda row: process_row(row, id_to_category, node_mapping), axis=1, meta=('object'))

    # Compute the result
    result = edges['result'].compute()

    # Create the `edge_type_mappings` dictionary
    edge_type_mappings = {}
    for edge_key, value in result:
        if edge_key not in edge_type_mappings:
            edge_type_mappings[edge_key] = []
        edge_type_mappings[edge_key].append(value)

    logger.info(f"➡️ Result done, will now do mapping")

    # Add edges to the graph
    for edge_key, edge_indices in edge_type_mappings.items():
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        data[edge_key].edge_index = edge_index

    logger.info(f"➡️ Edge done")

    return data


def mean(xs, n):
    return sum(xs) / n


def mean_sos(xs, n):
    return sum(x**2 for x in xs) / n


def variance(m, m2):
    return m2 - m * m

