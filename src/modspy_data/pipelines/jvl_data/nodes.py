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


# Function to convert DataFrame rows to tuples for the Bag
def create_edge_tuple(row):
    return (row['subject_category'], row['predicate'], row['object_category']), row['subject_id'], row['object_id']

def binop(accumulator, edge):
    return accumulator.append((edge[1], edge[2]))

def combine(accumulator1, accumulator2):
    return accumulator1.extend(accumulator2)



def kgx_to_pyg(nodes: dd.DataFrame, edges: dd.DataFrame) -> HeteroData:
    """Generate PyTorch Geometric HeteroData type graph from nodes and edges.

    Args:
        nodes (dd.DataFrame): _description_
        edges (dd.DataFrame): _description_

    Returns:
        HeteroData: _description_
    """
    logger = logging.getLogger(__name__)    


    # Group by 'node_type' and assign type-wise indices
    nodes_df['type_index'] = nodes_df.groupby('category').cumcount()
    # reseting index based on id
    nodes_df = nodes_df.set_index('id')
    
    ############# CHANGE ME #######################
    # Should be removed. It is added because the edge dataframe has erroneous column names.
    edges_df = edges_df.rename(columns={'subject_category': 'e_category', 'edge_category': 'subject_category'}).rename(columns={'e_category': 'edge_category'})
    ###############################################
    _edf = edges_df.merge(nodes_df, left_on='subject', right_index=True, suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on subject category: {_edf.columns}")
    _edf = _edf.rename(columns={'type_index': 'subject_id'})
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.merge(nodes_df, left_on='object', right_index=True, suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on object category: {_edf.columns}")
    _edf = _edf.rename(columns={'type_index': 'object_id'})
    # print(f"Columns after merging on object category: {_edf.columns}")
    # display(_edf.head())
    
    # Keep only the columns we need
    edges = _edf[['id','subject', 'subject_id', 'subject_category', 'predicate', 'edge_category', 'object_category', 'object_id', 'object']]
    # print(f"Columns after renaming and trimming: {edges.columns}")
        
    
    logger.info(f"🔵⚫🔘 Processing Nodes")
    ################## INITIALIZE NODES ##################
    # Prepare node mapping and node types
    node_mapping = {node_id: i for i, node_id in enumerate(nodes_df.index.unique())}
    node_types = nodes_df['category'].unique()

    # Initialize HeteroData for the heterogeneous graph
    data = HeteroData()
        
    # Prepare node mapping and node types
    # node_mapping = {node_id: i for i, node_id in enumerate(nodes_df['id'].unique())}
    node_types = nodes_df['category'].unique()

    # Add nodes to the graph
    for node_type in node_types:
        # get nodes of type `node_type`
        mask = nodes_df['category'] == node_type
        type_nodes = nodes_df[mask].compute()
        
        # node_features = torch.ones((type_nodes.shape[0], 1))
        data[node_type].num_nodes = type_nodes.shape[0] # node_features # torch.tensor(type_nodes.index.to_list(), dtype=torch.long).view(len(type_nodes.index.to_list()),1)

        # Add dummy features (e.g., a simple constant feature)
        data[node_type].x = torch.ones((data[node_type].num_nodes, 1))  # Each user has a feature vector of [1]

    logger.info(f"Node processing done 🔵⚫🔘 ")

    logger.info(f"➡️ Creating edges")

    ################## INITIALIZE EDGES ##################
    # Convert the edge categories to categoricals for efficiency
    edges['subject_category'] = edges['subject_category'].astype('category')
    edges['predicate'] = edges['predicate'].astype('category')
    edges['object_category'] = edges['object_category'].astype('category')
    edges['edge_category'] = edges['edge_category'].astype('category')

    # From unknown type categorical to known type categorical
    edges = edges.categorize(columns=['subject_category','predicate','object_category','edge_category'])

    # Prepare edge types and mappings
    edge_types = edges['edge_category'].unique().compute()  # Compute edge types on scheduler
        

    # Create a Bag from the DataFrame, and map the conversion function to each row
    edges_bag = edges.map_partitions(lambda df: df.apply(create_edge_tuple, axis=1)).to_bag()
    # display(edges_bag.compute())

    # Use foldby with the process_tuples function
    edge_type_mappings = edges_bag.foldby(key=lambda x: x[0], binop=binop, combine=combine, initial=[])
    edge_type_mappings = edge_type_mappings.compute()  # Trigger the computation


    logger.info(f"➡️ Result done, will now do mapping")

    # Add edges to the graph
    for edge_key, edge_indices in edge_type_mappings:
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

