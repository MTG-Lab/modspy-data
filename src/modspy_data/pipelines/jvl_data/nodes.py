"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

import ast
import pandas as pd
from modspy_data.helpers import KnowledgeGraphScores
from kedro.io import *
from kedro.pipeline import node, pipeline
import dask.dataframe as dd
import dask.bag as db
import logging
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
from kedro_datasets.dask import ParquetDataSet


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



def compute_similarity(df, kg, kg_name='GO', col_names=('gene_a', 'gene_b', 'target_GOs', 'modifier_GOs'), dask_params=None):
    # logger = logging.getLogger(__name__)
    
    # Deriving the name programattically. 
    # # Not reliable. Change to parameterized if time allows
    # Get the third element of the tuple, split it on '_', and get the second part
    # kg_name = col_names[2].split('_')[1][:2]
    kg_scores = KnowledgeGraphScores({kg_name: (kg, [])}, 
                                     col_names=col_names)
    cols = kg_scores.score_names
    cols_dtype = {
        col_names[0]: 'object',
        col_names[1]: 'object'
    }
    for c in cols:
        cols_dtype[c] = 'float32'
    dask_npartition = 1
    
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
    unique_ddf = ddf.drop_duplicates(subset=[col_names[0], col_names[1]])
    __df = unique_ddf.apply(kg_scores.get_scores, axis=1, meta=cols_dtype)
    return __df.compute()

# Define a function to apply to each row of the `edges` DataFrame
def process_row(row, id_to_category, node_mapping):
    subject_type = id_to_category[row['subject']]
    object_type = id_to_category[row['object']]
    edge_key = (subject_type, row['predicate'], object_type)
    return edge_key, (node_mapping[row['subject']], node_mapping[row['object']])


# Function to convert DataFrame rows to tuples for the Bag
def create_edge_tuple(row):
    return row['subject_category'], row['predicate'], row['object_category'], row['subject_id'], row['object_id']

# # Function to convert a tuple to a string
# def tuple_to_string(t):
#     # Convert the tuple to a string format of your choice
#     # This example uses a simple comma-separated format
#     return ','.join(map(str, t))

def binop(accumulator, edge):
    if accumulator is None:
        accumulator = []  # Initialize accumulator if it's None
    accumulator.append((edge[1], edge[2]))
    return accumulator

# def combine(accumulator1, accumulator2):
#     return accumulator1.extend(accumulator2)
def combine(accumulator1, accumulator2):
    if accumulator1 is None:
        accumulator1 = []
    if accumulator2 is None:
        accumulator2 = []
    return accumulator1 + accumulator2  # Use + for list concatenation



# Define a function to convert a line (string) back into a tuple
def parse_line(line):
    # Strip leading/trailing whitespace and newline characters
    line = line.strip()
    # Use `ast.literal_eval` to safely evaluate the string as a Python literal
    # This converts the string representation of a tuple back into an actual tuple
    return ast.literal_eval(line)



def make_edge_bag(nodes_df: ParquetDataSet, edges_df: ParquetDataSet) -> db.Bag:
    # Repartitioning for better memory management on Compute Canada
    edges_df = edges_df.repartition(npartitions=16)
    edges_df = edges_df.persist()  # if on a distributed system
    
    ############# CHANGE ME #######################
    # Should be removed. It is added because the edge dataframe has erroneous column names.
    edges_df = edges_df.rename(columns={'subject_category': 'e_category', 'edge_category': 'subject_category'}).rename(columns={'e_category': 'edge_category'})
    ###############################################
    # display(edges_df.head())
    _edf = edges_df.merge(nodes_df, left_on='subject', right_on='id', suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.rename(columns={'type_index': 'subject_id'})
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.merge(nodes_df, left_on='object', right_on='id', suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on object category: {_edf.columns}")
    _edf = _edf.rename(columns={'type_index': 'object_id'})
    # print(f"Columns after merging on object category: {_edf.columns}")
    # display(_edf.head())

    # Keep only the columns we need
    edges = _edf[['id','subject', 'subject_id', 'subject_category', 'predicate', 'edge_category', 'object_category', 'object_id', 'object']].copy()
    # print(f"Columns after renaming and trimming: {edges.columns}")
        
    # logger.info(f"➡️ Creating edges")

    ################## INITIALIZE EDGES ##################
    # Convert the edge categories to categoricals for efficiency
    edges['subject_category'] = edges['subject_category'].astype('category')
    edges['predicate'] = edges['predicate'].astype('category')
    edges['object_category'] = edges['object_category'].astype('category')
    # edges['edge_category'] = edges['edge_category'].astype('category')

    # From unknown type categorical to known type categorical
    edges = edges.categorize(columns=['subject_category','predicate','object_category'])

    # # Prepare edge types and mappings
    # edge_types = edges['edge_category'].unique().compute()  # Compute edge types on scheduler

    # Create a Bag from the DataFrame, and map the conversion function to each row
    edges_bag = edges.map_partitions(lambda df: df.apply(create_edge_tuple, axis=1)).to_bag()

    # # Map the conversion function over the bag to convert tuples to strings
    edges_bag = edges_bag.map(lambda x: str(x))
    
    return edges_bag



def bag_to_idx(edges_bag: db.Bag) -> db.Bag:
    edges_bag = edges_bag.map(parse_line).persist()
    
    # Use foldby with the process_tuples function
    edges_bag = edges_bag.foldby(key=lambda x: (x[0],x[1],x[2]), binop=binop, combine=combine, initial=[], split_every=8)
    # edge_type_mappings = edge_type_mappings.compute()  # Trigger the computation

    # etm = edges_bag.compute()  # Trigger the computation
    return edges_bag


def kgx_to_pyg(nodes_df: ParquetDataSet, edge_type_mappings: ParquetDataSet) -> HeteroData:
    """Generate PyTorch Geometric HeteroData type graph from nodes and edges.

    Args:
        nodes (dd.DataFrame): _description_
        edges (dd.DataFrame): _description_

    Returns:
        HeteroData: _description_
    """
    logger = logging.getLogger(__name__)
    
    # Initialize HeteroData for the heterogeneous graph
    data = HeteroData()

    # logger.info(nodes_df['category'].unique())
    # Prepare node mapping and node types
    # node_mapping = {node_id: i for i, node_id in enumerate(nodes_df['id'].unique())}
    node_types = nodes_df['category'].unique().compute()
    
    logger.info(f"🔵⚫🔘 Processing Nodes")
    ################## INITIALIZE NODES ##################
    # Prepare node mapping and node types
    # node_mapping = {node_id: i for i, node_id in enumerate(nodes_df.index.unique())}
    # node_types = nodes_df['category'].unique()

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
    
    logger.info(f"🕸️ 🕸️ 🕸️ MONARCH 🕸️ 🕸️ 🕸️ ")
    logger.info(nodes_df.dtypes)
    logger.info(edges_df.dtypes)


    ############# CHANGE ME #######################
    # Should be removed. It is added because the edge dataframe has erroneous column names.
    edges_df = edges_df.rename(columns={'subject_category': 'e_category', 'edge_category': 'subject_category'}).rename(columns={'e_category': 'edge_category'})
    ###############################################
    # display(edges_df.head())
    _edf = edges_df.merge(nodes_df, left_on='subject', right_on='id', suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.rename(columns={'type_index': 'subject_id'})
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.merge(nodes_df, left_on='object', right_on='id', suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on object category: {_edf.columns}")
    _edf = _edf.rename(columns={'type_index': 'object_id'})
    # print(f"Columns after merging on object category: {_edf.columns}")
    # display(_edf.head())

    # Keep only the columns we need
    edges = _edf[['id','subject', 'subject_id', 'subject_category', 'predicate', 'edge_category', 'object_category', 'object_id', 'object']]
    # print(f"Columns after renaming and trimming: {edges.columns}")
        
    logger.info(f"➡️ Creating edges")

    ################## INITIALIZE EDGES ##################
    # Convert the edge categories to categoricals for efficiency
    edges['subject_category'] = edges['subject_category'].astype('category')
    edges['predicate'] = edges['predicate'].astype('category')
    edges['object_category'] = edges['object_category'].astype('category')
    edges['edge_category'] = edges['edge_category'].astype('category')

    # From unknown type categorical to known type categorical
    edges = edges.categorize(columns=['subject_category','predicate','object_category','edge_category'])

    # # Prepare edge types and mappings
    # edge_types = edges['edge_category'].unique().compute()  # Compute edge types on scheduler
    
    # Create a Bag from the DataFrame, and map the conversion function to each row
    edges_bag = edges.map_partitions(lambda df: df.apply(create_edge_tuple, axis=1)).to_bag()
    # display(edges_bag.compute())

    # Use foldby with the process_tuples function
    edge_type_mappings = edges_bag.foldby(key=lambda x: x[0], binop=binop, combine=combine, initial=[])
    # edge_type_mappings = edge_type_mappings.compute()  # Trigger the computation
    
    
    # Add edges to HeteroData
    for edge_type in edges['edge_category'].unique():
        subset = edges_df[edges_df['edge_type'] == edge_type]
        # Map source and target IDs to the corresponding node indices in HeteroData
        source_indices = subset['source_id'].apply(lambda x: nodes_df.index[nodes_df['node_id'] == x].tolist()[0])
        target_indices = subset['target_id'].apply(lambda x: nodes_df.index[nodes_df['node_id'] == x].tolist()[0])
        # Assuming no additional edge features, otherwise add them here
        data['source_node_type', edge_type, 'target_node_type'].edge_index = torch.tensor([source_indices, target_indices], dtype=torch.long)

    
    logger.info(f"➡️ Edge done")
    
    logger.info(f"➡️ Result done, will now do mapping")

    # Add edges to the graph
    for edge_key, edge_indices in edge_type_mappings:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        data[edge_key].edge_index = edge_index

    return data
    

def kgx_to_edge_idx(nodes_df: ParquetDataSet, edges_df: ParquetDataSet) -> db.Bag:


    logger = logging.getLogger(__name__)

    logger.info(f"🕸️ 🕸️ 🕸️ MONARCH 🕸️ 🕸️ 🕸️ ")
    logger.info(nodes_df.dtypes)
    logger.info(edges_df.dtypes)


    ############# CHANGE ME #######################
    # Should be removed. It is added because the edge dataframe has erroneous column names.
    edges_df = edges_df.rename(columns={'subject_category': 'e_category', 'edge_category': 'subject_category'}).rename(columns={'e_category': 'edge_category'})
    ###############################################
    # display(edges_df.head())
    _edf = edges_df.merge(nodes_df, left_on='subject', right_on='id', suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.rename(columns={'type_index': 'subject_id'})
    # print(f"Columns after merging on subject category: {_edf.columns}")
    # display(_edf.head())
    _edf = _edf.merge(nodes_df, left_on='object', right_on='id', suffixes=('_ndf', '_edf'))
    # print(f"Columns after merging on object category: {_edf.columns}")
    _edf = _edf.rename(columns={'type_index': 'object_id'})
    # print(f"Columns after merging on object category: {_edf.columns}")
    # display(_edf.head())

    # Keep only the columns we need
    edges = _edf[['id','subject', 'subject_id', 'subject_category', 'predicate', 'edge_category', 'object_category', 'object_id', 'object']]
    # print(f"Columns after renaming and trimming: {edges.columns}")
        

    logger.info(f"➡️ Creating edges")

    ################## INITIALIZE EDGES ##################
    # Convert the edge categories to categoricals for efficiency
    edges['subject_category'] = edges['subject_category'].astype('category')
    edges['predicate'] = edges['predicate'].astype('category')
    edges['object_category'] = edges['object_category'].astype('category')
    edges['edge_category'] = edges['edge_category'].astype('category')

    # From unknown type categorical to known type categorical
    edges = edges.categorize(columns=['subject_category','predicate','object_category','edge_category'])

    # # Prepare edge types and mappings
    # edge_types = edges['edge_category'].unique().compute()  # Compute edge types on scheduler
    
    # Create a Bag from the DataFrame, and map the conversion function to each row
    edges_bag = edges.map_partitions(lambda df: df.apply(create_edge_tuple, axis=1)).to_bag()
    # display(edges_bag.compute())

    # Use foldby with the process_tuples function
    edge_type_mappings = edges_bag.foldby(key=lambda x: x[0], binop=binop, combine=combine, initial=[])
    # edge_type_mappings = edge_type_mappings.compute()  # Trigger the computation

    logger.info("======= Saving Edge Index =========")

    return edge_type_mappings


def mean(xs, n):
    return sum(xs) / n


def mean_sos(xs, n):
    return sum(x**2 for x in xs) / n


def variance(m, m2):
    return m2 - m * m

