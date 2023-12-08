"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

import pandas as pd
from modspy_data.helpers import KnowledgeGraphScores
from kedro.io import *
from kedro.pipeline import node, pipeline
import dask.dataframe as dd

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



def add_annotations(df, go):
    # io = DataCatalog()
    # go = io.load('go')
    # jvl = io.load('jvl_annotated')
    kg_scores = KnowledgeGraphScores({'go': (go, df)}, col_names=('QueryGene', 'SuppressorGene', 'QueryGene_GO', 'SuppressorGene_GO'))
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


def mean(xs, n):
    return sum(xs) / n


def mean_sos(xs, n):
    return sum(x**2 for x in xs) / n


def variance(m, m2):
    return m2 - m * m

