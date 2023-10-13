"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

import pandas as pd
from modspy_data.helpers import KnowledgeGraphScores
from kedro.io import *
from kedro.pipeline import node, pipeline

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


def clean_jvl(df):
    io = DataCatalog()
    goa = io.load("goa")
    jvl = io.load('jvl')
    jvl_pairs = jvl[['QueryGene', 'SuppressorGene']]
    _goa = goa.dropna(subset=['DB_Object_Synonym'])
    __jvl = jvl.apply(match_on_annotation, axis=1, goa=_goa, col_name=['QueryGene', 'SuppressorGene'])
    __jvl = __jvl[['QueryGene', 'SuppressorGene', 'QueryGene_GO', 'SuppressorGene_GO']]
    io.save('jvl_annotated', __jvl)
    # __jvl.to_csv('jvl_go.csv', index=False)


def add_annotations(df):
    io = DataCatalog()
    go = io.load('go')
    jvl = io.load('jvl_annotated')
    kg_scores = KnowledgeGraphScores({'go': (go, jvl)}, col_names=('QueryGene', 'SuppressorGene', 'QueryGene_GO', 'SuppressorGene_GO'))
    cols = kg_scores.score_names
    cols_dtype = {
        'QueryGene': 'object',
        'SuppressorGene': 'object'
    }
    for c in cols:
        cols_dtype[c] = 'float32'
    jvl_s = jvl.apply(kg_scores.get_scores, axis=1) 
    io.save('jvl_scored', jvl_s)


def mean(xs, n):
    return sum(xs) / n


def mean_sos(xs, n):
    return sum(x**2 for x in xs) / n


def variance(m, m2):
    return m2 - m * m

