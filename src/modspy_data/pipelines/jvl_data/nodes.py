"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

import pandas as pd
from modspy_data.helpers import KnowledgeGraphScores
from kedro.io import *

# Remove wraping quotation sign ("") from dataframe
def remove_quotes(df):
    df = df.applymap(lambda x: x.strip('"') if isinstance(x, str) else x)
    cols = [c.strip('"') for c in df.columns]
    df.rename(columns=dict(zip(df.columns, cols)), inplace=True)
    return df


# Add info from GO annotations to the JVL data
def match_on_annotation(row, goa, col_name='QueryGene'):
    __goa_filt = goa[goa['DB_Object_Synonym'].str.contains(row[col_name])]
    __annos = __goa_filt['GO_ID'].unique()
    row[f"{col_name}_GO"] = __annos
    return row


def clean_jvl(df):
    io = DataCatalog()
    go = io.load('go')
    goa = io.load("goa")
    jvl = io.load('jvl')
    jvl_pairs = jvl[['QueryGene', 'SuppressorGene']]
    _goa = goa.dropna(subset=['DB_Object_Synonym'])
    __jvl = jvl.apply(match_on_annotation, axis=1, goa=_goa, col_name='QueryGene')
    __jvl = __jvl[['QueryGene', 'SuppressorGene', 'QueryGene_GO']]
    __jvl = __jvl.apply(match_on_annotation, axis=1, goa=_goa, col_name='SuppressorGene')
    __jvl = __jvl[['QueryGene', 'SuppressorGene', 'QueryGene_GO', 'SuppressorGene_GO']]
    __jvl.to_csv('jvl_go.csv', index=False)


def add_annotations(df):
    pass

