
from loguru import logger
import numpy as np
import xarray as xr
import pandas as pd
from typing import List
from .extras.datasets.knowledge_graph_dataset import KGDataSet




class KnowledgeGraphScores:
        
    kg_name: str
    go_kg: KGDataSet
    nxo_info: List[str] = ['n_common_ancestors', 'n_union_ancestors', 'batet', 'batet_log', 'resnik', 'resnik_scaled', 'lin', 'jiang', 'jiang_seco']

    def __init__(self, graphs, col_names=('target', 'modifier', 'target_GO', 'modifier_GO')) -> None:
        # logger.info(graphs)
        self.kg_name = list(graphs.keys())    # Getting the name of the KG
        self.go_kg = graphs[self.kg_name[0]]
        self.col_names = col_names
        
        mix_methods = ['max', 'avg', 'bma']
        self.score_names = []
        for o in self.kg_name:
            for l in self.nxo_info:
                for p in mix_methods:
                    self.score_names.append(f"{o}_{l}_{p}")
        
        
    ##
    ## TODO Add https://github.com/MTG-Lab/GeMo/blob/5e146c2007e58cd61fafa74bf1ca4bf83ec380fa/src/gemo/features/build_features.py#L213 here to use fastsemsim.
    ## For publication purpose only
    ##
        
        
    def termgroup_sim(self, a, b, nxo, measures=None, mixing=['max','avg','bma']):
        """Get group similarity of a set of terms with another
        
        Arguments:        
            a {str} -- List of terms in group A
            b {str} -- List of terms in group B
            nxo {NXOntology} -- Nxontology object
            measures {[list]} -- any set of ['n_common_ancestors', 'n_union_ancestors', 'batet', 'batet_log', 'resnik', 'resnik_scaled', 'lin', 'jiang', 'jiang_seco']
            mixing {[list]} -- any from ['max','avg','bma']
        """
        if measures is None:
            measures = self.nxo_info
        
        # Creating xarray to store similarity
        # logger.debug(a.shape)
        # logger.debug(a)
        # logger.debug(b.shape)
        sim_arr = xr.DataArray(
            np.zeros((len(a), len(b), 9)), 
            dims=('target', 'candidate', 'score'), 
            coords={
                'target': a,
                'candidate': b,
                'score': self.nxo_info
            }
        )
        for i in sim_arr['target'].values:
            for j in sim_arr['candidate'].values:
                try:
                    sim = nxo.similarity(i, j).results() # Get similarity from NXOntology routine
                    # logger.debug(sim)
                except Exception as e:
                    logger.warning(repr(e))
                    continue
                sim_arr.loc[i,j] = [v for k, v in sim.items() if k in self.nxo_info] # Filterning only score from dictionary
        sim_arr = xr.where(sim_arr>0, sim_arr, np.nan) # Mask 0 for NaNs
        
        sim_scores = {}
        for i in measures:
            # Lin Similarity
            sim_scores[f"{i}_max"] = 0
            sim_scores[f"{i}_avg"] = 0
            sim_scores[f"{i}_bma"] = 0
            i_score = sim_arr[:,:,sim_arr['score']==i]
            if i_score.count():
                if 'max' in mixing:
                    i_max = i_score.max()
                    sim_scores[f"{i}_max"] = i_max.values
                if 'avg' in mixing:
                    i_avg = i_score.sum()/i_score.count()
                    sim_scores[f"{i}_avg"] = i_avg.values

                if 'bma' in mixing:
                    m_arr = np.max(i_score, axis=0)
                    n_arr = np.max(i_score, axis=1)
                    m_sum = m_arr.sum()
                    n_sum = n_arr.sum()
                    m = m_arr.count()
                    n = n_arr.count()
                    i_bma = (m_sum+n_sum)/(m+n)
                    sim_scores[f"{i}_bma"] = i_bma.values
        # logger.info(f'computation done: {sim_scores}')
        return sim_scores
            
        
    def get_feat_size(self):
        return len(self.nxo_info) * 3

    def get_scores(self, row):
        # logger.info(row)
        # logger.info(self.col_names)
        _g = row[self.col_names[0]]
        _tg = row[self.col_names[1]]
        
        result = {
            self.col_names[0]: _g,
            self.col_names[1]: _tg
        }
        # GO
        # logger.info(f"GO for {_g} and {_tg}")
        _g_annos = row[self.col_names[2]]
        _tg_annos = row[self.col_names[3]]
        if isinstance(_g_annos, str):
            _g_annos = pd.Series(_g_annos.split(','))
        elif np.isnan(_g_annos):
            _g_annos = pd.Series([])
        if isinstance(_tg_annos, str):
            _tg_annos = pd.Series(_tg_annos.split(','))
        elif np.isnan(_tg_annos):
            _tg_annos = pd.Series([])
        # logger.info(_g_annos)
        # logger.info(_tg_annos)
        _scores = self.termgroup_sim(_tg_annos, _g_annos, self.go_kg[0], measures=self.nxo_info, mixing=['max','avg','bma'])
        for k, s in _scores.items():
            result[f"{self.kg_name[0]}_{k}"] = s.item(0) if s else s
        # logger.debug(result)
        
        return pd.Series(result, index=[self.col_names[0], self.col_names[1]]+self.score_names)