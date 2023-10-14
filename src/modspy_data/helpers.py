
from loguru import logger
import numpy as np
import xarray as xr
import pandas as pd




class KnowledgeGraphScores:
        
    nxo_info = ['n_common_ancestors', 'n_union_ancestors', 'batet', 'batet_log', 'resnik', 'resnik_scaled', 'lin', 'jiang', 'jiang_seco']
    
    def __init__(self, graphs, col_names=('target', 'modifier', 'target_GO', 'modifier_GO'), saved_scores=None) -> None:
        logger.info(graphs)
        self.go_kg = graphs['go'] 
        # self.wpo_kg = graphs['wpo']
        self.saved_scores = None
        self.col_names = col_names
        
        # if saved_scores != None:
        #     self.saved_scores = pd.read_csv(config.data_dir/f"interim/jan_2023_all_combs_scores_2023-01-21.tsv", sep='\t')
        
        mix_methods = ['max', 'avg', 'bma']
        ontos = graphs.keys()
        self.score_names = []
        for o in ontos:
            for l in self.nxo_info:
                for p in mix_methods:
                    self.score_names.append(f"{o}_{l}_{p}")
        
        
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
        logger.debug(a.shape)
        logger.debug(a)
        logger.debug(b.shape)
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
                except Exception as e:
                    logger.warning(repr(e))
                    continue
                sim_arr.loc[i,j] = [v for k, v in sim.items() if k in self.nxo_info] # Filterning only score from dictionary
        sim_arr = xr.where(sim_arr>0, sim_arr, np.nan) # Mask 0 or NaNs
        
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
        _g = row[self.col_names[0]]
        _tg = row[self.col_names[1]]
        
        # # check if already calculated
        # if len(self.saved_scores):
        #     row = self.saved_scores[(self.saved_scores[self.col_names[0]]==_g) & (self.saved_scores[self.col_names[1]]==_tg)]
        #     if len(row):
        #         # ic(row.iloc[0])
        #         return row.iloc[0]
        
        result = {
            self.col_names[0]: _g,
            self.col_names[1]: _tg
        }
        # GO
        logger.info(f"GO for {_g} and {_tg}")
        _g_annos = row[self.col_names[2]]
        _tg_annos = row[self.col_names[3]]
        _scores = self.termgroup_sim(_tg_annos, _g_annos, self.go_kg[0], measures=self.nxo_info, mixing=['max','avg','bma'])
        for k, s in _scores.items():
            result[f"go_{k}"] = s.item(0) if s else s
        
        # # WPO
        # logger.info(f"WPO for {_g} and {_tg}")
        # _g_annos = self.wpo_kg.filtered_df[self.wpo_kg.filtered_df[1]==_g][4].unique()
        # _tg_annos = self.wpo_kg.filtered_df[self.wpo_kg.filtered_df[1]==_tg][4].unique()
        # _scores = self.termgroup_sim(_tg_annos, _g_annos, self.wpo_kg.nxo, measures=self.nxo_info, mixing=['max','avg','bma'])
        # for k, s in _scores.items():
        #     result[f"wpo_{k}"] = s.item(0) if s else s
        # ic(f"NA----",pd.Series(result, index=['wormbase_gene_id', 'target_gene_id']+self.score_names))
        return pd.Series(result, index=[self.col_names[0], self.col_names[1]]+self.score_names)