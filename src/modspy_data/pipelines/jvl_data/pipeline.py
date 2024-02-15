"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_annotations, clean_jvl, mean, mean_sos, variance, annotate_olida, compute_similarity, kgx_to_pyg

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(kgx_to_pyg, ['monarch_nodes_categorized', 'monarch_edges_categorized'], 'monarch_pyg', name='monarch_kgx_to_pyg'),
        ],


        # [
        #     node(len, "xs", "n"),
        #     node(mean, ["xs", "n"], "m", name="mean_node"),
        #     node(mean_sos, ["xs", "n"], "m2", name="mean_sos"),
        #     node(variance, ["m", "m2"], "v", name="variance_node"),
        # ],
        [
            # node(compute_similarity, ['olida_annotated', 'go', 
            #                           'olida_columns', 'params:dask'], 
            #                           'olida_scored', name='olida_similarity'),
            node(compute_similarity, ['olida_annotated', 'hpo', 
                                      'olida_po_columns', 'params:dask'], 
                                      'olida_scored_pheno', name='olida_PO_similarity'),
            node(compute_similarity, ['olida_annotated', 'do', 
                                      'olida_do_columns', 'params:dask'], 
                                      'olida_scored_disease', name='olida_DO_similarity'),
            # node(annotate_olida, ['olida_pairs', 'goa'], 'olida_annotated', name='annotate_olida'),
            # node(clean_jvl, ['jvl', 'goa'], 'jvl_annotated', name='clean_jvl'),
            # node(add_annotations, ['jvl_annotated', 'go'], 'jvl_scored', name='add_annotations'),
            node(compute_similarity, ['jvl_annotated', 'hpo', 
                                      'jvl_po_columns', 'params:dask'], 
                                      'jvl_scored_pheno', name='jvl_PO_similarity'),
            node(compute_similarity, ['jvl_annotated', 'do', 
                                      'jvl_do_columns', 'params:dask'], 
                                      'jvl_scored_disease', name='jvl_DO_similarity'),
            node(compute_similarity, ['zyg1_annotated', 'wb_po', 
                                      'zyg1_po_columns', 'params:dask'], 
                                      'zyg1_scored_pheno', name='zyg1_PO_similarity'),
            node(compute_similarity, ['zyg1_annotated', 'do', 
                                      'zyg1_do_columns', 'params:dask'], 
                                      'zyg1_scored_disease', name='zyg1_DO_similarity'),
        ]
    )
