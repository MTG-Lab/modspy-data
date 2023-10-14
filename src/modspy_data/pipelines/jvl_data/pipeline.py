"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import add_annotations, clean_jvl, mean, mean_sos, variance

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        # [
        #     node(len, "xs", "n"),
        #     node(mean, ["xs", "n"], "m", name="mean_node"),
        #     node(mean_sos, ["xs", "n"], "m2", name="mean_sos"),
        #     node(variance, ["m", "m2"], "v", name="variance_node"),
        # ],
        [
            node(clean_jvl, ['jvl', 'goa'], 'jvl_annotated', name='clean_jvl'),
            node(add_annotations, ['jvl_annotated', 'go'], 'jvl_scored', name='add_annotations'),
        ]
    )
