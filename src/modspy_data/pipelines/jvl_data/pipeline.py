"""
This is a boilerplate pipeline 'jvl_data'
generated using Kedro 0.18.12
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import mean, mean_sos, variance

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(len, "xs", "n"),
            node(mean, ["xs", "n"], "m", name="mean_node"),
            node(mean_sos, ["xs", "n"], "m2", name="mean_sos"),
            node(variance, ["m", "m2"], "v", name="variance_node"),
        ]
    )
