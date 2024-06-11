"""``KGDataSet`` Loads a knowledge graph.
"""
    
from pathlib import PurePosixPath
from typing import Any, Dict
from loguru import logger

import fsspec
import networkx as nx
import nxontology as nxo
import pronto
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path
from nxontology.imports import (
    from_file,
    multidigraph_to_digraph,
    pronto_to_multidigraph,
)


class KGDataSet(AbstractDataSet[pronto.Ontology, nx.DiGraph]):
    def __init__(self, filepath: str):
        """Creates a new NXOntology from the given filepath.

        Args:
            filepath: The location of the ontology file to load data.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)


    def parse_obo_file_custom(self, file_path, rel_type: str = "is_a") -> nx.DiGraph:
        """
        Parses an OBO file and returns a networkx graph.
        Each node in the graph is a term in the ontology, and edges represent relationships between terms.
        """
        graph = nx.DiGraph()
        current_term = None

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line == "[Term]" or line == "":
                    current_term = None
                elif line.startswith("id: "):
                    current_term = line.split("id: ")[1]
                    graph.add_node(current_term)
                elif line.startswith("name: "):
                    name = line.split("name: ")[1]
                    graph.nodes[current_term]['name'] = name
                elif line.startswith(rel_type+": "):
                    parent_term = line.split(rel_type+": ")[1].split(' ! ')[0]
                    graph.add_edge(current_term, parent_term)
                # TODO Additional relationships could be added here as needed
        digraph_nxo = nxo.NXOntology(graph)
        digraph_nxo.freeze()
        return digraph_nxo


    def _load(self) -> nx.DiGraph:
        """Loads data from the image file.

        Returns:
            Ontology as a NetworkX Ontology Digraph object
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)
        try: 
            with self._fs.open(load_path) as f:                
                po = pronto.Ontology(handle=load_path)
                mg = pronto_to_multidigraph(po, default_rel_type="is a")    # TODO consider adding other type of relationships
                # For future (may be) create seperate multi graphs for different relationships
                # DiGraph can not hold more than one edge.
                # anoter idea is to use edge attribute to indicate different relationships. 
                dg = multidigraph_to_digraph(mg, reduce=True)
                digraph_nxo = nxo.NXOntology(dg)
                digraph_nxo.freeze()
                return digraph_nxo
        except Exception as e:
            logger.warning(repr(e))
            return self.parse_obo_file_custom(load_path)

        
    def _describe(self) -> None:
        pass

    def _save(self) -> None:
        pass