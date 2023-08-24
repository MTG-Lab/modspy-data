"""``KGDataSet`` Loads a knowledge graph.
"""
    
from pathlib import PurePosixPath
from typing import Any, Dict

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

    def _load(self) -> nx.DiGraph:
        """Loads data from the image file.

        Returns:
            Ontology as a NetworkX Ontology Digraph object
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)
        with self._fs.open(load_path) as f:                
            po = pronto.Ontology(handle=load_path)
            mg = pronto_to_multidigraph(po, default_rel_type="is a")
            dg = multidigraph_to_digraph(mg, reduce=True)
            digraph_nxo = nxo.NXOntology(dg)
            digraph_nxo.freeze()    
            return digraph_nxo
        
    def _describe(self) -> None:
        pass

    def _save(self) -> None:
        pass