"""``KGDataSet`` Loads a knowledge graph.
"""
    
from pathlib import PurePosixPath
from typing import Any, Dict

import gzip
import fsspec
import pandas as pd
import torch
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class TorchFile(AbstractDataSet):
    def __init__(self, filepath: str):
        """Creates a Pytorch file

        Args:
            filepath: The location of the pytorch file to load.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)


    def _load(self):
        """Loads data from the .pt file.

        Returns:
            Annotation as pyotrch object
        """
        
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)

        return torch.load(load_path)
        
    def _describe(self) -> None:
        pass

    def _save(self, data) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        torch.save(data, save_path)