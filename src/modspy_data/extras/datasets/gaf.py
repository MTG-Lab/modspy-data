"""``KGDataSet`` Loads a knowledge graph.
"""
    
from pathlib import PurePosixPath
from typing import Any, Dict

import gzip
import fsspec
import pandas as pd
from kedro.io import AbstractDataSet
from kedro.io.core import get_filepath_str, get_protocol_and_path


class GAFDataFile(AbstractDataSet[pd.DataFrame, pd.DataFrame]):
    def __init__(self, filepath: str):
        """Creates a pandas dataframe from the given GAF annotation file.

        Args:
            filepath: The location of the gaf file to load.
        """
        # parse the path and protocol (e.g. file, http, s3, etc.)
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol)
        self.columns = [
                'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
                'DB_Reference', 'Evidence_Code', 'With_Or_From', 'Aspect',
                'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type', 'Taxon',
                'Date', 'Assigned_By', 'Annotation_Extension'
            ]


    def _load(self) -> pd.DataFrame:
        """Loads data from the .gaf file.

        Returns:
            Annotation as pd.Dataframe object
        """
        # using get_filepath_str ensures that the protocol and path are appended correctly for different filesystems
        load_path = get_filepath_str(self._filepath, self._protocol)

        # Try to open the file with gzip.open
        try:
            # Unzipping and reading the GAF file
            with gzip.open(load_path, 'rt') as file:
                # Skipping initial comment lines that start with '!'
                lines = [line for line in file if not line.startswith('!')]
        except OSError:
            # If gzip.open raises an OSError, the file is not gzipped
            with open(load_path, 'rt') as file:
                lines = [line for line in file if not line.startswith('!')]

        # Convert lines to DataFrame
        df = pd.DataFrame([line.strip().split('\t') for line in lines])
        # Renaming columns of the GO annotation DataFrame
        df.columns = self.columns

        return df
        
    def _describe(self) -> None:
        pass

    def _save(self) -> None:
        pass