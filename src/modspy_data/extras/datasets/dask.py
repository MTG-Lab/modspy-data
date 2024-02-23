from typing import Any, Dict
from copy import deepcopy

import dask.dataframe as dd
import dask.bag as db
from kedro.io import AbstractDataSet



class DaskBagDataset(AbstractDataSet):
    """``DaskBagDataset`` loads and saves Dask Bag."""

    DEFAULT_LOAD_ARGS: Dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: Dict[str, Any] = {}
    
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None):
        """Creates a new instance of ``DaskBagDataset``.

        Args:
            filepath: The path to the file to load or save the dataframe.
            load_args: Additional loading options `dask.dataframe.read_csv`:
                https://docs.dask.org/en/latest/generated/dask.dataframe.read_csv.html
            save_args: Additional saving options for `dask.dataframe.to_csv`:
                https://docs.dask.org/en/latest/generated/dask.dataframe.to_csv.html
        """
        self._filepath = filepath

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> db.Bag:
        """Loads the Dask dataframe from the file.

        Returns:
            The loaded Dask dataframe.
        """
        return db.read_text(self._filepath, **self._load_args)
        # return db.read_csv()

    def _save(self, data: db.Bag) -> None:
        """Saves the Dask dataframe to the file.

        Args:
            data: The Dask dataframe to save.
        """
        return data.to_textfiles(self._filepath, **self._save_args)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dictionary that describes the dataset.

        Returns:
            A dictionary that describes the dataset.
        """
        return {"filepath": self._filepath}
                 
                 
                 

class DaskDataFrameDataSet(AbstractDataSet):
    """``DaskDataFrameDataSet`` loads and saves Dask dataframes."""

    DEFAULT_LOAD_ARGS: Dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: Dict[str, Any] = {"write_index": False}

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] = None,
        save_args: Dict[str, Any] = None):
        """Creates a new instance of ``DaskDataFrameDataSet``.

        Args:
            filepath: The path to the file to load or save the dataframe.
            load_args: Additional loading options `dask.dataframe.read_csv`:
                https://docs.dask.org/en/latest/generated/dask.dataframe.read_csv.html
            save_args: Additional saving options for `dask.dataframe.to_csv`:
                https://docs.dask.org/en/latest/generated/dask.dataframe.to_csv.html
        """
        self._filepath = filepath

        # Handle default load and save arguments
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)

    def _load(self) -> dd.DataFrame:
        """Loads the Dask dataframe from the file.

        Returns:
            The loaded Dask dataframe.
        """
        return dd.read_csv(self._filepath, **self._load_args)

    def _save(self, data: dd.DataFrame) -> None:
        """Saves the Dask dataframe to the file.

        Args:
            data: The Dask dataframe to save.
        """
        data.to_csv(self._filepath, **self._save_args)

    def _describe(self) -> Dict[str, Any]:
        """Returns a dictionary that describes the dataset.

        Returns:
            A dictionary that describes the dataset.
        """
        return {"filepath": self._filepath}