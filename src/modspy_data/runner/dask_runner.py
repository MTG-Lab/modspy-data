"""``DaskRunner`` is an ``AbstractRunner`` implementation. It can be
used to distribute execution of ``Node``s in the ``Pipeline`` across
a Dask cluster, taking into account the inter-``Node`` dependencies.
"""
import logging
from collections import Counter
from itertools import chain
from typing import Any, Dict

from dask_jobqueue import SLURMCluster
from distributed import Client, as_completed, worker_client
from kedro.framework.hooks.manager import (
    _create_hook_manager,
    _NullPluginManager,
    _register_hooks,
    _register_hooks_setuptools,
)
from kedro.framework.project import settings
from kedro.io import AbstractDataSet, DataCatalog
from kedro.pipeline import Pipeline
from kedro.pipeline.node import Node
from kedro.runner import AbstractRunner, run_node
from pluggy import PluginManager


class _DaskDataSet(AbstractDataSet):
    """``_DaskDataSet`` publishes/gets named datasets to/from the Dask
    scheduler."""

    def __init__(self, name: str):
        self._name = name

    def _load(self) -> Any:
        try:
            with worker_client() as client:
                return client.get_dataset(self._name)
        except ValueError:
            # Upon successfully executing the pipeline, the runner loads
            # free outputs on the scheduler (as opposed to on a worker).
            Client.current().get_dataset(self._name)

    def _save(self, data: Any) -> None:
        with worker_client() as client:
            client.publish_dataset(data, name=self._name, override=True)

    def _exists(self) -> bool:
        return self._name in Client.current().list_datasets()

    def _release(self) -> None:
        Client.current().unpublish_dataset(self._name)

    def _describe(self) -> Dict[str, Any]:
        return dict(name=self._name)



class SLURMRunner(AbstractRunner):
    """``SLURMRunner`` is an ``AbstractRunner`` implementation. It can be
    used to distribute execution of ``Node``s in the ``Pipeline`` across
    a SLURM cluster using Dask API, taking into account the inter-``Node`` dependencies.
    It uses SLURM jobque instiation to run Dask jobs.
    """

    def __init__(self, n_workers: int, slurm_args: Dict[str, Any] = {}, 
                 client_args: Dict[str, Any] = {}, is_async: bool = False):
        """Instantiates the runner by creating a ``distributed.Client``.

        Args:
            n_workers: Number of workers to initiate
            slurm_args: Parameters to pass to the ``dask_jobqueue.SLURMCluster``
            client_args: Arguments to pass to the ``distributed.Client``
                constructor.
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
        """
        super().__init__(is_async=is_async)
        self.cluster = SLURMCluster(**slurm_args)
        self.n_workers = n_workers
        self._logger.info(self.cluster)
        self.client = Client(self.cluster)
        self._logger.info(self.client)

        self.cluster.scale(self.n_workers)

        
    def __del__(self):
        self.client.close()  # Close client
        self.cluster.close()  # Release resources

    def create_default_data_set(self, ds_name: str) -> _DaskDataSet:
        """Factory method for creating the default dataset for the runner.

        Args:
            ds_name: Name of the missing dataset.

        Returns:
            An instance of ``_DaskDataSet`` to be used for all
            unregistered datasets.
        """
        return _DaskDataSet(ds_name)

    @staticmethod
    def _run_node(
        node: Node,
        catalog: DataCatalog,
        is_async: bool = False,
        session_id: str = None,
        *dependencies: Node,
    ) -> Node:
        """Run a single `Node` with inputs from and outputs to the `catalog`.

        Wraps ``run_node`` to accept the set of ``Node``s that this node
        depends on. When ``dependencies`` are futures, Dask ensures that
        the upstream node futures are completed before running ``node``.

        A ``PluginManager`` instance is created on each worker because the
        ``PluginManager`` can't be serialised.

        Args:
            node: The ``Node`` to run.
            catalog: A ``DataCatalog`` containing the node's inputs and outputs.
            is_async: If True, the node inputs and outputs are loaded and saved
                asynchronously with threads. Defaults to False.
            session_id: The session id of the pipeline run.
            dependencies: The upstream ``Node``s to allow Dask to handle
                dependency tracking. Their values are not actually used.

        Returns:
            The node argument.
        """
        hook_manager = _create_hook_manager()
        _register_hooks(hook_manager, settings.HOOKS)
        _register_hooks_setuptools(hook_manager, settings.DISABLE_HOOKS_FOR_PLUGINS)
        
        return run_node(node, catalog, hook_manager, is_async, session_id)

    def _run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager,
        session_id: str = None,
    ) -> None:
        """The method implementing sequential pipeline running.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            Exception: in case of any downstream node failure.
        """
        nodes = pipeline.nodes
        done_nodes = set()

        load_counts = Counter(chain.from_iterable(n.inputs for n in nodes))

        # TODO: Variable/adaptive scaling
        # self.cluster.scale(4)
        for exec_index, node in enumerate(nodes):
            try:
                run_node(node, catalog, hook_manager, self._is_async, session_id)
                done_nodes.add(node)
            except Exception:
                self._suggest_resume_scenario(pipeline, done_nodes, catalog)
                raise

            # decrement load counts and release any data sets we've finished with
            for data_set in node.inputs:
                load_counts[data_set] -= 1
                if load_counts[data_set] < 1 and data_set not in pipeline.inputs():
                    catalog.release(data_set)
            for data_set in node.outputs:
                if load_counts[data_set] < 1 and data_set not in pipeline.outputs():
                    catalog.release(data_set)

            self._logger.info(
                "Completed %d out of %d tasks", exec_index + 1, len(nodes)
            )


    def run_only_missing(
        self, pipeline: Pipeline, catalog: DataCatalog
    ) -> Dict[str, Any]:
        """Run only the missing outputs from the ``Pipeline`` using the
        datasets provided by ``catalog``, and save results back to the
        same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be
                satisfied.

        Returns:
            Any node outputs that cannot be processed by the
            ``DataCatalog``. These are returned in a dictionary, where
            the keys are defined by the node outputs.
        """
        free_outputs = pipeline.outputs() - set(catalog.list())
        missing = {ds for ds in catalog.list() if not catalog.exists(ds)}
        to_build = free_outputs | missing
        to_rerun = pipeline.only_nodes_with_outputs(*to_build) + pipeline.from_inputs(
            *to_build
        )

        # We also need any missing datasets that are required to run the
        # `to_rerun` pipeline, including any chains of missing datasets.
        unregistered_ds = pipeline.data_sets() - set(catalog.list())
        # Some of the unregistered datasets could have been published to
        # the scheduler in a previous run, so we need not recreate them.
        missing_unregistered_ds = {
            ds_name
            for ds_name in unregistered_ds
            if not self.create_default_data_set(ds_name).exists()
        }
        output_to_unregistered = pipeline.only_nodes_with_outputs(
            *missing_unregistered_ds
        )
        input_from_unregistered = to_rerun.inputs() & missing_unregistered_ds
        to_rerun += output_to_unregistered.to_outputs(*input_from_unregistered)

        # We need to add any previously-published, unregistered datasets
        # to the catalog passed to the `run` method, so that it does not
        # think that the `to_rerun` pipeline's inputs are not satisfied.
        catalog = catalog.shallow_copy()
        for ds_name in unregistered_ds - missing_unregistered_ds:
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        return self.run(to_rerun, catalog)
        
    def run(
        self,
        pipeline: Pipeline,
        catalog: DataCatalog,
        hook_manager: PluginManager = None,
        session_id: str = None,
    ) -> Dict[str, Any]:
        """Run the ``Pipeline`` using the datasets provided by ``catalog``
        and save results back to the same objects.

        Args:
            pipeline: The ``Pipeline`` to run.
            catalog: The ``DataCatalog`` from which to fetch data.
            hook_manager: The ``PluginManager`` to activate hooks.
            session_id: The id of the session.

        Raises:
            ValueError: Raised when ``Pipeline`` inputs cannot be satisfied.

        Returns:
            Any node outputs that cannot be processed by the ``DataCatalog``.
            These are returned in a dictionary, where the keys are defined
            by the node outputs.

        """
        self._logger.info("================== 🏃 RUN 🏃 ==================")
        hook_manager = hook_manager or _NullPluginManager()
        catalog = catalog.shallow_copy()

        # Check which datasets used in the pipeline are in the catalog or match
        # a pattern in the catalog
        registered_ds = [ds for ds in pipeline.data_sets() if ds in catalog]

        # Check if there are any input datasets that aren't in the catalog and
        # don't match a pattern in the catalog.
        unsatisfied = pipeline.inputs() - set(registered_ds)

        if unsatisfied:
            raise ValueError(
                f"Pipeline input(s) {unsatisfied} not found in the DataCatalog"
            )

        # Check if there's any output datasets that aren't in the catalog and don't match a pattern
        # in the catalog.
        free_outputs = pipeline.outputs() - set(registered_ds)
        unregistered_ds = pipeline.data_sets() - set(registered_ds)

        # Create a default dataset for unregistered datasets
        for ds_name in unregistered_ds:
            catalog.add(ds_name, self.create_default_data_set(ds_name))

        if self._is_async:
            self._logger.info(
                "Asynchronous mode is enabled for loading and saving data"
            )

        # self.cluster.scale(jobs=3)  # Scale cluster based on number of nodes
        self._run(pipeline, catalog, hook_manager, session_id)
        self._logger.info("Closing Dask client and cluster")

        self.client.close()  # Close client
        self.cluster.close()  # Release resources

        return {ds_name: catalog.load(ds_name) for ds_name in free_outputs}

