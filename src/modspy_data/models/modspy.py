from comet_ml import Experiment

import os
import time
import torch
import typer
import random
import optuna
from pytorch_lightning.loggers import CometLogger
from torch_geometric.transforms import AddSelfLoops
from typing import List, Optional, Tuple, Union, cast

from traitlets import default
from typing_extensions import Annotated

# import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor

from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data.lightning.datamodule import LightningNodeData

from ray import tune
from ray.tune.schedulers import ASHAScheduler


import ray.train.lightning
from ray.air.integrations.comet import CometLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback


from embed import MetaPath2VecLightningModule, METAPATH_SCHEME

app = typer.Typer()


def train_m2vec(
    config,
    network_filepath: str = "",
    data_filepath: str = "",
    num_epochs=10,
    num_gpus=0,
    num_workers=8
):
    # Determing context size
    context_size = (
        config["context_size"]
        if "context_size" in config
        else random.randint(3, config["walk_length"])
    )
    model = MetaPath2VecLightningModule(
        network_filepath=network_filepath,
        data_filepath=data_filepath,
        embedding_dim=config["embedding_dim"],
        walk_length=config["walk_length"],
        context_size=context_size,
        walks_per_node=config["walks_per_node"],
        num_negative_samples=config["num_negative_samples"],
        lr=config["lr"],
        batch_size=config["batch_size"],
        metapath=config["metapath"],
        num_workers=num_workers,
    )

    print(f"Inside train_m2vec: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if not torch.cuda.is_available():
        print(
            "Lightning is not seeing any CUDA. Please check your system's configuration."
        )
    else:
        print(
            f"Available GPUs: {torch.cuda.device_count()}, First GPU Name: {torch.cuda.get_device_name(0)}"
        )

    comet_logger = CometLogger(
        project_name="modspy",
        save_dir="/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/06_models/modspy-experiments/comet-lightning-logs",
        offline=True,
    )  # We use Comet for logging

    early_stopper = EarlyStopping(
        monitor="val_accuracy", mode="max", patience=7, divergence_threshold=0.85
    )  # Early stopping criteria
    tune_reporter = TuneReportCheckpointCallback(
        metrics={"train_loss": "train_loss"}, save_checkpoints=True, on="train_end"
    )  # Ray Tune Reporer
    lr_monitor = LearningRateMonitor(logging_interval="epoch")  # LR optimization

    # checkpoiniting failing because of graham pyarrow does is having issue:
    # ImportError: The pyarrow installation is not built with support for 'S3FileSystem'
    trainer = pl.Trainer(
        default_root_dir=f"/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/06_models",
        accelerator="gpu",
        devices=1,
        strategy=pl.strategies.SingleDeviceStrategy("cuda:0"),
        max_epochs=num_epochs,
        enable_progress_bar=False,
        logger=comet_logger,
        log_every_n_steps=100,
        callbacks=[early_stopper, tune_reporter, lr_monitor],
        # logger=CSVLogger("logs"),
    )
    # Train model
    trainer.fit(model)


def tune_model(network_filepath: str, data_filepath: str, n_samples: int = 5, num_epochs: int = 100, num_workers: int = 8):
    print(f"CUDA_VISIBLE_DEVICES is {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(
        f"Received {os.environ['HEAD_NODE']} from environment variable. Going to initialize Ray"
    )
    # Connect to Ray cluster
    ray.init(
        address=f"{os.environ['ip_head']}", _node_ip_address=os.environ["HEAD_NODE"]
    )

    # Check that Ray sees two nodes and their status is 'Alive'
    print("Nodes for the Ray cluster:")
    print(ray.nodes())

    # Check that Ray sees 12 CPUs and 2 GPUs over 2 Nodes
    print(ray.available_resources())

    # Defining hyperparam search space
    search_space = {
        "batch_size": tune.choice([32, 64, 128]),
        "embedding_dim": tune.choice([128]),
        "num_negative_samples": tune.randint(5, 18),  # (5, 12)
        "lr": tune.loguniform(1e-4, 1e-1),
        "metapath": tune.choice([METAPATH_SCHEME]),
        "walk_length": tune.randint(3, 30),  # (10, 30)
        "walks_per_node": tune.randint(3, 30),  # (10, 30)
    }
    
    scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    trainable = tune.with_parameters(
        train_m2vec,
        network_filepath=network_filepath,
        data_filepath=data_filepath,
        num_epochs=num_epochs,
        num_gpus=1,
    )

    # Specify each trial resources
    resource_group = tune.PlacementGroupFactory([{"CPU": num_workers, "GPU": 1}])
    trainable = tune.with_resources(trainable, resource_group)

    analysis = tune.run(
        trainable,
        config=search_space,
        num_samples=n_samples,
        scheduler=scheduler,
        local_dir="/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/06_models/modspy-experiments/ray-reults",  # Directory where results are stored
        progress_reporter=tune.CLIReporter(
            metric_columns=["train_loss", "training_iteration"]
        ),  # Adjust the reported metrics if needed
        # resources_per_trial={"cpu": 4, "gpu": 1},
    )

    try:
        best_trial = analysis.get_best_trial(metric="val_accuracy", mode="max")
        print(f"Best trial config: {best_trial.config}")
        print(
            f"Best trial final validation loss: {best_trial.last_result['val_accuracy']}"
        )
    except:
        print("Could not print best result.")


@app.command()
def main(
    n_samples: int = typer.Option(
        5, help="Number of samples to draw from hyperparameter space for training."
    ),
    epoch: int = typer.Option(
        500, help="Number of samples to draw from hyperparameter space for training."
    )
):
    typer.echo(f"Number of samples: {n_samples}")

    # Ensure your data file path is correct
    network_filepath = "/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/05_model_input/2024-02-monarch_heterodata_v1.pt"
    # Datafilepath contains the output from `notebooks/monarch_gnn/3_create_dataset_pair_tensor_nd_validate.ipynb`
    data_filepath = "/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/05_model_input/2024-03-31-merged-dataset.pt"

    # Defining hyperparam search space
    params = {
        "batch_size": 128,
        "embedding_dim": 128,
        "context_size": 7,
        "num_negative_samples": 5,
        "lr": 0.004970498521366851,
        "metapath": METAPATH_SCHEME,
        "walk_length": 12,
        "walks_per_node": 5,
    }

    if os.path.exists(network_filepath) and os.path.exists(data_filepath):
        if n_samples <= 0:
            train_m2vec(
                params, network_filepath, data_filepath, num_epochs=epoch, num_gpus=1, num_workers=8
            )
        else:
            tune_model(network_filepath, data_filepath, n_samples=n_samples, num_epochs=epoch, num_workers=16)
    else:
        print(f"Network/Data file not found")


if __name__ == "__main__":
    app()
