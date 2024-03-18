from comet_ml import Experiment

import os
import time
import torch
from torch_geometric.transforms import AddSelfLoops
import optuna
from pytorch_lightning.loggers import CometLogger

# import wandb
import pytorch_lightning as pl
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data.lightning.datamodule import LightningNodeData

from ray import tune
from ray.tune.schedulers import ASHAScheduler


import ray.train.lightning
from ray.air.integrations.comet import CometLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from ray.train.lightning import (
    prepare_trainer,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
)


class MetaPath2VecLightningModule(pl.LightningModule):
    def __init__(
        self,
        data_filepath,
        embedding_dim=128,
        walk_length=50,
        context_size=7,
        walks_per_node=5,
        num_negative_samples=5,
        lr=0.01,
        batch_size=128,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Loading Data
        self.data = torch.load(data_filepath)
        
        print(f"Computing on {self.device}")
        print(torch.cuda.is_available())  # Should print True if CUDA is available
        print(torch.cuda.device_count())  # Should print the number of GPUs available
        print(torch.cuda.get_device_name(0))  # Should print the name of the first GPU
        print(
            f""" 
        Total nodes: {self.data.num_nodes}
        Total node types: {len(self.data.node_types)}

        Total edges: {self.data.num_edges}
        Total edge types: {len(self.data.edge_types)}                
        """
        )
        
        # Adding self loops to avoid 1. nodes without any edge, 2. consider intragenic modifier
        transform = AddSelfLoops()
        self.data = transform(self.data)

        # Defining metapath to explore
        self.metapaths = [
            ("biolink:Gene", "biolink:orthologous_to", "biolink:Gene"),
            ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
        ]

        # Model definition
        self.model = MetaPath2Vec(
            self.data.edge_index_dict,
            embedding_dim=self.hparams.embedding_dim,
            metapath=self.metapaths,
            walk_length=self.hparams.walk_length,
            context_size=self.hparams.context_size,
            walks_per_node=self.hparams.walks_per_node,
            num_negative_samples=self.hparams.num_negative_samples,
            sparse=True,
        ) #.to(self.device)

    def train_dataloader(self):
        loader = self.model.loader(
            batch_size=self.hparams.batch_size, shuffle=True, num_workers=4
        )
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        print(batch[0].device)
        pos_rw, neg_rw = batch
        print(f"Device pos_rw: {pos_rw.device}, neg_rw: {neg_rw.device}")
        # pos_rw = pos_rw.to(self.device)  # Ensure tensors are on the correct device
        # neg_rw = neg_rw.to(self.device)
        loss = self.model.loss(pos_rw, neg_rw)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return loss

    # Optional: Define a validation step
    def validation_step(self, batch, batch_idx):
        # Implement your validation logic here, possibly using a different dataset
        # and returning some metric of interest, e.g., accuracy.
        pass

    # Optional: If your 'test' function is essential, integrate it here similarly to validation_step
    # def test_step(self, batch, batch_idx):
    #     pass


############################################ CONSTRUCTION ######################################

# ## Adding Ray tune to fine tune model

# from ray.train import RunConfig, ScalingConfig, CheckpointConfig
# from ray.train.torch import TorchTrainer


# scaling_config = ScalingConfig(
#     num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
# )

# run_config = RunConfig(
#     checkpoint_config=CheckpointConfig(
#         num_to_keep=2,
#         checkpoint_score_attribute="ptl/val_accuracy",
#         checkpoint_score_order="max",
#     ),
# )

# # # Define a TorchTrainer without hyper-parameters for Tuner
# # ray_trainer = TorchTrainer(
# #     train_func,
# #     scaling_config=scaling_config,
# #     run_config=run_config,
# # )

# # def tune_mnist_asha(num_samples=10):
# #     scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

# #     tuner = tune.Tuner(
# #         ray_trainer,
# #         param_space={"train_loop_config": search_space},
# #         tune_config=tune.TuneConfig(
# #             metric="ptl/val_accuracy",
# #             mode="max",
# #             num_samples=num_samples,
# #             scheduler=scheduler,
# #         ),
# #     )
# #     return tuner.fit()

# # results = tune_mnist_asha(num_samples=num_samples)


# def objective(config):  # ①
#     train_loader, test_loader = load_data()  # Load some data
#     model = ConvNet().to("cpu")  # Create a PyTorch conv net
#     optimizer = torch.optim.SGD(  # Tune the optimizer
#         model.parameters(), lr=config["lr"], momentum=config["momentum"]
#     )

#     while True:
#         train(model, optimizer, train_loader)  # Train the model
#         acc = test(model, test_loader)  # Compute test accuracy
#         train.report({"mean_accuracy": acc})  # Report to Tune


#     # Data file
#     data_filepath = './data/05_model_input/2024-02-monarch_heterodata_v1.pt'
#     # tune_config = {
#     #     "lr": 1e-3,
#     #     "batch_size": tune.choice([32, 64]),
#     # }

#     # We will use Comet for logging
#     comet_logger = CometLogger(project_name="modspy-monarch",
#                                 save_dir="./data/06_models/monarch-lightning/comet",
#                                 rest_api_key='NpKZhU5E8qdb0tJdJtlzoUugy',  # Optional
#                                 experiment_name="lightning_logs",  # Optional
#                                 )

#     # Prepare your model
#     model = MetaPath2VecLightningModule(data_filepath)

#     strategy = pl.strategies.SingleDeviceStrategy('cuda:0')
#     # Setup the trainer
#     trainer = pl.Trainer(default_root_dir="./data/06_models/monarch-lightning", max_epochs=5,
#                         accelerator="gpu", strategy=strategy, devices=1, logger=comet_logger)

#     # Train the model
#     trainer.fit(model)


# search_space = {"lr": tune.loguniform(1e-4, 1e-2), "momentum": tune.uniform(0.1, 0.9)}
# algo = OptunaSearch()  # ②

# tuner = tune.Tuner(  # ③
#     objective,
#     tune_config=tune.TuneConfig(
#         metric="mean_accuracy",
#         mode="max",
#         search_alg=algo,
#     ),
#     run_config=train.RunConfig(
#         stop={"training_iteration": 5},
#     ),
#     param_space=search_space,
# )
# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)


# ====================================


# scaling_config = ScalingConfig(
#     num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 1}
# )

# run_config = RunConfig(
#     checkpoint_config=CheckpointConfig(
#         num_to_keep=2,
#         checkpoint_score_attribute="ptl/val_accuracy",
#         checkpoint_score_order="max",
#     ),
# )

# # # Define a TorchTrainer without hyper-parameters for Tuner
# # ray_trainer = TorchTrainer(
# #     train_func,
# #     scaling_config=scaling_config,
# #     run_config=run_config,
# # )

# # def tune_mnist_asha(num_samples=10):
# #     scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

# #     tuner = tune.Tuner(
# #         ray_trainer,
# #         param_space={"train_loop_config": search_space},
# #         tune_config=tune.TuneConfig(
# #             metric="ptl/val_accuracy",
# #             mode="max",
# #             num_samples=num_samples,
# #             scheduler=scheduler,
# #         ),
# #     )
# #     return tuner.fit()

# # results = tune_mnist_asha(num_samples=num_samples)


# Your MetaPath2VecLightningModule class definition remains the same


def train_m2vec(config, data_filepath=None, num_epochs=10, num_gpus=0):
    model = MetaPath2VecLightningModule(
        data_filepath,
        embedding_dim=config["embedding_dim"],
        walk_length=config["walk_length"],
        lr=config["lr"],
        batch_size=config["batch_size"],
    )
    
    print(f"Inside train_m2vec: {os.environ['CUDA_VISIBLE_DEVICES']}")
    if not torch.cuda.is_available():
        print("Lightning is not seeing any CUDA. Please check your system's configuration.")
    else:
        print(f"Available GPUs: {torch.cuda.device_count()}, First GPU Name: {torch.cuda.get_device_name(0)}")


    # We will use Comet for logging
    comet_logger = CometLogger(project_name="modspy",
                                save_dir="./data/06_models/monarch-lightning/comet"
                                )
            
    trainer = pl.Trainer(
        default_root_dir=f"/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/06_models/monarch-lightning/{int(time.time())}",
        accelerator="gpu",
        devices=1,
        strategy=pl.strategies.SingleDeviceStrategy('cuda:0'),
        max_epochs=num_epochs,
        enable_progress_bar=False,
        logger=comet_logger,
        log_every_n_steps=100,
        callbacks=[
            TuneReportCheckpointCallback(
                metrics=["train_loss"], filename="trainer.ckpt", on="train_end"
            )
        ]
        # accelerator="gpu", 
        # devices=1,
        # plugins=[RayLightningEnvironment()],
        # gpus=num_gpus,
        # accelerator="gpu" if use_gpu else "cpu",
        # logger=CSVLogger("logs"),
    )
    # trainer = prepare_trainer(trainer)

    # Train model
    trainer.fit(model)


def tune_model(data_filepath):
    print(f"CUDA_VISIBLE_DEVICES is {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"Received {os.environ['HEAD_NODE']} from environment variable. Going to initialize Ray")
    # Connect to Ray cluster
    ray.init(address=f"{os.environ['ip_head']}",_node_ip_address=os.environ['HEAD_NODE'])
    
    # Check that Ray sees two nodes and their status is 'Alive'
    print("Nodes in the Ray cluster:")
    print(ray.nodes())

    # Check that Ray sees 12 CPUs and 2 GPUs over 2 Nodes
    print(ray.available_resources())
    
    # Defining hyperparam search space
    search_space = {
        "embedding_dim": tune.choice([64, 128, 256, 512]),
        "walk_length": tune.randint(5, 100),
        "batch_size": tune.choice([32, 64, 128, 256, 512]),
        "lr": tune.loguniform(1e-4, 1e-1),
    }

    # The maximum training epochs
    num_epochs = 500
    # Number of sampls from parameter space
    num_samples = 100
    scheduler = ASHAScheduler(
        metric="train_loss",
        mode="min",
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2,
    )

    trainable = tune.with_parameters(
            train_m2vec, data_filepath=data_filepath, num_epochs=num_epochs, num_gpus=1
        )
    # trainable = tune.with_resources(trainable, {"cpu": 4, "gpu": 1})
    
    # Specify each trial resources
    resource_group = tune.PlacementGroupFactory([
            {"CPU": 4, "GPU": 1}
        ])
    trainable = tune.with_resources(trainable, resource_group)
                                                
    analysis = tune.run(trainable,
        config=search_space,
        num_samples=2,
        scheduler=scheduler,
        local_dir="/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/06_models/monarch-experiments/lightning-ray",  # Directory where results are stored
        progress_reporter=tune.CLIReporter(
            metric_columns=["train_loss", "training_iteration"]
        ),  # Adjust the reported metrics if needed
        callbacks=[
            CometLoggerCallback(
                False, ["ray", "test", "parallelism"], project_name="modspy"
            )
        ]
        # resources_per_trial={"cpu": 4, "gpu": 1},
    )

    best_trial = analysis.get_best_trial()
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['train_loss']}")

    # #---------------------------
    # num_workers=1
    # use_gpu=1
    # scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

    # run_config = RunConfig(
    #     name="ray-ptl-monarch-m2vec",
    #     storage_path="/tmp/ray_results",
    #     checkpoint_config=CheckpointConfig(
    #         num_to_keep=3,
    #         checkpoint_score_attribute="train_loss",
    #         checkpoint_score_order="min",
    #     ),
    # )

    # trainer = TorchTrainer(
    #     train_m2vec,
    #     scaling_config=scaling_config,
    #     run_config=run_config,
    # )

    ###############################


# import pytorch_lightning as pl
# from ray.train import RunConfig, ScalingConfig, CheckpointConfig
# from ray.train.torch import TorchTrainer
# from ray.train.lightning import (
#     RayDDPStrategy,
#     RayLightningEnvironment,
#     RayTrainReportCallback,
#     prepare_trainer,
# )
# """
# def train_func_per_worker():
#     model = MNISTClassifier(lr=1e-3, feature_dim=128)
#     datamodule = MNISTDataModule(batch_size=128)

#     trainer = pl.Trainer(
#         devices="auto",
#         strategy=RayDDPStrategy(),
#         plugins=[RayLightningEnvironment()],
#         callbacks=[RayTrainReportCallback()],
#         max_epochs=10,
#         accelerator="gpu" if use_gpu else "cpu",
#         log_every_n_steps=100,
#         logger=CSVLogger("logs"),
#     )

#     trainer = prepare_trainer(trainer)

#     # Train model
#     trainer.fit(model, datamodule=datamodule)

#     # Evaluation on the test dataset
#     trainer.test(model, datamodule=datamodule)
# """
# scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)

# run_config = RunConfig(
#     name="ptl-mnist-example",
#     storage_path="/tmp/ray_results",
#     checkpoint_config=CheckpointConfig(
#         num_to_keep=3,
#         checkpoint_score_attribute="val_accuracy",
#         checkpoint_score_order="max",
#     ),
# )

# trainer = TorchTrainer(
#     train_func_per_worker,
#     scaling_config=scaling_config,
#     run_config=run_config,
# )


###############################


def main():
    # Ensure your data file path is correct
    data_filepath = "/home/rahit/projects/def-mtarailo/rahit/from_scratch/modspy-data/data/05_model_input/2024-02-monarch_heterodata_v1.pt"
    if os.path.exists(data_filepath):
        tune_model(data_filepath)
    else:
        print(f"Data file not found at {data_filepath}")


################################# CONSTRUCTION #####################################


# def main():

#     # Data file
#     data_filepath = './data/05_model_input/2024-02-monarch_heterodata_v1.pt'
#     # lighting_datamodule = LightningNodeData(data_filepath)
#     # tune_config = {
#     #     "lr": 1e-3,
#     #     "batch_size": tune.choice([32, 64]),
#     # }

#     # We will use Comet for logging
#     comet_logger = CometLogger(project_name="modspy-monarch",
#                                 save_dir="./data/06_models/monarch-lightning/comet",
#                                 rest_api_key='NpKZhU5E8qdb0tJdJtlzoUugy',  # Optional
#                                 experiment_name="lightning_logs",  # Optional
#                                 )

#     # Prepare your model
#     model = MetaPath2VecLightningModule(data_filepath)

#     strategy = pl.strategies.SingleDeviceStrategy('cuda:0')
#     # strategy = pl.strategies.SingleDeviceStrategy('cuda:0')
#     # Setup the trainer
#     # trainer = pl.Trainer(default_root_dir="./data/06_models/monarch-lightning", max_epochs=5,
#     #                     accelerator="gpu", strategy=strategy, devices=1, logger=comet_logger)
#     trainer = pl.Trainer(default_root_dir="./data/06_models/monarch-lightning", max_epochs=5,
#                         accelerator="gpu", strategy=strategy, devices=1, logger=comet_logger)

#     # Train the model
#     trainer.fit(model)


if __name__ == "__main__":
    main()
