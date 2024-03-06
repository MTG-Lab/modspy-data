from comet_ml import Experiment
import torch
from torch_geometric.transforms import AddSelfLoops
import optuna
from pytorch_lightning.loggers import CometLogger
# import wandb
import pytorch_lightning as pl
from torch_geometric.nn import MetaPath2Vec



class MetaPath2VecLightningModule(pl.LightningModule):
    def __init__(self, data_filepath, embedding_dim=128, walk_length=50, 
                 context_size=7, walks_per_node=5, num_negative_samples=5):
        super().__init__()
        self.save_hyperparameters()
        
        # Loading Data
        self.data = torch.load(data_filepath)
        print(f""" 
        Total nodes: {self.data.num_nodes}
        Total node types: {len(self.data.node_types)}

        Total edges: {self.data.num_edges}
        Total edge types: {len(self.data.edge_types)}                
        """)
        # Adding self loops to avoid 1. nodes without any edge, 2. consider intragenic modifier
        transform = AddSelfLoops()
        self.data = transform(self.data)

        
        # Defining metapath to explore
        self.metapaths = [
            ('biolink:Gene', 'biolink:orthologous_to', 'biolink:Gene'),
            ('biolink:Gene', 'biolink:interacts_with', 'biolink:Gene'),
        ]
        
        # Model definition
        self.model = MetaPath2Vec(self.data.edge_index_dict, embedding_dim=self.hparams.embedding_dim,
                                  metapath=self.metapaths, walk_length=self.hparams.walk_length, 
                                  context_size=self.hparams.context_size,
                                  walks_per_node=self.hparams.walks_per_node, 
                                  num_negative_samples=self.hparams.num_negative_samples,
                                  sparse=True).to(self.device)

    def train_dataloader(self):
        loader = self.model.loader(batch_size=128, shuffle=True, num_workers=1)
        return loader

    def configure_optimizers(self):
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.01)
        return optimizer

    def training_step(self, batch, batch_idx):
        pos_rw, neg_rw = batch
        pos_rw = pos_rw.to(self.device)  # Ensure tensors are on the correct device
        neg_rw = neg_rw.to(self.device)
        loss = self.model.loss(pos_rw, neg_rw)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
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

## Adding Ray tune to fine tune model


# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.train import RunConfig, ScalingConfig, CheckpointConfig
# from ray.train.torch import TorchTrainer


# num_epochs = 100

# scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
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

# # Define a TorchTrainer without hyper-parameters for Tuner
# ray_trainer = TorchTrainer(
#     train_func,
#     scaling_config=scaling_config,
#     run_config=run_config,
# )

# def tune_mnist_asha(num_samples=10):
#     scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

#     tuner = tune.Tuner(
#         ray_trainer,
#         param_space={"train_loop_config": search_space},
#         tune_config=tune.TuneConfig(
#             metric="ptl/val_accuracy",
#             mode="max",
#             num_samples=num_samples,
#             scheduler=scheduler,
#         ),
#     )
#     return tuner.fit()

# results = tune_mnist_asha(num_samples=num_samples)

################################# CONSTRUCTION #####################################




def main():

    # Data file
    data_filepath = './data/05_model_input/2024-02-monarch_heterodata_v1.pt'
    # tune_config = {
    #     "lr": 1e-3,
    #     "batch_size": tune.choice([32, 64]),
    # }  

    # We will use Comet for logging
    comet_logger = CometLogger(project_name="modspy-monarch",
                                save_dir="./data/06_models/monarch-lightning/comet",
                                rest_api_key='NpKZhU5E8qdb0tJdJtlzoUugy',  # Optional
                                experiment_name="lightning_logs",  # Optional
                                )

    # Prepare your model
    model = MetaPath2VecLightningModule(data_filepath)

    strategy = pl.strategies.SingleDeviceStrategy('cuda:0')
    # Setup the trainer
    trainer = pl.Trainer(default_root_dir="./data/06_models/monarch-lightning", max_epochs=5, 
                        accelerator="gpu", strategy=strategy, devices=1, logger=comet_logger)

    # Train the model
    trainer.fit(model)


if __name__ == "__main__":
    main()