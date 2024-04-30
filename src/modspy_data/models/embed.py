from comet_ml import Experiment


import os
import random
from typing import List, Optional, Tuple, Union, cast

from traitlets import default
from typing_extensions import Annotated

import torch
import torchmetrics
from torch_geometric.transforms import AddSelfLoops
import optuna
from pytorch_lightning.loggers import CometLogger

# import wandb
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.typing import EdgeType
from torch_geometric.data import HeteroData
from torch_geometric.data.lightning.datamodule import LightningNodeData

from torch.nn.functional import cosine_similarity
from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import accuracy_score



##########################
# POSSIBLE METAPATH SCHEME
##########################
METAPATH_SCHEME = [
        ("biolink:Gene", "biolink:orthologous_to", "biolink:Gene"),
        ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
#     ]
#     [
        ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
        ("biolink:Gene", "biolink:orthologous_to", "biolink:Gene"),
        ("biolink:Gene", "biolink:interacts_with", "biolink:Gene"),
    # ],
    # [
        ("biolink:Gene", "biolink:participates_in", "biolink:Pathway"),
        ("biolink:Pathway", "biolink:participates_in", "biolink:Gene"),
    # ],
    # [
        ("biolink:Gene", "biolink:has_phenotype", "biolink:PhenotypicFeature"),
        ("biolink:PhenotypicFeature", "biolink:subclass_of", "biolink:PhenotypicFeature"),
        ("biolink:PhenotypicFeature", "biolink:has_phenotype", "biolink:Disease"),
        ("biolink:Disease", "biolink:has_phenotype", "biolink:PhenotypicFeature"),
        ("biolink:PhenotypicFeature", "biolink:has_phenotype", "biolink:Gene"),
    # ],
    # [
        ("biolink:Gene", "biolink:enables", "biolink:BiologicalProcessOrActivity"),
        ("biolink:BiologicalProcessOrActivity", "biolink:subclass_of", "biolink:BiologicalProcessOrActivity"),
        ("biolink:BiologicalProcessOrActivity", "biolink:actively_involved_in", "biolink:Gene"),
    # ],
]


class ModifierDataset(Dataset):
    def __init__(self, filepath: str = ""):
        """
        Args:
            data (Tensor): A tensor containing node pairs and their similarity label.
                           Shape: [num_pairs, 3], where each row is (node1, node2, label).
        """
        self.data = torch.load(filepath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        modifier, target, label = self.data[idx]
        return modifier, target, label



class MetaPath2VecLightningModule(pl.LightningModule):
    def __init__(
        self,
        network_filepath: str = "",
        data_filepath: str = "",
        embedding_dim=128,
        walk_length=50,
        context_size=7,
        walks_per_node=5,
        num_negative_samples=5,
        lr=0.01,
        batch_size=128,
        metapath=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Loading Data
        self.graph = torch.load(network_filepath)
        self.dataset = ModifierDataset(data_filepath)
        self.predictions = []
        self.ground_truths = []
        if self.hparams.metapath != None:
            self.metapath = self.hparams.metapath
        else:
            self.metapath = METAPATH_SCHEME

        print(f"Computing on {self.device}")
        print(torch.cuda.is_available())  # Should print True if CUDA is available
        print(torch.cuda.device_count())  # Should print the number of GPUs available
        print(torch.cuda.get_device_name(0))  # Should print the name of the first GPU
        print(
            f""" 
        Metapath: {self.metapath}
        Total nodes: {self.graph.num_nodes}
        Total node types: {len(self.graph.node_types)}

        Total edges: {self.graph.num_edges}
        Total edge types: {len(self.graph.edge_types)}                
        """
        )

        self.edge_types_to_reverse = [
            ("biolink:Gene", "biolink:acts_upstream_of_or_within", "biolink:BiologicalProcessOrActivity"),
            ("biolink:Gene", "biolink:actively_involved_in", "biolink:BiologicalProcessOrActivity"),
            ("biolink:Gene", "biolink:participates_in", "biolink:Pathway"),
            ("biolink:Disease", "biolink:has_phenotype", "biolink:PhenotypicFeature"),
            ('biolink:Gene', 'biolink:has_phenotype', 'biolink:PhenotypicFeature'),
            ("biolink:Gene", "biolink:gene_associated_with_condition", "biolink:Disease"),
            ("biolink:Gene", "biolink:expressed_in", "biolink:CellularComponent"),
        ]
        for _etype in self.edge_types_to_reverse:
            self.graph = self.reverse_edges(self.graph, _etype)

        # Adding self loops to avoid 1. nodes without any edge, 2. consider intragenic modifier
        transform = AddSelfLoops()
        self.graph = transform(self.graph)

        # Model definition
        self.model = MetaPath2Vec(
            self.graph.edge_index_dict,
            embedding_dim=self.hparams.embedding_dim,
            metapath=self.metapath,
            walk_length=self.hparams.walk_length,
            context_size=self.hparams.context_size,
            walks_per_node=self.hparams.walks_per_node,
            num_negative_samples=self.hparams.num_negative_samples,
            sparse=True,
        )

        # # Define the binary classifier layer
        # self.binary_classifier = nn.Linear(self.hparams.embedding_dim * 2, 1)

        # Permformance
        self.val_precision = torchmetrics.Precision(task="binary")
        self.val_recall = torchmetrics.Recall(task="binary")

    # Flip/Reverse source <-> target
    def reverse_edges(self, data: HeteroData, edge_type: EdgeType):
        rev_edge_index = data.edge_index_dict[edge_type].flip([0])
        data[(edge_type[2], edge_type[1], edge_type[0])].edge_index = rev_edge_index
        del data.edge_index_dict[edge_type]
        return data

    def train_dataloader(self):
        loader = self.model.loader(
            batch_size=self.hparams.batch_size, shuffle=True, num_workers=30
        )
        return loader

    #     train_batch_size = len(self.dataset)
    #     # self.log('val_batch_size', batch_size=val_batch_size)
    #     val_loader = DataLoader(
    #         self.dataset, batch_size=val_batch_size, shuffle=False
    #     )
    #     return loader

    def val_dataloader(self):
        val_batch_size = len(self.dataset)
        # self.log('val_batch_size', batch_size=val_batch_size)
        val_loader = DataLoader(self.dataset, batch_size=val_batch_size, shuffle=False, num_workers=30)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer

    # def configure_optimizers(self):
    #     # Create an optimizer that will optimize both the MetaPath2Vec model
    #     # and the binary classifier layer parameters
    #     # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.hparams.lr)
    #     # optimizer = torch.optim.SparseAdam(list(self.parameters()), lr=self.hparams.lr)
    #     return optimizer

    # def training_step(self, batch, batch_idx):
    #     pos_rw, neg_rw = batch
    #     loss = self.model.loss(pos_rw, neg_rw)
    #     self.log(
    #         "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
    #     )
    #     return loss
    
    def training_step(self, batch, batch_idx):
        # print(batch[0].device)
        pos_rw, neg_rw = batch
        # print(f"Device pos_rw: {pos_rw.device}, neg_rw: {neg_rw.device}")
        # pos_rw = pos_rw.to(self.device)  # Ensure tensors are on the correct device
        # neg_rw = neg_rw.to(self.device)
        loss = self.model.loss(pos_rw, neg_rw)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )
        return loss

    # def training_step(self, batch, batch_idx):
    #     node1, node2, label = batch
    #     node1_embedding = self.model.embedding(node1)
    #     node2_embedding = self.model.embedding(node2)
    #     combined_embedding = torch.cat([node1_embedding, node2_embedding], dim=1)
    #     logits = self.binary_classifier(combined_embedding)
    #     loss = BCEWithLogitsLoss()(logits, label.unsqueeze(1).float())
    #     self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    #     return loss

    def verify_format(self, predictions, ground_truths, num_classes=None):
        # Check shape compatibility
        assert predictions.dim() == ground_truths.dim() == 1, "Both tensors must be 1D"
        assert predictions.size(0) == ground_truths.size(0), "Mismatched batch sizes"

        # Check data type
        assert (
            predictions.dtype == torch.int32
        ), f"Predictions must be of type torch.int32. Found {predictions.dtype}"
        assert (
            ground_truths.dtype == torch.int32
        ), f"Ground truths must be of type torch.int32. Found {ground_truths.dtype}"

        # If num_classes is provided, check that class indices are valid
        if num_classes is not None:
            assert (
                predictions.max() < num_classes
            ), "Prediction contains invalid class indices"
            assert (
                ground_truths.max() < num_classes
            ), "Ground truth contains invalid class indices"

        print("Predictions and ground truths are in compatible format.")

    # Optional: Define a validation step
    def validation_step(self, batch, batch_idx):
        # Implement your validation logic here, possibly using a different dataset
        # and returning some metric of interest, e.g., accuracy.
        # print(f"batch {batch_idx}")
        # print(batch)
        # print(batch[0])
        # print("^^batch^^^")

        # Assuming 'batch' contains tuples of (node1, node2, ground_truth_similarity)
        node1_indices, node2_indices, ground_truth = batch

        # Get embeddings for node pairs
        node1_embeddings = self.model.embedding(
            node1_indices
        )  # You need to adjust this based on how your model stores embeddings
        node2_embeddings = self.model.embedding(node2_indices)
        # print("Printing Embeddings:")
        # print(node1_embeddings)
        # print(node2_embeddings)
        # print("^^^^^^^^^^^^^")
        # Compute cosine similarity between each pair of nodes
        similarities = cosine_similarity(node1_embeddings, node2_embeddings)

        # Apply threshold to decide if nodes are similar (1) or not (0)
        similarity_threshold = 0.5  # You might need to adjust this based on your validation set and similarity distribution
        predictions = (similarities >= similarity_threshold).long()
        # print(f"Predictions: {predictions}")
        # Compute accuracy or any other suitable metric
        accuracy = accuracy_score(ground_truth.cpu().numpy(), predictions.cpu().numpy())

        # Log validation accuracy
        self.log(
            "val_accuracy",
            torch.tensor(accuracy),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        predicted_labels = predictions.int().detach().cpu()
        ground_truths = ground_truth.int().detach().cpu()
        # Assuming num_classes is defined
        # self.verify_format(predicted_labels, ground_truths, num_classes=2)
        self.predictions.append(predicted_labels)
        self.ground_truths.append(ground_truths)

        # Convert tensors to lists or numpy arrays for logging
        predictions_list = predicted_labels.tolist()
        ground_truths_list = ground_truth.tolist()

        # Log predictions and ground truths
        self.logger.experiment.log_text(
            f"predictions: {predictions_list}", metadata={"batch": batch_idx}
        )
        self.logger.experiment.log_text(
            f"ground_truths: {ground_truths_list}", metadata={"batch": batch_idx}
        )

        # # Log embeddings and labels to Comet ML
        # # [only available for online experiments atm]
        # if batch_idx == 0:  # Optionally, you might want to log only for the first batch or a specific batch
        #     experiment = self.logger.experiment  # Assuming self.logger is your Comet ML logger
        #     experiment.log_embedding(node1_embeddings.cpu().detach().numpy(),
        #                             labels=ground_truth.cpu().tolist(),
        #                             title="Node1 Embeddings Validation")
        #     experiment.log_embedding(node2_embeddings.cpu().detach().numpy(),
        #                             labels=ground_truth.cpu().tolist(),
        #                             title="Node2 Embeddings Validation")

        ####### PR ########
        # Update precision and recall metrics
        self.val_precision(predictions, ground_truth.int())
        self.val_recall(predictions, ground_truth.int())
        # Log precision and recall
        self.log(
            "val_precision",
            self.val_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=False
        )
        ##################

        return {
            "val_accuracy": torch.tensor(accuracy),
            "val_recall": self.val_recall,
            "val_precision": self.val_precision,
        }

    # Optional: If your 'test' function is essential, integrate it here similarly to validation_step
    # def test_step(self, batch, batch_idx):
    #     pass

    def on_validation_epoch_end(self):
        predictions = torch.cat(self.predictions, dim=0)
        ground_truths = torch.cat(self.ground_truths, dim=0)

        # Log the confusion matrix to Comet ML
        self.logger.experiment.log_confusion_matrix(
            ground_truths,
            predictions,
            title="Confusion Matrix",
            step=self.current_epoch,
        )

        # Compute and log the average precision and recall over all validation batches
        avg_val_precision = self.val_precision.compute()
        avg_val_recall = self.val_recall.compute()

        self.log("avg_val_precision", avg_val_precision, prog_bar=False)
        self.log("avg_val_recall", avg_val_recall, prog_bar=False)

        # Reset the metrics after each epoch
        self.val_precision.reset()
        self.val_recall.reset()

        # Clear the lists for the next epoch
        self.predictions = []
        self.ground_truths = []

        return {
            "avg_val_precision": avg_val_precision,
            "avg_val_recall": avg_val_recall,
        }


# RotatE GNN based on PyG implementation wrapped by Pytorch Lightning

