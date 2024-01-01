import random
from functools import partial
from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, MeanAbsoluteError, MeanSquaredError
from torchmetrics.functional import pairwise_euclidean_distance
from torchvision import models
from torchvision.models import mobilenet_v2
from torchvision.transforms import v2
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
from transformers.modeling_outputs import ModelOutput

SUPPORTED_TASKS = ["classification", "regression"]


class EarthQuakeModel(pl.LightningModule):
    def __init__(self, **hparams):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams["task"] not in SUPPORTED_TASKS:
            raise ValueError(
                f"Task {self.hparams['task']} not supported. Supported tasks are: {SUPPORTED_TASKS}"
            )

        num_classes = 2 if self.hparams["task"] == "classification" else 1

        if "timm" in self.hparams["model_name"]:
            self.model = timm.create_model(
                self.hparams["model_name"],
                pretrained=False,
                num_classes=num_classes,
                in_chans=self.hparams["in_chans"],
            )
        elif "atarinet" in self.hparams["model_name"]:
            self.model = AtariNet(self.hparams["in_chans"])
        else:
            config = AutoConfig.from_pretrained(self.hparams["model_name"])
            config.num_channels = self.hparams["in_chans"]
            config.num_labels = num_classes
            self.model = AutoModelForImageClassification.from_config(config)

        self.accuracy = Accuracy("multiclass", num_classes=2)
        self.regr_metric = MeanAbsoluteError()

        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
        self.train_transform = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        if hasattr(x, "logits"):
            x = x.logits
        return x.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams["lr"], weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.hparams["lr"],
            pct_start=0.1,
            cycle_momentum=False,
            div_factor=1e9,
            final_div_factor=1e4,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def training_step(self, batch, batch_idx):
        sample, label, mag = (batch["sample"], batch["label"], batch["magnitude"])

        sample = self.train_transform(sample)

        y_r = self(sample)

        loss = 0.0
        if self.hparams["task"] == "classification":
            loss = self.classification_loss(y_r, label)
        elif self.hparams["task"] == "regression":
            loss = self.regression_loss(y_r, mag)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        sample, label, mag = (batch["sample"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        loss = 0.0
        if self.hparams["task"] == "regression":
            loss = self.regression_loss(y_r, mag)

            self.accuracy((y_r >= 1).to(torch.int), label)
            self.log("val_acc", self.accuracy)
            self.regr_metric(y_r, mag)
            self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)
        elif self.hparams["task"] == "classification":
            loss = self.classification_loss(y_r, label)

            self.accuracy(y_r, label)
            self.log("val_acc", self.accuracy)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        sample, label, mag = (batch["sample"], batch["label"], batch["magnitude"])

        y_r = self(sample)

        if self.hparams["task"] == "regression":
            self.accuracy((y_r >= 1).to(torch.int), label)
            self.log("val_acc", self.accuracy)
            self.regr_metric(y_r, mag)
            self.log(f"val_{self.regr_metric.__class__.__name__}", self.regr_metric)
        elif self.hparams["task"] == "classification":
            self.accuracy(y_r, label)
            self.log("test_acc", self.accuracy)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = batch["sample"]
        y_r = self(sample)
        return y_r
