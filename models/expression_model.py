import copy
from collections import defaultdict

import torch
import torchmetrics
from torch import nn, optim
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from models.DSN_model import ResNet, Reshape, DecoderFC, DecoderConv
from util import SIMSE


class ExpressionModel(pl.LightningModule):
    def __init__(
            self,
            code_size=100,
            n_class=12,
            AUs=[1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24],
            alpha=0.01
    ):
        super(ExpressionModel, self).__init__()
        self.classes = n_class
        self.alpha = alpha
        self.AUs = AUs

        ################################
        # expression encoder
        ################################

        self.expression_encoder = nn.Sequential(
            ResNet(),
            nn.Linear(in_features=512, out_features=code_size),
            nn.ReLU(True)
        )

        ################################
        # classifier
        ################################

        self.classifier = nn.Sequential(
            nn.Linear(in_features=code_size, out_features=100),
            nn.ReLU(True),
            nn.Linear(in_features=100, out_features=n_class)
        )

        ######################################
        # shared decoder
        ######################################

        self.shared_decoder = nn.Sequential(
            DecoderFC(),
            DecoderConv()
        )

        self.train_f1 = torchmetrics.F1(self.classes)
        self.val_f1 = torchmetrics.F1(self.classes)
        self.test_f1 = torchmetrics.F1(self.classes)

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

        self.AU_acc = {
            au: torchmetrics.Accuracy().cuda() for au in AUs
        }
        self.AU_f1 = {
            au: torchmetrics.F1(1).cuda() for au in AUs
        }

    def forward(self, input_data):
        expression_code = self.expression_encoder(input_data)
        class_pred = self.classifier(expression_code)
        reconstruction = self.shared_decoder(expression_code)

        return expression_code, class_pred, reconstruction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_epoch_end(self, outputs) -> None:
        self.log("train_f1", self.train_f1.compute())
        self.log("train_acc", self.train_acc.compute())

    def training_step(self, batch, batch_idx):
        imgs, labels = batch

        pred_loss = nn.BCEWithLogitsLoss()
        recon_loss = SIMSE()

        expression_code, class_pred, reconstruction = self(imgs)

        pred_loss = pred_loss(class_pred, labels)
        recon_loss = recon_loss(reconstruction, imgs) * self.alpha
        loss = pred_loss + recon_loss

        class_pred = torch.sigmoid(class_pred)

        self.train_f1(class_pred, labels)
        self.train_acc(class_pred, labels.int())

        self.log_dict(
            {
                "train_pred_loss": pred_loss.float(),
                "train_recon_loss": recon_loss.float(),
                "train_loss": loss.float()
            },
            on_step=False,
            on_epoch=True
        )

        return loss

    def validation_epoch_end(self, outputs) -> None:
        self.log("val_f1", self.val_f1.compute())
        self.log("val_acc", self.val_acc.compute())

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch

        pred_loss = nn.BCEWithLogitsLoss()
        recon_loss = SIMSE()

        expression_code, class_pred, reconstruction = self(imgs)

        pred_loss = pred_loss(class_pred, labels)
        recon_loss = recon_loss(reconstruction, imgs) * self.alpha
        loss = pred_loss + recon_loss

        class_pred = torch.sigmoid(class_pred)

        self.val_f1(class_pred, labels)
        self.val_acc(class_pred, labels.int())

        self.log_dict(
            {
                "val_pred_loss": pred_loss.float(),
                "val_recon_loss": recon_loss.float(),
                "val_loss": loss.float()
            },
            on_step=False,
            on_epoch=True
        )

        return loss

    def test_epoch_end(self, outputs) -> None:
        self.log("test_f1", self.test_f1.compute())
        self.log("test_acc", self.test_acc.compute())

        for j, au in enumerate(self.AUs):
            self.log(f"test_AU{au}_f1", self.AU_f1[au].compute())
            self.log(f"test_AU{au}_acc", self.AU_acc[au].compute())

    def test_step(self, batch, batch_idx):
        imgs, labels = batch

        pred_loss = nn.BCEWithLogitsLoss()
        recon_loss = SIMSE()

        expression_code, class_pred, reconstruction = self(imgs)

        pred_loss = pred_loss(class_pred, labels)
        recon_loss = recon_loss(reconstruction, imgs) * self.alpha
        loss = pred_loss + recon_loss

        class_pred = torch.sigmoid(class_pred)

        self.test_f1(class_pred, labels)
        self.test_acc(class_pred, labels.int())

        for j, au in enumerate(self.AUs):
            self.AU_f1[au](class_pred[:, j], labels[:, j])
            self.AU_acc[au](class_pred[:, j], labels[:, j].int())

        self.log_dict(
            {
                "test_pred_loss": pred_loss.float(),
                "test_recon_loss": recon_loss.float(),
                "test_loss": loss.float()
            },
            on_step=False,
            on_epoch=True
        )

        return loss