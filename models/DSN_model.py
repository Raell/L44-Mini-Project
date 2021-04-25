import argparse
import json
from collections import defaultdict
from typing import Union, Dict, Optional, Any

import torch.nn as nn
import torch
import pickle
import torchmetrics
from pytorch_lightning.loggers import LightningLoggerBase
import re

from torchvision import models

from data import AU_CROP
from util import SIMSE, DiffLoss
import pytorch_lightning as pl


class MetricsLogger(LightningLoggerBase):

    def __init__(self, log_file):
        super().__init__()
        self.filename = log_file
        self.metrics = {
            "train": defaultdict(list),
            "val": defaultdict(list),
            "test": defaultdict(list)
        }

    @property
    def experiment(self) -> Any:
        pass

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        for k, v in metrics.items():
            match = re.match("^(train|val|test)_(.*)$", k)
            if match:
                self.metrics[match.group(1)][match.group(2)].append(v)

    def save_file(self):
        with open(self.filename, "w") as fp:
            json.dump(self.metrics, fp, indent="\t")

    def log_hyperparams(self, params: argparse.Namespace, *args, **kwargs):
        pass

    @property
    def name(self) -> str:
        return "MetricsLogger"

    @property
    def version(self) -> Union[int, str]:
        pass




class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class DecoderFC(nn.Module):
    def __init__(self):
        super(DecoderFC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 6 * 6),
            nn.ReLU(True),
            nn.BatchNorm1d(64 * 6 * 6),
            nn.ReLU(True),
            Reshape(-1, 64, 6, 6)
        )

    def forward(self, x):
        return self.fc(x)


class DecoderConv(nn.Module):
    def __init__(self, img_type="full"):
        super(DecoderConv, self).__init__()
        if img_type == "upper":
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 96, (1, 3), padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(96, 128, (1, 3), padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, (1, 5), padding=1),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 3, 12, stride=2, padding=5)
            )
        elif img_type == "lower":
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 96, (1, 3), padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(96, 128, (1, 3), padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, (3, 5), padding=1),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 3, 12, stride=2, padding=5)
            )
        else:
            self.conv = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 96, 3, padding=0),
                nn.ReLU(True),
                nn.ConvTranspose2d(96, 128, 3, padding=1),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 5, padding=1),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose2d(64, 3, 12, stride=2, padding=5)
            )

    def forward(self, x):
        return self.conv(x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=8631)

        modules = list(self.model.children())[:-1] + [torch.nn.Flatten()]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class VGGFace2ResNet(nn.Module):
    def __init__(self, use_weights=True):
        super(VGGFace2ResNet, self).__init__()
        self.model = models.resnet50(pretrained=False, num_classes=8631)

        if use_weights:
            with open("models/resnet50_ft_weight.pkl", 'rb') as f:
                weights = pickle.load(f, encoding='latin1')
            own_state = self.model.state_dict()
            for name, param in weights.items():
                if name in own_state:
                    own_state[name].copy_(torch.from_numpy(param))

        modules = list(self.model.children())[:-1] + [torch.nn.Flatten()]
        self.model = nn.Sequential(*modules)

        if use_weights:
            for param in self.parameters():
                param.requires_grad = False


    def forward(self, x):
        return self.model(x)


class DSN(pl.LightningModule):
    def __init__(
            self,
            code_size=100,
            use_vgg=True,
            AUs=[1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24],
            alpha=0.01,
            beta=0.05
    ):
        super(DSN, self).__init__()
        self.classes = len(AUs)
        self.alpha = alpha
        self.beta = beta
        self.AUs = AUs

        if len(AUs) == 1:
            self.crop = AU_CROP[AUs[0]]
        else:
            self.crop = "full"

        ##########################################
        # person-specific encoder
        ##########################################

        encoder = VGGFace2ResNet(use_vgg)
        self.person_encoder = nn.Sequential(
            encoder,
            nn.Linear(in_features=2048, out_features=code_size),
            nn.ReLU(True)
        )

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
            nn.Linear(in_features=100, out_features=self.classes)
        )

        ######################################
        # shared decoder
        ######################################

        self.shared_decoder = nn.Sequential(
            DecoderFC(),
            DecoderConv(self.crop)
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

    def forward(self, input_data, rec_scheme='all'):
        person_code = self.person_encoder(input_data)
        expression_code = self.expression_encoder(input_data)
        class_pred = self.classifier(expression_code)

        if rec_scheme == 'expression':
            union_code = expression_code
        elif rec_scheme == 'all':
            union_code = person_code + expression_code

        reconstruction = self.shared_decoder(union_code)

        return person_code, expression_code, class_pred, reconstruction

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
        diff_loss = DiffLoss()

        person_code, expression_code, class_pred, reconstruction = self(imgs)

        pred_loss = pred_loss(class_pred, labels)
        recon_loss = recon_loss(reconstruction, imgs) * self.alpha
        diff_loss = diff_loss(person_code, expression_code) * self.beta
        loss = pred_loss + recon_loss + diff_loss

        class_pred = torch.sigmoid(class_pred)

        self.train_f1(class_pred, labels)
        self.train_acc(class_pred, labels.int())

        self.log_dict(
            {
                "train_pred_loss": pred_loss.float(),
                "train_recon_loss": recon_loss.float(),
                "train_diff_loss": diff_loss.float(),
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
        diff_loss = DiffLoss()

        person_code, expression_code, class_pred, reconstruction = self(imgs)

        pred_loss = pred_loss(class_pred, labels)
        recon_loss = recon_loss(reconstruction, imgs) * self.alpha
        diff_loss = diff_loss(person_code, expression_code) * self.beta
        loss = pred_loss + recon_loss + diff_loss

        class_pred = torch.sigmoid(class_pred)

        self.val_f1(class_pred, labels)
        self.val_acc(class_pred, labels.int())

        self.log_dict(
            {
                "val_pred_loss": pred_loss.float(),
                "val_recon_loss": recon_loss.float(),
                "val_diff_loss": diff_loss.float(),
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
        diff_loss = DiffLoss()

        person_code, expression_code, class_pred, reconstruction = self(imgs)

        pred_loss = pred_loss(class_pred, labels)
        recon_loss = recon_loss(reconstruction, imgs) * self.alpha
        diff_loss = diff_loss(person_code, expression_code) * self.beta
        loss = pred_loss + recon_loss + diff_loss

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
                "test_diff_loss": diff_loss.float(),
                "test_loss": loss.float()
            },
            on_step=False,
            on_epoch=True
        )

        return loss