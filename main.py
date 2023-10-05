from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
import torch
import torch.nn as nn
from torchvision import transforms
import os
from torchvision.models import resnet18, resnet34, resnet101, resnet50
from torch.optim import SGD, Adam, adamw
from torchmetrics import Accuracy
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class Model(pl.LightningModule):
  def __init__(self,
               num_classes,
               model,
               train_dir,
               val_dir,
               test_dir = None,
               image_size = (128, 128),
               optimizer = "Adam",
               lr = 1e-3,
               batch_size = 16,
               transfer = True):
    super().__init__()

    self.num_classes = num_classes
    self.train_dir = train_dir
    self.val_dir = val_dir
    self.test_dir = test_dir
    self.lr = lr
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.img_size = image_size


    # instantiate loss criterion
    self.loss_fn = (nn.BCEWithLogitsLoss() if self.num_classes == 1 else nn.CrossEntropyLoss())

    self.acc = Accuracy(task='binary' if self.num_classes == 1 else "multiclass", num_classes = self.num_classes)

    # instantiate the models

    self.model = model(pretrained=True)
    linear_size = list(self.model.children())[-1].in_features
    self.model.fc = nn.Linear(linear_size, self.num_classes)

  def forward(self, x):
    return self.model(x)

  def configure_optimizers(self):
    return self.optimizer(self.parameters(), lr=self.lr)

  def _step(self, batch):
    x, y = batch
    preds = self(x)

    if self.num_classes == 1:
      preds = preds.flatten()
      y = y.float()

    loss = self.loss_fn(preds, y)
    acc = self.acc(preds, y)
    return loss, acc

  def _dataloader(self, data_dir, shuffle=False):
    transform = transforms.Compose(
        [
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.48232,), (0.23051,)),
            transforms.RandomHorizontalFlip()
        ]
    )

    img_folder = ImageFolder(data_dir, transform=transform)

    return DataLoader(img_folder, batch_size=self.batch_size, shuffle=shuffle)

  def train_dataloader(self):
    return self._dataloader(self.train_dir, shuffle=True)

  def training_step(self, batch, batch_indx):
    loss, acc = self._step(batch)

    self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    return loss

  def val_dataloader(self):
    return self._dataloader(self.val_dir, shuffle=False)


  def validation_step(self, batch, batch_indx):

    loss, acc = self._step(batch)

    self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
    self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    return loss

  def test_dataloader(self):
    return self._dataloader(self.test_dir)

  def test_step(self, batch, batch_indx):
    loss, acc = self._step(batch)

    self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)


if __name__ == "__main__":
  model = Model(
        num_classes=36,
        model=resnet18,
        train_dir="dataset/train",
        val_dir="dataset/val",
        test_dir="dataset/test",
        lr=1e-4,
        batch_size=16,
        transfer=True,
    )
  
