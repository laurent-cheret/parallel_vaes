# models/lightning_classifier.py

import torch
import torch.nn as nn
import pytorch_lightning as pl

class LightningClassifier(pl.LightningModule):
    def __init__(self, latent_size, num_classes):
        super(LightningClassifier, self).__init__()

        self.bn = nn.BatchNorm1d(latent_size)
        self.fc1 = nn.Linear(latent_size, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
