import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import pytorch_lightning as pl
import wandb

from .feature_extractors import VGG16FeatureExtractor, SmallCNNFeatureExtractor
from .multi_encoders_vae import MultiEncodersVAE

class VFE_classifier(pl.LightningModule):
    def __init__(self, num_vaes, input_dim, hidden_dim, latent_dim, num_classes):
        super(VFE_classifier, self).__init__()
        self.feat_extract = VGG16FeatureExtractor()
        # self.feat_extract = SmallCNNFeatureExtractor(3)
        self.vfe = MultiEncodersVAE(num_vaes, input_dim, hidden_dim, latent_dim, num_classes)


    def forward(self, x):
        # Forward pass of the model

        features = self.feat_extract(x)
        features = features.view(features.size(0), -1)  # Flatten the features
        # Forward pass through the classifier
        decoded, qz_params, latents, concat_latents, class_logits = self.vfe(features)
        return decoded,features,qz_params, concat_latents, class_logits

    def configure_optimizers(self):
        # Define the optimizer
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def training_step(self, batch, batch_idx):
        # Training step
        inputs, labels = batch
        decoded,x,qz_params, concat_latents, class_logits = self(inputs)
        preds = torch.argmax(class_logits, dim=1)
        accuracy = (preds == labels).float().mean()
        total_loss, rc_loss, kl_loss, mi_loss, cl_loss= self.vfe.compute_total_loss(decoded,x,qz_params, concat_latents, class_logits, labels)
        loss_dict = {'total_loss':total_loss, 'rec_loss': rc_loss, 'KL_Div':kl_loss, 'Mutual_info':mi_loss, 'classification_loss':cl_loss, 'accuracy': accuracy}

        wandb.log(loss_dict)
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Validation step
        inputs, labels = batch
        decoded,x,qz_params, concat_latents, class_logits = self(inputs)
        preds = torch.argmax(class_logits, dim=1)
        accuracy = (preds == labels).float().mean()
        total_loss, rc_loss, kl_loss, mi_loss, cl_loss= self.vfe.compute_total_loss(decoded,x,qz_params, concat_latents, class_logits, labels)
        loss_dict = {'val_total_loss':total_loss, 'val_rec_loss': rc_loss, 'val_KL_Div':kl_loss, 'val_Mutual_info':mi_loss, 'Val_classification_loss':cl_loss, 'Val_accuracy': accuracy}

        wandb.log(loss_dict)
        return total_loss

        return loss
