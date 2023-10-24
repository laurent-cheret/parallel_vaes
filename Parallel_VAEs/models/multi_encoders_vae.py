import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from .vae import VAE_encoder, VAE_decoder
from .nn_classifier import LightningClassifier




class MultiEncodersVAE(pl.LightningModule):
    def __init__(self, num_encoders,input_size, hidden_size, latent_size, num_classes):
        super(MultiEncodersVAE, self).__init__()

        # Define the encoders
        self.ENCODERS = nn.ModuleList([VAE_encoder(input_size, hidden_size, latent_size).cuda() for _ in range(num_encoders)])
        combined_latent_size = num_encoders*latent_size

        self.num_encoders = num_encoders
        self.cross_ent = nn.CrossEntropyLoss()

        self.classifier = LightningClassifier(combined_latent_size, num_classes)
        self.decoder = VAE_decoder(combined_latent_size, hidden_size, input_size).cuda()

    def forward(self, x):

        latents = []
        qz_params = []
        for encoder in self.ENCODERS:
            mu, logvar, z = encoder(x)
            qz_params.append([mu,logvar])
            latents.append(z)
        concat_latents = torch.cat(latents, dim=1)
        decoded = self.decoder(concat_latents)
        class_logits = self.classifier(concat_latents)


        return decoded, qz_params, latents, concat_latents, class_logits

    def training_step(self, batch, batch_idx):

        x, labels = batch

        decoded, qz_params, latents, concat_latents, class_logits = self(x)
        preds = torch.argmax(class_logits, dim=1)
        accuracy = (preds == labels).float().mean()

        total_loss, rc_loss, kl_loss ,mi_loss, cl_loss= self.compute_total_loss(decoded,x,qz_params, concat_latents, class_logits, labels)
        loss_dict = {'total_loss':total_loss,'rec_loss': rc_loss,'KL_Div':kl_loss,'Mutual_info':mi_loss, 'Classification_loss':cl_loss, 'Accuracy':accuracy}

        wandb.log(loss_dict)

        return total_loss

    def validation_step(self, batch, batch_idx):

        x, labels = batch
        decoded, qz_params, latents, concat_latents, class_logits = self(x)
        preds = torch.argmax(class_logits, dim=1)
        accuracy = (preds == labels).float().mean()

        total_loss, rc_loss, kl_loss, mi_loss, cl_loss= self.compute_total_loss(decoded,x,qz_params, concat_latents, class_logits, labels)
        loss_dict = {'val_total_loss':total_loss, 'val_rec_loss': rc_loss, 'val_KL_Div':kl_loss, 'val_Mutual_info':mi_loss, 'Val_classification_loss':cl_loss, 'Val_accuracy': accuracy}

        wandb.log(loss_dict)

        return total_loss


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def reconstruction_loss(self,recon_x, x):
        return F.mse_loss(recon_x, x)

    def kl_divergence(self,mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def classif_loss(self, class_logits, labels):

        CL = self.cross_ent(class_logits, labels)
        return CL

    def pairwise_correlation_loss(self,qz_params):
        # qz_params is a list of parameters (mu and logvar) for each VAE's latent space
        num_vaes = len(qz_params)
        pairwise_loss = 0.0

        for i in range(num_vaes):
            for j in range(i + 1, num_vaes):
                mu_i, logvar_i = qz_params[i]
                mu_j, logvar_j = qz_params[j]

                # Calculate the pairwise correlation
                correlation = torch.sum((mu_i - mu_j) ** 2) / (torch.sum(logvar_i.exp()) + torch.sum(logvar_j.exp()))

                # Add the pairwise correlation to the loss
                pairwise_loss += correlation

        return pairwise_loss

    def mutual_information_loss(self, qz_params):
      
        means = torch.stack([params[0] for params in qz_params], dim=1)  # Shape: [batch_size, num_vaes, latent_dim]
        log_variances = torch.stack([params[1] for params in qz_params], dim=1)  # Shape: [batch_size, num_vaes, latent_dim]

        # Calculate pair-wise differences between means and log_variances
        means_diff = means.unsqueeze(2) - means.unsqueeze(1)  # Shape: [batch_size, num_vaes, num_vaes, latent_dim]
        log_variances_diff = log_variances.unsqueeze(2) - log_variances.unsqueeze(1)

        # Calculate pair-wise squared distances
        squared_distance = (means_diff ** 2 + torch.exp(log_variances_diff) - log_variances_diff - 1) / 2
        # Shape: [batch_size, num_vaes, num_vaes, latent_dim]

        # Sum over the latent dimensions
        squared_distance = torch.sum(squared_distance, dim=-1)  # Shape: [batch_size, num_vaes, num_vaes]

        # Calculate the mutual information loss using broadcasting
        mi_loss = torch.mean(torch.exp(-squared_distance))

        return mi_loss


    def compute_total_loss(self,recon_x, x, qz_params, concat_latents, class_logits, labels):
        # Compute the reconstruction loss
        reconstruction_term = self.reconstruction_loss(recon_x, x)

        # Compute the KL divergence term of the concatenated latent variable, forcing it to be near gaussian
        concat_mean = torch.cat([mu for i, (mu, logvar) in enumerate(qz_params)], dim=1)
        concat_logvar = torch.cat([logvar for i, (mu, logvar) in enumerate(qz_params)], dim=1)

        kl_term = self.kl_divergence(concat_mean, concat_logvar)
        # Sum up the individual KL divergence terms


        mi_term = self.mutual_information_loss(qz_params)

        cl_term = self.classif_loss(class_logits, labels)

        # Calculate the total loss
        total_loss = reconstruction_term + 0.00001*kl_term + mi_term + cl_term

        return total_loss, reconstruction_term, kl_term, mi_term, cl_term