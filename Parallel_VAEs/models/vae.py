# models/vae.py

import torch
import torch.nn as nn

class VAE_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE_encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.3)
        self.input_dim = input_dim

    def encode(self, x):
        h1 = self.bn1(self.dropout(self.fc1(x)))
        return self.fc2_mean(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_dim))
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


class VAE_decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAE_decoder, self).__init__()

        self.fc3 = nn.Linear(latent_dim, 2*latent_dim)
        self.fc4 = nn.Linear(2*latent_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        # self.dropout = nn.Dropout(0.1)

    def decode(self, z):
        h3 = self.fc3(z)
        return self.fc4(h3)

    def forward(self, x):
        return self.decode(x)
