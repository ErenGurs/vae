#
# Written by Eren Gurses - 1/14/2024
#
# VAE network. Mainly based on https://github.com/AntixK/PyTorch-VAE/models/vanilla_vae.py

import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self, latent_dim, patch_size):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim  # Latent variable dimension: 128
        self.patch_size = patch_size  # Patches are: 64x64
        hidden_dims = [32, 64, 128, 256, 512]
        self.final_dim = hidden_dims[-1]
        in_channels = 3
        modules = []

        # Legacy Fully Connected (FC) encoder
        #self.fc1 = nn.Linear(784, 400)
        #self.fc21 = nn.Linear(400, 20)
        #self.fc22 = nn.Linear(400, 20)
        #self.fc3 = nn.Linear(20, 400)
        #self.fc4 = nn.Linear(400, 784)

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

        # 5 conv layers defined by hidden_dims = [...] and each has stride=2, downscales input 64x64 to 2x2
        self.size_bottleneck = int(patch_size / pow(2, len(hidden_dims)))
        self.fc_mu = nn.Linear(hidden_dims[-1] * self.size_bottleneck * self.size_bottleneck, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * self.size_bottleneck * self.size_bottleneck, self.latent_dim)

        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1] * self.size_bottleneck * self.size_bottleneck)
        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())        

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, self.final_dim, self.size_bottleneck, self.size_bottleneck)
        result = self.decoder(result)
        result = self.final_layer(result)
        #result = celeb_transform1(result)
        #result = torch.flatten(result, start_dim=1)
        #result = torch.nan_to_num(result)
        return result

    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var