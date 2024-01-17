# Written by Eren Gurses - 1/12/2024
#
# For training the VAE model in vae.py
#
# 

import argparse
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from vae import VAE

parser = argparse.ArgumentParser(description='VAE Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='Use cpu for training (default is CUDA)')
                          
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")   
else:
    device = torch.device("cuda")


PATCH_SIZE = 64
LATENT_DIM = 128

#
# Prepare Data
#
# transforms applied
celeb_transform = transforms.Compose([
    transforms.CenterCrop(148),
    transforms.Resize(PATCH_SIZE),
    transforms.ToTensor(),])  # used when transforming image to tensor

kwargs = {'num_workers': 1, 'pin_memory': True} if not args.cpu else {}
train_loader = torch.utils.data.DataLoader(
    #datasets.MNIST('../data', train=True, download=True,
    datasets.CelebA('./data/', split='train', download=False,transform=celeb_transform), # transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CelebA('./data/', split='test', transform=celeb_transform), # transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=False, **kwargs)




model = VAE(LATENT_DIM, PATCH_SIZE).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3) #0.005)

# Define loss: Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var):
    MSE =F.mse_loss(recon_x, x) # .view(-1, image_dim)
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    #KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
    kld_weight = 0.00025  #0.025
    loss = MSE + kld_weight * KLD  
    return loss


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # Run the model
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        
        optimizer.step()
        #if batch_idx % args.log_interval == 0:
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch[:n]])
                                      #recon_batch.view(args.batch_size, 3, PATCH_SIZE, PATCH_SIZE)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        torch.save(model, f'results.model/vae_model_{epoch}.pth')
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, LATENT_DIM).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 3, PATCH_SIZE, PATCH_SIZE),
                       'results/sample_' + str(epoch) + '.png')


