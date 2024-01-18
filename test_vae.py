#
# Written by Eren Gurses - 1/14/2024
#
# For testing the model trained by train_vae.py (epoch#50 weights: checkpoints/vae_model_50_kld_weight0.025.pth)
#

import argparse
import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.image as mpimg
import numpy as np
#from PIL import Image
from moviepy.editor import ImageSequenceClip

parser = argparse.ArgumentParser(description='VAE Inference Example')
parser.add_argument('--cpu', action='store_true', default=False,
                    help='Use cpu for inference (default is CUDA)')
                          
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")   
else:
    device = torch.device("cuda")


PATCH_SIZE = 64
MODEL_FILE = 'checkpoints/vae_model_50_kld_weight0.00025.pth'
#MODEL_FILE = 'checkpoints/vae_model_50_kld_weight0.025.pth'

# transforms applied
celeb_transform = transforms.Compose([
    transforms.CenterCrop(148),
    transforms.Resize(PATCH_SIZE),
    transforms.ToTensor(),])  # used when transforming image to tensor

dataset = datasets.CelebA('data/', split='all', download=True, transform=celeb_transform)
loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True) #False)
model = torch.load(MODEL_FILE, map_location=device)
#model = VAE(LATENT_DIM, PATCH_SIZE).to(device)


def interpolate(vae, x_1, x_2, n=12):
    with torch.no_grad():
        mu, log_var = vae.encode(x_1)
        z_1 = vae.reparameterize(mu, log_var)
        mu, log_var = vae.encode(x_2)
        z_2 = vae.reparameterize(mu, log_var)
        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
        interpolate_list = vae.decode(z)
        interpolate_list = torch.cat((x_1, interpolate_list), dim=0)
        interpolate_list = torch.cat((interpolate_list, x_2), dim=0)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()



    w = PATCH_SIZE
    img = np.zeros((3, w, (n+2)*w))
    for i, x_hat in enumerate(interpolate_list):
        img[:, :, i*w:(i+1)*w] = x_hat.reshape(3, PATCH_SIZE, PATCH_SIZE)
    mpimg.imsave("rndpics_interpolate.png", img.transpose(1,2,0))


def interpolate_gif(vae, x_1, x_2, n=50):
    with torch.no_grad():
        mu, log_var = vae.encode(x_1)
        z_1 = vae.reparameterize(mu, log_var)
        mu, log_var = vae.encode(x_2)
        z_2 = vae.reparameterize(mu, log_var)
        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
        interpolate_list = vae.decode(z)
        #interpolate_list = torch.cat((x_1, interpolate_list), dim=0)
        #interpolate_list = torch.cat((interpolate_list, x_2), dim=0)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

        clip = ImageSequenceClip(list(interpolate_list.transpose(0,2,3,1)), fps=2*(n))
        clip.write_gif('rndpics_interpolate.gif', fps=2*(n))


# Interpolate for each latent vector entry for a range "range_latent"
# to see if they correspond to some meaningul features
def interpolate_latent_gif(vae, x_1, latent_vector_idx, n=50):
    mu, log_var = vae.encode(x_1)
    z_1 = vae.reparameterize(mu, log_var)
    std = torch.exp(0.5 * log_var[0,latent_vector_idx])

    # Find the range of the latent variable that we will modify (given by latent_vector_idx)
    one_hot_vec = torch.zeros_like(log_var)
    one_hot_vec[0,latent_vector_idx] = 3   # range_latent

    z = torch.stack([z_1 + one_hot_vec *t for t in np.linspace(-1, 1, n)])
    interpolate_list = vae.decode(z)
    interpolate_list = interpolate_list.to('cpu').detach().numpy()*255

    clip = ImageSequenceClip(list(interpolate_list.transpose(0,2,3,1)), fps=2*(n))
    clip.write_gif(f"rndpics_interpolate_latent{latent_vector_idx}.gif", fps=2*(n))


#for i, (pic0, _) in enumerate(loader):
for pic0, _ in loader:  # batch size is 1, loader is shuffled, so this gets one random pic
    pic0 = pic0.to(device)
    break
for pic1, _ in loader:  # batch size is 1, loader is shuffled, so this gets one random pic
    pic1 = pic1.to(device)
    break

recon0 = model(pic0)
recon1 = model(pic1)
save_image(pic0, 'rndpics_orig0.png')
save_image(recon0[0], 'rndpics_recon0.png')
save_image(pic1, 'rndpics_orig1.png')
save_image(recon1[0], 'rndpics_recon1.png')

interpolate(model, pic0, pic1)
interpolate_gif(model, pic0, pic1)

# Interpolate for some of the latent vector entries (check every 10)
for i in range (3, 128, 10):
    interpolate_latent_gif(model, pic0, i)

