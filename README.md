
# Variational Auto Encoders (VAE)

Implements Variational Auto Encoder (VAE) [Kingma and Welling's ["Auto-Encoding Variational Bayes"](https://arxiv.org/pdf/1312.6114.pdf)] and associated training code for CelebA, plus the inference code for latent space sampling. You need to get the [CelebA](https://github.com/AntixK/PyTorch-VAE/#:~:text=to%20download%20the-,file,-from%20google%20drive) data separately and unzip under `./data/` due to download restrictions imposed by Google.

## How To Run:
Set up the environment (using miniconda)
```
$ conda create --name "vae" python=3.6.12
$ conda activate vae
$ pip install -r requirements.txt
```

## Running training: 
If you don't specify any flags, it trains for `batch=128` and `epochs=50`. During training, generated reconstruction and random sampled results are saved under `results/` folder. Also the model weights after each epoch is saved under `results.model/` folder.

```
$ python train_vae.py
```

## Running the inference:
Interpolation results between two random CelebA pictures are saved as `rndpics_interpolate.{gif,png}`
```
python test_vae.py
```
Some examples of interpolation between two pictures by sampling from the latent space and generating the images using the posterior $p(z|x)$

<table>
  <tr>
    <td> <img src="examples/examples1/rndpics_interpolate.png" width="1000"/> </td>
  </tr>
  <tr>
    <td> <img src="examples/examples1/rndpics_interpolate.gif" width="200"/> </td>
  </tr>

  <tr>
    <td> <img src="examples/examples2/rndpics_interpolate.png" width="1000"/> </td>
  </tr>
  <tr>
    <td> <img src="examples/examples2/rndpics_interpolate.gif" width="200"/> </td>
  </tr>

  <tr>
    <td> <img src="examples/examples3/rndpics_interpolate.png" width="1000"/> </td>
  </tr>
  <tr>
    <td> <img src="examples/examples3/rndpics_interpolate.gif" width="200"/> </td>
  </tr>
</table>  

### Code Reference:
   
   [1] Training code (train_vae.py) is mainly from Official Pytorch [examples/vae](https://github.com/pytorch/examples/tree/main/vae)

   [2] [AntixK github page](https://github.com/AntixK/PyTorch-VAE/) is a nice resource for various VAE algorithms. I mainly borrowed the code in `vae.py` from "class VanillaVAE" in `vanilla_vae.py`. Code is written very organized and modular by using pytorch-lightning that automatically uses DDP for multi-GPU training. However the modularity made it too complex to debug/understand for someone who is new. Besides there is no inference script for quick testing.
    

   [3] Got some inspiration from Moshe Sipper's [Medium post](https://medium.com/the-generator/a-basic-variational-autoencoder-in-pytorch-trained-on-the-celeba-dataset-f29c75316b26) and [github repo](https://github.com/moshesipper/vae-torch-celeba), for the inference code (test_vae.py)
 

   [4] Used some ideas/code about sampling from latent space, plotting them using matplotlib, generating GIF and using PCA. Tingsong Ou's [Medium Post](https://medium.com/@outerrencedl/variational-autoencoder-and-a-bit-kl-divergence-with-pytorch-ce04fd55d0d7) and [Alexander van de Kleut [github repo](https://avandekleut.github.io/vae/)