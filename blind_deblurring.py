import os
import yaml
import numpy as np
import tqdm
import torch
from torch import nn
import sys
sys.path.insert(0,'./')
from guided_diffusion.models import Model
import random
from ddim_inversion_utils import *
from utils import *

with open('configs/blind_deblurring.yml', 'r') as f:
    task_config = yaml.safe_load(f)


### Reproducibility
torch.set_printoptions(sci_mode=False)
ensure_reproducibility(task_config['seed'])


with open( "data/celeba_hq.yml", "r") as f:
    config1 = yaml.safe_load(f)
config = dict2namespace(config1)
model, device = load_pretrained_diffusion_model(config)

### Define the DDIM scheduler
ddim_scheduler=DDIMScheduler(beta_start=config.diffusion.beta_start, beta_end=config.diffusion.beta_end, beta_schedule=config.diffusion.beta_schedule)
ddim_scheduler.set_timesteps(config.diffusion.num_diffusion_timesteps // task_config['delta_t'])#task_config['Denoising_steps']

#scale=41
l2_loss= nn.MSELoss() #nn.L1Loss() 
net_kernel = fcn(200, task_config['kernel_size'] * task_config['kernel_size']).cuda()
net_input_kernel = get_noise(200, 'noise', (1, 1)).cuda()
net_input_kernel.squeeze_()


img_pil, downsampled_torch = generate_blurry_image('data/imgs/00287.png')
radii =  torch.ones([1, 1, 1]).cuda() * (np.sqrt(256*256*3))

latent = torch.nn.parameter.Parameter(torch.randn( 1, config.model.in_channels, config.data.image_size, config.data.image_size).to(device))  
optimizer = torch.optim.Adam([{'params':latent,'lr':task_config['lr_img']}, {'params':net_kernel.parameters(),'lr':task_config['lr_blur']}])


for iteration in range(task_config['Optimization_steps']):
    optimizer.zero_grad()
    x_0_hat = DDIM_efficient_feed_forward(latent, model, ddim_scheduler)   
    out_k = net_kernel(net_input_kernel)
    out_k_m = out_k.view(-1, 1, task_config['kernel_size'], task_config['kernel_size'])

    blurred_xt = nn.functional.conv2d(x_0_hat.view(-1, 1, config.data.image_size, config.data.image_size), out_k_m, padding="same", bias=None).view(1, 3, config.data.image_size, config.data.image_size) 
    loss = l2_loss(blurred_xt, downsampled_torch)
    loss.backward()  
    optimizer.step()  

    #Project to the Sphere of radius sqrt(D)
    for param in latent:
        param.data.div_((param.pow(2).sum(tuple(range(0, param.ndim)), keepdim=True) + 1e-9).sqrt())
        param.data.mul_(radii)

    if iteration % 10 == 0:
        #psnr = psnr_orig(np.array(img_pil).astype(np.float32), process(x_0_hat, 0))
        #print(iteration, 'loss:', loss.item(), torch.norm(latent.detach()), psnr)
        Image.fromarray(np.concatenate([ process(downsampled_torch, 0), process(x_0_hat, 0), np.array(img_pil).astype(np.uint8)], 1)).save('results/blind_deblurring.png')

   


