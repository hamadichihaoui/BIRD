from PIL import Image
import argparse
import numpy as np
import torch
from torch import nn
import random
from math import log10, sqrt
import sys
sys.path.insert(0,'./')
from guided_diffusion.models import Model

def ensure_reproducibility(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


# Code form RePaint   
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a



def process(sample, i):
    image_processed = sample.detach().cpu().permute(0, 2, 3, 1)
    image_processed = image_processed.squeeze(0)
    image_processed = torch.clip((image_processed + 1.0) * 127.5, 0., 255.)
    image_processed = image_processed.numpy().astype(np.uint8)
    return image_processed

def process_gray(sample, i):
    image_processed = sample.detach().cpu().permute(0, 2, 3, 1)
    image_processed = image_processed.squeeze(0)
    image_processed = torch.clip((image_processed + 1.0) * 127.5, 0., 255.)
    image_processed = image_processed.numpy().astype(np.uint8)
    init_image=Image.fromarray(image_processed).convert('L') 
    img1 = np.expand_dims(np.array(init_image).astype(np.uint8), axis=2)
    #mask1 = np.where(img1 < 50)
    img2 = np.tile(img1, [1, 1, 3])   
    return img2

def process_gray_thresh(sample, i, thresh=170):
    image_processed = sample.detach().cpu().permute(0, 2, 3, 1)
    image_processed = image_processed.squeeze(0)
    image_processed = torch.clip((image_processed + 1.0) * 127.5, 0., 255.)
    image_processed = image_processed.numpy().astype(np.uint8)
    init_image=Image.fromarray(image_processed).convert('L') 
    img1 = np.expand_dims(np.array(init_image).astype(np.uint8), axis=2)
    img3 = (np.where((img1 > thresh), 255, 0)).astype(np.uint8)
    img2 = np.tile(img3, [1, 1, 3])   
   
    return img2


def get_mask(sample, i, thresh=170):
    image_processed = sample.detach().cpu().permute(0, 2, 3, 1)
    image_processed = image_processed.squeeze(0)
    image_processed = torch.clip((image_processed + 1.0) * 127.5, 0., 255.)
    image_processed = image_processed.numpy().astype(np.uint8)
    init_image=Image.fromarray(image_processed).convert('L') 
    img1 = np.expand_dims(np.array(init_image).astype(np.uint8), axis=2)
    img3 = (np.where((img1 > thresh), 0., 1.)).astype(np.float32)
   
   
    return img3[:,:,0]



def psnr_orig(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def psnr_mask(original, compressed, mask):
    mse = ((original*mask - compressed*mask) ** 2).sum() / mask.sum()
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def generate_noisy_image(path, speckle_coef=0.8, speckle_lambda=0.4, gauss_coef=0.2, gauss_sigma=0.15):
    pil_img =  Image.open(path).resize((256, 256))
    gauss = np.random.normal(0, speckle_lambda, 3*256*256)
    gauss = gauss.reshape(256, 256, 3).astype(np.float32)
    x = np.array(pil_img).astype(np.float32)   
    img_np =  speckle_coef * np.array(x + x * gauss, dtype=np.float32) + gauss_coef  * (x + np.random.normal(size=np.array(pil_img).shape).astype(np.float32) * gauss_sigma * 255) 
    img_np = np.clip(img_np, 0., 255.)
    #print(path1, 'std', np.std(img_np-x) / 255.)
    img_np = img_np/ 255 * 2 - 1
    return pil_img, img_np

def generate_noisy_image_and_mask(path='imgs/00205.png', speckle_coef=0.8, speckle_lambda=0.12, gauss_coef=0.2, gauss_sigma=0.05):
    init_image =  Image.open(path).resize((256, 256))
    gauss = np.random.normal(0,speckle_lambda, 3*256*256)
    gauss = gauss.reshape(256,256,3).astype(np.float32)
    x = np.array(init_image).astype(np.float32)   
    img_np =  speckle_coef * np.array(x + x * gauss, dtype=np.float32) +   gauss_coef * (x + np.random.normal(size=np.array(init_image).shape).astype(np.float32) * gauss_sigma * 255.)  
    img_np = np.clip(img_np, 0., 255.)
    img_np = img_np/ 255 * 2 - 1
    mask = np.ones((256, 256))   
    for i in range(128-40, 128+40):
        for j in range(128-40, 128+40):
            mask[i, j]=0.    

    return  init_image, img_np, mask


def generate_lr_image(path, downsampling_ratio, speckle_coef=0.8, speckle_lambda=0.12, gauss_coef=0.2, gauss_sigma=0.05):

    init_image = Image.open(path).resize((256, 256))
    img_np = np.array(init_image).astype(np.float32) / 255 * 2 - 1
    img = torch.tensor(img_np).permute(2,0,1).unsqueeze(0)
    downsampling_op = torch.nn.AdaptiveAvgPool2d((256//downsampling_ratio,256//downsampling_ratio)).cuda() 
    for param in downsampling_op.parameters():
        param.requires_grad = False
    #b, c, h, w = img.shape
    downsampled = downsampling_op(img.cuda())
    downsampled_resc1 = (downsampled + 1.) / 2.
    gauss = torch.randn_like(downsampled) * speckle_lambda

    downsampled_resc = speckle_coef *(downsampled_resc1 + downsampled_resc1 * gauss) + gauss_coef * (downsampled_resc1 + gauss_sigma * torch.randn_like(downsampled))
    #print('std', (downsampled_resc - downsampled_resc1).std())
    downsampled = downsampled_resc * 2. - 1.
    return init_image, downsampled, downsampling_op

def MeanUpsample(x, scale):
    n, c, h, w = x.shape
    out = torch.zeros(n, c, h, scale, w, scale).to(x.device) + x.view(n,c,h,1,w,1)
    out = out.view(n, c, scale*h, scale*w)
    return out


def get_conv(scale):
    conv = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=scale, stride=1, padding=scale//2, bias=False)  
    kernel = np.load('data/kernel.npy')
    kernel_torch = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()
    conv.weight = torch.nn.Parameter(kernel_torch)
    return conv

def generate_blurry_image(path='imgs/00205.png', kernel_size=41, speckle_coef=0.8, speckle_lambda=0.12, gauss_coef=0.2, gauss_sigma=0.05):

    pil_image = Image.open(path).resize((256, 256))
    img_np = np.array(pil_image).astype(np.float32) / 255 * 2 - 1
    img = torch.tensor(img_np).permute(2,0,1).unsqueeze(0).cuda()
    conv = get_conv(kernel_size).cuda()

    for param in conv.parameters():
        param.requires_grad = False
    b, c, h, w = img.shape
    blurred = conv(img.view(-1, 1, h, w))
    blurred = blurred.view(1, c, h, w)
    downsampled = blurred#[:,:, ::32, ::32]
 
    downsampled_resc1 = (downsampled + 1.) / 2.
    gauss = torch.randn_like(downsampled) * speckle_lambda
    downsampled_resc = speckle_coef *(downsampled_resc1 + downsampled_resc1 * gauss) + gauss_coef * (downsampled_resc1 + gauss_sigma * torch.randn_like(downsampled))
    #print('std', (downsampled_resc - downsampled_resc1).std())
    downsampled = downsampled_resc * 2. - 1.
    return pil_image, downsampled

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    torch.manual_seed(0)
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def fcn(num_input_channels=200, num_output_channels=1, num_hidden=1000):

    layers = []
    layers.append(nn.Linear(num_input_channels, num_hidden,bias=True))
    layers.append(nn.ReLU6())
#
    layers.append(nn.Linear(num_hidden, num_output_channels))
    layers.append(nn.Softmax())
    model2 = nn.Sequential(*layers)

    return model2

def load_pretrained_diffusion_model(config):
    model = Model(config)
    ckpt = "checkpoints/celeba_hq.ckpt"
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config.device = device
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model = torch.nn.DataParallel(model)    
    return model, device
