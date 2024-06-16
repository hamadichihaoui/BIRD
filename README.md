# Blind Image Restoration via Fast Diffusion Inversion (BIRD)

This is the official implementation of "Blind Image Restoration via Fast Diffusion Inversion
". [arxiv](https://arxiv.org/abs/2405.19572)

## Environment 
```
pip install numpy torch blobfile tqdm pyYaml pillow diffusers   
```

## Pre-Trained Models

For face restoration, the pretrained model can be found [here](https://drive.google.com/file/d/1qMs7tNGV3tkOZNKH5L130dkwsmobEJdh/view?usp=sharing).


## Blind Deblurring

```
python blind_deblurring.py    
```
![image info](results/blind_deblurring.png)

## Non-uniform Deblurring

```
python non_uniform_deblurring.py    
```
![image info](results/non_uniform_deblurring.png)

## Inpainting

```
python inpainting.py    
```

![image info](results/inpainted.png)

## Denoising

```
python denoising.py    
```
![image info](results/denoised.png)

## Superresolution

```
python super_resolution.py    
```
![image info](results/super_resolution.png)


## References

If you find this repository useful for your research, please cite the following work.



```
@article{chihaoui2024blind,
  title={Blind Image Restoration via Fast Diffusion Inversion},
  author={Chihaoui, Hamadi and Lemkhenter, Abdelhak and Favaro, Paolo},
  journal={arXiv preprint arXiv:2405.19572},
  year={2024}
}

```


