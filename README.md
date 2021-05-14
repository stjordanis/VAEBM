# CS698X_Project

## Introduction
Repository for CS698X course project (Topics in Probabilistic Modeling and Inference, Winter 2021, IIT Kanpur), on energy-based VAEs. Contains implementation of VAEBM for Vanilla VAE, $$\beta$$-TCVAE and Factor-VAE for MNIST, CelebA-64 and Chairs dataset. Based on the ICLR 2021 paper ***[VAEBM: A Symbiosis between Variational Autoencoders and Energy-based Models (Xiao et al. 2021)](https://arxiv.org/abs/2010.00654)***

## Running the models
The Energy based model can be trained for a given VAE and dataset.
To train the model on default parameters, run the following command in the CS698X directory:

```bash
python3 train_vaebm.py
```
Args:
```python
  --vae_type, type=str, default='VAE'
  --num_workers,type=int, default=2
  --dataset,type=str, default='mnist'
  --batch_size,type=int, default=32
  --l2_reg_weight, type=float, default=1.0
  --spectral_norm_weight, type=float, default=0.2
  --sample_type,type=str, default='lang',
  --sample_step_size, type=float, default=8e-5
  --sample_steps, type=int, default=10
  --train_step_size, type=float, default=4e-5
  --train_steps, type=int, default=15
```
To generate samples and traversals, run the following command :
```bash
python3 test_vaebm.py
```

Args:
```python
  --vae_type,type=str, default='VAE'
  --dataset,type=str, default='mnist'
  --batch_size,type=int, default=1
  --step_size, type=float, default=8e-3
  --steps, type=int, default=16
  ```
Samples are stored according to the VAE used, dataset used and the step size involved. Pre-Trained models can be found here: [Google Drive](https://drive.google.com/drive/folders/1RW8uu5ZDbvm8dOZ0nWSHhhz76AY5F0Tf?usp=sharing). Please change the ROOT_DIR in `test_vaebm.py` to wherever the models are stored before running.

## References
Code for IGEBM from https://github.com/rosinality/igebm-pytorch

FID Calculations using pytorch-fid library

Pretrained VAE from https://github.com/YannDubs/disentangling-vae

