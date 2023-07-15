import torch
from config import configs

def create_noise(batch_size, z_size, mode_z):
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size)*2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z

fixed_z = create_noise(batch_size=configs['batch_size'], z_size=configs['z_size'], mode_z=configs['mode_z']).to(configs['device'])

def create_samples(g_model, input_z):
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (configs['batch_size'], *configs['image_size']))
    return (images+1)/2.0