from gen_and_dis import make_generator, make_discriminator
from dataset import mnist_dl
from config import configs
from create import create_noise
import numpy as np
import torch
import torch.nn as nn


input_real, label = next(iter(mnist_dl))
input_real = input_real.view(configs['batch_size'], -1)

torch.manual_seed(configs['torch_manual_seed'])

mode_z = configs['mode_z']

input_z = create_noise(configs['batch_size'], configs['z_size'], mode_z)

print('input_z shape : ', input_z.shape)
print('input_real shape: ', input_real.shape)

gen_model = make_generator(
    input_size=configs['z_size'],
    num_hidden_layers=configs['gen_hidden_layers'],
    num_hidden_units=configs['gen_hidden_size'],
    num_output_units=np.prod(configs['image_size'])
)

disc_model = make_discriminator(
    input_size=np.prod(configs['image_size']),
    num_hidden_layers=configs['disc_hidden_layers'],
    num_hidden_units=configs['disc_hidden_size']
)


g_output = gen_model(input_z)
print('Output of generator shape : ', g_output.shape)

d_proba_real = disc_model(input_real)
d_proba_fake = disc_model(g_output)

print('Output of discriminator shape on real : ', d_proba_real.shape)
print('Output of discriminator shape on fake : ', d_proba_fake.shape)

loss_fn = nn.BCELoss()

g_labels_real = torch.ones_like(d_proba_real)
g_loss = loss_fn(d_proba_fake, g_labels_real)
print(f'Generator Loss : {g_loss:.4f}')

d_labels_real = torch.ones_like(d_proba_real)
d_labels_fake = torch.zeros_like(d_proba_fake)

d_loss_real = loss_fn(d_labels_real, d_proba_real)
d_loss_fake = loss_fn(d_labels_fake, d_proba_fake)

print(f'Discriminator Loss for real : {d_loss_real:.4f}, for fake : {d_loss_fake:.4f}')


from torchviz import make_dot

make_dot(g_output, params=dict(list(gen_model.named_parameters()))).render(filename="gen_model",directory="C:/Users/psvka/OneDrive/Desktop/code/GAN/Vanilla_GAN/imgs", format="png")