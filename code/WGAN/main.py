import torch
import torch.nn as nn
from gen_dis import make_gen, make_dis
import numpy as np
from config import configs
from create import create_noise, create_samples, fixed_z
import torch.nn as nn
from dataset import mnist_dl
from train import d_train_wgan, g_train_wgan
import matplotlib.pyplot as plt
import itertools

torch.manual_seed(configs['torch_manual_seed'])
np.random.seed(configs['np_random_seed'])
gen_model = make_gen(input_size=100,
                      n_filters=32).to(configs['device'])

dis_model = make_dis(n_filters=32).to(configs['device'])


g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0002)
d_optimizer = torch.optim.Adam(dis_model.parameters(), 0.0002)

epoch_samples_wgan = []
lambda_gp = 10.0
num_epochs = 50
torch.manual_seed(1)
critic_iterations = 5
for epoch in range(1, num_epochs+1):
    gen_model.train()
    d_losses, g_losses = [], []
    for i, (x, _) in enumerate(mnist_dl):
        for _ in range(critic_iterations):
            d_loss = d_train_wgan(dis_model, x, d_optimizer, gen_model)
        d_losses.append(d_loss)
        g_losses.append(g_train_wgan(gen_model, x, g_optimizer, dis_model))

    print(f'Epoch {epoch:03d} | D Loss >>'
    f' {torch.FloatTensor(d_losses).mean():.4f}')
    gen_model.eval()
    epoch_samples_wgan.append(create_samples(gen_model, fixed_z).detach().cpu().numpy())

fig = plt.figure(figsize=(16, 6))
## Plotting the losses
ax = fig.add_subplot(1, 2, 1)
plt.plot(g_losses, label='Generator loss')
half_d_losses = [all_d_loss/2 for all_d_loss in d_losses]
plt.plot(half_d_losses, label='Discriminator loss')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Loss', size=15)

plt.show()

selected_epochs = [1, 2, 4, 15, 25, 50]
fig = plt.figure(figsize=(10, 14))
for i,e in enumerate(selected_epochs):
    for j in range(5):
        ax = fig.add_subplot(6, 5, i*5+j+1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.06, 0.5, f'Epoch {e}',
                rotation=90, size=8, color='red',
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)
        image = epoch_samples_wgan[e-1][j]
        ax.imshow(image, cmap='gray_r')

plt.show()