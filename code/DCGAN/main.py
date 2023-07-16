import torch
import torch.nn as nn
from gen_and_dis import generator, discriminator
import numpy as np
from config import configs
from create import create_noise, create_samples, fixed_z
import torch.nn as nn
from dataset import mnist_dl
from train import d_train, g_train
import matplotlib.pyplot as plt
import itertools


## Setting the random seed
torch.manual_seed(configs['torch_manual_seed'])
np.random.seed(configs['np_random_seed'])
gen_model = generator(input_size=100,
                      n_filters=32).to(configs['device'])

dis_model = discriminator(n_filters=32).to(configs['device'])

loss_fn = nn.BCELoss()
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0003)
d_optimizer = torch.optim.Adam(dis_model.parameters(), 0.0002)


## Training loop

epoch_samples = []
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []
num_epochs = 100

for epoch in range(num_epochs):
    d_losses, g_losses = [], []
    d_vals_real, d_vals_fake = [], []

    for i, (x, _) in enumerate(mnist_dl):
        d_loss, d_proba_real, d_proba_fake = d_train(dis_model, x, loss_fn, d_optimizer, gen_model)
        d_losses.append(d_loss)
        g_losses.append(g_train(gen_model, x, loss_fn, g_optimizer, dis_model))
        d_vals_real.append(d_proba_real.mean().cpu())
        d_vals_fake.append(d_proba_fake.mean().cpu())


    all_d_losses.append(torch.tensor(d_losses).mean())
    all_g_losses.append(torch.tensor(g_losses).mean())
    all_d_real.append(torch.tensor(d_vals_real).mean())
    all_d_fake.append(torch.tensor(d_vals_fake).mean())

    print(f'Epoch {epoch:03d} | Avg Losses >>'
    f' G/D {all_g_losses[-1]:.4f}/{all_d_losses[-1]:.4f}'
    f' [D-Real: {all_d_real[-1]:.4f}'
    f' D-Fake: {all_d_fake[-1]:.4f}]')

    epoch_samples.append(
    create_samples(gen_model, fixed_z).detach().cpu().numpy())



fig = plt.figure(figsize=(16, 6))
## Plotting the losses
ax = fig.add_subplot(1, 2, 1)
plt.plot(all_g_losses, label='Generator loss')
half_d_losses = [all_d_loss/2 for all_d_loss in all_d_losses]
plt.plot(half_d_losses, label='Discriminator loss')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Loss', size=15)

## Plotting the outputs of the discriminator
ax = fig.add_subplot(1, 2, 2)
plt.plot(all_d_real, label=r'Real: $D(\mathbf{x})$')
plt.plot(all_d_fake, label=r'Fake: $D(G(\mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Discriminator output', size=15)
plt.show()

selected_epochs = [1, 2, 4, 10, 50, 100]
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
        image = epoch_samples[e-1][j]
        ax.imshow(image, cmap='gray_r')

plt.show()
