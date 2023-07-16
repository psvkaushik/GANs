from config import configs
import torch
from create import create_noise
def d_train(model, x, loss_fn, optimizer, g_model, device=configs['device']):
    """
    The method used for training the discriminator
    inputs ->
    model : the discriminator model to be trained
    x : the input
    loss_fn = the loss_fn used to measure the loss
    optimizer = the optimizer being used
    g_model = the adversary model -> the generator
    device : the device being used

    outputs ->
    loss_item : the total loss which is its combined loss on both real and fake images
    d_proba_real : the probability that a real image is real
    d_proba_fake : the probability that a fake image is fake

    """
    model.zero_grad()
    batch_size = x.size(0)

    x = x.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_proba_real = model(x)

    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    input_z = create_noise(batch_size=batch_size, z_size=configs['z_size'], mode_z=configs['mode_z']).to(device)
    g_output = g_model(input_z)
    d_proba_fake = model(g_output)

    d_loss_fake = loss_fn(d_proba_fake.view(-1, 1).squeeze(0), d_labels_fake)
    d_loss_real = loss_fn(d_proba_real.view(-1, 1).squeeze(0), d_labels_real)

    d_loss = d_loss_fake + d_loss_real
    d_loss.backward()
    optimizer.step()
    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()


def g_train(model, x, loss_fn, optimizer, d_model, device=configs['device']):
    """
    The method used for training the generator
    inputs ->
    model : the generator model to be trained
    x : the input
    loss_fn = the loss_fn used to measure the loss
    optimizer = the optimizer being used
    d_model = the adversary model -> the discriminator
    device : the device being used

    outputs ->
    loss_item : the loss of the generator

    """
    model.zero_grad()
    batch_size = x.size(0)

    x = x.to(device)
    input_z = create_noise(batch_size=batch_size, z_size=configs['z_size'], mode_z=configs['mode_z']).to(device)
    g_output = model(input_z)
    d_proba_fake = d_model(g_output).view(-1, 1).squeeze(0)
    d_labels_real = torch.ones(batch_size, 1, device=device)

    g_loss = loss_fn(d_proba_fake, d_labels_real)
    g_loss.backward()
    optimizer.step()

    return g_loss.data.item()