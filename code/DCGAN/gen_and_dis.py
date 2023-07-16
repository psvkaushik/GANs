
import torch.nn as nn

def generator(input_size,n_filters):
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters*4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(n_filters*4),
        nn.LeakyReLU(0.02),
        nn.ConvTranspose2d(n_filters*4, n_filters*2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters*2),
        nn.LeakyReLU(0.02),
        nn.ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(0.02),
        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    return model

def discriminator(n_filters):
    model = nn.Sequential(
        nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(0.02),
        nn.Conv2d(n_filters, n_filters*2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters*2),
        nn.LeakyReLU(0.02),
        nn.Conv2d(n_filters*2, n_filters*4, 3, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters*4),
        nn.LeakyReLU(0.02),
        nn.Conv2d(n_filters*4, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
    return model

