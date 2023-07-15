import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


## Define the generator network

def make_generator(
        input_size=20,
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=784):
    """
    The method which returns an instance of the generator
    """
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module(f'fc_g{i}', nn.Linear(input_size, num_hidden_units))
        model.add_module(f'relu_g{i}', nn.LeakyReLU())
        input_size=num_hidden_units
    model.add_module(f'fc_g{num_hidden_layers}', nn.Linear(num_hidden_units, num_output_units))
    model.add_module(f'tanh_g', nn.Tanh())
    return model


def make_discriminator(
        input_size,
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=1):
    """
    The method which returns an instance of the discriminator 
    """

    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module(f'fc_d{i}', nn.Linear(input_size, num_hidden_units))
        model.add_module(f'relu_d{i}', nn.LeakyReLU())
        input_size=num_hidden_units
    model.add_module(f'fc_d{num_hidden_layers}', nn.Linear(num_hidden_units, num_output_units))
    model.add_module(f'sigmoid_d', nn.Sigmoid())
    return model
