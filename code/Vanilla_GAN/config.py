import torch


configs = {
    'torch_manual_seed': 1,
    'np_random_seed':1,
    'batch_size':64,
    'mode_z':'uniform', # or 'normal
    'z_size': 20,
    'image_size' :(28, 28),
    'z_size' : 20, # initial white noise size
    'gen_hidden_layers' :1,
    'gen_hidden_size' : 100,
    'disc_hidden_layers' : 1,
    'disc_hidden_size': 100,
    'device' : torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

}
