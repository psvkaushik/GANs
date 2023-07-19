import torch
from torch.autograd import grad as torch_grad
from config import configs

def gradient_penalty(disc_model, real_images, fake_images, lambda_gp=10.0, device=configs['device']):
    """
    Return the gradient penalty to be added to the discriminator
    """

    batch_size = real_images.size(0)

    alpha = torch.rand(real_images.shape[0], 1, 1, 1, requires_grad=True, device=device)
    interpolated_images = alpha*real_images + (1-alpha)*fake_images

    proba_interpolated = disc_model(interpolated_images)

    gradients = torch_grad(outputs=proba_interpolated,
                           inputs=interpolated_images,
                           grad_outputs=torch.ones(proba_interpolated.size(), device=device),
                           retain_graph=True,
                           create_graph=True)[0]
    
    gradients = gradients.view(batch_size, -1)

    gradients_norm = gradients.norm(2, dim=1)

    return lambda_gp*((gradients_norm - 1)**2).mean()