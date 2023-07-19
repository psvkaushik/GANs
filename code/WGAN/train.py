from config import configs
from create import create_noise
from gradient_penalty import gradient_penalty

def d_train_wgan(model, x, optimizer, g_model, device=configs['device']):
    model.zero_grad()
    batch_size = x.size(0)
    x = x.to(device)

    d_real = model(x)
    input_z = create_noise(batch_size, z_size=configs['z_size'], mode_z=configs['mode_z']).to(device)
    g_output = g_model(input_z)
    d_generated = model(g_output)

    d_loss = d_generated.mean() -d_real.mean()  + gradient_penalty(model, x, g_output, device=device)

    d_loss.backward()
    optimizer.step()
    return d_loss.data.item()

def g_train_wgan(model, x, optimizer, d_model, device=configs['device']):
    model.zero_grad()

    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size=configs['z_size'], mode_z=configs['mode_z']).to(device)
    g_output = model(input_z)

    d_generated = d_model(g_output)

    g_loss = -d_generated.mean()

    g_loss.backward()
    optimizer.step()
    return g_loss.data.item()