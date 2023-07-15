# GANs

This repository consists of code for implementations of three different GAN arctitectures, and this readme consists of the results observed for each implmentation.
[VanillaGAN](##VanillaGAN)
[DCGAN](##DCGAN)
[WGAN](##WGAN)

## VanillaGAN

The first implemented architecture is the VanillaGAN whose Generator and Discriminator consist of two fully connected non-linear layers.

### Architectures of VanillaGAN

The generator architecture is:

![image](https://github.com/psvkaushik/GANs/assets/86014345/748168dd-93e0-4d63-acd9-b7798f74bcbc)

The discriminator architecture is:

![image](https://github.com/psvkaushik/GANs/assets/86014345/7f1337c5-44d4-4104-aac9-8c98cbc89fcb)

The Loss Function used to optimize both the generator and discriminator is taken from the [original GAN paper](https://arxiv.org/pdf/1406.2661.pdf) by I.GoodFellow which is: ![image](https://github.com/psvkaushik/GANs/assets/86014345/19e4f266-b643-45a2-a753-9cf946030541)

The dataset used is the MNIST digits dataset. Each image is of shape (28,28). The objective of this generator model was to just produce an image which would fool the discriminator.

The losses observed were:
![image](https://github.com/psvkaushik/GANs/assets/86014345/16b5c202-d468-434e-b386-05559a5187c0)

The outputs of the generator after epochs  (1, 2, 4, 10, 50, 100)

![image](https://github.com/psvkaushik/GANs/assets/86014345/cbe8001f-e202-4fb5-be17-dfc7bf400a12)



## DCGAN
*In Progress*

## WGAN
*In Progress*



