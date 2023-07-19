import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from config import configs

image_path = './'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

mnist_dataset = torchvision.datasets.MNIST(
    root=image_path,
    train=True,
    transform=transform,
    download=False
)

example, label = next(iter(mnist_dataset))

print(f'Min: {example.min()}, Max: {example.max()}')
print(example.shape)

batch_size = 32
mnist_dl = DataLoader(mnist_dataset, batch_size=configs['batch_size'], shuffle=False)
