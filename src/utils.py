import torchvision
from torchvision import transforms


def get_dataset(dataset_name="mnist", trans=transforms.Compose([transforms.ToTensor()])):
    if dataset_name == "mnist":
        return torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=trans), \
            torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=trans)
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=trans), \
            torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=trans)