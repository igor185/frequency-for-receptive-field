import torchvision
from torchvision import transforms


def get_dataset(dataset_name="mnist", trans=transforms.Compose([transforms.ToTensor(), transforms.Resize([32, 32])])):
    if dataset_name == "mnist":
        test_trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([48, 48])])
        return torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=trans), \
            torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=test_trans)
    if dataset_name == "cifar10":
        return torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=trans), \
            torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=trans)