import os

import torch

from src.resnet import resnet18_small, resnet34_small
from src.training import eval_model
from src.utils import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_accuracy(type, load_path):
    _, test_ds = get_dataset("cifar10")

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64)

    use_fourier = type != "conv"
    os.environ["TYPE"] = type
    model = resnet18_small(num_classes=10, use_fourier=use_fourier, in_channels=3).to(device)

    return eval_model(model, test_loader, device, load_path)


if __name__ == '__main__':
    print("Conv accuracy: ", eval_accuracy("conv", "logs/cifar10/resnet_conv/seed_42/model_best.pth"))
    print("Fourier accuracy: ", eval_accuracy("fourier", "logs/cifar10/resnet_fourier/seed_42/model_best.pth"))
    print("Wavelet accuracy: ", eval_accuracy("wavelet", "logs/cifar10/resnet_wavelet/seed_42/model_best.pth"))