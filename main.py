import torch
from torch import nn, optim

import os
import sys

from src.model import Classifier
from src.utils import get_dataset
from src.training import train_epoch, eval_model
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "model": "fourier",  # conv, fourier, wavelet
    "dataset": "cifar10",
    "batch_size": 128,
    "lr": 0.001,
    "experiment_name": "mnist/fourier",
    "validate_frequency": 300,
}


if __name__ == '__main__':
    torch.manual_seed(42)
    config["experiment_name"] = config["dataset"] + "/2resnet_" + config["model"] + "/" + str(config["batch_size"])
    print(config["experiment_name"])
    writer = SummaryWriter(log_dir="./logs/" + config["experiment_name"])

    train_ds, test_ds = get_dataset(config["dataset"])
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"])

    in_channels = 3 if config["dataset"] == "cifar10" else 1
    model = Classifier(config["model"], in_channels, 32, image_size=(32, 32)).to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    steps = 0
    for i in range(100):
        print("Epoch: ", i + 1)
        steps += train_epoch(model, train_loader, val_loader, loss, optimizer, writer, steps, config, device, total_len=len(train_ds) // config["batch_size"])

    eval_accuracy = eval_model(model, test_loader, device)
    writer.add_scalar('test/accuracy', eval_accuracy, 0)


