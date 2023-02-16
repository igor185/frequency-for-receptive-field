import random
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
from torch import nn, optim

import sys

from src.resnet import resnet18_small
from src.utils import get_dataset
from src.training import train_epoch, eval_model
from tensorboardX import SummaryWriter

seed = 48
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.use_deterministic_algorithms(True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "model": "fourier",  # conv, fourier
    "dataset": "cifar10",
    "batch_size": 256,
    "lr": 0.1,
    "experiment_name": "mnist/fourier",
    "validate_frequency": 200,
}

if __name__ == '__main__':

    config["experiment_name"] = config["dataset"] + \
                                "/resnet_" + config["model"] + \
                                "/batch_" + str(config["batch_size"]) + "/seed_" + str(seed)
    print(config["experiment_name"])
    writer = SummaryWriter(log_dir="./logs/" + config["experiment_name"])

    train_ds, test_ds = get_dataset(config["dataset"])
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"])

    use_fourier = config["model"] == "fourier"
    model = resnet18_small(num_classes=10, use_fourier=use_fourier).to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    steps = 0
    for i in range(100):
        print("Epoch: ", i + 1)
        steps += train_epoch(model, train_loader, val_loader, loss, optimizer, writer, steps, config, device,
                             total_len=len(train_ds) // config["batch_size"])

    eval_accuracy = eval_model(model, test_loader, device)
    writer.add_scalar('test/accuracy', eval_accuracy, 0)


