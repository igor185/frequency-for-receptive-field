import argparse
import random
import os

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
from torch import nn

from src.resnet import resnet18_small, resnet34_small
from src.utils import get_dataset
from src.training import train_epoch, eval_model
from tensorboardX import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = {
    "model": "fourier",  # conv or fourier
    "model_type": "resnet18",  # resnet18 or resnet34
    "dataset": "cifar10",
    "batch_size": 256,
    "lr": 0.1,
    "experiment_name": "",
    "validate_frequency": 200,
    "epoch": 100
}

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default='resnet18')
parser.add_argument('--model', default='conv')
parser.add_argument('--seed', default='42')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--epoch', default='100')

if __name__ == '__main__':
    args = parser.parse_args()

    config["seed"] = int(args.seed)
    config["dataset"] = args.dataset
    config["model_type"] = args.model_type
    config["model"] = args.model
    config["epoch"] = int(args.epoch)

    seed = int(args.seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

    config["experiment_name"] = config["dataset"] + \
                                "/" + config["model_type"] + "_" + config["model"] + \
                                "/batch_" + str(config["batch_size"]) + "/seed_" + str(config["seed"])
    print(config["experiment_name"])
    writer = SummaryWriter(log_dir="./logs/" + config["experiment_name"])

    train_ds, test_ds = get_dataset(config["dataset"])
    train_ds, val_ds = torch.utils.data.random_split(train_ds, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"])
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config["batch_size"]//4)

    use_fourier = config["model"] != "conv"
    model = resnet18_small if config["model_type"] == "resnet18_small" else resnet34_small
    os.environ["TYPE"] = type
    in_channels = 1 if config["dataset"] == "mnist" else 3
    model = model(num_classes=10, use_fourier=use_fourier, in_channels=in_channels).to(device)

    loss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), config["lr"], momentum=0.9, weight_decay=1e-4)

    steps = 0
    for i in range(config["epoch"]):
        print("Epoch: ", i + 1)
        steps += train_epoch(model, train_loader, val_loader, loss, optimizer, writer, steps, config, device,
                             total_len=len(train_ds) // config["batch_size"])

    path = "./logs/" + config["experiment_name"] + "/model_best.pth"
    eval_accuracy = eval_model(model, test_loader, device, path)
    writer.add_scalar('test/accuracy', eval_accuracy, 0)
