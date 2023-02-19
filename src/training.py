import os

from tqdm import tqdm
import torch


def train_epoch(model, data, val_loader, criterion, optimizer, writer, init_step, config, device, total_len=None):
    model.train()
    best_val_accuracy = -1.
    i = 0
    for i, (x, y) in tqdm(enumerate(data), total=total_len):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_pred = model(x)

        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss, init_step+i)

        if (init_step + i) % config["validate_frequency"] == 0:
            path = "./logs/" + config["experiment_name"] + "/model_best.pth"
            accuracy = eval_model(model, val_loader, device)
            print("Accuracy: %.5f" % accuracy)
            if accuracy > best_val_accuracy:
                torch.save(model.state_dict(), path)
            writer.add_scalar('val/accuracy', accuracy, init_step + i)
            model.train()
    return i


def eval_model(model, data, device, path=None):
    if path and os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print("Loaded for eval")
    model.eval()
    with torch.no_grad():
        total = 0
        corr = 0
        for i, (x, y) in enumerate(data):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            corr += (y == y_pred.argmax(1)).sum()
            total += y.shape[0]

        return corr / total