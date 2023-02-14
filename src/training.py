from tqdm import tqdm
import torch


def train_epoch(model, data, val_loader, criterion, optimizer, writer, init_step, config, device, total_len=None):
    model.train()
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
            accuracy = eval_model(model, val_loader, device)
            print("Accuracy: %.5f" % accuracy)
            writer.add_scalar('val/accuracy', accuracy, init_step + i)
            model.train()
    return i


def eval_model(model, data, device):
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