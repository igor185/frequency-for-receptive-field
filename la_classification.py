import numpy as np
import torchvision
from torchvision import transforms
import torch
from sklearn.svm import LinearSVC


def fft(data):
    data = torch.fft.fft2(data, dim=(1, 2))
    return torch.absolute(data)


if __name__ == '__main__':
    torch.manual_seed(42)
    trans = transforms.Compose([transforms.ToTensor()])

    train, test = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=trans), \
        torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=trans)

    train_x = train.data.numpy()
    train_x_fft = fft(train.data).numpy()
    train_y = train.train_labels.numpy()

    test_x = test.data.numpy()
    test_x_fft = fft(test.data).numpy()
    test_y = test.train_labels.numpy()

    svm = LinearSVC(dual=False)  # TODO explore more different classifiers (and faster)
    print("Fit model")
    svm.fit(train_x.reshape(train_x.shape[0], -1), train_y)
    pred = svm.predict(test_x.reshape(test_x.shape[0], -1))
    print("Result")
    print(np.mean(test_y == pred))  # 0.9171

    svm = LinearSVC(dual=False)
    print("Fit model fft")
    svm.fit(train_x.reshape(train_x.shape[0], -1), train_y)
    pred = svm.predict(test_x.reshape(test_x.shape[0], -1))
    print("Result fft")
    print(np.mean(test_y == pred))  # 0.8389
