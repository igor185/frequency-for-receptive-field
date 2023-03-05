import numpy as np
import torch
from sklearn.svm import LinearSVC
from src.frequency_transformation import fft_torch_native
from src.utils import get_dataset


if __name__ == '__main__':
    torch.manual_seed(42)

    train, test = get_dataset()

    train_x = train.data.numpy()
    train_x_fft = fft_torch_native(train.data, combine="abs").numpy()
    train_y = train.train_labels.numpy()

    test_x = test.data.numpy()
    test_x_fft = fft_torch_native(test.data, combine="abs").numpy()
    test_y = test.train_labels.numpy()

    svm = LinearSVC(dual=False, verbose=True)  # TODO explore more different classifiers (and faster)
    print("Fit model")
    svm.fit(train_x.reshape(train_x.shape[0], -1), train_y)
    pred = svm.predict(test_x.reshape(test_x.shape[0], -1))
    print("Result")
    print(np.mean(test_y == pred))  # 0.9171

    svm = LinearSVC(dual=False, verbose=True)
    print("Fit model fft")
    svm.fit(train_x_fft.reshape(train_x_fft.shape[0], -1), train_y)
    pred = svm.predict(test_x_fft.reshape(test_x_fft.shape[0], -1))
    print("Result fft")
    print(np.mean(test_y == pred))  # 0.8389
