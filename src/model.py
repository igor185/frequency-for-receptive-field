from torch import nn
import torch
import numpy as np

class ConvFeatures(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = nn.SiLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return self.bn(x)


class FourierFeatures(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.conv = ConvFeatures(in_channels * 2, out_channels)

    def forward(self, x):  # x is (batch, channels, w, h)
        x = torch.fft.rfftn(x, dim=(2, 3), norm='ortho')
        x = torch.cat([torch.real(x), torch.imag(x)], dim=1)
        x = self.conv(x)
        x = torch.fft.irfftn(x, dim=(2, 3), norm="ortho")
        return x


class Classifier(nn.Module):
    def __init__(self, model_type="conv", in_channels=1, out_channels=32, image_size=(28, 28)):
        super().__init__()
        if model_type == "conv":
            feature_extractor = ConvFeatures(in_channels, out_channels)
        elif model_type == "fourier":
            feature_extractor = FourierFeatures(in_channels, out_channels)
        else:
            raise ValueError(model_type)

        self.feature_extractor = nn.Sequential(
            feature_extractor,
            nn.Conv2d(out_channels, 2*out_channels, kernel_size=3, stride=2),
            nn.SiLU(),
            nn.BatchNorm2d(2*out_channels),
            nn.Conv2d(2*out_channels, 3*out_channels, kernel_size=3, stride=2),
            nn.BatchNorm2d(3 * out_channels),
            nn.SiLU(),
            nn.Dropout2d(p=0.2),
        )
        map = {28: 5, 32: 7} # TODO
        size_x = map[image_size[0]]
        size_y = map[image_size[1]]
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*out_channels*size_x*size_y, 128),
            nn.Linear(128, 10),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classification_head(x)
        return x