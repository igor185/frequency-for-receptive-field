import torch


def fft_torch_native(data, combine="abs"):
    data = torch.fft.fft2(data, dim=(1, 2))
    if combine == "abs":
        return torch.absolute(data)
    elif combine == "stack":
        return torch.stack([data.real, data.imag], dim=1)
    else:
        return data


def fft_torch(data, combine="abs"):
    raise NotImplementedError  # TODO


def wavelet_torch(data, combine="abs"):
    raise NotImplementedError  # TODO
