import math
import numpy as np
import torch

def fft_torch_native(data, dim=[1, 2], combine="abs"):
    data = torch.fft.fft2(data, dim=dim)
    if combine == "abs":
        return torch.absolute(data)
    elif combine == "stack":
        return torch.stack([data.real, data.imag], dim=1)
    else:
        return data


def fft_complex_1d(x):
    """
    A non-recursive implementation of
    the 1D Cooley-Tukey FFT, the
    input should have a length of
    power of 2.
    """
    shape = x.shape[0]
    T = x.shape[2]
    n_min = 2  # # only if N > 32
    n = torch.arange(n_min)
    k = n[:, None]
    m = torch.exp(-2j * np.pi * n * k / n_min)
    X = m @ x.cfloat().reshape((n_min, -1))
    X = X.reshape((X.shape[0], -1, T))

    while X.shape[0] < shape:
        x_even = X[:, :X.shape[1] // 2]
        x_odd = X[:, X.shape[1] // 2:]
        factor = torch.exp(-1j * np.pi * torch.arange(X.shape[0]) / X.shape[0])[:, None]

        fx = (factor * x_odd.reshape((factor.shape[0], -1))).reshape((factor.shape[0], -1, T))

        X = torch.vstack([x_even + fx, x_even - fx])

    return X


def rfft(x_r, dim=0):
    x_r = x_r.transpose(0, dim)
    n = x_r.shape[0]
    t = math.prod(x_r.shape[1:])
    x = fft_complex_1d(x_r.reshape(n, 1, t))
    x /= math.sqrt(n)
    x = x[0:n // 2 + 1]
    return (x.view((n // 2 + 1,) + x_r.shape[1:])).transpose(0, dim)


def my_fft(x_r, dim=0):
    x_r = x_r.transpose(0, dim)
    n = x_r.shape[0]
    t = math.prod(x_r.shape[1:])
    x = fft_complex_1d(x_r.reshape(n, 1, t))
    x /= math.sqrt(n)
    return x.view((n,) + x_r.shape[1:]).transpose(0, dim)


def irfft(x, dim=0):
    x_r = x.transpose(0, dim)
    x_r = torch.concatenate([x_r, x_r[1:-1].flip(0)])
    n = x_r.shape[0]
    t = math.prod(x_r.shape[1:])
    x = fft_complex_1d(x_r.reshape(n, 1, t))
    x /= math.sqrt(n)
    return x.view((n,) + x_r.shape[1:]).transpose(0, dim)


def ifft(x, dim=0):
    x_r = x.transpose(0, dim)
    n = x_r.shape[0]
    t = math.prod(x_r.shape[1:])
    x = fft_complex_1d(x_r.reshape(n, 1, t))
    x /= math.sqrt(n)
    return x.view((n,) + x_r.shape[1:]).transpose(0, dim)


def rfftn(x_r, dim):
    return my_fft(rfft(x_r, dim=dim[1]), dim=dim[0])


def irfftn(x, dim):
    return irfft(ifft(x, dim=dim[0]), dim=dim[1])


def fft_torch(x, dim=(2, 3), combine=""):
    data = rfftn(x, dim=dim)
    if combine == "abs":
        return torch.absolute(data)
    elif combine == "stack":
        return torch.stack([data.real, data.imag], dim=1)
    else:
        return data


def wavelet_torch(data, combine="abs"):
    raise NotImplementedError  # TODO


if __name__ == '__main__':
    from pytorch_wavelets import DTCWTForward, DTCWTInverse
    img = torch.randn((1, 3, 512, 512)).float()


    xfm = DTCWTForward(J=3, biort='near_sym_b', qshift='qshift_b')
    Yl, Yh = xfm(img)
    ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
    Y = ifm((Yl, Yh))

    f_4d_torch = torch.fft.rfftn(img, dim=(2, 3), norm='ortho')
    f_4d_my_c = rfftn(img, dim=(2, 3))

    torch.testing.assert_close(f_4d_torch, f_4d_my_c, check_stride=False, rtol=1e-3, atol=1e-3)
