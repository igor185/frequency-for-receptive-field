from torch import nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, use_fourier, stride=1, skip_connection=None, groups=1,
                 base_width=64, ratio_gin=0.5, ratio_gout=0.5, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = FFCBlock(inplanes, width, use_fourier, kernel_size=3, padding=1, stride=stride,
                              ratio_gin=ratio_gin, ratio_gout=ratio_gout, norm_layer=norm_layer,
                              activation_layer=nn.ReLU)
        self.conv2 = FFCBlock(width, planes * self.expansion, use_fourier, kernel_size=3, padding=1,
                              ratio_gin=ratio_gout, ratio_gout=ratio_gout, norm_layer=norm_layer)
        self.relu_l = nn.Identity() if ratio_gout == 1 else nn.ReLU(inplace=True)
        self.relu_g = nn.Identity() if ratio_gout == 0 else nn.ReLU(inplace=True)
        self.skip = skip_connection
        self.stride = stride

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x if self.skip is None else self.skip(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x_l, x_g = x

        x_l = self.relu_l(x_l + id_l)
        x_g = self.relu_g(x_g + id_g)

        return x_l, x_g


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, use_fourier, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False):
        super(FFC, self).__init__()

        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias)
        module = nn.Identity if not use_fourier or (in_cg == 0 or out_cg == 0) else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFCBlock(nn.Module):

    def __init__(self, in_channels, out_channels, use_fourier,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity):
        super(FFCBlock, self).__init__()
        self.ffc = FFC(in_channels, out_channels, use_fourier, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        self.bn_l = lnorm(int(out_channels * (1 - ratio_gout)))
        self.bn_g = gnorm(int(out_channels * ratio_gout))

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(SpectralTransform, self).__init__()
        self.down_sample = nn.AvgPool2d(kernel_size=(2, 2), stride=2) if stride == 2 else nn.Identity()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)

        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        if self.down_sample:
            x = self.down_sample(x)
        x = self.conv1(x)
        output = self.fu(x)

        output = self.conv2(x + output)

        return output


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, h, w = x.size()

        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, dim=2, norm="ortho")
        ffted = torch.stack([ffted.real, ffted.imag], dim=-1)
        # (batch, c, 2, h, w/2+1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

        splitted = torch.split(ffted, split_size_or_sections=1, dim=-1)
        ffted = torch.complex(splitted[0][:, :, :, :, 0], splitted[1][:, :, :, :, 0])
        output = torch.fft.irfft(ffted, dim=2, norm="ortho")

        return output


class FFCResNet(nn.Module):

    def __init__(self, layers, use_fourier, num_classes=1000, width_per_group=64, ratio=0.5, lfu=False,
                 use_se=False):
        super(FFCResNet, self).__init__()

        self._norm_layer = nn.BatchNorm2d

        self.channels = 64
        self.use_fourier = use_fourier
        self.dilation = 1
        self.base_width = width_per_group
        self.lfu = lfu
        self.use_se = use_se
        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.channels)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64 * 1, layers[0], stride=1, ratio_gin=0, ratio_gout=ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(32, num_classes), nn.Softmax())


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1, ratio_gin=0.5, ratio_gout=0.5):
        skip_connection = None
        if stride != 1 or self.channels != planes or ratio_gin == 0:
            skip_connection = FFCBlock(self.channels, planes, kernel_size=1, stride=stride,
                                         ratio_gin=ratio_gin, ratio_gout=ratio_gout, use_fourier=self.use_fourier)

        layers = []
        layers.append(BasicBlock(self.channels, planes, self.use_fourier, stride, skip_connection, self.base_width,
                                 self.dilation, ratio_gin, ratio_gout, ))
        self.channels = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(self.channels, planes, self.use_fourier, base_width=self.base_width,
                           ratio_gin=ratio_gout, ratio_gout=ratio_gout))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.avg_pool(x[0])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18_small(**kwargs):
    return FFCResNet([2], **kwargs)


def resnet34_small(**kwargs):
    return FFCResNet([3], **kwargs)
