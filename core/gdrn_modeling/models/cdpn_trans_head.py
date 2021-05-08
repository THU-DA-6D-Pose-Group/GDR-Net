import torch.nn as nn
import torch
from mmcv.cnn import normal_init, constant_init
from core.utils.layer_utils import get_norm
from torch.nn.modules.batchnorm import _BatchNorm


class TransHeadNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_layers=3,
        num_filters=256,
        kernel_size=3,
        output_dim=3,
        freeze=False,
        norm="BN",
        num_gn_groups=32,
    ):
        super().__init__()

        self.freeze = freeze

        if kernel_size == 3:
            padding = 1
        elif kernel_size == 2:
            padding = 0

        self.features = nn.ModuleList()
        for i in range(num_layers):
            _in_channels = in_channels if i == 0 else num_filters
            self.features.append(
                nn.Conv2d(_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))

        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(256 * 8 * 8, 4096))
        self.linears.append(nn.ReLU(inplace=True))
        self.linears.append(nn.Linear(4096, 4096))
        self.linears.append(nn.ReLU(inplace=True))
        self.linears.append(nn.Linear(4096, output_dim))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                for i, l in enumerate(self.features):
                    x = l(x)
                x = x.view(-1, 256 * 8 * 8)
                for i, l in enumerate(self.linears):
                    x = l(x)
                return x.detach()
        else:
            for i, l in enumerate(self.features):
                x = l(x)
            x = x.view(-1, 256 * 8 * 8)
            for i, l in enumerate(self.linears):
                x = l(x)
            return x
