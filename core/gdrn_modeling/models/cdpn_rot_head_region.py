import torch.nn as nn
import torch
from mmcv.cnn import normal_init, kaiming_init, constant_init
from core.utils.layer_utils import get_norm
from torch.nn.modules.batchnorm import _BatchNorm
from .resnet_backbone import resnet_spec


class RotWithRegionHead(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        num_layers=3,
        num_filters=256,
        kernel_size=3,
        output_kernel_size=1,
        rot_output_dim=3,
        mask_output_dim=1,
        freeze=False,
        num_classes=1,
        rot_class_aware=False,
        mask_class_aware=False,
        region_class_aware=False,
        num_regions=8,
        norm="BN",
        num_gn_groups=32,
    ):
        super().__init__()

        self.freeze = freeze
        self.concat = cfg.MODEL.CDPN.ROT_HEAD.ROT_CONCAT
        assert kernel_size == 2 or kernel_size == 3 or kernel_size == 4, "Only support kenerl 2, 3 and 4"
        assert num_regions > 1, f"Only support num_regions > 1, but got {num_regions}"
        padding = 1
        output_padding = 0
        if kernel_size == 3:
            output_padding = 1
        elif kernel_size == 2:
            padding = 0

        assert output_kernel_size == 1 or output_kernel_size == 3, "Only support kenerl 1 and 3"
        if output_kernel_size == 1:
            pad = 0
        elif output_kernel_size == 3:
            pad = 1

        if self.concat:
            _, _, channels, _ = resnet_spec[cfg.MODEL.CDPN.BACKBONE.NUM_LAYERS]
            self.features = nn.ModuleList()
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(
                        num_filters + channels[-2 - i], num_filters, kernel_size=3, stride=1, padding=1, bias=False
                    )
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))

                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))
        else:
            self.features = nn.ModuleList()
            self.features.append(
                nn.ConvTranspose2d(
                    in_channels,
                    num_filters,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
            for i in range(num_layers):
                # _in_channels = in_channels if i == 0 else num_filters
                # self.features.append(
                #    nn.ConvTranspose2d(_in_channels, num_filters, kernel_size=kernel_size, stride=2, padding=padding,
                #                       output_padding=output_padding, bias=False))
                # self.features.append(nn.BatchNorm2d(num_filters))
                # self.features.append(nn.ReLU(inplace=True))
                if i >= 1:
                    self.features.append(nn.UpsamplingBilinear2d(scale_factor=2))
                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))

                self.features.append(
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, bias=False)
                )
                self.features.append(get_norm(norm, num_filters, num_gn_groups=num_gn_groups))
                self.features.append(nn.ReLU(inplace=True))

        self.rot_output_dim = rot_output_dim
        if rot_class_aware:
            self.rot_output_dim *= num_classes

        self.mask_output_dim = mask_output_dim
        if mask_class_aware:
            self.mask_output_dim *= num_classes

        self.region_output_dim = num_regions + 1  # add one channel for bg
        if region_class_aware:
            self.region_output_dim *= num_classes

        self.features.append(
            nn.Conv2d(
                num_filters,
                self.mask_output_dim + self.rot_output_dim + self.region_output_dim,
                kernel_size=output_kernel_size,
                padding=pad,
                bias=True,
            )
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)

    def forward(self, x, x_f64=None, x_f32=None, x_f16=None):
        if self.concat:
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        if i == 3:
                            x = l(torch.cat([x, x_f16], 1))
                        elif i == 12:
                            x = l(torch.cat([x, x_f32], 1))
                        elif i == 21:
                            x = l(torch.cat([x, x_f64], 1))
                        x = l(x)
                    return x.detach()
            else:
                for i, l in enumerate(self.features):
                    if i == 3:
                        x = torch.cat([x, x_f16], 1)
                    elif i == 12:
                        x = torch.cat([x, x_f32], 1)
                    elif i == 21:
                        x = torch.cat([x, x_f64], 1)
                    x = l(x)
                return x
        else:
            if self.freeze:
                with torch.no_grad():
                    for i, l in enumerate(self.features):
                        x = l(x)
                    mask = x[:, : self.mask_output_dim, :, :]
                    xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
                    region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
                    bs, c, h, w = xyz.shape
                    xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
                    coor_x = xyz[:, 0, :, :, :]
                    coor_y = xyz[:, 1, :, :, :]
                    coor_z = xyz[:, 2, :, :, :]
                    return (mask.detach(), coor_x.detach(), coor_y.detach(), coor_z.detach(), region.detach())
            else:
                for i, l in enumerate(self.features):
                    x = l(x)
                mask = x[:, : self.mask_output_dim, :, :]
                xyz = x[:, self.mask_output_dim : self.mask_output_dim + self.rot_output_dim, :, :]
                region = x[:, self.mask_output_dim + self.rot_output_dim :, :, :]
                bs, c, h, w = xyz.shape
                xyz = xyz.view(bs, 3, self.rot_output_dim // 3, h, w)
                coor_x = xyz[:, 0, :, :, :]
                coor_y = xyz[:, 1, :, :, :]
                coor_z = xyz[:, 2, :, :, :]
                return mask, coor_x, coor_y, coor_z, region
