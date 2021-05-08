import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init
from core.utils.layer_utils import get_norm
from core.utils.dropblock import DropBlock2D, LinearScheduler


def topk_pool_with_region(x, region, k=32):
    """
    Args:
        x: (B, F, M, P)
        region: (B, M, P)
        k: top k

    Returns:
        (B, F, M, topk)
    """
    featdim = x.shape[1]
    _, region_idx = torch.topk(region, k=k, dim=2)  # (B, M, k)
    index = region_idx.unsqueeze(1).repeat(1, featdim, 1, 1)  # (B, F, M, k)
    pooled = torch.gather(x, dim=3, index=index)
    return pooled


def topk_pool(x, k=32):
    """
    Args:
        x: (B, F, M, P)
        k: top k

    Returns:
        (B, F, M, topk)
    """
    _, idx = torch.topk(x, k=k, dim=3)  # (B, F, M, k)
    pooled = torch.gather(x, dim=3, index=idx)
    return pooled


class ConvPnPNet(nn.Module):
    def __init__(
        self,
        nIn,
        featdim=128,
        rot_dim=4,
        num_layers=3,
        norm="GN",
        num_gn_groups=32,
        num_regions=8,
        drop_prob=0.0,
        dropblock_size=5,
        mask_attention_type="none",
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
        """
        super().__init__()
        self.featdim = featdim
        self.num_regions = num_regions
        self.mask_attention_type = mask_attention_type
        # -----------------------------------
        self.drop_prob = drop_prob
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=drop_prob,
            nr_steps=5000,
        )

        assert num_layers >= 3, num_layers
        self.features = nn.ModuleList()
        for i in range(3):
            _in_channels = nIn if i == 0 else featdim
            self.features.append(nn.Conv2d(_in_channels, featdim, kernel_size=3, stride=2, padding=1, bias=False))
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))
        for i in range(num_layers - 3):  # when num_layers > 3
            self.features.append(nn.Conv2d(featdim, featdim, kernel_size=3, stride=1, padding=1, bias=False))
            self.features.append(get_norm(norm, featdim, num_gn_groups=num_gn_groups))
            self.features.append(nn.ReLU(inplace=True))

        # self.fc1 = nn.Linear(featdim * 8 * 8 + 128, 1024)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(featdim * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc_r = nn.Linear(256, rot_dim)  # quat or rot6d
        # TODO: predict centroid and z separately
        self.fc_t = nn.Linear(256, 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        # feature for extent
        # self.extent_fc1 = nn.Linear(3, 64)
        # self.extent_fc2 = nn.Linear(64, 128)

        # init ------------------------------------
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_r, std=0.01)
        normal_init(self.fc_t, std=0.01)

    def forward(self, coor_feat, region=None, extents=None, mask_attention=None):
        """
        Args:
             since this is the actual correspondence
            x: (B,C,H,W)
            extents: (B, 3)
        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape
        if in_c == 3 or in_c == 5:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1, 1)
        # convs
        if region is not None:
            x = torch.cat([coor_feat, region], dim=1)
        else:
            x = coor_feat

        if self.mask_attention_type != "none":
            assert mask_attention is not None
            if self.mask_attention_type == "mul":
                x = x * mask_attention
            elif self.mask_attention_type == "concat":
                x = torch.cat([x, mask_attention], dim=1)
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        if self.drop_prob > 0:
            self.dropblock.step()  # increment number of iterations
            x = self.dropblock(x)

        for _i, layer in enumerate(self.features):
            x = layer(x)

        x = x.view(-1, self.featdim * 8 * 8)
        # extent feature
        # # TODO: use extent the other way: denormalize coords
        # x_extent = self.act(self.extent_fc1(extents))
        # x_extent = self.act(self.extent_fc2(x_extent))
        # x = torch.cat([x, x_extent], dim=1)
        #
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        #
        rot = self.fc_r(x)
        t = self.fc_t(x)
        return rot, t
