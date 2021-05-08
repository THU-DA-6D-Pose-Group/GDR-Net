import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn.functional as F
from mmcv.cnn import normal_init, constant_init


def SoftPool(x, N_p=32):
    """
    Args:
        x: (B, F, P)
        N_p: top k

    Returns:
        (B, F, N_p, F)
    """
    x = F.softmax(x, dim=1)
    bs, featdim = x.shape[:2]
    # (B, F, k, F)
    sp_cube = torch.zeros(bs, featdim, N_p, featdim).cuda()
    for idx in range(featdim):
        # x_val, x_idx = torch.sort(x[:, idx, :], dim=1, descending=True)  # (B, P)
        x_val, x_idx = torch.topk(x[:, idx, :], k=N_p, dim=1)  # (B, N_p)
        index = x_idx[:, :N_p].unsqueeze(1).repeat(1, featdim, 1)  # (B, F, N_p)
        sp_cube[:, :, :, idx] = torch.gather(x, dim=2, index=index)
    return sp_cube


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


class PointPnPNet(nn.Module):
    def __init__(
        self,
        nIn,
        spatial_pooltype="max",
        spatial_topk=1,
        region_softpool=False,
        num_regions=8,
        region_topk=8,
        rot_dim=4,
        mask_attention_type="none",
    ):  # NOTE: not used!!!
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
            region_softpool (bool): if not softpool, just flatten
        """
        super().__init__()
        self.mask_attention_type = mask_attention_type
        self.spatial_pooltype = spatial_pooltype
        self.spatial_topk = spatial_topk
        self.region_softpool = region_softpool
        self.num_regions = num_regions
        self.region_topk = region_topk
        # -----------------------------------

        self.conv1 = torch.nn.Conv1d(nIn, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)

        if self.spatial_pooltype == "topk":
            self.conv_topk = nn.Conv2d(128, 128, kernel_size=(1, self.spatial_topk), stride=(1, 1))

        if not region_softpool:
            in_dim = 128 * num_regions
        else:
            in_dim = 128 * region_topk
            self.conv_sp = nn.Conv2d(128, 128, kernel_size=(1, 128), stride=(1, 1))

        # self.fc1 = nn.Linear(in_dim + 128, 512)  # NOTE: 128 for extents feature
        self.fc1 = nn.Linear(in_dim, 512)  # NOTE: no extent feature
        self.fc2 = nn.Linear(512, 256)
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
            x: (B,C,H,W),
            region: (B, M, H, W) after softmax
            extents: (B, 3)
        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape
        if in_c == 3 or in_c == 5:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1, 1)

        # B,F,M,k
        # x = topk_pool_with_region(x, region.view(bs, self.num_regions, -1), k=self.spatial_topk)

        x = coor_feat

        x = x.view(bs, in_c, -1)  # (B,C,N)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)  # (B,128,N)

        x = x.view(bs, 128, 1, fh, fw) * region.view(bs, 1, self.num_regions, fh, fw)
        # BCMHW

        x = x.view(bs, 128, self.num_regions, -1)
        # -----------------------------------------
        # spatial pooling for each region
        if self.spatial_pooltype == "max":
            # (B, 128, num_regions, 1)
            x = torch.max(x, dim=-1, keepdim=True)[0]
        elif self.spatial_pooltype == "mean":
            x = torch.mean(x, dim=-1, keepdim=True)
        elif self.spatial_pooltype == "topk":  # seems bad
            # pool topk features with region prob as reference
            # (B,128,num_regions,topk)
            # x = topk_pool_with_region(x, region.view(bs, self.num_regions, -1), k=self.spatial_topk)
            x = topk_pool(x, k=self.spatial_topk)
            x = self.act(self.conv_topk(x))
        elif self.spatial_pooltype == "soft":
            # (B, 128, num_regions, N)
            # x_rs = []
            # for r in range(self.num_regions):  # Very slow !!!
            #     x_r = x[:, :, r, :]  # B,128,HW
            #     x_r_sp = SoftPool(x_r, N_p=self.spatial_topk)  # B,128, topk, 128
            #     x_rs.append(x_r_sp)
            # x_rs = torch.cat(x_rs, dim=2)  # B,128,num_regions*topk, 128
            # NOTE: slow
            x = x.permute(0, 2, 1, 3).reshape(bs * self.num_regions, 128, -1)
            x_sp = SoftPool(x, N_p=self.spatial_topk).reshape(bs, self.num_regions, 128, -1)
            x = x_sp.permute(0, 2, 1, 3).contiguous()
            x = torch.max(x, dim=3, keepdim=True)[0]  # NOTE: need to reduce dim for each region
        else:
            raise ValueError(f"Unknown spatial pool type: {self.spatial_pooltype}")

        # -----------------------------------------
        # softpool for regions (or just flatten)
        # Bxnum_regionsxF --> BxkFxF
        x = x.view(bs, 128, self.num_regions)
        if self.region_softpool:
            x_sp = SoftPool(x, N_p=self.region_topk)  # B,128,topk,128
            x = self.conv_sp(x_sp)  # B,128,topk
            x = self.act(x)

        x = x.view(bs, -1)

        # extent feature
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


class SimplePointPnPNet(nn.Module):
    """https://github.com/cvlab-epfl/single-stage-pose/blob/master/model.py."""

    def __init__(self, nIn, rot_dim=6, use_softpool=False, softpool_topk=32, mask_attention_type="none"):
        # NOTE: softpool is much slower
        super(SimplePointPnPNet, self).__init__()
        self.mask_attention_type = mask_attention_type
        self.use_softpool = use_softpool
        self.softpool_topk = softpool_topk

        self.conv1 = torch.nn.Conv1d(nIn, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        if self.use_softpool:
            self.conv3 = torch.nn.Conv1d(128, 128, 1)
            self.conv_sp = nn.Conv2d(128, 128, kernel_size=(1, 128), stride=(1, 1))
            self.fc1 = nn.Linear(128 * self.softpool_topk, 512)
        else:
            self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.fc1 = nn.Linear(1024, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc_pose = nn.Linear(256, rot_dim + 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.rot_dim = rot_dim

    def forward(self, coor_feat, region=None, extents=None, mask_attention=None):
        """
        Args:
            x: (B,C,N)

        Returns:

        """
        bs, in_c, fh, fw = coor_feat.shape
        if in_c == 3 or in_c == 5:
            coor_feat[:, :3, :, :] = (coor_feat[:, :3, :, :] - 0.5) * extents.view(bs, 3, 1, 1)

        if region is not None:
            features = torch.cat([coor_feat, region], dim=1)
            feat_dim = features.size(1)
        else:
            features = coor_feat
            feat_dim = in_c

        if self.mask_attention_type != "none":
            assert mask_attention is not None
            if self.mask_attention_type == "mul":
                features = features * mask_attention
            elif self.mask_attention_type == "concat":
                features = torch.cat([features, mask_attention], dim=1)
                feat_dim += 1
            else:
                raise ValueError(f"Wrong mask attention type: {self.mask_attention_type}")

        x = features.view(bs, feat_dim, -1)

        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)  # (B,1024,N)
        if self.use_softpool:
            x = x.view(bs, 128, -1)
            x_sp = SoftPool(x, N_p=self.softpool_topk)  # B,128,topk,128
            x = self.conv_sp(x_sp)  # B,128,topk
            x = self.act(x)
            x = x.view(bs, 128 * self.softpool_topk)
        else:
            x = x.view(bs, 1024, -1)
            # TODO: maybe here can be replaced by SoftPool
            x = torch.max(x, dim=2, keepdim=True)[0]  # (B, 1024, 1)
            # x = torch.mean(x, dim=2, keepdim=True)
            x = x.view(bs, 1024)
        #
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        #
        pose = self.fc_pose(x)
        rot = pose[:, : self.rot_dim]
        t = pose[:, self.rot_dim : self.rot_dim + 3]
        return rot, t


def test_softpool():
    dim_pn = 128
    bottleneck_size = 256
    N_p = 32
    x = torch.rand(6, dim_pn, 64 * 64 * 64).to(device="cuda", dtype=torch.float32)
    print(x.shape)
    y = SoftPool(x, N_p)
    # y2 = SoftPoolv2(x, N_p)
    # import ipdb; ipdb.set_trace()
    print("after softpool: ", y.shape)
    # print("diff: ", torch.max(y - y2), torch.mean(y - y2))
    conv8 = torch.nn.Conv2d(dim_pn, bottleneck_size, kernel_size=(1, dim_pn), stride=(1, 1)).to("cuda")
    conv9 = torch.nn.Conv2d(bottleneck_size, bottleneck_size, kernel_size=(N_p, 1), stride=(1, 1)).to("cuda")
    z = conv8(y)
    print(f"after conv (1, {dim_pn}): ", z.shape)
    z = conv9(z)
    print(f"after conv ({N_p}, 1): ", z.shape)  # (B, 256, 1, 1)
    # concat with the original point feature ?
    # z = z.view(-1, 256, 1).repeat(npoints)
    # z = torch.cat([z, pointfeat], 1)


if __name__ == "__main__":
    test_softpool()
