import torch
import numpy as np
from lib.pysixd.pose_error import re, te


def get_2d_coord(bs, width, height, dtype=torch.float32, device="cuda"):
    """
    Args:
        bs: batch size
        width:
        height:
    """
    # coords values are in [-1, 1]
    x = np.linspace(-1, 1, width, dtype=np.float32)
    y = np.linspace(-1, 1, height, dtype=np.float32)
    xy = np.meshgrid(x, y)
    coord = np.stack([xy for _ in range(bs)])
    coord_tensor = torch.tensor(coord, dtype=dtype, device=device)
    coord_tensor = coord_tensor.view(bs, 2, height, width)

    return coord_tensor  # [bs, 2, h, w]


def get_mask_prob(cfg, pred_mask):
    # (b,c,h,w)
    # output: (b, 1, h, w)
    mask_loss_type = cfg.MODEL.CDPN.ROT_HEAD.MASK_LOSS_TYPE
    bs, c, h, w = pred_mask.shape
    if mask_loss_type == "L1":
        assert c == 1, c
        mask_max = torch.max(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        mask_min = torch.min(pred_mask.view(bs, -1), dim=-1)[0].view(bs, 1, 1, 1)
        # [0, 1]
        mask_prob = (pred_mask - mask_min) / (mask_max - mask_min)  # + 1e-6)
    elif mask_loss_type == "BCE":
        assert c == 1, c
        mask_prob = torch.sigmoid(pred_mask)
    elif mask_loss_type == "CE":
        mask_prob = torch.softmax(pred_mask, dim=1, keepdim=True)[:, 1:2, :, :]
    else:
        raise NotImplementedError(f"unknown mask loss type: {mask_loss_type}")
    return mask_prob


def compute_mean_re_te(pred_transes, pred_rots, gt_transes, gt_rots):
    pred_transes = pred_transes.detach().cpu().numpy()
    pred_rots = pred_rots.detach().cpu().numpy()
    gt_transes = gt_transes.detach().cpu().numpy()
    gt_rots = gt_rots.detach().cpu().numpy()

    bs = pred_rots.shape[0]
    R_errs = np.zeros((bs,), dtype=np.float32)
    T_errs = np.zeros((bs,), dtype=np.float32)
    for i in range(bs):
        R_errs[i] = re(pred_rots[i], gt_rots[i])
        T_errs[i] = te(pred_transes[i], gt_transes[i])
    return R_errs.mean(), T_errs.mean()
