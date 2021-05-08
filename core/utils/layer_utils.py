import torch
from torch import nn
import torch.nn.functional as F
from detectron2.layers.batch_norm import BatchNorm2d, FrozenBatchNorm2d, NaiveSyncBatchNorm
from detectron2.utils import comm, env


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


def get_norm(norm, out_channels, num_gn_groups=32):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.
    Returns:
        nn.Module or None: the normalization layer
    """
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "BN": BatchNorm2d,
            # Fixed in https://github.com/pytorch/pytorch/pull/36382
            "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(num_gn_groups, channels),
            # for debugging:
            "nnSyncBN": nn.SyncBatchNorm,
            "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels)


def soft_argmax(x, beta=1000.0, dim=1, base_index=0, step_size=1, keepdim=False):
    """Compute the forward pass of the soft arg-max function as defined below:

    SoftArgMax(x) = \sum_i (i * softmax(x)_i)
    :param x: The input to the soft arg-max layer
    :return: Output of the soft arg-max layer
    """
    smax = F.softmax(x * beta, dim=dim)
    end_index = base_index + x.shape[dim] * step_size
    indices = torch.arange(start=base_index, end=end_index, step=step_size).to(x)
    view_shape = [1 for _ in x.shape]
    view_shape[dim] = x.shape[dim]
    indices = indices.view(view_shape)
    return torch.sum(smax * indices, dim=dim, keepdim=keepdim)


def gumbel_soft_argmax(x, tau=1.0, dim=1, hard=True, eps=1e-10, base_index=0, step_size=1, keepdim=False):
    """
    NOTE: this is stochastic
    """
    gsmax = F.gumbel_softmax(x, tau=tau, dim=dim, hard=hard, eps=eps)
    end_index = base_index + x.shape[dim] * step_size
    indices = torch.arange(start=base_index, end=end_index, step=step_size).to(x)
    view_shape = [1 for _ in x.shape]
    view_shape[dim] = x.shape[dim]
    indices = indices.view(view_shape)
    return torch.sum(gsmax * indices, dim=dim, keepdim=keepdim)
