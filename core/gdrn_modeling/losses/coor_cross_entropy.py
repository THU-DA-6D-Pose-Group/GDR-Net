import torch
from torch import nn


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, (
        "nn criterions don't compute the gradient w.r.t. targets - please "
        "mark these tensors as not requiring gradients"
    )


class CrossEntropyHeatmapLoss(nn.Module):
    def __init__(self, reduction, weight=None):
        super(CrossEntropyHeatmapLoss, self).__init__()
        self.m = nn.LogSoftmax(dim=1)
        if weight is not None:  # bin_size+1
            weight_ = torch.ones(weight)
            weight_[weight - 1] = 0  # bg
            self.loss = nn.NLLLoss(reduction=reduction, weight=weight_)
        else:
            self.loss = nn.NLLLoss(reduction=reduction)

    def forward(self, coor, gt_coor):
        _assert_no_grad(gt_coor)
        loss = self.loss(self.m(coor), gt_coor)
        return loss
