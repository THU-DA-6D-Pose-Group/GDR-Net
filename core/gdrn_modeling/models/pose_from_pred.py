import numpy as np
import torch

from core.utils.utils import allocentric_to_egocentric, allocentric_to_egocentric_torch, allo_to_ego_mat_torch

from lib.pysixd import RT_transform
from core.utils.pose_utils import quat2mat_torch
from lib.utils.utils import dprint


# TODO: pred normalized translation like single-stage


def pose_from_pred(pred_rots, pred_transes, eps=1e-4, is_allo=True, is_train=True):
    if is_train:
        return pose_from_predictions_train(pred_rots, pred_transes, eps=eps, is_allo=is_allo)
    else:
        return pose_from_predictions_test(pred_rots, pred_transes, eps=eps, is_allo=is_allo)


def pose_from_predictions_test(pred_rots, pred_transes, eps=1e-4, is_allo=True):
    """NOTE: for test, non-differentiable"""
    translation = pred_transes

    # quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
    # quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
    # use numpy since it is more accurate
    if pred_rots.shape[-1] == 4 and pred_rots.ndim == 2:
        pred_quats = pred_rots.detach().cpu().numpy()  # allo
        ego_rot_preds = np.zeros((pred_quats.shape[0], 3, 3), dtype=np.float32)
        for i in range(pred_quats.shape[0]):
            # try:
            if is_allo:
                # this allows unnormalized quat
                cur_ego_mat = allocentric_to_egocentric(
                    RT_transform.quat_trans_to_pose_m(pred_quats[i], translation[i].detach().cpu().numpy()),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = RT_transform.quat_trans_to_pose_m(pred_quats[i], translation[i].detach().cpu().numpy())
            ego_rot_preds[i] = cur_ego_mat
            # except:

    # rot mat
    if pred_rots.shape[-1] == 3 and pred_rots.ndim == 3:
        pred_rots = pred_rots.detach().cpu().numpy()
        ego_rot_preds = np.zeros_like(pred_rots)
        for i in range(pred_rots.shape[0]):
            if is_allo:
                cur_ego_mat = allocentric_to_egocentric(
                    np.hstack([pred_rots[i], translation[i].detach().cpu().numpy().reshape(3, 1)]),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = pred_rots[i]
            ego_rot_preds[i] = cur_ego_mat
    return torch.from_numpy(ego_rot_preds), translation


def pose_from_predictions_train(pred_rots, pred_transes, eps=1e-4, is_allo=True):
    """for train
    Args:
        pred_rots:
        pred_transes:
        eps:
        is_allo:

    Returns:

    """
    translation = pred_transes

    if pred_rots.ndim == 2 and pred_rots.shape[-1] == 4:
        pred_quats = pred_rots
        quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
        if is_allo:
            quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
        else:
            quat_ego = quat_allo
        rot_ego = quat2mat_torch(quat_ego)
    if pred_rots.ndim == 3 and pred_rots.shape[-1] == 3:  # Nx3x3
        if is_allo:
            rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)
        else:
            rot_ego = pred_rots
    return rot_ego, translation
