"""
ref:
https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
https://github.com/arraiyopensource/kornia/blob/master/kornia/geometry/conversions.py
"""
from math import acos, cos, pi, sin

import numpy as np
import torch
import torch.nn.functional as F
from numba import jit, njit
from numpy import linalg as LA
from transforms3d.axangles import axangle2mat, mat2axangle
from transforms3d.euler import _AXES2TUPLE, _NEXT_AXIS, _TUPLE2AXES, euler2mat, euler2quat, mat2euler, quat2euler
from transforms3d.quaternions import mat2quat, quat2mat

from lib.pysixd.pose_error import re

pixel_coords = None


def qmul_torch(q, r):
    """Multiply quaternion(s) q with quaternion(s) r.

    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def qrot_torch(q, v):
    """Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,

    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    # qmult(q, qmult(varr, qconjugate(q)))[1:]
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def qrot_points_th(q, points):
    """
    q: (4,)
    points: (N, 3)
    """
    assert q.numel() == 4, q.numel()
    assert points.shape[1] == 3, points.shape[1]
    N = points.shape[0]
    points_q = qrot_torch(q.expand(N, 4), points)
    return points_q


def euler2quat_torch(ai, aj, ak, axes="sxyz"):
    """slower than numpy version batch."""
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis + 1
    j = _NEXT_AXIS[i + parity - 1] + 1
    k = _NEXT_AXIS[i - parity] + 1
    # print(i, j, k)

    ai, aj, ak = ai.clone(), aj.clone(), ak.clone()
    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = torch.cos(ai)
    si = torch.sin(ai)
    cj = torch.cos(aj)
    sj = torch.sin(aj)
    ck = torch.cos(ak)
    sk = torch.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk

    B = len(ai)
    if repetition:
        qw = cj * (cc - ss)
        qi = cj * (cs + sc)
        qj = sj * (cc + ss)
        qk = sj * (cs - sc)
    else:
        qw = cj * cc + sj * ss
        qi = cj * sc - sj * cs
        qj = cj * ss + sj * cc
        qk = cj * cs - sj * sc
    if parity:
        qj *= -1.0
    order = {i: 1, j: 2, k: 3}
    q = torch.stack((qw, qi, qj, qk), dim=1)[:, [0, order[1], order[2], order[3]]]
    if B == 1:
        q = q.view(4)
    return q


def quat2euler_torch(q, order="zyx", epsilon=0):
    """NOTE: zyx is the same as sxyz in transforms3d
    https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
    # i,j,k ==> zyx
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    norm_quat = q.norm(p=2, dim=-1, keepdim=True)
    # print('norm_quat: ', norm_quat)  # Bx1
    q = q / norm_quat
    # print(q)

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert all(condition), "wrong size for {}, expected {}, got  {}".format(
        input_name, "x".join(expected), list(input.size())
    )


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.

    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat_torch(angle):
    """Convert euler angles to rotation matrix.

    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach() * 0
    ones = zeros.detach() + 1
    zmat = torch.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat


def axangle2mat_torch(axis, angle, is_normalized=False):
    """Rotation matrix for rotation angle `angle` around `axis`
    Parameters
    ----------
    axis : [B, 3] element sequence
       vector specifying axis for rotation.
    angle :[B, ] scalar
       angle of rotation in radians.
    is_normalized : bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False.
    Returns
    -------
    mat : array shape (B, 3,3)
       rotation matrix for specified rotation
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    """
    B = axis.shape[0]

    if not is_normalized:
        norm_axis = axis.norm(p=2, dim=1, keepdim=True)
        normed_axis = axis / norm_axis
    else:
        normed_axis = axis
    x, y, z = normed_axis[:, 0], normed_axis[:, 1], normed_axis[:, 2]
    c = torch.cos(angle)
    s = torch.sin(angle)
    C = 1 - c
    # yapf: disable
    xs  = x * s;   ys = y * s;   zs = z * s  # noqa
    xC  = x * C;   yC = y * C;   zC = z * C  # noqa
    xyC = x * yC; yzC = y * zC; zxC = z * xC  # noqa
    # yapf: enable
    return torch.stack(
        [x * xC + c, xyC - zs, zxC + ys, xyC + zs, y * yC + c, yzC - xs, zxC - ys, yzC + xs, z * zC + c], dim=1
    ).reshape(B, 3, 3)


def quat2mat_torch(quat, eps=0.0):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: [B, 4]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, quat.shape
    norm_quat = quat.norm(p=2, dim=1, keepdim=True)
    # print('quat', quat) # Bx4
    # print('norm_quat: ', norm_quat)  # Bx1
    norm_quat = quat / (norm_quat + eps)
    # print('normed quat: ', norm_quat)
    qw, qx, qy, qz = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    B = quat.size(0)

    s = 2.0  # * Nq = qw*qw + qx*qx + qy*qy + qz*qz
    X = qx * s
    Y = qy * s
    Z = qz * s
    wX = qw * X
    wY = qw * Y
    wZ = qw * Z
    xX = qx * X
    xY = qx * Y
    xZ = qx * Z
    yY = qy * Y
    yZ = qy * Z
    zZ = qz * Z
    rotMat = torch.stack(
        [1.0 - (yY + zZ), xY - wZ, xZ + wY, xY + wZ, 1.0 - (xX + zZ), yZ - wX, xZ - wY, yZ + wX, 1.0 - (xX + yY)], dim=1
    ).reshape(B, 3, 3)

    # rotMat = torch.stack([
    #     qw * qw + qx * qx - qy * qy - qz * qz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    #     2 * (qx * qy + qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz - qw * qx),
    #     2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz],
    #     dim=1).reshape(B, 3, 3)

    # w2, x2, y2, z2 = qw*qw, qx*qx, qy*qy, qz*qz
    # wx, wy, wz = qw*qx, qw*qy, qw*qz
    # xy, xz, yz = qx*qy, qx*qz, qy*qz

    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


def pose_vec2mat(vec, rotation_mode="quat"):
    """Convert 6DoF parameters to transformation matrix.

    Args:s
        vec: 6DoF parameters in the order of
            "euler": rx, ry, rz, tx, ty, tz -- [B, 6]
            "quat": qw, qx, qy, qz, tx, ty, tz -- [B, 7]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    if rotation_mode == "euler":
        rot = vec[:, :3]
        translation = vec[:, 3:6].unsqueeze(-1)  # [B, 3, 1]
        rot_mat = euler2mat_torch(rot)  # [B, 3, 3]
    elif rotation_mode == "quat":
        rot = vec[:, :4]
        translation = vec[:, 4:7].unsqueeze(-1)  # [B, 3, 1]
        rot_mat = quat2mat_torch(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def inverse_warp(img, depth, pose, intrinsics, rotation_mode="euler", padding_mode="zeros"):
    """Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: 6DoF pose parameters from target to source -- [B, 6] / [B, 7]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    check_sizes(img, "img", "B3HW")
    check_sizes(depth, "depth", "BHW")
    check_sizes(pose, "pose", "B6")
    check_sizes(intrinsics, "intrinsics", "B33")

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr, padding_mode)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points


def get_closest_rot(rot_est, rot_gt, sym_info):
    """get the closest rot_gt given rot_est and sym_info.

    rot_est: ndarray
    rot_gt: ndarray
    sym_info: None or Kx3x3 ndarray, m2m
    """
    if sym_info is None:
        return rot_gt
    if isinstance(sym_info, torch.Tensor):
        sym_info = sym_info.cpu().numpy()
    if len(sym_info.shape) == 2:
        sym_info = sym_info.reshape((1, 3, 3))
    # find the closest rot_gt with smallest re
    r_err = re(rot_est, rot_gt)
    closest_rot_gt = rot_gt
    for i in range(sym_info.shape[0]):
        # R_gt_m2c x R_sym_m2m ==> R_gt_sym_m2c
        rot_gt_sym = rot_gt.dot(sym_info[i])
        cur_re = re(rot_est, rot_gt_sym)
        if cur_re < r_err:
            r_err = cur_re
            closest_rot_gt = rot_gt_sym

    return closest_rot_gt


def get_closest_rot_batch(pred_rots, gt_rots, sym_infos):
    """
    get closest gt_rots according to current predicted poses_est and sym_infos
    --------------------
    pred_rots: [B, 4] or [B, 3, 3]
    gt_rots: [B, 4] or [B, 3, 3]
    sym_infos: list [Kx3x3 or None],
        stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_gt_rots: [B, 3, 3]
    """
    batch_size = pred_rots.shape[0]
    device = pred_rots.device
    if pred_rots.shape[-1] == 4:
        pred_rots = quat2mat_torch(pred_rots[:, :4])
    if gt_rots.shape[-1] == 4:
        gt_rots = quat2mat_torch(gt_rots[:, :4])

    closest_gt_rots = gt_rots.clone().cpu().numpy()  # B,3,3

    for i in range(batch_size):
        closest_rot = get_closest_rot(pred_rots[i].detach().cpu().numpy(), gt_rots[i].cpu().numpy(), sym_infos[i])
        # TODO: automatically detect rot_gt's format in PM_Loss to avoid converting multiple times
        closest_gt_rots[i] = closest_rot
    closest_gt_rots = torch.tensor(closest_gt_rots, device=device, dtype=gt_rots.dtype)
    return closest_gt_rots


def get_closest_pose_batch(poses_est, poses_gt, sym_infos):
    """
    get closest poses_gt according to current predicted poses_est and sym_infos
    --------------------
    poses_est: [B, 8]
    poses_gt: [B, 8]
    sym_infos: dict {label_idx: Kx3x3 or None}, stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_poses_gt: [B, 8]
    """
    batch_size = poses_est.shape[0]
    device = poses_est.device

    rots_est = quat2mat_torch(poses_est[:, :4])
    rots_gt = quat2mat_torch(poses_gt[:, :4])
    labels = poses_est[:, 7].long()

    closest_poses_gt = poses_gt.clone().cpu().numpy()

    for i in range(batch_size):
        closest_rot = get_closest_rot(
            rots_est[i].detach().cpu().numpy(), rots_gt[i].cpu().numpy(), sym_infos[int(labels[i])]
        )
        # TODO: automatically detect rot_gt's format in PM_Loss to avoid converting multiple times
        closest_poses_gt[i][:4] = mat2quat(closest_rot)
    closest_poses_gt = torch.tensor(closest_poses_gt, device=device, dtype=poses_gt.dtype)
    return closest_poses_gt


def get_closest_pose_batch_cpu(poses_est, poses_gt, sym_infos):
    """
    get closest poses_gt according to current predicted poses_est and sym_infos
    --------------------
    poses_est: [B, 8] ndarray
    poses_gt: [B, 8] ndarray
    sym_infos: dict {label_idx: Kx3x3 or None}, stores K rotations regarding symmetries, if not symmetric, None
    -----
    closest_poses_gt: [B, 8]
    """
    batch_size = poses_est.shape[0]

    rots_est = [quat2mat(poses_est[i, :4]) for i in range(batch_size)]
    rots_gt = [quat2mat(poses_gt[i, :4]) for i in range(batch_size)]
    labels = poses_est[:, 7].astype(int)

    closest_poses_gt = poses_gt.copy()

    for i in range(batch_size):
        closest_rot = get_closest_rot(rots_est[i], rots_gt[i], sym_infos[int(labels[i])])
        closest_poses_gt[i][:4] = mat2quat(closest_rot)
    return closest_poses_gt


def R_transform_th(R_src, R_delta, rot_coord="CAMERA"):
    """transform R_src use R_delta.

    :param R_src: matrix
    :param R_delta:
    :param rot_coord:
    :return:
    """
    if rot_coord.lower() == "model":
        R_output = torch.matmul(R_src, R_delta)
    elif rot_coord.lower() == "camera" or rot_coord.lower() == "naive" or rot_coord.lower() == "camera_new":
        # dR_m2c x R_src_m2c
        R_output = torch.matmul(R_delta, R_src)
    else:
        raise Exception("Unknown rot_coord in R_transform: {}".format(rot_coord))
    return R_output


def T_transform_batch(T_src, T_delta, zoom_factor, labels_pred=None):
    """inv_zoom T_delta; T_delta + T_src --> T_tgt.
    T_src: [B, 3] (x1, y1, z1)
    T_delta: [B, 3xnum_classes] (dx, dy, dz)
    zoom_factor: [B, 4]
            wx = crop_height / height
            wy = crop_height / height
            tx = zoom_c_x / width * 2 - 1
            ty = zoom_c_y / height * 2 - 1
            affine_matrix = [[wx, 0, tx], [0, wy, ty]]
    ---------
    T_tgt: [B, 3] (x2, y2, z2)
    """
    batch_size = T_delta.shape[0]
    if T_delta.shape[1] > 3:  # class aware
        assert labels_pred is not None, "labels_pred should not be None when class aware"
        inds = torch.arange(0, batch_size, dtype=torch.long, device=T_delta.device)
        T_delta_selected = T_delta.view(batch_size, -1, 3)[inds, labels_pred]  # [B, 3]
    else:
        T_delta_selected = T_delta
    factor_x = zoom_factor[:, 0]  # [B,]
    factor_y = zoom_factor[:, 1]  # [B,]

    vx_0 = T_delta_selected[:, 0] * factor_x
    vy_0 = T_delta_selected[:, 1] * factor_y

    vz = torch.div(T_src[:, 2], torch.exp(T_delta_selected[:, 2]))
    vx = vz * torch.addcdiv(vx_0, 1.0, T_src[:, 0], T_src[:, 2])
    vy = vz * torch.addcdiv(vy_0, 1.0, T_src[:, 1], T_src[:, 2])
    # import pdb; pdb.set_trace()

    T_tgt = torch.stack([vx, vy, vz], 1)
    return T_tgt


def R_transform_batch(quats_delta, poses_src):
    """
    # R_tgt_m2c = dR_c2c x R_src_m2c
    quats_delta: [B, 4] or [B, 4*num_classes]
    poses_src: [B, 8]
    --------------
    rots_tgt: [B, 3, 3]
    """
    batch_size = quats_delta.shape[0]
    rots_src = quat2mat_torch(poses_src[:, :4])  # [B, 3, 3]  m2c
    if quats_delta.shape[1] > 4:  # class aware
        labels = poses_src[:, 7].long()  # [B,]
        quats_delta = quats_delta.view(batch_size, -1, 4)  # [B, num_classes, 4]
        inds = torch.arange(0, batch_size, dtype=torch.long, device=quats_delta.device)
        quats_delta_selected = quats_delta[inds, labels]  # [B, 4]
        rots_delta = quat2mat_torch(quats_delta_selected)
    else:
        rots_delta = quat2mat_torch(quats_delta)  # [B,3,3]  c2c
    rots_tgt = torch.matmul(rots_delta, rots_src)  # [B,3,3] # m2c
    return rots_tgt


def RT_transform_batch_gpu(quaternion_delta, translation, poses_src_batch):
    quaternion_delta = quaternion_delta.detach().cpu().numpy()
    translation = translation.detach().cpu().numpy()
    poses_src = poses_src_batch.cpu().numpy()
    poses_tgt = RT_transform_batch_cpu(quaternion_delta, translation, poses_src)
    poses_tgt = torch.cuda.FloatTensor(poses_tgt, device=poses_src_batch.device)
    return poses_tgt


def RT_transform_batch_cpu(quaternion_delta, translation, poses_src):
    poses_tgt = poses_src.copy()
    for i in range(poses_src.shape[0]):
        cls = int(poses_src[i, 1]) if quaternion_delta.shape[1] > 4 else 0
        if all(poses_src[i, 2:] == 0):
            poses_tgt[i, 2:] = 0
        else:
            poses_tgt[i, 2:6] = mat2quat(
                np.dot(quat2mat(quaternion_delta[i, 4 * cls : 4 * cls + 4]), quat2mat(poses_src[i, 2:6]))
            )
            poses_tgt[i, 6:] = translation[i, 3 * cls : 3 * cls + 3]
    return poses_tgt


def get_closest_pose(est_rot, gt_rot, sym_info):
    # sym_info: (sym_axis, sym_angle) [0:3, 3]
    def gen_mat(axis, degree):
        axis = axis / LA.norm(axis)
        return quat2mat(
            [
                cos(degree / 360.0 * pi),
                axis[0] * sin(degree / 360.0 * pi),
                axis[1] * sin(degree / 360.0 * pi),
                axis[2] * sin(degree / 360.0 * pi),
            ]
        )

    sym_angle = int(sym_info[3])
    sym_axis = np.copy(sym_info[:3])
    if sym_angle == -1:
        closest_rot = gt_rot
    elif sym_angle == 0:
        angle = 180.0
        gt_rot_1 = np.copy(gt_rot)
        rd_1 = re(gt_rot_1, est_rot)
        gt_rot_2 = np.matmul(gt_rot, gen_mat(sym_axis, angle))
        rd_2 = re(gt_rot_2, est_rot)
        if rd_1 < rd_2:
            gt_rot_1 = np.matmul(gt_rot, gen_mat(sym_axis, -90))
            gt_rot_2 = np.matmul(gt_rot, gen_mat(sym_axis, 90))
        else:
            gt_rot_1 = np.matmul(gt_rot, gen_mat(sym_axis, 90))
            gt_rot_2 = np.matmul(gt_rot, gen_mat(sym_axis, 270))
        rd_1 = re(gt_rot_1, est_rot)
        rd_2 = re(gt_rot_2, est_rot)
        count = 1
        thresh = 0.1
        while angle > thresh:
            angle /= 2
            count += 1
            if rd_1 < rd_2:
                gt_rot_2 = np.matmul(gt_rot_2, gen_mat(sym_axis, -angle))
                rd_2 = re(gt_rot_2, est_rot)
            else:
                gt_rot_1 = np.matmul(gt_rot_1, gen_mat(sym_axis, angle))
                rd_1 = re(gt_rot_1, est_rot)

        # print("rd_1: {}, rd_2: {}, angle: {}, count: {}".format(rd_1, rd_2, angle, count))
        closest_rot = gt_rot_1 if rd_1 < rd_2 else gt_rot_2
    else:
        assert 180 % sym_angle == 0
        rot_delta = gen_mat(sym_axis, sym_angle)
        cur_rot = np.copy(gt_rot)
        closest_rot = np.copy(cur_rot)
        closest_angle = re(cur_rot, est_rot)
        for i in range(180 / sym_angle):
            cur_rot = np.matmul(cur_rot, rot_delta)
            rd = re(cur_rot, est_rot)
            if rd < closest_angle:
                closest_rot = np.copy(cur_rot)
                closest_angle = rd

    return closest_rot


def se3_inverse_torch(RT):
    # RT is a 3x4 matrix
    R = RT[0:3, 0:3]
    T = RT[0:3, 3]
    R_inv = R.t()
    T_inv = -1 * torch.matmul(R.t(), T)
    RT_inv = torch.cat([R_inv, T_inv.view(3, 1)], dim=1)
    return RT_inv


def se3_mul_torch(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    R_new = torch.matmul(R1, R2)
    T_new = torch.matmul(R1, T2) + T1
    RT_new = torch.cat([R_new, T_new.view(3, 1)], dim=1)
    return RT_new


def se3_inverse_torch_batch(RT):
    # RT is a Bx3x4 matrix
    B = RT.shape[0]
    R_inv = RT[:, :3, :3].permute(0, 2, 1)
    T_inv = -1 * torch.matmul(R_inv, RT[:, :3, 3].view(B, 3, 1))
    RT_inv = torch.cat([R_inv, T_inv], dim=-1)
    return RT_inv


def se3_mul_torch_batch(RT1, RT2):
    assert RT1.shape[0] == RT2.shape[0]
    B = RT1.shape[0]

    R_new = torch.matmul(RT1[:, :3, :3], RT2[:, :3, :3])
    T_new = torch.matmul(RT1[:, :3, :3], RT2[:, :3, 3].view(B, 3, 1)) + RT1[:, :3, 3].view(B, 3, 1)
    RT_new = torch.cat([R_new, T_new], dim=-1)
    return RT_new


def calc_se3_torch(pose_src, pose_tgt):
    """
    :param pose_src: pose matrix of soucre, [R|T], 3x4
    :param pose_tgt: pose matrix of target, [R|T], 3x4
    """
    se3_src2tgt = se3_mul_torch(pose_tgt, se3_inverse_torch(pose_src))
    return se3_src2tgt


def calc_se3_torch_batch(poses_src, poses_tgt):
    """Bx3x4."""
    # B = poses_src.shape[0]
    se3_src2tgt_batch = se3_mul_torch_batch(poses_tgt, se3_inverse_torch_batch(poses_src))
    return se3_src2tgt_batch


def blender_euler_to_blender_pose(euler):
    euler_0 = (-euler[0] + 90) % 360
    euler_1 = euler[1] + 90
    return euler2mat(euler_0 * np.pi / 180, euler_1 * np.pi / 180, euler[2] * np.pi / 180, axes="szxz")


def blender_pose_to_blender_euler(pose):
    euler = [r / np.pi * 180 for r in mat2euler(pose, axes="szxz")]
    euler[0] = -(euler[0] + 90) % 360
    euler[1] = euler[1] - 90
    return np.array(euler)


###########################################################################
# NOTE: tests
def test_calc_se3_torch():
    from lib.pysixd.RT_transform import calc_se3

    B = 8
    device = "cuda"

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    np.random.seed(1)
    axis = np.random.rand(B, 3)
    angle = np.random.rand(B)
    axis_tensor = to_tensor(axis)
    angle_tensor = to_tensor(angle)
    mat_torch = axangle2mat_torch(axis_tensor, angle_tensor, is_normalized=False)
    RT1 = torch.rand(B, 3, 4)
    RT1[:, :3, :3] = mat_torch

    axis = np.random.rand(B, 3)
    angle = np.random.rand(B)
    axis_tensor = to_tensor(axis)
    angle_tensor = to_tensor(angle)
    mat_torch = axangle2mat_torch(axis_tensor, angle_tensor, is_normalized=False)
    RT2 = torch.rand(B, 3, 4)
    RT2[:, :3, :3] = mat_torch
    runs = 10000
    import time

    t1 = time.perf_counter()
    for _ in range(runs):
        se3_numpy = []
        for i in range(B):
            se3_r, se3_t = calc_se3(RT1[i].cpu().numpy(), RT2[i].cpu().numpy())
            se3_numpy.append(np.hstack([se3_r, se3_t.reshape((3, 1))]))
        se3_numpy = np.array(se3_numpy)
    print("numpy: {}s".format((time.perf_counter() - t1) / runs))

    t2 = time.perf_counter()
    for _ in range(runs):
        se3_torch_single = torch.empty_like(RT1)
        for i in range(B):
            se3_torch_single[i] = calc_se3_torch(RT1[i], RT2[i])
    print("torch_single: {}s".format((time.perf_counter() - t2) / runs))

    t3 = time.perf_counter()
    for _ in range(runs):
        se3_torch = calc_se3_torch_batch(RT1, RT2)
    print("torch: {}s".format((time.perf_counter() - t3) / runs))

    print(np.allclose(se3_numpy, se3_torch.cpu().numpy()))
    print(torch.allclose(se3_torch_single, se3_torch))
    """
    numpy: 0.000260600209236145s
    torch_single: 0.0005826136112213135s
    torch: 0.00012000765800476074s
    True
    True
    """


def test_pose_vec2mat():
    B = 8
    qt1 = torch.rand(B, 7).to("cuda", torch.float32)
    RT1 = pose_vec2mat(qt1)

    RT_np = []
    for i in range(B):
        r = quat2mat(qt1[i, :4].cpu().numpy())
        t = qt1[i, 4:7].cpu().numpy().reshape((3, 1))
        RT_np.append(np.hstack([r, t]))
    RT_np = np.array(RT_np)
    print(RT_np.dtype)
    print(RT1.dtype)
    print(np.allclose(RT_np, RT1.cpu().numpy()))
    if np.allclose(RT_np, RT1.cpu().numpy()) is False:
        print(np.abs(RT_np - RT1.cpu().numpy()).mean(), np.abs(RT_np - RT1.cpu().numpy()).max())
        # 4.28597091408379e-08 1.8986341143722996e-07


def test_axangle2mat_torch():
    B = 8
    device = "cuda"

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    np.random.seed(1)
    axis = np.random.rand(B, 3)
    angle = np.random.rand(B)
    axis_tensor = to_tensor(axis)
    angle_tensor = to_tensor(angle)
    mat_torch = axangle2mat_torch(axis_tensor, angle_tensor, is_normalized=False)
    mat_np = []
    for i in range(B):
        mat_np.append(axangle2mat(axis[i], angle[i]))
    mat_np = np.array(mat_np)
    print(mat_np)
    print(mat_torch)
    print(np.allclose(mat_np, mat_torch.cpu().numpy()))


def test_quat2euler():
    B = 8
    quat = np.random.rand(B, 4)
    euler = []
    for i in range(quat.shape[0]):
        euler.append(quat2euler(quat[i]))
    euler = np.array(euler)

    # torch
    quat_torch = torch.from_numpy(quat)
    euler_torch = quat2euler_torch(quat_torch)
    print(euler)
    print(euler_torch)
    print(np.allclose(euler, euler_torch.cpu().numpy()))


def test_euler2quat():
    B = 8
    quat = np.random.rand(B, 4)
    euler = []
    for i in range(quat.shape[0]):
        euler.append(quat2euler(quat[i]))
    euler = np.array(euler)

    # torch
    quat_torch = torch.from_numpy(quat).to("cuda")
    euler_torch = quat2euler_torch(quat_torch)
    print(euler_torch)
    print(euler)

    ######
    """
    torch  0.0002950624704360962
    numpy  0.0001986891746520996
    """
    runs = 10000
    import time

    t1 = time.perf_counter()
    for _ in range(runs):
        quat_from_euler_torch = euler2quat_torch(euler_torch[:, 0], euler_torch[:, 1], euler_torch[:, 2])
        # print(quat_from_euler_torch.shape)
    print("torch ", (time.perf_counter() - t1) / runs)

    euler_np = euler_torch.cpu().numpy()
    quat_from_euler_np = torch.zeros_like(quat_torch)
    t1 = time.perf_counter()
    for _ in range(runs):
        for i in range(B):
            quat_from_euler_np[i].copy_(torch.tensor(euler2quat(euler_np[i, 0], euler_np[i, 1], euler_np[i, 2])))
    print("numpy ", (time.perf_counter() - t1) / runs)
    print(np.allclose(quat_from_euler_np.cpu().numpy(), quat_from_euler_torch.cpu().numpy()))
    print(quat_from_euler_np)
    print(quat_from_euler_torch)


def test_qrot_points():
    from lib.pysixd.inout import load_ply

    data_root = osp.normpath(osp.join(cur_dir, "../../datasets/BOP_DATASETS/lm/"))
    models_cad_files = [osp.join(data_root, "models/obj_{:06d}.ply".format(i)) for i in range(1, 15 + 1)]
    obj_id = 0
    points = load_ply(models_cad_files[obj_id])["pts"]
    axis = np.array([1, 2, 0])
    rot = axangle2mat(axis, -pi / 3)
    quat = mat2quat(rot)
    points_q = qrot_points_th(torch.from_numpy(quat), torch.from_numpy(points))
    # N = points.shape[0]
    # points_q = qrot_torch(torch.from_numpy(quat).expand(N, 4), torch.from_numpy(points))
    points_r = rot.dot(points.T).T
    print(np.allclose(points_q.numpy(), points_r))


if __name__ == "__main__":
    import time
    import os
    import os.path as osp
    import sys
    import mmcv

    os.environ["PYOPENGL_PLATFORM"] = "egl"
    cur_dir = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, osp.join(cur_dir, "../../"))
    test_qrot_points()
    exit(0)
    # test_axangle2mat_torch()
    # exit(0)

    # test_quat2euler()
    # exit(0)
    # test_calc_se3_torch()
    # test_pose_vec2mat()
    test_euler2quat()
    exit(0)

    from lib.meshrenderer.meshrenderer_color import Renderer
    from lib.vis_utils.image import grid_show
    from lib.pysixd.misc import get_symmetry_transformations

    IDX2CLASS = {
        1: "ape",
        2: "benchvise",
        3: "bowl",
        4: "camera",
        5: "can",
        6: "cat",
        7: "cup",
        8: "driller",
        9: "duck",
        10: "eggbox",
        11: "glue",
        12: "holepuncher",
        13: "iron",
        14: "lamp",
        15: "phone",
    }
    CLASSES = IDX2CLASS.values()
    CLASSES = sorted(CLASSES)
    CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}
    idx2class = IDX2CLASS
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])
    ZNEAR = 0.25
    ZFAR = 6.0
    width = 640
    height = 480

    data_root = osp.normpath(osp.join(cur_dir, "../../data/BOP_DATASETS/lm_full/"))

    models_cad_files = [osp.join(data_root, "models/obj_{:06d}.ply".format(i)) for i in range(1, 15 + 1)]
    models_info = mmcv.load(osp.join(data_root, "models/models_info.pkl"))
    renderer = Renderer(
        models_cad_files,
        K,
        samples=1,
        vertex_tmp_store_folder=".",
        vertex_scale=0.001,
        height=height,
        width=width,
        near=ZNEAR,
        far=ZFAR,
    )

    # obj_id = 10 - 1
    # trans = np.array([-0.0021458883, 0.0804758, 0.78142926])
    # axis_est = np.array([1, 2, 0])
    # axis_gt = np.array([0,2,1])
    # est_rot = axangle2mat(axis_est, -pi/3)
    # gt_rot = axangle2mat(axis_gt, pi)
    # sym_info = axangle2mat([0, 0, 1], pi)
    # print('sym_info', sym_info)

    cls_idx = 3
    obj_id = cls_idx - 1
    trans = np.array([-0.0021458883, 0.0804758, 0.78142926])
    axis_est = np.array([1, 2, 0])
    axis_gt = np.array([0, 2, 1])
    est_rot = axangle2mat(axis_est, -pi / 3)
    gt_rot = axangle2mat(axis_gt, pi)

    transforms_sym = get_symmetry_transformations(models_info[cls_idx], max_sym_disc_step=0.01)
    # sym_info = axangle2mat([0, 0, 1], pi)
    sym_info = np.array([sym["R"] for sym in transforms_sym])
    print("sym_info", sym_info.shape)

    est_pose = np.random.rand(3, 4)
    est_pose[:, :3] = est_rot
    gt_pose = np.random.rand(3, 4)
    gt_pose[:, :3] = gt_rot
    rd_ori = re(est_rot, gt_rot)

    t = time.perf_counter()
    for i in range(3000):
        closest_rot = get_closest_rot(est_rot, gt_rot, sym_info)
    print(("calculate closest rot {}s".format((time.perf_counter() - t) / 3000)))
    closest_pose = np.copy(gt_pose)
    closest_pose[:, :3] = closest_rot

    rd_closest = re(est_rot, closest_pose[:, :3])
    print(
        (
            "rot_est: {}, rot_gt: {}, closest rot_gt: {}".format(
                mat2axangle(est_rot), mat2axangle(gt_rot), mat2axangle(closest_rot)
            )
        )
    )
    print(("original rot dist: {}, closest rot dist: {}".format(rd_ori, rd_closest)))

    est_img, _ = renderer.render(obj_id, est_rot, trans)
    gt_img, _ = renderer.render(obj_id, gt_rot, trans)
    closest_img, _ = renderer.render(obj_id, closest_rot, trans)
    show_imgs = [est_img[:, :, [2, 1, 0]], gt_img[:, :, [2, 1, 0]], closest_img[:, :, [2, 1, 0]]]
    show_titles = ["est", "gt_ori", "gt_closest"]
    grid_show(show_imgs, show_titles, row=1, col=3)

    # import cv2
    # while(1):
    #     est_img = render(renderer, est_rot, trans)
    #     cv2.imshow('test', cv2.cvtColor(est_img, cv2.COLOR_RGB2BGR))
    #     q = cv2.waitKey(16)
    #     if q == ord('w'):
    #         trans[1] += 0.05
    #     elif q == ord('s'):
    #         trans[1] -= 0.05
    #     elif q == ord('a'):
    #         trans[0] -= 0.1
    #     elif q == ord('d'):
    #         trans[0] += 0.1
    #     elif q == ord('q'):
    #         trans[2] += 0.01
    #     elif q == ord('e'):
    #         trans[2] -= 0.01
    #     print trans

    # import cv2
    # rot = aa2mat([0,0,1], 0)
    # a = aa2mat([0, 0, 1], 5)
    # d = aa2mat([0, 0, 1], -5)
    # q = aa2mat([0, 1, 0], 5)
    # e = aa2mat([0, 1, 0], -5)
    # w = aa2mat([1, 0, 0], 5)
    # s = aa2mat([1, 0, 0], -5)
    # while(1):
    #     est_img = render(renderer, rot, trans)
    #     cv2.imshow('test', cv2.cvtColor(est_img, cv2.COLOR_RGB2BGR))
    #     key = cv2.waitKey(16)
    #     if key == ord('w'):
    #         rot = np.matmul(w, rot)
    #     elif key == ord('s'):
    #         rot = np.matmul(s, rot)
    #     elif key == ord('a'):
    #         rot = np.matmul(a, rot)
    #     elif key == ord('d'):
    #         rot = np.matmul(d, rot)
    #     elif key == ord('q'):
    #         rot = np.matmul(q, rot)
    #     elif key == ord('e'):
    #         rot = np.matmul(e, rot)
    #     print mat2quat(rot)
