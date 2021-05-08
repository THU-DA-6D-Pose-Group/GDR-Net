# https://github.com/papagina/RotationContinuity/blob/master/Inverse_Kinematics/code/tools.py
import torch
import numpy as np
import torch.nn as nn
from transforms3d.quaternions import quat2mat, mat2quat, axangle2quat
import torch.nn.functional as F


def normalize_vector(v):
    # bxn
    # batch = v.shape[0]
    # v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    # v_mag = torch.max(v_mag, torch.FloatTensor([1e-8]).to(v))
    # v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    # v = v / v_mag
    v = F.normalize(v, p=2, dim=1)
    return v


def cross_product(u, v):
    # u, v bxn
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # bx3

    return out


def ortho6d_to_mat_batch(poses):
    # poses bx6
    # poses
    x_raw = poses[:, 0:3]  # bx3
    y_raw = poses[:, 3:6]  # bx3

    x = normalize_vector(x_raw)  # bx3
    z = cross_product(x, y_raw)  # bx3
    z = normalize_vector(z)  # bx3
    y = cross_product(z, x)  # bx3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # bx3x3
    return matrix


def mat_to_ortho6d_batch(rots):
    """
    bx3x3
    ---
    bx6
    """
    x = rots[:, :, 0]  # col x
    y = rots[:, :, 1]  # col y
    ortho6d = torch.cat([x, y], 1)  # bx6
    return ortho6d


def mat_to_ortho6d_np(rot):
    """
    3x3
    ---
    (6,)
    """
    x = rot[:3, 0]  # col x
    y = rot[:3, 1]  # col y
    ortho6d = np.concatenate([x, y])  # (6,)
    return ortho6d


def quat2mat_batch(quaternion):
    # quaternion bx4
    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion)

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # bx3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # bx3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # bx3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # bx3x3

    return matrix


def normalize_5d_rotation(r5d):
    # rotation5d bx5
    batch = r5d.shape[0]
    sin_cos = r5d[:, 0:2]  # bx2
    sin_cos_mag = torch.max(torch.sqrt(sin_cos.pow(2).sum(1)), torch.DoubleTensor([1e-8]).to(r5d))  # b
    sin_cos_mag = sin_cos_mag.view(batch, 1).expand(batch, 2)  # bx2
    sin_cos = sin_cos / sin_cos_mag  # bx2

    axis = r5d[:, 2:5]  # bx3
    axis_mag = torch.max(torch.sqrt(axis.pow(2).sum(1)), torch.DoubleTensor([1e-8]).to(axis))  # b

    axis_mag = axis_mag.view(batch, 1).expand(batch, 3)  # bx3
    axis = axis / axis_mag  # bx3
    out_rotation = torch.cat((sin_cos, axis), 1)  # bx5

    return out_rotation


def rotation5d_to_mat_batch(r5d):
    # rotation5d bx5
    # out matrix bx3x3
    batch = r5d.shape[0]
    sin = r5d[:, 0].view(batch, 1)  # bx1
    cos = r5d[:, 1].view(batch, 1)  # bx1

    x = r5d[:, 2].view(batch, 1)  # bx1
    y = r5d[:, 3].view(batch, 1)  # bx1
    z = r5d[:, 4].view(batch, 1)  # bx1

    row1 = torch.cat((cos + x * x * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin), 1)  # b*3
    row2 = torch.cat((y * x * (1 - cos) + z * sin, cos + y * y * (1 - cos), y * z * (1 - cos) - x * sin), 1)  # b*3
    row3 = torch.cat((z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z * z * (1 - cos)), 1)  # b*3

    matrix = torch.cat((row1.view(-1, 1, 3), row2.view(-1, 1, 3), row3.view(-1, 1, 3)), 1)  # b*3*3*seq_len
    matrix = matrix.view(batch, 3, 3)
    return matrix


def compute_pose_from_rotation_matrix(T_pose, r_matrix):
    # T_poses numx3
    # r_matrix batch*3*3
    batch = r_matrix.shape[0]
    joint_num = T_pose.shape[0]
    r_matrices = r_matrix.view(batch, 1, 3, 3).expand(batch, joint_num, 3, 3).contiguous().view(batch * joint_num, 3, 3)
    src_poses = (
        T_pose.view(1, joint_num, 3, 1).expand(batch, joint_num, 3, 1).contiguous().view(batch * joint_num, 3, 1)
    )

    out_poses = torch.matmul(r_matrices, src_poses)  # (b*joint_num)x3x1

    return out_poses.view(batch, joint_num, 3)


def stereographic_unproject_old(a):
    # in bx5
    # out bx6
    s2 = torch.pow(a, 2).sum(1)  # b
    unproj = 2 * a / (s2 + 1).view(-1, 1).repeat(1, 5)  # bx5
    w = (s2 - 1) / (s2 + 1)  # b
    out = torch.cat((unproj, w.view(-1, 1)), 1)  # bx6

    return out


def stereographic_unproject(a, axis=None):
    """# in a batch*5, axis int Inverse of stereographic projection: increases
    dimension by one."""
    batch = a.shape[0]
    if axis is None:
        axis = a.shape[1]
    s2 = torch.pow(a, 2).sum(1)  # batch
    ans = torch.zeros(batch, a.shape[1] + 1).to(a)  # batch*6
    unproj = 2 * a / (s2 + 1).view(batch, 1).repeat(1, a.shape[1])  # batch*5
    if axis > 0:
        ans[:, :axis] = unproj[:, :axis]  # batch*(axis-0)
    ans[:, axis] = (s2 - 1) / (s2 + 1)  # batch
    # NOTE: this is a no-op if the default option (last axis) is used
    ans[:, axis + 1 :] = unproj[:, axis:]  # batch*(5-axis)
    return ans


def ortho5d_to_mat_batch(a):
    # a batch*5
    # out batch*3*3
    batch = a.shape[0]
    proj_scale_np = np.array([np.sqrt(2) + 1, np.sqrt(2) + 1, np.sqrt(2)])  # 3
    proj_scale = torch.FloatTensor(proj_scale_np).to(a).view(1, 3).repeat(batch, 1)  # batch,3

    u = stereographic_unproject(a[:, 2:5] * proj_scale, axis=0)  # batch*4
    norm = torch.sqrt(torch.pow(u[:, 1:], 2).sum(1))  # batch
    u = u / norm.view(batch, 1).repeat(1, u.shape[1])  # batch*4
    b = torch.cat((a[:, 0:2], u), 1)  # batch*6
    matrix = ortho6d_to_mat_batch(b)
    return matrix


def axisAngle2mat_batch(axisAngle):
    # axisAngle batch*4 angle, x,y,z
    batch = axisAngle.shape[0]

    theta = torch.tanh(axisAngle[:, 0]) * np.pi  # [-180, 180]
    sin = torch.sin(theta)
    axis = normalize_vector(axisAngle[:, 1:4])  # batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # b*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # b*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # b*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # b*3*3

    return matrix


def hopf2mat_batch(hopf):
    # axisAngle batch*3 a,b,c
    batch = hopf.shape[0]

    theta = (torch.tanh(hopf[:, 0]) + 1.0) * np.pi / 2.0  # [0, pi]
    phi = (torch.tanh(hopf[:, 1]) + 1.0) * np.pi  # [0,2pi)
    tao = (torch.tanh(hopf[:, 2]) + 1.0) * np.pi  # [0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def euler2mat_batch(euler):
    # euler bx4
    # output cuda bx3x3 matrices in the rotation order of XZ'Y'' (intrinsic) or YZX (extrinsic)
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1

    row1 = torch.cat((c2 * c3, -s2, c2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((c1 * s2 * c3 + s1 * s3, c1 * c2, c1 * s2 * s3 - s1 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2 * c3 - c1 * s3, s1 * c2, s1 * s2 * s3 + c1 * c3), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


def proj_u_a(u, a):
    # u,a batch*3
    # out batch*3
    batch = u.shape[0]
    top = u[:, 0] * a[:, 0] + u[:, 1] * a[:, 1] + u[:, 2] * a[:, 2]
    bottom = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1] + u[:, 2] * u[:, 2]
    bottom = torch.max(torch.zeros(batch).to(a) + 1e-8, bottom)
    factor = (top / bottom).view(batch, 1).expand(batch, 3)
    out = factor * u
    return out


def compute_rotation_matrix_from_matrix(matrices):
    # matrices bx3x3
    b = matrices.shape[0]
    a1 = matrices[:, :, 0]  # bx3
    a2 = matrices[:, :, 1]
    a3 = matrices[:, :, 2]

    u1 = a1
    u2 = a2 - proj_u_a(u1, a2)
    u3 = a3 - proj_u_a(u1, a3) - proj_u_a(u2, a3)

    e1 = normalize_vector(u1)
    e2 = normalize_vector(u2)
    e3 = normalize_vector(u3)

    rmat = torch.cat((e1.view(b, 3, 1), e2.view(b, 3, 1), e3.view(b, 3, 1)), 2)

    return rmat


def get_44_rotation_matrix_from_33_rotation_matrix(m):
    # m bx3x3
    # out bx4x4
    batch = m.shape[0]

    row4 = torch.zeros(batch, 1, 3).cuda()

    m43 = torch.cat((m, row4), 1)  # bx4x3

    col4 = torch.zeros(batch, 4, 1).cuda()
    col4[:, 3, 0] = col4[:, 3, 0] + 1

    out = torch.cat((m43, col4), 2)  # bx4x4

    return out


def compute_geodesic_distance_from_two_matrices(m1, m2):
    # matrices batch*3*3
    # both matrix are orthogonal rotation matrices
    # out theta between 0 to 180 degree batch
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.ones(batch).cuda())
    cos = torch.max(cos, torch.ones(batch).cuda() * -1)

    theta = torch.acos(cos)

    # theta = torch.min(theta, 2*np.pi - theta)

    return theta


def compute_angle_from_r_matrices(m):
    # matrices batch*3*3
    # both matrix are orthogonal rotation matrices
    # out theta between 0 to 180 degree batch
    batch = m.shape[0]

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.ones(batch).cuda())
    cos = torch.max(cos, torch.ones(batch).cuda() * -1)

    theta = torch.acos(cos)

    return theta


def get_sampled_rotation_matrices_by_quat(batch):
    quat = torch.randn(batch, 4).cuda()
    matrix = quat2mat_batch(quat)
    return matrix


def get_sampled_rotation_matrices_by_hpof(batch):

    theta = torch.FloatTensor(np.random.uniform(0, 1, batch) * np.pi).cuda()  # [0, pi]
    phi = torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).cuda()  # [0,2pi)
    tao = torch.FloatTensor(np.random.uniform(0, 2, batch) * np.pi).cuda()  # [0,2pi)

    qw = torch.cos(theta / 2) * torch.cos(tao / 2)
    qx = torch.cos(theta / 2) * torch.sin(tao / 2)
    qy = torch.sin(theta / 2) * torch.cos(phi + tao / 2)
    qz = torch.sin(theta / 2) * torch.sin(phi + tao / 2)

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def get_sampled_rotation_matrices_by_axisAngle(batch):
    # axisAngle bx4 angle, x,y,z
    # [0, pi] #[-180, 180]
    theta = torch.FloatTensor(np.random.uniform(-1, 1, batch) * np.pi).cuda()
    sin = torch.sin(theta)
    axis = torch.randn(batch, 3).cuda()
    axis = normalize_vector(axis)  # bx3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix


def compute_euler_angles_from_rotation_matrices(rotation_matrices):
    # input batch*4*4 or batch*3*3
    # output torch batch*3 x, y, z in radiant
    # the rotation is in the sequence of x,y,z
    batch = rotation_matrices.shape[0]
    R = rotation_matrices
    sy = torch.sqrt(R[:, 0, 0] * R[:, 0, 0] + R[:, 1, 0] * R[:, 1, 0])
    singular = sy < 1e-6
    singular = singular.float()

    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    y = torch.atan2(-R[:, 2, 0], sy)
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])

    xs = torch.atan2(-R[:, 1, 2], R[:, 1, 1])
    ys = torch.atan2(-R[:, 2, 0], sy)
    zs = R[:, 1, 0] * 0

    out_euler = torch.zeros(batch, 3).cuda()
    out_euler[:, 0] = x * (1 - singular) + xs * singular
    out_euler[:, 1] = y * (1 - singular) + ys * singular
    out_euler[:, 2] = z * (1 - singular) + zs * singular

    return out_euler


def axisAngle2quat_batch(axisAngles):
    # input batch*4
    # output batch*4
    w = torch.cos(axisAngles[:, 0] / 2)
    sin = torch.sin(axisAngles[:, 0] / 2)
    x = sin * axisAngles[:, 1]
    y = sin * axisAngles[:, 2]
    z = sin * axisAngles[:, 3]

    quat = torch.cat((w.view(-1, 1), x.view(-1, 1), y.view(-1, 1), z.view(-1, 1)), 1)

    return quat


# NOTE: this is from code of 6d rot paper
# def mat2quat_batch(matrices, eps=1e-8):
#     # matrices batch*4*4 or batch*3*3
#     # quaternions batch*4
#     batch = matrices.shape[0]

#     w = torch.sqrt(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0
#     w = torch.max(w, torch.zeros(batch).to(matrices) + eps)  # batch
#     w4 = 4.0 * w
#     x = (matrices[:, 2, 1] - matrices[:, 1, 2]) / w4
#     y = (matrices[:, 0, 2] - matrices[:, 2, 0]) / w4
#     z = (matrices[:, 1, 0] - matrices[:, 0, 1]) / w4

#     quats = torch.cat((w.view(batch, 1), x.view(batch, 1), y.view(batch, 1), z.view(batch, 1)), 1)

#     return quats


def mat2quat_batch(rotation_matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # NOTE: from kornia, but they use (x, y, z, w) quat format
    # NOTE: it seems this is still not stable, leeding to NaN losses
    r"""Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (w, x, y, z) format.
    Args:
        rotation_matrix (torch.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
        torch.Tensor: the rotation in quaternion.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    Example:
        >>> input = torch.rand(4, 3, 3)  # Nx3x3
        >>> output = kornia.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input size must be a (*, 3, 3) tensor. Got {}".format(rotation_matrix.shape))

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps: float = torch.finfo(numerator.dtype).tiny  # type: ignore
        return numerator / torch.clamp(denominator, min=eps)

    rotation_matrix_vec: torch.Tensor = rotation_matrix.view(*rotation_matrix.shape[:-2], 9)

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.chunk(rotation_matrix_vec, chunks=9, dim=-1)

    trace: torch.Tensor = m00 + m11 + m22

    def trace_positive_cond():
        sq = torch.sqrt(trace + 1.0) * 2.0  # sq = 4 * qw.
        qw = 0.25 * sq
        qx = safe_zero_division(m21 - m12, sq)
        qy = safe_zero_division(m02 - m20, sq)
        qz = safe_zero_division(m10 - m01, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_1():
        sq = torch.sqrt(1.0 + m00 - m11 - m22 + eps) * 2.0  # sq = 4 * qx.
        qw = safe_zero_division(m21 - m12, sq)
        qx = 0.25 * sq
        qy = safe_zero_division(m01 + m10, sq)
        qz = safe_zero_division(m02 + m20, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_2():
        sq = torch.sqrt(1.0 + m11 - m00 - m22 + eps) * 2.0  # sq = 4 * qy.
        qw = safe_zero_division(m02 - m20, sq)
        qx = safe_zero_division(m01 + m10, sq)
        qy = 0.25 * sq
        qz = safe_zero_division(m12 + m21, sq)
        return torch.cat([qw, qx, qy, qz], dim=-1)

    def cond_3():
        sq = torch.sqrt(1.0 + m22 - m00 - m11 + eps) * 2.0  # sq = 4 * qz.
        qw = safe_zero_division(m10 - m01, sq)
        qx = safe_zero_division(m02 + m20, sq)
        qy = safe_zero_division(m12 + m21, sq)
        qz = 0.25 * sq
        return torch.cat([qw, qx, qy, qz], dim=-1)

    where_2 = torch.where(m11 > m22, cond_2(), cond_3())
    where_1 = torch.where((m00 > m11) & (m00 > m22), cond_1(), where_2)

    quaternion: torch.Tensor = torch.where(trace > 0.0, trace_positive_cond(), where_1)
    return quaternion


#####################################
# tests
def test_ortho6d_rot():
    axis = np.random.rand(3)
    angle = np.random.rand(1)
    # quat = axangle2quat([1, 2, 3], 0.7)
    quat = axangle2quat(axis, angle)
    print("quat:\n", quat)
    mat = quat2mat(quat)
    print("mat:\n", mat)
    mat_th = torch.tensor(mat.astype("float32"))[None].to("cuda")
    print("mat_th:\n", mat_th)
    ortho6d = mat_to_ortho6d_batch(mat_th)
    print("ortho6d:\n", ortho6d)
    mat_2 = ortho6d_to_mat_batch(ortho6d)
    print("mat_2:\n", mat_2)
    diff_mat = mat_th - mat_2
    print("mat_diff:\n", diff_mat)


def test_mat2quat_torch():
    from core.utils.pose_utils import quat2mat_torch

    axis = np.random.rand(3)
    angle = np.random.rand(1)
    # quat = axangle2quat([1, 2, 3], 0.7)
    quat = axangle2quat(axis, angle)
    print("quat:\n", quat)
    mat = quat2mat(quat)
    print("mat:\n", mat)
    mat_th = torch.tensor(mat.astype("float32"))[None].to("cuda")
    print("mat_th:\n", mat_th)
    quat_th = mat2quat_batch(mat_th)
    print("quat_th:\n", quat_th)
    mat_2 = quat2mat_torch(quat_th)
    print("mat_2:\n", mat_2)
    diff_mat = mat_th - mat_2
    print("mat_diff:\n", diff_mat)
    diff_quat = quat - quat_th.cpu().numpy()
    print("diff_quat:\n", diff_quat)


def test_mat2rot6d_np():
    axis = np.random.rand(3)
    angle = np.random.rand(1)
    # quat = axangle2quat([1, 2, 3], 0.7)
    quat = axangle2quat(axis, angle)
    print("quat:\n", quat)
    mat = quat2mat(quat)
    print("mat:\n", mat)
    rot6d = mat_to_ortho6d_np(mat)
    print("rot6d: \n", rot6d)


if __name__ == "__main__":
    import os.path as osp
    import sys

    sys.path.insert(0, osp.join(osp.dirname(__file__), "../../"))

    # test_ortho6d_rot()

    # test_mat2quat_torch()
    test_mat2rot6d_np()
