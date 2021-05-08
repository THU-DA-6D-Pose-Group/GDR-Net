# modified from: https://github.com/NVlabs/latentfusion/blob/master/latentfusion/three/quaternion.py
import math

import torch
from torch.nn import functional as F


@torch.jit.script
def acos_safe(t, eps: float = 1e-7):
    return torch.acos(torch.clamp(t, min=-1.0 + eps, max=1.0 - eps))


@torch.jit.script
def ensure_batch_dim(tensor, num_dims: int):
    unsqueezed = False
    if len(tensor.shape) == num_dims:
        tensor = tensor.unsqueeze(0)
        unsqueezed = True

    return tensor, unsqueezed


def identity(n: int, device: str = "cpu"):
    return torch.tensor((1.0, 0.0, 0.0, 0.0), device=device).view(1, 4).expand(n, 4)


def normalize(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Normalizes a quaternion.
    The quaternion should be in (w, x, y, z) format.
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          normalized. The tensor can be of shape :math:`(*, 4)`.
        eps (Optional[bool]): small value to avoid division by zero.
          Default: 1e-12.
    Return:
        torch.Tensor: the normalized quaternion of shape :math:`(*, 4)`.
    """
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(quaternion)))

    # if not quaternion.shape[-1] == 4:
    #     raise ValueError(
    #         "Input must be a tensor of shape (*, 4). Got {}".format(
    #             quaternion.shape))
    return F.normalize(quaternion, p=2.0, dim=-1, eps=eps)


def quat_to_mat(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Converts a quaternion to a rotation matrix.
    The quaternion should be in (w, x, y, z) format.
    Adapted from:
        https://github.com/kornia/kornia/blob/d729d7c4357ca73e4915a42285a0771bca4436ce/kornia/geometry/conversions.py#L235
    Args:
        quaternion (torch.Tensor): a tensor containing a quaternion to be
          converted. The tensor can be of shape :math:`(*, 4)`.
    Return:
        torch.Tensor: the rotation matrix of shape :math:`(*, 3, 3)`.
    Example:
        >>> quaternion = torch.tensor([0., 0., 1., 0.])
        >>> quat_to_mat(quaternion)
        tensor([[[-1.,  0.,  0.],
                 [ 0., -1.,  0.],
                 [ 0.,  0.,  1.]]])
    """
    quaternion, unsqueezed = ensure_batch_dim(quaternion, 1)

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape (*, 4). Got {}".format(quaternion.shape))
    # normalize the input quaternion
    quaternion_norm = normalize(quaternion)

    # unpack the normalized quaternion components
    w, x, y, z = torch.chunk(quaternion_norm, chunks=4, dim=-1)

    # compute the actual conversion
    tx: torch.Tensor = 2.0 * x
    ty: torch.Tensor = 2.0 * y
    tz: torch.Tensor = 2.0 * z
    twx: torch.Tensor = tx * w
    twy: torch.Tensor = ty * w
    twz: torch.Tensor = tz * w
    txx: torch.Tensor = tx * x
    txy: torch.Tensor = ty * x
    txz: torch.Tensor = tz * x
    tyy: torch.Tensor = ty * y
    tyz: torch.Tensor = tz * y
    tzz: torch.Tensor = tz * z
    one: torch.Tensor = torch.tensor(1.0)

    matrix: torch.Tensor = torch.stack(
        [
            one - (tyy + tzz),
            txy - twz,
            txz + twy,
            txy + twz,
            one - (txx + tzz),
            tyz - twx,
            txz - twy,
            tyz + twx,
            one - (txx + tyy),
        ],
        dim=-1,
    ).view(-1, 3, 3)

    if unsqueezed:
        matrix = matrix.squeeze(0)

    return matrix


def mat_to_quat(rotation_matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 4d quaternion vector.
    The quaternion vector has components in (w, x, y, z) format.
    Adapted From:
        https://github.com/kornia/kornia/blob/d729d7c4357ca73e4915a42285a0771bca4436ce/kornia/geometry/conversions.py#L235
    Args:
        rotation_matrix (torch.Tensor): the rotation matrix to convert.
        eps (float): small value to avoid zero division. Default: 1e-8.
    Return:
        torch.Tensor: the rotation in quaternion.
    Shape:
        - Input: :math:`(*, 3, 3)`
        - Output: :math:`(*, 4)`
    """
    rotation_matrix, unsqueezed = ensure_batch_dim(rotation_matrix, 2)

    if not isinstance(rotation_matrix, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(rotation_matrix)))

    if not rotation_matrix.shape[-2:] == (3, 3):
        raise ValueError("Input size must be a (*, 3, 3) tensor. Got {}".format(rotation_matrix.shape))

    def safe_zero_division(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(numerator.dtype).tiny
        return numerator / torch.clamp(denominator, min=eps)

    if not rotation_matrix.is_contiguous():
        rotation_matrix_vec: torch.Tensor = rotation_matrix.reshape(*rotation_matrix.shape[:-2], 9)
    else:
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

    if unsqueezed:
        quaternion = quaternion.squeeze(0)

    return quaternion


@torch.jit.script
def random(k: int = 1, device: str = "cpu"):
    """Return uniform random unit quaternion.

    rand: array like or None
        Three independent random variables that are uniformly distributed
        between 0 and 1.
    """
    rand = torch.rand(k, 3, device=device)
    r1 = torch.sqrt(1.0 - rand[:, 0])
    r2 = torch.sqrt(rand[:, 0])
    pi2 = math.pi * 2.0
    t1 = pi2 * rand[:, 1]
    t2 = pi2 * rand[:, 2]

    return torch.stack([torch.cos(t2) * r2, torch.sin(t1) * r1, torch.cos(t1) * r1, torch.sin(t2) * r2], dim=1)


def qmul(q1, q2):
    """Quaternion multiplication.

    Use the Hamilton product to perform quaternion multiplication.
    References:
        http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
        https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    ham_prod = torch.bmm(q2.view(-1, 4, 1), q1.view(-1, 1, 4))

    w = ham_prod[:, 0, 0] - ham_prod[:, 1, 1] - ham_prod[:, 2, 2] - ham_prod[:, 3, 3]
    x = ham_prod[:, 0, 1] + ham_prod[:, 1, 0] - ham_prod[:, 2, 3] + ham_prod[:, 3, 2]
    y = ham_prod[:, 0, 2] + ham_prod[:, 1, 3] + ham_prod[:, 2, 0] - ham_prod[:, 3, 1]
    z = ham_prod[:, 0, 3] - ham_prod[:, 1, 2] + ham_prod[:, 2, 1] + ham_prod[:, 3, 0]

    return torch.stack((w, x, y, z), dim=1).view(q1.shape)


def rotate_vector(quat, vector):
    """
    References:
            https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/quaternions.py#L419
    """
    assert quat.shape[-1] == 4
    assert vector.shape[-1] == 3
    assert quat.shape[:-1] == vector.shape[:-1]

    original_shape = list(vector.shape)
    quat = quat.view(-1, 4)
    vector = vector.view(-1, 3)

    pure_quat = quat[:, 1:]
    uv = torch.cross(pure_quat, vector, dim=1)
    uuv = torch.cross(pure_quat, uv, dim=1)
    return (vector + 2 * (quat[:, :1] * uv + uuv)).view(original_shape)


def from_spherical(theta, phi, r=1.0):
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(phi)
    w = torch.zeros_like(x)

    return torch.stack((w, x, y, z), dim=-1)


def from_axis_angle(axis, angle):
    """Compute a quaternion from the axis angle representation.

    Reference:
        https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    Args:
        axis: axis to rotate about
        angle: angle to rotate by
    Returns:
        Tensor of shape (*, 4) representing a quaternion.
    """
    if torch.is_tensor(axis) and isinstance(angle, float):
        angle = torch.tensor(angle, dtype=axis.dtype, device=axis.device)
        angle = angle.expand(axis.shape[0])

    axis = axis / torch.norm(axis, dim=-1, keepdim=True)

    c = torch.cos(angle / 2.0)
    s = torch.sin(angle / 2.0)

    w = c
    x = s * axis[..., 0]
    y = s * axis[..., 1]
    z = s * axis[..., 2]

    return torch.stack((w, x, y, z), dim=-1)


def qexp(q, eps=1e-8, is_normalized=False):
    """allow unnormalized Computes the quaternion exponent.

    Reference:
        https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions
    Args:
        q (tensor): (*, 3) or (*, 4)  the quaternion to compute the exponent of
    Returns:
        (tensor): Tensor of shape (*, 4) representing exp(q)
    """
    if is_normalized:
        q = normalize(q, eps=eps)
    if q.shape[1] == 4:
        # Let q = (s; v).
        s, v = torch.split(q, (1, 3), dim=-1)
    else:
        s = torch.zeros_like(q[:, :1])
        v = q

    theta = torch.norm(v, dim=-1, keepdim=True)
    exp_s = torch.exp(s)
    w = torch.cos(theta)
    xyz = 1.0 / theta.clamp(min=eps) * torch.sin(theta) * v

    return exp_s * torch.cat((w, xyz), dim=-1)


def qlog(q, eps=1e-8):
    """Computes the quaternion logarithm.

    Reference:
        https://en.wikipedia.org/wiki/Quaternion#Exponential,_logarithm,_and_power_functions
        https://users.aalto.fi/~ssarkka/pub/quat.pdf
    Args:
        q (tensor): the quaternion to compute the logarithm of
    Returns:
        (tensor): Tensor of shape (*, 4) representing ln(q)
    """

    mag = torch.norm(q, dim=-1, keepdim=True)
    # Let q = (s; v).
    s, v = torch.split(q, (1, 3), dim=-1)
    w = torch.log(mag)
    xyz = v / torch.norm(v, dim=-1, keepdim=True).clamp(min=eps) * acos_safe(s / mag.clamp(min=eps))

    return torch.cat((w, xyz), dim=-1)


def qdelta(n, std, device=None):
    omega = torch.cat((torch.zeros(n, 1, device=device), torch.randn(n, 3, device=device)), dim=-1)
    delta_q = qexp(std / 2.0 * omega)
    return delta_q


def perturb(q, std):
    """Perturbs the unit quaternion `q`.

    References:
        https://math.stackexchange.com/questions/2992016/how-to-linearize-quaternions
        http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_aa10_appendix.pdf
        https://math.stackexchange.com/questions/473736/small-angular-displacements-with-a-quaternion-representation
    Args:
        q (tensor): the quaternion to perturb (the mean of the perturbation)
        std (float): the stadnard deviation of the perturbation
    Returns:
        (tensor): Tensor of shape (*, 4), the perturbed quaternion
    """
    q, unsqueezed = ensure_batch_dim(q, num_dims=1)

    n = q.shape[0]
    delta_q = qdelta(n, std, device=q.device)
    q_out = qmul(delta_q, q)

    if unsqueezed:
        q_out = q_out.squeeze(0)

    return q_out


def angular_distance(q1, q2, eps: float = 1e-7):
    q1 = normalize(q1)
    q2 = normalize(q2)
    dot = q1 @ q2.t()
    dist = 2 * acos_safe(dot.abs(), eps=eps)
    return dist
