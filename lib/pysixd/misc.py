# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
# modified
# NOTE: some bbox calculation functions are wrong in original,
# and may not changed in this file.
"""Miscellaneous functions."""
import os
import math
import cv2
import mmcv
import subprocess
import numpy as np
from numba import jit, njit
from PIL import Image, ImageDraw
from scipy.spatial import distance
from lib.pysixd.inout import load_ply
from lib.pysixd import transform
from lib.utils import logger
from lib.vis_utils.colormap import colormap


log = logger.info


def pnp(points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_ITERATIVE):
    # try:
    #     dist_coeffs = pnp.dist_coeffs
    # except:
    #     print('except')
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    assert points_3d.shape[0] == points_2d.shape[0], "points 3D and points 2D must have same number of vertices"
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    _, R_exp, t = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=method)
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)
    # trans_3d=np.matmul(points_3d,R.transpose())+t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t], axis=-1)


def pnp_ransac_custom(
    image_points,
    model_points,
    K,
    pnp_type=cv2.SOLVEPNP_ITERATIVE,
    ransac_iter=100,
    ransac_min_iter=10,
    ransac_reprojErr=2,
):
    """ransac_reprojErr: 6.3  (2: lzg)
    ransac by lzg
    """
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    random_sample_num = 10
    confidence = 0.995
    # 'SOLVEPNP_AP3P', 'SOLVEPNP_DLS', 'SOLVEPNP_EPNP', 'SOLVEPNP_ITERATIVE',
    # 'SOLVEPNP_MAX_COUNT', 'SOLVEPNP_P3P', 'SOLVEPNP_UPNP'
    # cv2.SOLVEPNP_ITERATIVE
    best_err = float("inf")
    best_inliers_num = 0
    best_R_exp = None
    best_T_vector = None
    assert len(model_points) == len(image_points)
    corrs_num = len(model_points)
    # minInliersCount = int(0.5 * corrs_num)
    # end_flag = 0
    model_points = np.asarray(model_points)
    image_points = np.asarray(image_points)
    i_ransac = 0
    for iter in range(ransac_iter):
        i_ransac += 1
        # random sampling
        idx = np.random.choice(corrs_num, random_sample_num, replace=False)
        model_points_choose = model_points[idx]
        image_points_choose = image_points[idx]

        # x1, y1, z1 = model_points_choose[1] - model_points_choose[0]
        # x2, y2, z2 = model_points_choose[2] - model_points_choose[0]
        # x3, y3, z3 = model_points_choose[3] - model_points_choose[0]
        # if ((x1*y2*z3) + (x2*y3*z2) - (x3*y2*z1) - (z3*x2*y1) <= 1e-10) and \
        #     ((x1 * y2 * z3) + (x2 * y3 * z2) - (x3 * y2 * z1) - (z3 * x2 * y1) >= -1e-10):
        #     i_ransac -= 1
        #     continue

        # compute pose
        success, R_exp, T_vector = cv2.solvePnP(
            model_points_choose, image_points_choose, K, dist_coeffs, flags=pnp_type
        )
        # compute reprojection error
        pts_2d, _ = cv2.projectPoints(model_points, R_exp, T_vector, K, dist_coeffs)
        errs = np.linalg.norm(pts_2d.squeeze() - image_points, axis=1)
        # collect inliers
        inliers_idx = errs < ransac_reprojErr
        inliers_num = sum(inliers_idx)
        inliers_model_pts = model_points[inliers_idx]
        inliers_image_pts = image_points[inliers_idx]
        err_mean = errs.mean()
        if err_mean < best_err:
            best_err = errs.mean()
            best_R_exp = R_exp
            best_T_vector = T_vector
        # update
        if (err_mean < best_err or inliers_num > best_inliers_num) and inliers_num >= 4:
            best_inliers_num = inliers_num
            success, R_exp, T_vector = cv2.solvePnP(
                inliers_model_pts, inliers_image_pts, K, dist_coeffs, flags=pnp_type
            )
            # compute reprojection error
            pts_2d, _ = cv2.projectPoints(model_points, R_exp, T_vector, K, dist_coeffs)
            errs = np.linalg.norm(pts_2d.squeeze() - image_points, axis=1)
            if errs.mean() < best_err:
                best_err = errs.mean()
                best_R_exp = R_exp
                best_T_vector = T_vector
        # compute k
        w = inliers_num / corrs_num
        k = np.log10(1 - confidence) / np.log10(1 - pow(w, 10))
        if i_ransac > np.max([k, ransac_min_iter]):
            # print("best err: {}".format(best_err))
            break
    R_exp = best_R_exp
    T_vector = best_T_vector
    R, _ = cv2.Rodrigues(R_exp)
    return np.concatenate([R, T_vector.reshape((3, 1))], axis=-1)


def pnp_v2(
    points_3d, points_2d, camera_matrix, method=cv2.SOLVEPNP_EPNP, ransac=False, ransac_reprojErr=3.0, ransac_iter=100
):
    """
    method: cv2.SOLVEPNP_P3P, cv2.SOLVEPNP_DLS, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
        DLS seems to be similar to EPNP
        SOLVEPNP_EPNP does not work with no ransac
    RANSAC:
        CDPN: 3.0, 100
        default ransac params:   float reprojectionError=8.0, int iterationsCount=100, double confidence=0.99
        in DPOD paper: reproj error=1.0, ransac_iter=150
    """
    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")

    assert points_3d.shape[0] == points_2d.shape[0], "points 3D and points 2D must have same number of vertices"
    if method == cv2.SOLVEPNP_EPNP:
        points_3d = np.expand_dims(points_3d, 0)
        points_2d = np.expand_dims(points_2d, 0)

    points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
    points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
    camera_matrix = camera_matrix.astype(np.float64)
    if not ransac:
        _, R_exp, t = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, flags=method)
    else:
        _, R_exp, t, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            camera_matrix,
            dist_coeffs,
            # flags=cv2.SOLVEPNP_EPNP,
            flags=method,
            reprojectionError=ransac_reprojErr,
            iterationsCount=ransac_iter,
        )
    # , None, None, False, cv2.SOLVEPNP_UPNP)

    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)

    R, _ = cv2.Rodrigues(R_exp)  # R = cv2.Rodrigues(R_exp, jacobian=0)[0]
    # trans_3d=np.matmul(points_3d, R.transpose()) + t.transpose()
    # if np.max(trans_3d[:,2]<0):
    #     R=-R
    #     t=-t

    return np.concatenate([R, t.reshape((3, 1))], axis=-1)


def ensure_dir(path):
    """Ensures that the specified directory exists.

    :param path: Path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def get_symmetry_transformations(model_info, max_sym_disc_step):
    """Returns a set of symmetry transformations for an object model.

    :param model_info: See files models_info.json provided with the datasets.
    :param max_sym_disc_step: The maximum fraction of the object diameter which
      the vertex that is the furthest from the axis of continuous rotational
      symmetry travels between consecutive discretized rotations.
    :return: The set of symmetry transformations.
    """
    # NOTE: t is in mm, so may need to devide 1000
    # Discrete symmetries.
    trans_disc = [{"R": np.eye(3), "t": np.array([[0, 0, 0]]).T}]  # Identity.
    if "symmetries_discrete" in model_info:
        for sym in model_info["symmetries_discrete"]:
            sym_4x4 = np.reshape(sym, (4, 4))
            R = sym_4x4[:3, :3]
            t = sym_4x4[:3, 3].reshape((3, 1))
            trans_disc.append({"R": R, "t": t})

    # Discretized continuous symmetries.
    trans_cont = []
    if "symmetries_continuous" in model_info:
        for sym in model_info["symmetries_continuous"]:
            axis = np.array(sym["axis"])
            offset = np.array(sym["offset"]).reshape((3, 1))

            # (PI * diam.) / (max_sym_disc_step * diam.) = discrete_steps_count
            discrete_steps_count = int(np.ceil(np.pi / max_sym_disc_step))

            # Discrete step in radians.
            discrete_step = 2.0 * np.pi / discrete_steps_count

            for i in range(1, discrete_steps_count):
                R = transform.rotation_matrix(i * discrete_step, axis)[:3, :3]
                t = -R.dot(offset) + offset
                trans_cont.append({"R": R, "t": t})

    # Combine the discrete and the discretized continuous symmetries.
    trans = []
    for tran_disc in trans_disc:
        if len(trans_cont):
            for tran_cont in trans_cont:
                R = tran_cont["R"].dot(tran_disc["R"])
                t = tran_cont["R"].dot(tran_disc["t"]) + tran_cont["t"]
                trans.append({"R": R, "t": t})
        else:
            trans.append(tran_disc)

    return trans


def draw_rect(vis, rect, color=(255, 255, 255)):
    vis_pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(vis_pil)
    draw.rectangle((rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), outline=color, fill=None)
    del draw
    return np.asarray(vis_pil)


def points_to_2D(points, R, T, K):
    """
    discription: project 3D points to 2D image plane

    :param points: (N, 3)
    :param R: (3, 3)
    :param T: (3, )
    :param K: (3, 3)
    :return: points_2D: (N, 2), z: (N,)
    """
    # using opencv
    # cv2Proj2d = cv2.projectPoints(model_points, R_vec, T, K, None)[0]
    points_in_world = np.matmul(R, points.T) + T.reshape((3, 1))  # (3, N)
    points_in_camera = np.matmul(K, points_in_world)  # (3, N) # z is not changed in this step
    N = points_in_world.shape[1]
    points_2D = np.zeros((2, N))
    points_2D[0, :] = points_in_camera[0, :] / (points_in_camera[2, :] + 1e-15)
    points_2D[1, :] = points_in_camera[1, :] / (points_in_camera[2, :] + 1e-15)
    z = points_in_world[2, :]
    return points_2D.T, z


# @jit
def calc_emb_bp_fast(depth, R, T, K):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # ProjEmb = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    ProjEmb = (
        np.einsum(
            "ijkl,ijlm->ijkm",
            R.T.reshape(1, 1, 3, 3),
            depth.reshape(height, width, 1, 1)
            * np.einsum("ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1))
            - T.reshape(1, 1, 3, 1),
        ).squeeze()
        * mask.reshape(height, width, 1)
    )

    return ProjEmb


calc_xyz_bp_fast = calc_emb_bp_fast


def backproject(depth, K):
    """Backproject a depth map to a cloud map.

    :param depth: Input depth map [H, W]
    :param K: Intrinsics of the camera
    :return: An organized cloud map
    """
    H, W = depth.shape[:2]
    X = np.asarray(range(W)) - K[0, 2]
    X = np.tile(X, (H, 1))
    Y = np.asarray(range(H)) - K[1, 2]
    Y = np.tile(Y, (W, 1)).transpose()
    return np.stack((X * depth / K[0, 0], Y * depth / K[1, 1], depth), axis=2)


def backproject_th(depth, K):
    """Backproject a depth map to a cloud map.

    :param depth: Input depth map [H, W]
    :param K: Intrinsics of the camera
    :return: An organized cloud map
    """
    import torch

    assert depth.ndim == 2, depth.ndim
    H, W = depth.shape[:2]
    X = torch.tensor(range(W)).to(depth) - K[0, 2]
    X = X.repeat(H, 1)
    Y = torch.tensor(range(H)).to(depth) - K[1, 2]
    Y = Y.repeat(W, 1).t()
    return torch.stack((X * depth / K[0, 0], Y * depth / K[1, 1], depth), dim=2)


def backproject_v2(depth, K):
    """Backproject a depth map to a cloud map
    seems slower
    depth:  depth
    ----
    organized cloud map: (H,W,3)
    """
    Kinv = np.linalg.inv(K)

    height, width = depth.shape
    # pc_cam = np.zeros((height, width, 3)).astype(np.float32)

    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    grid_2d = np.stack([grid_x, grid_y, np.ones((height, width))], axis=2)
    mask = (depth != 0).astype(depth.dtype)
    pc_cam = depth.reshape(height, width, 1, 1) * np.einsum(
        "ijkl,ijlm->ijkm", Kinv.reshape(1, 1, 3, 3), grid_2d.reshape(height, width, 3, 1)
    )
    pc_cam = pc_cam.squeeze()
    return pc_cam


def calc_emb_bp_torch(depth, R, T, K, ProjEmb):
    """
    depth: rendered depth
    ----
    ProjEmb: (H,W,3)
    """
    import torch

    Kinv = torch.inverse(K)

    height, width = depth.shape
    mask = (depth != 0).to(depth)
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width))
    grid_2d = torch.stack([grid_x, grid_y, torch.ones(height, width, dtype=torch.long)], axis=2).to(depth)
    Rinv_expand = R.t().expand(height, width, 3, 3)
    Kinv_expand = Kinv.expand(height, width, 3, 3)
    T_expand = T.view((3, 1)).expand(height, width, 3, 1)
    # [H,W,3]
    ProjEmb.data = (
        torch.matmul(
            Rinv_expand,
            depth.view(height, width, 1, 1) * torch.matmul(Kinv_expand, grid_2d.view(height, width, 3, 1)) - T_expand,
        ).squeeze()
        * mask.view(height, width, 1)
    )


calc_xyz_bp_torch = calc_emb_bp_torch


def calc_emb(model_points_, R, T, K, height=480, width=640):
    # directly project 3d points onto 2d plane
    # numerical error due to round
    points_2d, z = points_to_2D(model_points_, R, T, K)
    image_points = np.round(points_2d).astype(np.int32)
    ProjEmb = np.zeros((height, width, 3)).astype(np.float32)
    depth = np.zeros((height, width, 1)).astype(np.float32)
    for i, (x, y) in enumerate(image_points):
        if x >= width or y >= height or x < 0 or y < 0:
            continue
        if depth[y, x, 0] == 0:
            depth[y, x, 0] = z[i]
            ProjEmb[y, x] = model_points_[i]
        elif z[i] < depth[y, x, 0]:
            depth[y, x, 0] = z[i]
            ProjEmb[y, x] = model_points_[i]
        else:
            pass
    # print("ProjEmb: min {} max {}".format(ProjEmb.min(), ProjEmb.max()))
    return ProjEmb


calc_xyz = calc_emb


def calc_emb_proj(vertices, R, T, K, attributes=None, width=640, height=480):
    # directly project 3d points onto 2d plane
    # numerical error due to round
    if attributes is None:
        # default project 3d coordinates
        attributes = vertices
    elif attributes == "nocs":
        nocs = np.copy(vertices)
        xmin, xmax = nocs[:, 0].min(), nocs[:, 0].max()
        ymin, ymax = nocs[:, 1].min(), nocs[:, 1].max()
        zmin, zmax = nocs[:, 2].min(), nocs[:, 2].max()
        # # move (xmin, ymin, zmin) to origin, model centered at the 3b bbox center
        nocs[:, 0] -= xmin
        nocs[:, 1] -= ymin
        nocs[:, 2] -= zmin
        # scale = max(max(xmax - xmin, ymax - ymin), zmax - zmin)
        diagonal = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
        # unit diagonal
        nocs /= diagonal
        attributes = nocs
    elif attributes == "normalized_coords":
        normalizedCoords = np.copy(vertices)
        xmin, xmax = vertices[:, 0].min(), vertices[:, 0].max()
        ymin, ymax = vertices[:, 1].min(), vertices[:, 1].max()
        zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
        # normalize every axis to [0, 1]
        normalizedCoords[:, 0] = (normalizedCoords[:, 0] - xmin) / (xmax - xmin)
        normalizedCoords[:, 1] = (normalizedCoords[:, 1] - ymin) / (ymax - ymin)
        normalizedCoords[:, 2] = (normalizedCoords[:, 2] - zmin) / (zmax - zmin)

    assert vertices.shape[0] == attributes.shape[0], "points and attributes shape mismatch"
    points_2d, z = points_to_2D(vertices, R, T, K)
    image_points = np.round(points_2d).astype(np.int32)

    n_c = attributes.shape[1]
    ProjEmb = np.zeros((height, width, n_c)).astype(np.float32)

    depth = np.zeros((height, width, 1)).astype(np.float32)
    for i, (x, y) in enumerate(image_points):
        if x >= width or y >= height or x < 0 or y < 0:
            continue
        if depth[y, x, 0] == 0:
            depth[y, x, 0] = z[i]
            ProjEmb[y, x] = attributes[i]
        elif z[i] < depth[y, x, 0]:
            depth[y, x, 0] = z[i]
            ProjEmb[y, x] = attributes[i]
        else:
            pass
    # print("ProjEmb: min {} max {}".format(ProjEmb.min(), ProjEmb.max()))
    return ProjEmb


calc_xyz_proj = calc_emb_proj


def points2d_to_mask(points2d, height=480, width=640):
    # points2d: (N,2)
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in points2d:
        x = int(round(x))
        y = int(round(y))
        # print(x, y)
        if x >= 0 and x < width and y >= 0 and y < height:
            mask[y, x] = 1
    return mask


def project_model(model, pose, K):
    """
    model: Nx3
    pose: 3x4
    K: 3x3
    ----------
    Return
    projected points: Nx2
    """
    camera_points_3d = np.dot(model, pose[:, :3].T) + pose[:, 3].reshape((1, 3))  # Nx3
    camera_points_3d = np.dot(camera_points_3d, K.T)  # Nx3
    return camera_points_3d[:, :2] / camera_points_3d[:, 2:]


def project_pts(pts, K, R, t):
    """Projects 3D points.

    :param pts: nx3 ndarray with the 3D points.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx2 ndarray with 2D image coordinates of the projections.
    """
    assert pts.shape[1] == 3
    P = K.dot(np.hstack((R, t.reshape(3, 1))))  # 3x4
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))  # Nx4
    pts_im = P.dot(pts_h.T)  # 3xN
    pts_im /= pts_im[2, :]  # 3xN
    return pts_im[:2, :].T  # Nx2


@jit
def get_obj_im_c(K, t):
    # t: cam_t_m2c
    K = K.astype(np.float32)
    t = t.astype(np.float32)
    obj_c = K.dot(t)
    c_x = obj_c[0] / obj_c[2]
    c_y = obj_c[1] / obj_c[2]
    return c_x, c_y


class Precomputer(object):
    """Caches pre_Xs, pre_Ys for a 30% speedup of depth_im_to_dist_im()"""

    xs, ys = None, None
    pre_Xs, pre_Ys = None, None
    depth_im_shape = None
    K = None

    @staticmethod
    def precompute_lazy(depth_im, K):
        """Lazy precomputation for depth_im_to_dist_im() if depth_im.shape or K
        changes.

        :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
          is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
          or 0 if there is no such 3D point (this is a typical output of the
          Kinect-like sensors).
        :param K: 3x3 ndarray with an intrinsic camera matrix.
        :return: hxw ndarray (Xs/depth_im, Ys/depth_im)
        """
        if depth_im.shape != Precomputer.depth_im_shape:
            Precomputer.depth_im_shape = depth_im.shape
            Precomputer.xs, Precomputer.ys = np.meshgrid(np.arange(depth_im.shape[1]), np.arange(depth_im.shape[0]))

        if depth_im.shape != Precomputer.depth_im_shape or not np.all(K == Precomputer.K):
            Precomputer.K = K
            Precomputer.pre_Xs = (Precomputer.xs - K[0, 2]) / np.float64(K[0, 0])
            Precomputer.pre_Ys = (Precomputer.ys - K[1, 2]) / np.float64(K[1, 1])

        return Precomputer.pre_Xs, Precomputer.pre_Ys


def depth_im_to_dist_im_fast(depth_im, K):
    """Converts a depth image to a distance image.

    :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
      is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y],
      or 0 if there is no such 3D point (this is a typical output of the
      Kinect-like sensors).
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :return: hxw ndarray with the distance image, where dist_im[y, x] is the
      distance from the camera center to the 3D point [X, Y, Z] that projects to
      pixel [x, y], or 0 if there is no such 3D point.
    """
    # Only recomputed if depth_im.shape or K changes.
    pre_Xs, pre_Ys = Precomputer.precompute_lazy(depth_im, K)

    dist_im = np.sqrt(
        np.multiply(pre_Xs, depth_im) ** 2 + np.multiply(pre_Ys, depth_im) ** 2 + depth_im.astype(np.float64) ** 2
    )

    return dist_im


def depth_im_to_dist_im(depth_im, K):
    """Converts a depth image to a distance image.

    :param depth_im: hxw ndarray with the input depth image, where depth_im[y, x]
     is the Z coordinate of the 3D point [X, Y, Z] that projects to pixel [x, y], or 0 if there is
    no such 3D point (this is a typical output of the
     Kinect-like sensors).
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :return: hxw ndarray with the distance image, where dist_im[y, x] is the
     distance from the camera center to the 3D point [X, Y, Z] that projects to
      pixel [x, y], or 0 if there is no such 3D point.
    """
    xs, ys = np.meshgrid(np.arange(depth_im.shape[1]), np.arange(depth_im.shape[0]))

    Xs = np.multiply(xs - K[0, 2], depth_im) * (1.0 / K[0, 0])
    Ys = np.multiply(ys - K[1, 2], depth_im) * (1.0 / K[1, 1])

    dist_im = np.sqrt(Xs.astype(np.float64) ** 2 + Ys.astype(np.float64) ** 2 + depth_im.astype(np.float64) ** 2)
    # dist_im = np.linalg.norm(np.dstack((Xs, Ys, depth_im)), axis=2)  # Slower.

    return dist_im


def norm_depth(depth, valid_start=0.2, valid_end=1.0):
    mask = depth > 0
    depth_n = depth.astype(np.float)
    depth_n[mask] -= depth_n[mask].min()
    depth_n[mask] /= depth_n[mask].max() / (valid_end - valid_start)
    depth_n[mask] += valid_start
    return depth_n


def rgbd_to_point_cloud(K, depth, rgb=np.array([])):
    vs, us = depth.nonzero()
    zs = depth[vs, us]
    xs = ((us - K[0, 2]) * zs) / float(K[0, 0])
    ys = ((vs - K[1, 2]) * zs) / float(K[1, 1])
    pts = np.array([xs, ys, zs]).T
    pts_im = np.vstack([us, vs]).T
    if rgb != np.array([]):
        colors = rgb[vs, us, :]
    else:
        colors = None
    return pts, colors, pts_im


def clip_pt_to_im(pt, width, height):
    pt_c = [min(max(pt[0], 0), width - 1), min(max(pt[1], 0), height - 1)]
    return pt_c


def calc_2d_bbox_xywh(xs, ys, width, height, clip=False):
    bb_lt = [xs.min(), ys.min()]
    bb_rb = [xs.max(), ys.max()]
    if clip:
        bb_lt = clip_pt_to_im(bb_lt, width, height)
        bb_rb = clip_pt_to_im(bb_rb, width, height)
    # why not +1
    # return [bb_tl[0], bb_tl[1], bb_br[0] - bb_tl[0], bb_br[1] - bb_tl[1]]
    x1, y1 = bb_lt
    x2, y2 = bb_rb
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def calc_2d_bbox_xyxy(xs, ys, width=640, height=480, clip=False):
    bb_lt = [xs.min(), ys.min()]
    bb_rb = [xs.max(), ys.max()]
    if clip:
        bb_lt = clip_pt_to_im(bb_lt, width, height)
        bb_rb = clip_pt_to_im(bb_rb, width, height)
    x1, y1 = bb_lt
    x2, y2 = bb_rb
    return [x1, y1, x2, y2]


def calc_2d_bbox_xyxy_v2(xs, ys, width=640, height=480, clip=False):
    """br is excluded."""
    bb_lt = [xs.min(), ys.min()]
    bb_rb = [xs.max(), ys.max()]
    x1, y1 = bb_lt
    x2, y2 = bb_rb
    if clip:
        x1 = min(max(x1, 0), width - 1)
        x2 = min(max(x1, 0), width - 1)
        y1 = min(max(x1, 0), height - 1)
        y2 = min(max(x1, 0), height - 1)
    return [x1, y1, x2 + 1, y2 + 1]


# def calc_pose_2d_bbox_old(model, im_size, K, R_m2c, t_m2c):
#     pts_im = project_pts(model["pts"], K, R_m2c, t_m2c)
#     pts_im = np.round(pts_im).astype(np.int)
#     return calc_2d_bbox(pts_im[:, 0], pts_im[:, 1], im_size)


def calc_pose_2d_bbox_xywh(points, width, height, K, R_m2c, t_m2c):
    pts_im = project_pts(points, K, R_m2c, t_m2c)
    # pts_im = np.round(pts_im).astype(np.int)
    return calc_2d_bbox_xywh(pts_im[:, 0], pts_im[:, 1], width, height)


def calc_pose_2d_bbox_xyxy(points, width, height, K, R_m2c, t_m2c):
    pts_im = project_pts(points, K, R_m2c, t_m2c)
    # pts_im = np.round(pts_im).astype(np.int)
    return calc_2d_bbox_xyxy(pts_im[:, 0], pts_im[:, 1], width, height)


def calc_pose_2d_bbox_xyxy_v2(points, width, height, K, R_m2c, t_m2c):
    """br is excluded."""
    pts_im = project_pts(points, K, R_m2c, t_m2c)
    # pts_im = np.round(pts_im).astype(np.int)
    return calc_2d_bbox_xyxy_v2(pts_im[:, 0], pts_im[:, 1], width, height)


@jit
def compute_2d_bbox_xyxy_from_pose(points, pose, K, width=640, height=480, clip=False):
    x3d = np.ones((4, points.shape[0]), dtype=np.float32)
    x3d[0, :] = points[:, 0]
    x3d[1, :] = points[:, 1]
    x3d[2, :] = points[:, 2]

    RT = pose.astype(np.float32)  # numba should operate on the same dtype
    K = K.astype(np.float32)
    # _K = K.copy()
    # _K[0, 2] -= 0.5
    # _K[1, 2] -= 0.5
    x2d = K.dot(RT.dot(x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    x1 = np.min(x2d[0, :])
    y1 = np.min(x2d[1, :])
    x2 = np.max(x2d[0, :])
    y2 = np.max(x2d[1, :])
    if clip:
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
    return np.array([x1, y1, x2, y2])  # NOTE: this is not rounded


@jit
def compute_2d_bbox_xyxy_from_pose_v2(points, pose, K, width=640, height=480, clip=False):
    """br excluded."""
    x3d = np.ones((4, points.shape[0]), dtype=np.float32)
    x3d[0, :] = points[:, 0]
    x3d[1, :] = points[:, 1]
    x3d[2, :] = points[:, 2]

    RT = pose.astype(np.float32)
    K = K.astype(np.float32)
    # _K = K.copy()
    # _K[0, 2] -= 0.5
    # _K[1, 2] -= 0.5
    x2d = np.matmul(K, np.matmul(RT, x3d))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    x1 = np.min(x2d[0, :])
    y1 = np.min(x2d[1, :])
    x2 = np.max(x2d[0, :])
    y2 = np.max(x2d[1, :])
    if clip:
        x1 = min(max(x1, 0), width - 1)
        y1 = min(max(y1, 0), height - 1)
        x2 = min(max(x2, 0), width - 1)
        y2 = min(max(y2, 0), height - 1)
    return np.array([x1, y1, x2 + 1, y2 + 1])  # NOTE: this is not rounded


@jit
def compute_2d_bbox_xywh_from_pose(points, pose, K, width=640, height=480, clip=False):
    x1, y1, x2, y2 = compute_2d_bbox_xyxy_from_pose(points, pose, K, width=width, height=height, clip=clip)
    return np.array([x1, y1, x2 - x1 + 1, y2 - y1 + 1])  # NOTE: this is not rounded


def calc_3d_bbox(xs, ys, zs):
    """Calculates 3D bounding box of the given set of 3D points.

    :param xs: 1D ndarray with x-coordinates of 3D points.
    :param ys: 1D ndarray with y-coordinates of 3D points.
    :param zs: 1D ndarray with z-coordinates of 3D points.
    :return: 3D bounding box (x, y, z, w, h, d), where (x, y, z) is the top-left
      corner and (w, h, d) is width, height and depth of the bounding box.
    """
    bb_min = [xs.min(), ys.min(), zs.min()]
    bb_max = [xs.max(), ys.max(), zs.max()]
    # NOTE: this need to +1, do not use this !!!!!!!!!!!!!!!
    return [bb_min[0], bb_min[1], bb_min[2], bb_max[0] - bb_min[0], bb_max[1] - bb_min[1], bb_max[2] - bb_min[2]]


def crop_im(im, roi):
    # roi: xywh
    im_h, im_w = im.shape[:2]
    if im.ndim == 3:
        crop = im[max(roi[1], 0) : min(roi[1] + roi[3] + 1, im_h), max(roi[0], 0) : min(roi[0] + roi[2] + 1, im_w), :]
    else:
        crop = im[max(roi[1], 0) : min(roi[1] + roi[3] + 1, im_h), max(roi[0], 0) : min(roi[0] + roi[2] + 1, im_w)]
    return crop


def paste_im(src, trg, pos):
    """Pastes src to trg with the top left corner at pos."""
    assert src.ndim == trg.ndim

    # Size of the region to be pasted
    w = min(src.shape[1], trg.shape[1] - pos[0])
    h = min(src.shape[0], trg.shape[0] - pos[1])

    if src.ndim == 3:
        trg[pos[1] : (pos[1] + h), pos[0] : (pos[0] + w), :] = src[:h, :w, :]
    else:
        trg[pos[1] : (pos[1] + h), pos[0] : (pos[0] + w)] = src[:h, :w]


def iou(bb_a, bb_b):
    """Calculates the Intersection over Union (IoU) of two 2D bounding boxes.

    :param bb_a: 2D bounding box (x1, y1, w1, h1) -- see calc_2d_bbox.
    :param bb_b: 2D bounding box (x2, y2, w2, h2) -- see calc_2d_bbox.
    :return: The IoU value.
    """
    # [x1, y1, width, height] --> [x1, y1, x2, y2]
    tl_a, br_a = (bb_a[0], bb_a[1]), (bb_a[0] + bb_a[2], bb_a[1] + bb_a[3])
    tl_b, br_b = (bb_b[0], bb_b[1]), (bb_b[0] + bb_b[2], bb_b[1] + bb_b[3])

    # Intersection rectangle.
    tl_inter = max(tl_a[0], tl_b[0]), max(tl_a[1], tl_b[1])
    br_inter = min(br_a[0], br_b[0]), min(br_a[1], br_b[1])

    # Width and height of the intersection rectangle.
    w_inter = br_inter[0] - tl_inter[0]
    h_inter = br_inter[1] - tl_inter[1]

    if w_inter > 0 and h_inter > 0:
        area_inter = w_inter * h_inter
        area_a = bb_a[2] * bb_a[3]
        area_b = bb_b[2] * bb_b[3]
        iou = area_inter / float(area_a + area_b - area_inter)
    else:
        iou = 0.0

    return iou


def paste_emb_to_im(src, bbox_xyxy, height=480, width=640):
    """Pastes cropped emb to img size: (height, width, c) at bbox_xyxy.

    src: (h, w, c)
    """
    if src.ndim == 3:
        channel = src.shape[-1]
        tgt = np.zeros((height, width, channel), dtype=np.float32)
    else:
        tgt = np.zeros((height, width), dtype=np.float32)
    x1, y1, x2, y2 = bbox_xyxy[:4]
    if src.ndim == 3:
        tgt[y1 : y2 + 1, x1 : x2 + 1, :] = src
    else:
        tgt[y1 : y2 + 1, x1 : x2 + 1] = src
    return tgt


def paste_emb_to_im_batch(embs, bboxes_xyxy, height=480, width=640):
    """Pastes cropped embs to img size: (height, width, c) at bbox_xyxy.

    src: (h, w, c)
    """
    num = len(embs)
    tgt_embs = [None for _ in embs]
    for i in range(num):
        if embs[i] is not None:
            if embs[i].ndim == 3:
                channel = embs[i].shape[-1]
                tgt_embs = [
                    np.zeros((height, width, channel), dtype=np.float32) if emb is not None else None for emb in embs
                ]
            else:
                tgt_embs = [np.zeros((height, width), dtype=np.float32) if emb is not None else None for emb in embs]
            break

    for i in range(num):
        src = embs[i]
        if src is None:
            tgt_embs[i] = None
            continue
        x1, y1, x2, y2 = bboxes_xyxy[i, :4].astype(np.int)
        tgt_embs[i][y1 : y2 + 1, x1 : x2 + 1] = src
    return tgt_embs


def paste_im_mask(src, trg, pos, mask):
    assert src.ndim == trg.ndim
    assert src.shape[:2] == mask.shape[:2]
    src_pil = Image.fromarray(src)
    trg_pil = Image.fromarray(trg)
    mask_pil = Image.fromarray(mask.astype(np.uint8))
    trg_pil.paste(src_pil, pos, mask_pil)
    trg[:] = np.array(trg_pil)[:]


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def transform_pts_Rt_th(pts, R, t):
    """Applies a rigid transformation to 3D points.

    # NOTE: this is not for batched points
    :param pts: nx3 tensor with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 tensor with transformed 3D points.
    """
    import torch

    assert pts.shape[1] == 3
    if not isinstance(pts, torch.Tensor):
        pts = torch.as_tensor(pts)
    if not isinstance(R, torch.Tensor):
        R = torch.as_tensor(R).to(pts)
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t).to(pts)
    pts_res = R.view(1, 3, 3) @ pts.view(10, 3, 1) + t.view(1, 3, 1)
    return pts_res.squeeze(-1)  # (n, 3)


def transform_pts_batch(pts, R, t=None):
    """
    Args:
        pts: (B,P,3)
        R: (B,3,3)
        t: (B,3,1)

    Returns:

    """
    bs = R.shape[0]
    n_pts = pts.shape[1]
    assert pts.shape == (bs, n_pts, 3)
    if t is not None:
        assert t.shape[0] == bs

    pts_transformed = R.view(bs, 1, 3, 3) @ pts.view(bs, n_pts, 3, 1)
    if t is not None:
        pts_transformed += t.view(bs, 1, 3, 1)
    return pts_transformed.squeeze(-1)  # (B, P, 3)


def calc_pts_diameter(pts):
    """Calculates the diameter of a set of 3D points (i.e. the maximum distance
    between any two points in the set).

    :param pts: nx3 ndarray with 3D points.
    :return: The calculated diameter.
    """
    diameter = -1.0
    for pt_id in range(pts.shape[0]):
        pt_dup = np.tile(np.array([pts[pt_id, :]]), [pts.shape[0] - pt_id, 1])
        pts_diff = pt_dup - pts[pt_id:, :]
        max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
        if max_dist > diameter:
            diameter = max_dist
    return diameter


def calc_pts_diameter2(pts):
    """Calculates the diameter of a set of 3D points (i.e. the maximum distance
    between any two points in the set). Faster but requires more memory than
    calc_pts_diameter.

    :param pts: nx3 ndarray with 3D points.
    :return: The calculated diameter.
    """
    dists = distance.cdist(pts, pts, "euclidean")
    diameter = np.max(dists)
    return diameter


def get_bbox3d_and_center(pts):
    """
    pts: Nx3
    ---
    bb: bb3d+center, 9x3
    """
    bb = []
    minx, maxx = min(pts[:, 0]), max(pts[:, 0])
    miny, maxy = min(pts[:, 1]), max(pts[:, 1])
    minz, maxz = min(pts[:, 2]), max(pts[:, 2])
    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])
    # (000)-->
    # bb.append([minx, miny, minz])
    # bb.append([minx, maxy, minz])
    # bb.append([minx, miny, maxz])
    # bb.append([minx, maxy, maxz])
    # bb.append([maxx, miny, minz])
    # bb.append([maxx, maxy, minz])
    # bb.append([maxx, miny, maxz])
    # bb.append([maxx, maxy, maxz])
    # bb.append([avgx, avgy, avgz])
    # bb = np.asarray(bb, dtype=np.float32)
    # NOTE: we use a different order from roi10d
    """
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    bb = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )
    return bb


def get_axis3d_and_center(pts, scale=0.5):
    """
    pts: 4x3
    ---
    kpts: 3d axis points + center, 4x3
    """
    bb = []
    minx, maxx = min(pts[:, 0]), max(pts[:, 0])
    miny, maxy = min(pts[:, 1]), max(pts[:, 1])
    minz, maxz = min(pts[:, 2]), max(pts[:, 2])
    avgx = np.average(pts[:, 0])
    avgy = np.average(pts[:, 1])
    avgz = np.average(pts[:, 2])

    """
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
            2
            |
            3 ---1
           /
          0
    """
    bb = np.array(
        [
            [maxx, maxy, maxz],
            [minx, maxy, maxz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, maxy, minz],
            [minx, maxy, minz],
            [minx, miny, minz],
            [maxx, miny, minz],
            [avgx, avgy, avgz],
        ],
        dtype=np.float32,
    )
    # front, right, up, center
    kpts = np.array(
        [
            (bb[2] + bb[3] + bb[6] + bb[7]) / 4,
            (bb[0] + bb[3] + bb[4] + bb[7]) / 4,
            (bb[0] + bb[1] + bb[2] + bb[3]) / 4,
            bb[-1],
        ],
        dtype=np.float32,
    )
    kpts = (kpts - bb[-1][None]) * scale + bb[-1][None]
    return kpts


def get_3D_corners(pts):
    """
    Args:
        pts: nx3
    Return:
        corners: 8x3
    """
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])
    min_z = np.min(pts[:, 2])
    max_z = np.max(pts[:, 2])

    corners = np.array(
        [
            [max_x, max_y, max_z],
            [min_x, max_y, max_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [min_x, max_y, min_z],
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
        ]
    )
    # the last row is 1: RP + T
    # corners = np.concatenate((np.transpose(corners), np.ones((1, 8))), axis=0)
    return corners


def overlapping_sphere_projections(radius, p1, p2):
    """Checks if projections of two spheres overlap (approximated).

    :param radius: Radius of the two spheres.
    :param p1: [X1, Y1, Z1] center of the first sphere.
    :param p2: [X2, Y2, Z2] center of the second sphere.
    :return: True if the projections of the two spheres overlap.
    """
    if p1[2] == 0 or p2[2] == 0:
        return False

    # 2D projections of centers of the spheres.
    proj1 = (p1 / p1[2])[:2]
    proj2 = (p2 / p2[2])[:2]

    # Distance between the center projections.
    proj_dist = np.linalg.norm(proj1 - proj2)

    # The max. distance of the center projections at which the sphere projections,
    # i.e. sphere silhouettes, still overlap (approximated).
    proj_dist_thresh = radius * (1.0 / p1[2] + 1.0 / p2[2])

    return proj_dist < proj_dist_thresh


def get_error_signature(error_type, n_top, **kwargs):
    """Generates a signature for the specified settings of pose error
    calculation.

    :param error_type: Type of error.
    :param n_top: Top N pose estimates (with the highest score) to be evaluated
      for each object class in each image.
    :return: Generated signature.
    """
    error_sign = "error:" + error_type + "_ntop:" + str(n_top)
    if error_type == "vsd":
        if kwargs["vsd_tau"] == float("inf"):
            vsd_tau_str = "inf"
        else:
            vsd_tau_str = "{:.3f}".format(kwargs["vsd_tau"])
        error_sign += "_delta:{:.3f}_tau:{}".format(kwargs["vsd_delta"], vsd_tau_str)
    return error_sign


def get_score_signature(correct_th, visib_gt_min):
    """Generates a signature for a performance score.

    :param visib_gt_min: Minimum visible surface fraction of a valid GT pose.
    :return: Generated signature.
    """
    eval_sign = "th:" + "-".join(["{:.3f}".format(t) for t in correct_th])
    eval_sign += "_min-visib:{:.3f}".format(visib_gt_min)
    return eval_sign


def run_meshlab_script(meshlab_server_path, meshlab_script_path, model_in_path, model_out_path, attrs_to_save):
    """Runs a MeshLab script on a 3D model.

    meshlabserver depends on X server. To remove this dependence (on linux), run:
    1) Xvfb :100 &
    2) export DISPLAY=:100.0
    3) meshlabserver <my_options>

    :param meshlab_server_path: Path to meshlabserver.exe.
    :param meshlab_script_path: Path to an MLX MeshLab script.
    :param model_in_path: Path to the input 3D model saved in the PLY format.
    :param model_out_path: Path to the output 3D model saved in the PLY format.
    :param attrs_to_save: Attributes to save:
      - vc -> vertex colors
      - vf -> vertex flags
      - vq -> vertex quality
      - vn -> vertex normals
      - vt -> vertex texture coords
      - fc -> face colors
      - ff -> face flags
      - fq -> face quality
      - fn -> face normals
      - wc -> wedge colors
      - wn -> wedge normals
      - wt -> wedge texture coords
    """
    meshlabserver_cmd = [meshlab_server_path, "-s", meshlab_script_path, "-i", model_in_path, "-o", model_out_path]

    if len(attrs_to_save):
        meshlabserver_cmd += ["-m"] + attrs_to_save

    log(" ".join(meshlabserver_cmd))
    if subprocess.call(meshlabserver_cmd) != 0:
        exit(-1)


def draw_projected_box3d(image, qs, color=(255, 0, 255), middle_color=None, bottom_color=None, thickness=2):
    """Draw 3d bounding box in image
    qs: (8,2), projected 3d points array of vertices for the 3d box in following order:
        1 -------- 0
       /|         /|
      2 -------- 3 .
      | |        | |
      . 5 -------- 4
      |/         |/
      6 -------- 7
    """
    # Ref: https://github.com/charlesq34/frustum-pointnets/blob/master/kitti/kitti_util.py
    qs = qs.astype(np.int32)
    color = mmcv.color_val(color)  # top color
    colors = colormap(rgb=False, maximum=255)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        # use LINE_AA for opencv3
        # CV_AA for opencv2?
        # bottom: blue
        i, j = k + 4, (k + 1) % 4 + 4
        if bottom_color is None:
            _bottom_color = tuple(int(_c) for _c in colors[k % len(colors)])
        else:
            _bottom_color = tuple(int(_c) for _c in mmcv.color_val(bottom_color))
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _bottom_color, thickness, cv2.LINE_AA)

        # middle: colormap
        i, j = k, k + 4
        if middle_color is None:
            _middle_color = tuple(int(_c) for _c in colors[k % len(colors)])
        else:
            _middle_color = tuple(int(_c) for _c in mmcv.color_val(middle_color))
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), _middle_color, thickness, cv2.LINE_AA)

        # top: pink/red
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)

    # # method 2
    # draw pillars in blue color-------------------
    # for i, j in zip(range(4), range(4, 8)):
    #     image = cv2.line(image, tuple(qs[i]), tuple(qs[j]), (255), thickness)

    # # draw bottom layer in red color
    # image = cv2.drawContours(image, [qs[4:]], -1, (0, 0, 255), thickness)
    # # draw top layer in red color
    # image = cv2.drawContours(image, [qs[:4]], -1, (0, 255, 0), thickness)
    # ---------------------------
    return image


def ply_vtx_color_expand(model):
    """
    model loaded by inout.load_ply()
    discription: read all vertices from a ply file and expand vertices using polygon info.
    (borrow from https://github.com/paroj/linemod_dataset/blob/master/read.py)

    -------
    add terms model["pts_expand"], model['colors_expand']
    """
    assert "pts" in model and "faces" in model, "wrong model, no pts and faces"

    pts = model["pts"]
    colors = model["colors"]
    faces = model["faces"]
    ptsExpand = []
    colorsExpand = []
    for f_i in range(len(faces)):
        # num, *ptsIdx = line.strip().split()
        # num = len(faces[i])
        num = 3  # assume triangle
        face = faces[f_i]
        for i in range(int(num)):
            for j in range(int(num)):
                if i < j:
                    pts_i = pts[int(face[i])]
                    pts_j = pts[int(face[j])]
                    pts_bias = 1 / 3.0 * (pts_j - pts_i)
                    ptsExpand.append(pts_i + pts_bias)
                    ptsExpand.append(pts_i + 2 * pts_bias)

                    colors_i = colors[int(face[i])]
                    colors_j = colors[int(face[j])]
                    colors_bias = 1 / 3.0 * (colors_j - colors_i)
                    colorsExpand.append(colors_i + colors_bias)
                    colorsExpand.append(colors_i + 2 * colors_bias)

        ptsExpand.append(1 / 3.0 * (pts[int(face[0])] + pts[int(face[1])] + pts[int(face[2])]))
        colorsExpand.append(1 / 3.0 * (colors[int(face[0])] + colors[int(face[1])] + colors[int(face[2])]))
    ptsExpand = np.array(ptsExpand, dtype=np.float)
    colorsExpand = np.array(colorsExpand, dtype=np.float)
    pts_expand = np.concatenate((pts, ptsExpand), axis=0)
    colors_expand = np.concatenate((colors, colorsExpand), axis=0)
    model["pts_expand"] = pts_expand
    model["colors_expand"] = colors_expand
    return model


def calc_uv_emb_proj(uv_model_path_or_model, R, T, K, height=480, width=640, expand=False):
    """calculate uv map emb via projection it seems to be better not to use
    expand."""
    if isinstance(uv_model_path_or_model, str):
        model = load_ply(uv_model_path_or_model)
        if expand:
            model = ply_vtx_color_expand(model)
    else:
        model = uv_model_path_or_model
    if expand:
        points = model["pts_expand"]
        uv_gb = model["colors_expand"][:, [1, 2]]
    else:
        points = model["pts"]
        uv_gb = model["colors"][:, [1, 2]]
    points_2d, z = points_to_2D(points, R, T, K)
    image_points = np.round(points_2d).astype(np.int32)
    # image_points = (points_2d + 0.5).astype(np.int32)
    uv_ProjEmb = np.zeros((height, width, 2)).astype(np.float32)
    depth = np.zeros((height, width, 1)).astype(np.float32)
    for i, (x, y) in enumerate(image_points):
        if x >= width or y >= height or x < 0 or y < 0:
            continue
        if depth[y, x, 0] == 0:
            depth[y, x, 0] = z[i]
            uv_ProjEmb[y, x] = uv_gb[i]
        elif z[i] < depth[y, x, 0]:
            depth[y, x, 0] = z[i]
            uv_ProjEmb[y, x] = uv_gb[i]
        else:
            pass
    # print("ProjEmb: min {} max {}".format(ProjEmb.min(), ProjEmb.max()))
    return uv_ProjEmb


def calc_texture_uv_emb_proj(uv_model_path_or_model, R, T, K, height=480, width=640):
    """calculate uv map emb via projection it seems to be better not to use
    expand the models are generated by blender, where texture_u, texture_v are
    provided."""
    if isinstance(uv_model_path_or_model, str):
        model = load_ply(uv_model_path_or_model)
    else:
        model = uv_model_path_or_model

    points = model["pts"]
    uv_gb = model["texture_uv"]
    points_2d, z = points_to_2D(points, R, T, K)
    image_points = np.round(points_2d).astype(np.int32)
    # image_points = (points_2d + 0.5).astype(np.int32)
    uv_ProjEmb = np.zeros((height, width, 2)).astype(np.float32)
    depth = np.zeros((height, width, 1)).astype(np.float32)
    for i, (x, y) in enumerate(image_points):
        if x >= width or y >= height or x < 0 or y < 0:
            continue
        if depth[y, x, 0] == 0:
            depth[y, x, 0] = z[i]
            uv_ProjEmb[y, x] = uv_gb[i]
        elif z[i] < depth[y, x, 0]:
            depth[y, x, 0] = z[i]
            uv_ProjEmb[y, x] = uv_gb[i]
        else:
            pass
    # print("ProjEmb: min {} max {}".format(ProjEmb.min(), ProjEmb.max()))
    return uv_ProjEmb


def test_draw_3d_bbox():
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  # LM6d
    model_dir = os.path.join(cur_dir, "../../datasets/BOP_DATASETS/lm/models")
    class_name = "ape"
    cls_idx = 1
    model_path = os.path.join(model_dir, "obj_{:06d}.ply".format(cls_idx))
    pts_3d = load_ply(model_path, vertex_scale=0.001)["pts"]
    corners_3d = get_3D_corners(pts_3d)
    image_path = os.path.join(cur_dir, "../../datasets/BOP_DATASETS/lm/test/000001/rgb/000011.png")
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    gt_dict = mmcv.load(os.path.join(cur_dir, "../../datasets/BOP_DATASETS/lm/test/000001/scene_gt.json"))
    R = np.array(gt_dict["11"][0]["cam_R_m2c"]).reshape(3, 3)
    t = np.array(gt_dict["11"][0]["cam_t_m2c"]) / 1000.0

    corners_2d, _ = points_to_2D(corners_3d, R, t, K)
    # print(pts_2d.shape)
    image_3dbb = draw_projected_box3d(image, corners_2d, thickness=1)
    cv2.imshow("image with 3d bbox", image_3dbb)
    cv2.waitKey()


if __name__ == "__main__":
    from lib.pysixd.inout import ply_vtx

    # import scipy.io as sio
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  # LM6d
    # depth_path = os.path.join(cur_dir, '../../data/LINEMOD_6D/LM6d_render_v1/data/real/01/000001-depth.png')
    # depth_gt = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED) / 1000.
    # dist_gt = depth_im_to_dist_im(depth_gt, K)
    # print(dist_gt.shape)
    # import matplotlib.pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.axis('off')
    # plt.imshow(depth_gt)
    # print('depth_gt: ', np.min(depth_gt), np.max(depth_gt))
    # plt.subplot(1, 2, 2)
    # plt.axis('off')
    # plt.imshow(dist_gt)
    # print('dist_gt: ', np.min(dist_gt), np.max(dist_gt))
    # plt.show()
    test_draw_3d_bbox()
