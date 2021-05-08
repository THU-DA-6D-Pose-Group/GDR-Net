# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

"""Implementation of the pose error functions described in:

Hodan, Michel et al., "BOP: Benchmark for 6D Object Pose Estimation",
ECCV'18 Hodan et al., "On Evaluation of 6D Object Pose Estimation",
ECCVW'16
"""
# Modified
# --------------------------------------------------------
import os
import math
import numpy as np
from scipy import spatial
from scipy.linalg import logm
import numpy.linalg as LA
from lib.utils import logger
from lib.pysixd import misc, visibility


def vsd(
    R_est,
    t_est,
    R_gt,
    t_gt,
    depth_test,
    K,
    delta,
    taus,
    normalized_by_diameter,
    diameter,
    renderer,
    obj_id,
    cost_type="step",
    renderer_type="python",
):
    """Visible Surface Discrepancy -- by Hodan, Michel et al. (ECCV 2018).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param depth_test: hxw ndarray with the test depth image.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks.
    :param taus: A list of misalignment tolerance values.
    :param normalized_by_diameter: Whether to normalize the pixel-wise distances
        by the object diameter.
    :param diameter: Object diameter.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :param cost_type: Type of the pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW'16
        'step' - Used for SIXD Challenge 2017 onwards.
    :return: List of calculated errors (one for each misalignment tolerance).
    """
    # Render depth images of the model in the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if renderer_type in ["cpp", "python"]:
        # import pdb; pdb.set_trace()
        depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
        depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]
    elif renderer_type == "egl":
        import torch

        pc_cam_tensor = torch.cuda.FloatTensor(renderer.height, renderer.width, 4).detach()
        poses_gt = [np.hstack([R_gt, t_gt.reshape((3, 1))])]
        poses_est = [np.hstack([R_est, t_est.reshape((3, 1))])]
        renderer.render([obj_id - 1], poses=poses_est, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_est = pc_cam_tensor[:, :, 2].cpu().numpy()
        # print(depth_est.min(), depth_est.max(), depth_est.mean())
        renderer.render([obj_id - 1], poses=poses_gt, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_gt = pc_cam_tensor[:, :, 2].cpu().numpy()
        # print(depth_gt.min(), depth_gt.max(), depth_gt.mean())
    elif renderer_type == "aae":
        _, depth_est = renderer.render(obj_id - 1, R_est, t_est, K=K)
        _, depth_gt = renderer.render(obj_id - 1, R_gt, t_gt, K=K)
    else:
        raise ValueError("renderer type: {} is not supported".format(renderer_type))
    # import pdb; pdb.set_trace();

    # Convert depth images to distance images.
    dist_test = misc.depth_im_to_dist_im_fast(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im_fast(depth_est, K)

    # Visibility mask of the model in the ground-truth pose.
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta, visib_mode="bop19")

    # Visibility mask of the model in the estimated pose.
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta, visib_mode="bop19")

    # Intersection and union of the visibility masks.
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()

    # Pixel-wise distances.
    dists = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])

    # Normalization of pixel-wise distances by object diameter.
    if normalized_by_diameter:
        dists /= diameter

    # Calculate VSD for each provided value of the misalignment tolerance.
    if visib_union_count == 0:
        errors = [1.0] * len(taus)
    else:
        errors = []
        for tau in taus:

            # Pixel-wise matching cost.
            if cost_type == "step":
                costs = dists >= tau
            elif cost_type == "tlinear":  # Truncated linear function.
                costs = dists / tau
                costs[costs > 1.0] = 1.0
            else:
                raise ValueError("Unknown pixel matching cost.")

            e = (np.sum(costs) + visib_comp_count) / float(visib_union_count)
            errors.append(e)

    return errors


def mssd(R_est, t_est, R_gt, t_gt, pts, syms):
    """Maximum Symmetry-Aware Surface Distance (MSSD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    pts_est = misc.transform_pts_Rt(pts, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_sym = misc.transform_pts_Rt(pts, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(pts_est - pts_gt_sym, axis=1).max())
    return min(es)


def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms):
    """Maximum Symmetry-Aware Projection Distance (MSPD).

    See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with the intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :param syms: Set of symmetry transformations, each given by a dictionary with:
      - 'R': 3x3 ndarray with the rotation matrix.
      - 't': 3x1 ndarray with the translation vector.
    :return: The calculated error.
    """
    proj_est = misc.project_pts(pts, K, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = misc.project_pts(pts, K, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())
    return min(es)


##########################
# symmetry-aware re/te/proj
def re_sym(R_est, R_gt, syms):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        rotation_diff = np.dot(R_est, R_gt_sym.T)
        trace = np.trace(rotation_diff)
        trace = trace if trace <= 3 else 3
        # Avoid invalid values due to numerical errors
        error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
        rd_deg = np.rad2deg(np.arccos(error_cos))
        es.append(rd_deg)

    return min(es)


def te_sym(t_est, t_gt, R_gt, syms):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    assert t_est.size == t_gt.size == 3
    es = []
    for sym in syms:
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        error = np.linalg.norm(t_gt_sym - t_est)
        es.append(error)
    return min(es)


def arp_2d_sym(R_est, t_est, R_gt, t_gt, pts, K, syms):
    """# NOTE: the same as proj average re-projection error in 2d."""
    pts_est_2d = transform_pts_Rt_2d(pts, R_est, t_est, K)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        pts_gt_2d_sym = transform_pts_Rt_2d(pts, R_gt_sym, t_gt_sym, K)
        e = np.linalg.norm(pts_est_2d - pts_gt_2d_sym, axis=1).mean()
        es.append(e)
    return min(es)


def proj_sym(R_est, t_est, R_gt, t_gt, K, pts, syms):
    """Average distance of projections of object model vertices [px]

    - by Brachmann et al. (CVPR'16).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    proj_est = misc.project_pts(pts, K, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym["R"])
        t_gt_sym = R_gt.dot(sym["t"]) + t_gt
        proj_gt_sym = misc.project_pts(pts, K, R_gt_sym, t_gt_sym)
        e = np.linalg.norm(proj_est - proj_gt_sym, axis=1).mean()
        es.append(e)
    return min(es)


################################################################


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def transform_pts_Rt_2d(pts, R, t, K):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :param K: 3x3 intrinsic matrix
    :return: nx2 ndarray with transformed 2D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))  # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    return pts_2d


def add(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with no indistinguishable.

    views - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable
    views.

    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def calc_rt_dist_q(Rq_src, Rq_tgt, T_src, T_tgt):

    rd_rad = np.arccos(np.inner(Rq_src, Rq_tgt) ** 2 * 2 - 1)
    rd_deg = rd_rad / np.pi * 180
    td = LA.norm(T_tgt - T_src)
    return rd_deg, td


def calc_rt_dist_m(pose_src, pose_tgt, errtol=2.5e-6):
    R_src = pose_src[:, :3]
    T_src = pose_src[:, 3]
    R_tgt = pose_tgt[:, :3]
    T_tgt = pose_tgt[:, 3]
    # # method 1
    # temp, errest = logm(np.dot(np.transpose(R_src), R_tgt), disp=False)
    # if not np.isfinite(errest) or errest >= errtol:
    #     logger.info("logm result may be inaccurate, approximate err ={}".format(errest))

    # rd_rad = LA.norm(temp, "fro") / np.sqrt(2)
    # # rd_deg = rd_rad / pi * 180
    # rd_deg = np.rad2deg(rd_rad)

    # method 2
    rotation_diff = np.dot(R_src, R_tgt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    td = LA.norm(T_tgt - T_src)

    return rd_deg, td


# def re_old(R_est, R_gt):
#     """
#     Rotational Error.

#     :param R_est: Rotational element of the estimated pose (3x1 vector).
#     :param R_gt: Rotational element of the ground truth pose (3x1 vector).
#     :return: Error of t_est w.r.t. t_gt.
#     """
#     assert R_est.shape == R_gt.shape == (3, 3)
#     error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
#     error_cos = min(1.0, max(-1.0, error_cos))  # Avoid invalid values due to numerical errors
#     error = math.acos(error_cos)
#     error = 180.0 * error / np.pi  # [rad] -> [deg]
#     return error

# def re(R_est, R_gt, errtol=2.5e-6):
#     assert R_est.shape == R_gt.shape == (3, 3)
#     temp, errest = logm(np.dot(np.transpose(R_est), R_gt), disp=False)
#     if not np.isfinite(errest) or errest >= errtol:
#         logger.info("logm result may be inaccurate, approximate err ={}".format(errest))
#     rd_rad = LA.norm(temp, "fro") / np.sqrt(2)
#     rd_deg = rd_rad / np.pi * 180
#     return rd_deg


def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def re_q(q1, q2):
    """compute r_err from quaternions."""
    normed_q1 = q1 / np.linalg.norm(q1)
    normed_q2 = q2 / np.linalg.norm(q2)
    return np.arccos(max(min(1, 2 * np.power(np.dot(normed_q1, normed_q2), 2) - 1), -1)) * 180.0 / np.pi


def te(t_est, t_gt):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error


def arp_2d(R_est, t_est, R_gt, t_gt, pts, K):
    """# NOTE: the same as proj average re-projection error in 2d."""
    pts_est_2d = transform_pts_Rt_2d(pts, R_est, t_est, K)
    pts_gt_2d = transform_pts_Rt_2d(pts, R_gt, t_gt, K)
    e = np.linalg.norm(pts_est_2d - pts_gt_2d, axis=1).mean()
    return e


def proj(R_est, t_est, R_gt, t_gt, K, pts):
    """Average distance of projections of object model vertices [px]

    - by Brachmann et al. (CVPR'16).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    proj_est = misc.project_pts(pts, K, R_est, t_est)
    proj_gt = misc.project_pts(pts, K, R_gt, t_gt)
    e = np.linalg.norm(proj_est - proj_gt, axis=1).mean()
    return e


def cou_mask(mask_est, mask_gt):
    """Complement over Union of 2D binary masks.

    :param mask_est: hxw ndarray with the estimated mask.
    :param mask_gt: hxw ndarray with the ground-truth mask.
    :return: The calculated error.
    """
    mask_est_bool = mask_est.astype(np.bool)
    mask_gt_bool = mask_gt.astype(np.bool)

    inter = np.logical_and(mask_gt_bool, mask_est_bool)
    union = np.logical_or(mask_gt_bool, mask_est_bool)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def cus(R_est, t_est, R_gt, t_gt, K, renderer, obj_id, renderer_type="python"):
    """Complement over Union of projected 2D masks.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :return: The calculated error.
    """
    # Render depth images of the model at the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if renderer_type in ["cpp", "python"]:
        depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
        depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]
    elif renderer_type == "egl":
        import torch

        pc_cam_tensor = torch.cuda.FloatTensor(renderer.height, renderer.width, 4).detach()
        poses_gt = [np.hstack([R_gt, t_gt.reshape((3, 1))])]
        poses_est = [np.hstack([R_est, t_est.reshape((3, 1))])]
        renderer.render([obj_id - 1], poses=poses_est, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_est = pc_cam_tensor[:, :, 2].cpu().numpy()
        renderer.render([obj_id - 1], poses=poses_gt, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_gt = pc_cam_tensor[:, :, 2].cpu().numpy()
    elif renderer_type == "aae":
        _, depth_est = renderer.render(obj_id - 1, R_est, t_est, K=K)
        _, depth_gt = renderer.render(obj_id - 1, R_gt, t_gt, K=K)
    else:
        raise ValueError("renderer type: {} is not supported".format(renderer_type))

    # Masks of the rendered model and their intersection and union.
    mask_est = depth_est > 0
    mask_gt = depth_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


def cou_bb(bb_est, bb_gt):
    """Complement over Union of 2D bounding boxes.

    :param bb_est: The estimated bounding box (x1, y1, w1, h1).
    :param bb_gt: The ground-truth bounding box (x2, y2, w2, h2).
    :return: The calculated error.
    """
    e = 1.0 - misc.iou(bb_est, bb_gt)
    return e


def cou_bb_proj(R_est, t_est, R_gt, t_gt, K, renderer, obj_id, renderer_type="python"):
    """Complement over Union of projected 2D bounding boxes.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param K: 3x3 ndarray with an intrinsic camera matrix.
    :param renderer: Instance of the Renderer class (see renderer.py).
    :param obj_id: Object identifier.
    :return: The calculated error.
    """
    # Render depth images of the model at the estimated and the ground-truth pose.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if renderer_type in ["cpp", "python"]:
        depth_est = renderer.render_object(obj_id, R_est, t_est, fx, fy, cx, cy)["depth"]
        depth_gt = renderer.render_object(obj_id, R_gt, t_gt, fx, fy, cx, cy)["depth"]
    elif renderer_type == "egl":
        import torch

        pc_cam_tensor = torch.cuda.FloatTensor(renderer.height, renderer.width, 4).detach()
        poses_gt = [np.hstack([R_gt, t_gt.reshape((3, 1))])]
        poses_est = [np.hstack([R_est, t_est.reshape((3, 1))])]
        renderer.render([obj_id - 1], poses=poses_est, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_est = pc_cam_tensor[:, :, 2].cpu().numpy()
        renderer.render([obj_id - 1], poses=poses_gt, K=K, pc_cam_tensor=pc_cam_tensor)
        depth_gt = pc_cam_tensor[:, :, 2].cpu().numpy()
    elif renderer_type == "aae":
        _, depth_est = renderer.render(obj_id - 1, R_est, t_est, K=K)
        _, depth_gt = renderer.render(obj_id - 1, R_gt, t_gt, K=K)
    else:
        raise ValueError("renderer type: {} is not supported".format(renderer_type))

    # Masks of the rendered model and their intersection and union
    mask_est = depth_est > 0
    mask_gt = depth_gt > 0

    ys_est, xs_est = mask_est.nonzero()
    bb_est = misc.calc_2d_bbox(xs_est, ys_est, im_size=None, clip=False)

    ys_gt, xs_gt = mask_gt.nonzero()
    bb_gt = misc.calc_2d_bbox(xs_gt, ys_gt, im_size=None, clip=False)

    e = 1.0 - misc.iou(bb_est, bb_gt)
    return e


def load_object_points(point_path):
    print(point_path)
    assert os.path.exists(point_path), "Path does not exist: {}".format(point_path)
    points = np.loadtxt(point_path)
    return points


def load_object_extents(extent_path, num_classes):
    assert os.path.exists(extent_path), "Path does not exist: {}".format(extent_path)
    extents = np.zeros((num_classes, 3), dtype=np.float32)
    extents[1:, :] = np.loadtxt(extent_path)  # assume class 0 is '__background__'
    return extents


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    lov_path = os.path.join(cur_dir, "../../data/LOV")
    point_file = os.path.join(lov_path, "models", "003_cracker_box", "points.xyz")
    points = load_object_points(point_file)
    print(points.min(0))
    print(points.max(0))
    print(points.max(0) - points.min(0))

    extent_file = os.path.join(lov_path, "extents.txt")
    extents = load_object_extents(extent_file, num_classes=22)  # 21 + 1
