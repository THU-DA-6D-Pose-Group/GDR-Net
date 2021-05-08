# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Implementation of the pose error functions described in:
# Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016
# --------------------------------------------------------
# Modified
# --------------------------------------------------------
import os.path as osp
import numpy as np
from lib.utils import logger

from . import misc, visibility  # , renderer


def vsd(R_est, t_est, R_gt, t_gt, model, depth_test, K, delta, tau, cost_type="step", renderer=None):
    """Visible Surface Discrepancy.

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param depth_test: Depth image of the test scene.
    :param K: Camera matrix.
    :param delta: Tolerance used for estimation of the visibility masks. in mm, default 15
    :param tau: Misalignment tolerance. in mm, default 20
    :param cost_type: Pixel-wise matching cost:
        'tlinear' - Used in the original definition of VSD in:
            Hodan et al., On Evaluation of 6D Object Pose Estimation, ECCVW 2016
        'step' - Used for SIXD Challenge 2017. It is easier to interpret.
    :return: Error of pose_est w.r.t. pose_gt.
    typically err_vsd < 0.3, correct
    """

    # im_size = (depth_test.shape[1], depth_test.shape[0])
    height, width = depth_test.shape[:2]

    # Render depth images of the model in the estimated and the ground truth pose
    # depth_est = renderer.render(model, im_size, K, R_est, t_est, clip_near=100,
    #                             clip_far=10000, mode='depth')

    # depth_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
    #                            clip_far=10000, mode='depth')
    _, depth_est = renderer.render(0, R_est, t_est, K=K, W=width, H=height, near=100, far=10000, to_255=True)
    _, depth_gt = renderer.render(0, R_gt, t_gt, K=K, W=width, H=height, near=100, far=10000, to_255=True)

    # Convert depth images to distance images
    dist_test = misc.depth_im_to_dist_im(depth_test, K)
    dist_gt = misc.depth_im_to_dist_im(depth_gt, K)
    dist_est = misc.depth_im_to_dist_im(depth_est, K)

    # Visibility mask of the model in the ground truth pose
    visib_gt = visibility.estimate_visib_mask_gt(dist_test, dist_gt, delta)

    # Visibility mask of the model in the estimated pose
    visib_est = visibility.estimate_visib_mask_est(dist_test, dist_est, visib_gt, delta)

    # Intersection and union of the visibility masks
    visib_inter = np.logical_and(visib_gt, visib_est)
    visib_union = np.logical_or(visib_gt, visib_est)

    # Pixel-wise matching cost
    costs = np.abs(dist_gt[visib_inter] - dist_est[visib_inter])
    if cost_type == "step":
        costs = costs >= tau
    elif cost_type == "tlinear":  # Truncated linear
        costs *= 1.0 / tau
        costs[costs > 1.0] = 1.0
    else:
        logger.error("Error: Unknown pixel matching cost.")
        exit(-1)

    # costs_vis = np.ones(dist_gt.shape)
    # costs_vis[visib_inter] = costs
    # import matplotlib.pyplot as plt
    # plt.matshow(costs_vis)
    # plt.colorbar()
    # plt.show()

    # Visible Surface Discrepancy
    visib_union_count = visib_union.sum()
    visib_comp_count = visib_union_count - visib_inter.sum()
    if visib_union_count > 0:
        e = (costs.sum() + visib_comp_count) / float(visib_union_count)
    else:
        e = 1.0
    return e


def cou(R_est, t_est, R_gt, t_gt, model, im_size, K, renderer=None):
    """Complement over Union, i.e. the inverse of the Intersection over Union
    used.

    in the PASCAL VOC challenge - by Everingham et al. (IJCV 2010).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :param im_size: Test image size.
    :param K: Camera matrix.
    :return: Error of pose_est w.r.t. pose_gt.
    """

    # Render depth images of the model in the estimated and the ground truth pose
    # d_est = renderer.render(model, im_size, K, R_est, t_est, clip_near=100,
    #                         clip_far=10000, mode='depth')

    # d_gt = renderer.render(model, im_size, K, R_gt, t_gt, clip_near=100,
    #                        clip_far=10000, mode='depth')
    width, height = im_size
    _, d_est = renderer.render(0, R_est, t_est, K=K, W=width, H=height, near=100, far=10000, to_255=True)
    _, d_gt = renderer.render(0, R_gt, t_gt, K=K, W=width, H=height, near=100, far=10000, to_255=True)

    # Masks of the rendered model and their intersection and union
    mask_est = d_est > 0
    mask_gt = d_gt > 0
    inter = np.logical_and(mask_gt, mask_est)
    union = np.logical_or(mask_gt, mask_est)

    union_count = float(union.sum())
    if union_count > 0:
        e = 1.0 - inter.sum() / union_count
    else:
        e = 1.0
    return e


if __name__ == "__main__":
    cur_dir = osp.dirname(osp.abspath(__file__))
