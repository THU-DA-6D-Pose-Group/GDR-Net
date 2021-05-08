# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with custom evaluation
funcs."""
import contextlib
import copy
import datetime
import io
import itertools
import logging
import os.path as osp
import random
import subprocess
import time
from collections import OrderedDict

import cv2
from matplotlib.pyplot import sca
import mmcv
import numpy as np
from numpy.lib.npyio import save
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, inference_context
from detectron2.layers import paste_masks_in_image
from detectron2.structures import BoxMode
from detectron2.utils.logger import create_small_table, log_every_n_seconds, log_first_n
from tabulate import tabulate
from tqdm import tqdm
from transforms3d.quaternions import quat2mat

cur_dir = osp.dirname(osp.abspath(__file__))
import ref
from core.utils.my_comm import all_gather, is_main_process, synchronize
from core.utils.pose_utils import get_closest_rot
from core.utils.my_visualizer import MyVisualizer, _RED, _GREEN, _BLUE, _GREY
from core.utils.data_utils import crop_resize_by_warp_affine
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import add, adi, arp_2d, re, te
from lib.utils.mask_utils import binary_mask_to_rle
from lib.utils.utils import dprint
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2

from .engine_utils import get_out_coor, get_out_mask

PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))


class GDRN_EvaluatorCustom(DatasetEvaluator):
    """custom evaluation of 6d pose."""

    def __init__(self, cfg, dataset_name, distributed, output_dir, train_objs=None):
        self.cfg = cfg
        self._distributed = distributed
        self._output_dir = output_dir
        mmcv.mkdir_or_exist(output_dir)

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs

        self.dataset_name = dataset_name
        self._metadata = MetadataCatalog.get(dataset_name)
        self.data_ref = ref.__dict__[self._metadata.ref_key]
        self.obj_names = self._metadata.objs
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(self._metadata.json_file)
        self.model_paths = [
            osp.join(self.data_ref.model_eval_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in self.obj_ids
        ]
        self.diameters = [self.data_ref.diameters[self.data_ref.objects.index(obj_name)] for obj_name in self.obj_names]
        self.models_3d = [
            inout.load_ply(model_path, vertex_scale=self.data_ref.vertex_scale) for model_path in self.model_paths
        ]

        if cfg.DEBUG:
            from lib.render_vispy.model3d import load_models
            from lib.render_vispy.renderer import Renderer

            self.get_gts()

            self.kpts_3d = [misc.get_bbox3d_and_center(m["pts"]) for m in self.models_3d]
            self.kpts_axis_3d = [misc.get_axis3d_and_center(m["pts"], scale=0.5) for m in self.models_3d]

            self.ren = Renderer(size=(self.data_ref.width, self.data_ref.height), cam=self.data_ref.camera_matrix)
            self.ren_models = load_models(
                model_paths=self.data_ref.model_paths,
                scale_to_meter=0.001,
                cache_dir=".cache",
                texture_paths=self.data_ref.texture_paths,
                center=False,
                use_cache=True,
            )

        self.eval_precision = cfg.VAL.get("EVAL_PRECISION", False)
        self._logger.info(f"eval precision: {self.eval_precision}")
        # eval cached
        self.use_cache = False
        if cfg.VAL.EVAL_CACHED or cfg.VAL.EVAL_PRINT_ONLY:
            self.use_cache = True
            if self.eval_precision:
                self._eval_predictions_precision()
            else:
                self._eval_predictions()  # recall
            exit(0)

    def reset(self):
        self._predictions = OrderedDict()

    def _maybe_adapt_label_cls_name(self, label):
        if self.train_objs is not None:
            cls_name = self.obj_names[label]
            if cls_name not in self.train_objs:
                return None, None  # this class was not trained
            label = self.train_objs.index(cls_name)
        else:
            cls_name = self.obj_names[label]
        return label, cls_name

    def get_fps_and_center(self, pts, num_fps=8, init_center=True):
        from core.csrc.fps.fps_utils import farthest_point_sampling

        avgx = np.average(pts[:, 0])
        avgy = np.average(pts[:, 1])
        avgz = np.average(pts[:, 2])
        fps_pts = farthest_point_sampling(pts, num_fps, init_center=init_center)
        res_pts = np.concatenate([fps_pts, np.array([[avgx, avgy, avgz]])], axis=0)
        return res_pts

    def get_img_model_points_with_coords2d(
        self, mask_pred_crop, xyz_pred_crop, coord2d_crop, im_H, im_W, extent, max_num_points=-1, mask_thr=0.5
    ):
        """
        from predicted crop_and_resized xyz, bbox top-left,
        get 2D-3D correspondences (image points, 3D model points)
        Args:
            mask_pred_crop: HW, predicted mask in roi_size
            xyz_pred_crop: HWC, predicted xyz in roi_size(eg. 64)
            coord2d_crop: HW2 coords 2d in roi size
            im_H, im_W
            extent: size of x,y,z
        """
        # [0, 1] --> [-0.5, 0.5] --> original
        xyz_pred_crop[:, :, 0] = (xyz_pred_crop[:, :, 0] - 0.5) * extent[0]
        xyz_pred_crop[:, :, 1] = (xyz_pred_crop[:, :, 1] - 0.5) * extent[1]
        xyz_pred_crop[:, :, 2] = (xyz_pred_crop[:, :, 2] - 0.5) * extent[2]

        coord2d_crop[:, :, 0] = coord2d_crop[:, :, 0] * im_W
        coord2d_crop[:, :, 1] = coord2d_crop[:, :, 1] * im_H

        sel_mask = (
            (mask_pred_crop > mask_thr)
            & (abs(xyz_pred_crop[:, :, 0]) > 0.0001 * extent[0])
            & (abs(xyz_pred_crop[:, :, 1]) > 0.0001 * extent[1])
            & (abs(xyz_pred_crop[:, :, 2]) > 0.0001 * extent[2])
        )
        model_points = xyz_pred_crop[sel_mask].reshape(-1, 3)
        image_points = coord2d_crop[sel_mask].reshape(-1, 2)

        if max_num_points >= 4:
            num_points = len(image_points)
            max_keep = min(max_num_points, num_points)
            indices = [i for i in range(num_points)]
            random.shuffle(indices)
            model_points = model_points[indices[:max_keep]]
            image_points = image_points[indices[:max_keep]]
        return image_points, model_points

    def process(self, inputs, outputs, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs:
        """
        cfg = self.cfg
        if cfg.TEST.USE_PNP:
            if cfg.TEST.PNP_TYPE.lower() == "ransac_pnp":
                return self.process_pnp_ransac(inputs, outputs, out_dict)
            elif cfg.TEST.PNP_TYPE.lower() == "net_iter_pnp":
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="iter")
            elif cfg.TEST.PNP_TYPE.lower() == "net_ransac_pnp":
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="ransac")
            elif cfg.TEST.PNP_TYPE.lower() == "net_ransac_pnp_rot":
                # use rot from PnP/RANSAC and translation from Net
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="ransac_rot")
            else:
                raise NotImplementedError

        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            start_process_time = time.perf_counter()
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1
                file_name = _input["file_name"][inst_i]

                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                K = _input["cam"][inst_i].cpu().numpy().copy()

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])

                # get pose
                rot_est = out_rots[inst_i]
                trans_est = out_transes[inst_i]

                if cfg.DEBUG:  # visualize pose
                    pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])
                    file_name = _input["file_name"][inst_i]

                    if f"{int(scene_id)}/{im_id}" != "9/499":
                        continue

                    im_ori = mmcv.imread(file_name, "color")

                    bbox = _input["bbox_est"][inst_i].cpu().numpy().copy()
                    x1, y1, x2, y2 = bbox
                    # center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    # scale = max(x2 - x1, y2 - y1) * 1.5

                    test_label = _input["roi_cls"][inst_i]
                    kpt_3d = self.kpts_3d[test_label]
                    # kpt_3d = self.kpts_axis_3d[test_label]
                    kpt_2d = misc.project_pts(kpt_3d, K, rot_est, trans_est)

                    gt_dict = self.gts[cls_name][file_name]
                    gt_rot = gt_dict["R"]
                    gt_trans = gt_dict["t"]
                    kpt_2d_gt = misc.project_pts(kpt_3d, K, gt_rot, gt_trans)

                    maxx, maxy, minx, miny = 0, 0, 1000, 1000
                    for i in range(len(kpt_2d)):
                        maxx, maxy, minx, miny = (
                            max(maxx, kpt_2d[i][0]),
                            max(maxy, kpt_2d[i][1]),
                            min(minx, kpt_2d[i][0]),
                            min(miny, kpt_2d[i][1]),
                        )
                        maxx, maxy, minx, miny = (
                            max(maxx, kpt_2d_gt[i][0]),
                            max(maxy, kpt_2d_gt[i][1]),
                            min(minx, kpt_2d_gt[i][0]),
                            min(miny, kpt_2d_gt[i][1]),
                        )
                    center = np.array([(minx + maxx) / 2, (miny + maxy) / 2])
                    scale = max(maxx - minx, maxy - miny) + 5

                    out_size = 256
                    zoomed_im = crop_resize_by_warp_affine(im_ori, center, scale, out_size)
                    save_path = osp.join(
                        cfg.OUTPUT_DIR, "vis", "{}_{}_{:06d}_no_bbox.png".format(cls_name, scene_id, im_id)
                    )
                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    mmcv.imwrite(zoomed_im, save_path)
                    # yapf: disable
                    kpt_2d = np.array(
                        [
                            [(x - (center[0] - scale / 2)) * out_size / scale,
                             (y - (center[1] - scale / 2)) * out_size / scale]
                            for [x, y] in kpt_2d
                        ]
                    )

                    kpt_2d_gt = np.array(
                        [
                            [(x - (center[0] - scale / 2)) * out_size / scale,
                             (y - (center[1] - scale / 2)) * out_size / scale]
                            for [x, y] in kpt_2d_gt
                        ]
                    )
                    # yapf: enable
                    # draw est bbox
                    linewidth = 3
                    visualizer = MyVisualizer(zoomed_im[:, :, ::-1], self._metadata)
                    # zoomed_im_vis = visualizer.draw_axis3d_and_center(
                    #     kpt_2d, linewidth=linewidth, draw_center=True
                    # )
                    # visualizer.draw_bbox3d_and_center(
                    #     kpt_2d_gt, top_color=_BLUE, bottom_color=_GREY, linewidth=linewidth, draw_center=True
                    # )
                    zoomed_im_vis = visualizer.draw_bbox3d_and_center(
                        kpt_2d, top_color=_GREEN, bottom_color=_GREY, linewidth=linewidth, draw_center=True
                    )
                    save_path = osp.join(
                        cfg.OUTPUT_DIR, "vis", "{}_{}_{:06d}_gt_est.png".format(cls_name, scene_id, im_id)
                    )
                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    zoomed_im_vis.save(save_path)
                    print("zoomed_in_vis saved to:", save_path)

                    im_vis = vis_image_bboxes_cv2(im_ori, [bbox], [f"{cls_name}_{score}"])

                    self.ren.clear()
                    self.ren.draw_background(mmcv.bgr2gray(im_ori, keepdim=True))
                    self.ren.draw_model(self.ren_models[self.data_ref.objects.index(cls_name)], pose_est)
                    ren_im, _ = self.ren.finish()
                    grid_show(
                        [ren_im[:, :, ::-1], im_vis[:, :, ::-1]],
                        [f"ren_im_{cls_name}", f"{scene_id}/{im_id}_{score}"],
                        row=1,
                        col=2,
                    )

                output["time"] += time.perf_counter() - start_process_time

                if cls_name not in self._predictions:
                    self._predictions[cls_name] = OrderedDict()

                result = {"score": score, "R": rot_est, "t": trans_est, "time": output["time"]}
                self._predictions[cls_name][file_name] = result

    def process_net_and_pnp(self, inputs, outputs, out_dict, pnp_type="iter"):
        """Initialize with network prediction (learned PnP) + iter PnP
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            pnp_type: iter | ransac (use ransac+EPnP)
            outputs:
        """
        cfg = self.cfg
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(self._cpu_device).numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(self._cpu_device).numpy()

        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            start_process_time = time.perf_counter()
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1

                bbox_center_i = _input["bbox_center"][inst_i]
                cx_i, cy_i = bbox_center_i
                scale_i = _input["scale"][inst_i]

                coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"][inst_i].item()
                im_W = _input["im_W"][inst_i].item()

                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                K = _input["cam"][inst_i].cpu().numpy().copy()

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                # scene_id = int(scene_im_id_split[0])
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])

                # get pose
                xyz_i = out_xyz[out_i].transpose(1, 2, 0)
                mask_i = np.squeeze(out_mask[out_i])

                img_points, model_points = self.get_img_model_points_with_coords2d(
                    mask_i,
                    xyz_i,
                    coord_2d_i,
                    im_H=im_H,
                    im_W=im_W,
                    extent=_input["roi_extent"][inst_i].cpu().numpy(),
                    mask_thr=cfg.MODEL.CDPN.ROT_HEAD.MASK_THR_TEST,
                )

                rot_est_net = out_rots[out_i]
                trans_est_net = out_transes[out_i]

                num_points = len(img_points)
                if num_points >= 4:
                    dist_coeffs = np.zeros(shape=[8, 1], dtype="float64")
                    points_2d = np.ascontiguousarray(img_points.astype(np.float64))
                    points_3d = np.ascontiguousarray(model_points.astype(np.float64))
                    camera_matrix = K.astype(np.float64)

                    rvec0, _ = cv2.Rodrigues(rot_est_net)

                    if pnp_type in ["ransac", "ransac_rot"]:
                        points_3d = np.expand_dims(points_3d, 0)
                        points_2d = np.expand_dims(points_2d, 0)
                        _, rvec, t_est, _ = cv2.solvePnPRansac(
                            objectPoints=points_3d,
                            imagePoints=points_2d,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            flags=cv2.SOLVEPNP_EPNP,
                            useExtrinsicGuess=True,
                            rvec=rvec0,
                            tvec=trans_est_net,
                            reprojectionError=3.0,  # default 8.0
                            iterationsCount=20,
                        )
                    else:  # iter PnP
                        # points_3d = np.expand_dims(points_3d, 0)  # uncomment for EPNP
                        # points_2d = np.expand_dims(points_2d, 0)
                        _, rvec, t_est = cv2.solvePnP(
                            objectPoints=points_3d,
                            imagePoints=points_2d,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                            # flags=cv2.SOLVEPNP_EPNP,
                            useExtrinsicGuess=True,
                            rvec=rvec0,
                            tvec=trans_est_net,
                        )
                    rot_est, _ = cv2.Rodrigues(rvec)
                    if pnp_type not in ["ransac_rot"]:
                        diff_t_est = te(t_est, trans_est_net)
                        if diff_t_est > 1:  # diff too large
                            self._logger.warning("translation error too large: {}".format(diff_t_est))
                            t_est = trans_est_net
                    else:
                        t_est = trans_est_net
                    pose_est = np.concatenate([rot_est, t_est.reshape((3, 1))], axis=-1)
                else:
                    self._logger.warning("num points: {}".format(len(img_points)))
                    pose_est_net = np.hstack([rot_est_net, trans_est_net.reshape(3, 1)])
                    pose_est = pose_est_net

                output["pose_est"] = pose_est
                output["time"] += time.perf_counter() - start_process_time

                # result
                file_name = _input["file_name"][inst_i]

                if cls_name not in self._predictions:
                    self._predictions[cls_name] = OrderedDict()

                result = {"score": score, "R": pose_est[:3, :3], "t": pose_est[:3, 3], "time": output["time"]}
                self._predictions[cls_name][file_name] = result

    def process_pnp_ransac(self, inputs, outputs, out_dict):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id", "scene_id".
            outputs:
        """
        cfg = self.cfg
        out_coor_x = out_dict["coor_x"].detach()
        out_coor_y = out_dict["coor_y"].detach()
        out_coor_z = out_dict["coor_z"].detach()
        out_xyz = get_out_coor(cfg, out_coor_x, out_coor_y, out_coor_z)
        out_xyz = out_xyz.to(self._cpu_device).numpy()

        out_mask = get_out_mask(cfg, out_dict["mask"].detach())
        out_mask = out_mask.to(self._cpu_device).numpy()

        out_trans = out_dict["trans"].detach().to(self._cpu_device).numpy()
        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            start_process_time = time.perf_counter()
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1
                bbox_center_i = _input["bbox_center"][inst_i]
                cx_i, cy_i = bbox_center_i
                scale_i = _input["scale"][inst_i]

                coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"][inst_i].item()
                im_W = _input["im_W"][inst_i].item()

                scene_im_id_split = _input["scene_im_id"][inst_i].split("/")
                K = _input["cam"][inst_i].cpu().numpy().copy()

                roi_label = _input["roi_cls"][inst_i]  # 0-based label
                score = _input["score"][inst_i]
                roi_label, cls_name = self._maybe_adapt_label_cls_name(roi_label)
                if cls_name is None:
                    continue

                # scene_id = int(scene_im_id_split[0])
                scene_id = scene_im_id_split[0]
                im_id = int(scene_im_id_split[1])

                # get pose
                if "rot" in cfg.MODEL.CDPN.TASK.lower():
                    xyz_i = out_xyz[out_i].transpose(1, 2, 0)
                    mask_i = np.squeeze(out_mask[out_i])

                    img_points, model_points = self.get_img_model_points_with_coords2d(
                        mask_i,
                        xyz_i,
                        coord_2d_i,
                        im_H=im_H,
                        im_W=im_W,
                        extent=_input["roi_extent"][inst_i].cpu().numpy(),
                        mask_thr=cfg.MODEL.CDPN.ROT_HEAD.MASK_THR_TEST,
                    )

                    pnp_method = cv2.SOLVEPNP_EPNP
                    # pnp_method = cv2.SOLVEPNP_ITERATIVE
                    num_points = len(img_points)
                    if num_points >= 4:
                        pose_est = misc.pnp_v2(
                            model_points,
                            img_points,
                            K,
                            method=pnp_method,
                            ransac=True,
                            ransac_reprojErr=3,
                            ransac_iter=100,
                            # ransac_reprojErr=1,  # more accurate but ~10ms slower
                            # ransac_iter=150,
                        )
                    else:
                        self._logger.warning("num points: {}".format(len(img_points)))
                        pose_est = -100 * np.ones((3, 4), dtype=np.float32)

                if "trans" in cfg.MODEL.CDPN.TASK.lower():
                    # compute T from trans head
                    trans_i = out_trans[out_i]

                    test_bbox_type = cfg.TEST.TEST_BBOX_TYPE
                    if test_bbox_type == "gt":
                        bbox_key = "bbox"
                    else:
                        bbox_key = f"bbox_{test_bbox_type}"
                    bbox_ori = _input[bbox_key][inst_i]
                    bw_ori = bbox_ori[2] - bbox_ori[0]
                    bh_ori = bbox_ori[3] - bbox_ori[1]

                    ox_2d = trans_i[0] * bw_ori + cx_i
                    oy_2d = trans_i[1] * bh_ori + cy_i

                    resize_ratio = _input["resize_ratio"][inst_i]  # out_res / scale
                    tz = trans_i[2] * resize_ratio

                    tx = (ox_2d - K[0, 2]) * tz / K[0, 0]
                    ty = (oy_2d - K[1, 2]) * tz / K[1, 1]

                    pred_trans = np.asarray([tx, ty, tz])
                    if "rot" in cfg.MODEL.CDPN.TASK.lower():
                        pose_est[:3, 3] = pred_trans
                    else:
                        pose_est = np.concatenate([np.eye(3), np.asarray(pred_trans.reshape(3, 1))], axis=1)

                output["pose_est"] = pose_est
                output["time"] += time.perf_counter() - start_process_time

                # result
                file_name = _input["file_name"][inst_i]

                if cls_name not in self._predictions:
                    self._predictions[cls_name] = OrderedDict()

                result = {"score": score, "R": pose_est[:3, :3], "t": pose_est[:3, 3], "time": output["time"]}
                self._predictions[cls_name][file_name] = result

    def evaluate(self):
        # bop toolkit eval in subprocess, no return value
        if self._distributed:
            synchronize()
            _predictions = all_gather(self._predictions)
            # NOTE: gather list of OrderedDict
            self._predictions = OrderedDict()
            for preds in _predictions:
                for _k, _v in preds.items():
                    self._predictions[_k] = _v
            # self._predictions = list(itertools.chain(*_predictions))
            if not is_main_process():
                return
        if self.eval_precision:
            return self._eval_predictions_precision()
        return self._eval_predictions()
        # return copy.deepcopy(self._eval_predictions())

    def get_gts(self):
        # NOTE: it is cached by dataset dicts loader
        self.gts = OrderedDict()

        dataset_dicts = DatasetCatalog.get(self.dataset_name)
        self._logger.info("load gts of {}".format(self.dataset_name))
        for im_dict in tqdm(dataset_dicts):
            file_name = im_dict["file_name"]
            annos = im_dict["annotations"]
            K = im_dict["cam"]
            for anno in annos:
                quat = anno["quat"]
                R = quat2mat(quat)
                trans = anno["trans"]
                obj_name = self._metadata.objs[anno["category_id"]]
                if obj_name not in self.gts:
                    self.gts[obj_name] = OrderedDict()
                self.gts[obj_name][file_name] = {"R": R, "t": trans, "K": K}

    def _eval_predictions(self):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        self._logger.info("Eval results ...")
        cfg = self.cfg
        method_name = f"{cfg.EXP_ID.replace('_', '-')}"
        cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_preds.pkl")
        if osp.exists(cache_path) and self.use_cache:
            self._logger.info("load cached predictions")
            self._predictions = mmcv.load(cache_path)
        else:
            if hasattr(self, "_predictions"):
                mmcv.dump(self._predictions, cache_path)
            else:
                raise RuntimeError("Please run inference first")

        recalls = OrderedDict()
        errors = OrderedDict()
        self.get_gts()

        error_names = ["ad", "re", "te", "proj"]
        metric_names = [
            "ad_2",
            "ad_5",
            "ad_10",
            "rete_2",
            "rete_5",
            "rete_10",
            "re_2",
            "re_5",
            "re_10",
            "te_2",
            "te_5",
            "te_10",
            "proj_2",
            "proj_5",
            "proj_10",
        ]

        for obj_name in self.gts:
            if obj_name not in self._predictions:
                continue
            cur_label = self.obj_names.index(obj_name)
            if obj_name not in recalls:
                recalls[obj_name] = OrderedDict()
                for metric_name in metric_names:
                    recalls[obj_name][metric_name] = []

            if obj_name not in errors:
                errors[obj_name] = OrderedDict()
                for err_name in error_names:
                    errors[obj_name][err_name] = []

            #################
            obj_gts = self.gts[obj_name]
            obj_preds = self._predictions[obj_name]
            for file_name, gt_anno in obj_gts.items():
                if file_name not in obj_preds:  # no pred found
                    for metric_name in metric_names:
                        recalls[obj_name][metric_name].append(0.0)
                    continue
                # compute each metric
                R_pred = obj_preds[file_name]["R"]
                t_pred = obj_preds[file_name]["t"]

                R_gt = gt_anno["R"]
                t_gt = gt_anno["t"]

                t_error = te(t_pred, t_gt)

                if obj_name in cfg.DATASETS.SYM_OBJS:
                    R_gt_sym = get_closest_rot(R_pred, R_gt, self._metadata.sym_infos[cur_label])
                    r_error = re(R_pred, R_gt_sym)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt_sym, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = adi(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )
                else:
                    r_error = re(R_pred, R_gt)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = add(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )

                #########
                errors[obj_name]["ad"].append(ad_error)
                errors[obj_name]["re"].append(r_error)
                errors[obj_name]["te"].append(t_error)
                errors[obj_name]["proj"].append(proj_2d_error)
                ############
                recalls[obj_name]["ad_2"].append(float(ad_error < 0.02 * self.diameters[cur_label]))
                recalls[obj_name]["ad_5"].append(float(ad_error < 0.05 * self.diameters[cur_label]))
                recalls[obj_name]["ad_10"].append(float(ad_error < 0.1 * self.diameters[cur_label]))
                # deg, cm
                recalls[obj_name]["rete_2"].append(float(r_error < 2 and t_error < 0.02))
                recalls[obj_name]["rete_5"].append(float(r_error < 5 and t_error < 0.05))
                recalls[obj_name]["rete_10"].append(float(r_error < 10 and t_error < 0.1))

                recalls[obj_name]["re_2"].append(float(r_error < 2))
                recalls[obj_name]["re_5"].append(float(r_error < 5))
                recalls[obj_name]["re_10"].append(float(r_error < 10))

                recalls[obj_name]["te_2"].append(float(t_error < 0.02))
                recalls[obj_name]["te_5"].append(float(t_error < 0.05))
                recalls[obj_name]["te_10"].append(float(t_error < 0.1))
                # px
                recalls[obj_name]["proj_2"].append(float(proj_2d_error < 2))
                recalls[obj_name]["proj_5"].append(float(proj_2d_error < 5))
                recalls[obj_name]["proj_10"].append(float(proj_2d_error < 10))

        # summarize
        obj_names = sorted(list(recalls.keys()))
        header = ["objects"] + obj_names + [f"Avg({len(obj_names)})"]
        big_tab = [header]
        for metric_name in metric_names:
            line = [metric_name]
            this_line_res = []
            for obj_name in obj_names:
                res = recalls[obj_name][metric_name]
                if len(res) > 0:
                    line.append(f"{100 * np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(0.0)
                    this_line_res.append(0.0)
            # average
            if len(obj_names) > 0:
                line.append(f"{100 * np.mean(this_line_res):.2f}")
            big_tab.append(line)

        for error_name in ["re", "te"]:
            line = [error_name]
            this_line_res = []
            for obj_name in obj_names:
                res = errors[obj_name][error_name]
                if len(res) > 0:
                    line.append(f"{np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(float("nan"))
                    this_line_res.append(float("nan"))
            # mean
            if len(obj_names) > 0:
                line.append(f"{np.mean(this_line_res):.2f}")
            big_tab.append(line)
        ### log big tag
        self._logger.info("recalls")
        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
            # floatfmt=floatfmt
        )
        self._logger.info("\n{}".format(res_log_tab_str))
        errors_cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_errors.pkl")
        recalls_cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_recalls.pkl")
        mmcv.dump(errors, errors_cache_path)
        mmcv.dump(recalls, recalls_cache_path)

        dump_tab_name = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_tab.txt")
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))

        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")

        return {}

    def _eval_predictions_precision(self):
        """NOTE: eval precision instead of recall
        Evaluate self._predictions on 6d pose.
        Return results with the metrics of the tasks.
        """
        self._logger.info("Eval results ...")
        cfg = self.cfg
        method_name = f"{cfg.EXP_ID.replace('_', '-')}"
        cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_preds.pkl")
        if osp.exists(cache_path) and self.use_cache:
            self._logger.info("load cached predictions")
            self._predictions = mmcv.load(cache_path)
        else:
            if hasattr(self, "_predictions"):
                mmcv.dump(self._predictions, cache_path)
            else:
                raise RuntimeError("Please run inference first")

        precisions = OrderedDict()
        errors = OrderedDict()
        self.get_gts()

        error_names = ["ad", "re", "te", "proj"]
        metric_names = [
            "ad_2",
            "ad_5",
            "ad_10",
            "rete_2",
            "rete_5",
            "rete_10",
            "re_2",
            "re_5",
            "re_10",
            "te_2",
            "te_5",
            "te_10",
            "proj_2",
            "proj_5",
            "proj_10",
        ]

        for obj_name in self.gts:
            if obj_name not in self._predictions:
                continue
            cur_label = self.obj_names.index(obj_name)
            if obj_name not in precisions:
                precisions[obj_name] = OrderedDict()
                for metric_name in metric_names:
                    precisions[obj_name][metric_name] = []

            if obj_name not in errors:
                errors[obj_name] = OrderedDict()
                for err_name in error_names:
                    errors[obj_name][err_name] = []

            #################
            obj_gts = self.gts[obj_name]
            obj_preds = self._predictions[obj_name]
            for file_name, gt_anno in obj_gts.items():
                # compute precision as in DPOD paper
                if file_name not in obj_preds:  # no pred found
                    # NOTE: just ignore undetected
                    continue
                # compute each metric
                R_pred = obj_preds[file_name]["R"]
                t_pred = obj_preds[file_name]["t"]

                R_gt = gt_anno["R"]
                t_gt = gt_anno["t"]

                t_error = te(t_pred, t_gt)

                if obj_name in cfg.DATASETS.SYM_OBJS:
                    R_gt_sym = get_closest_rot(R_pred, R_gt, self._metadata.sym_infos[cur_label])
                    r_error = re(R_pred, R_gt_sym)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt_sym, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = adi(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )
                else:
                    r_error = re(R_pred, R_gt)

                    proj_2d_error = arp_2d(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[cur_label]["pts"], K=gt_anno["K"]
                    )

                    ad_error = add(
                        R_pred, t_pred, R_gt, t_gt, pts=self.models_3d[self.obj_names.index(obj_name)]["pts"]
                    )

                #########
                errors[obj_name]["ad"].append(ad_error)
                errors[obj_name]["re"].append(r_error)
                errors[obj_name]["te"].append(t_error)
                errors[obj_name]["proj"].append(proj_2d_error)
                ############
                precisions[obj_name]["ad_2"].append(float(ad_error < 0.02 * self.diameters[cur_label]))
                precisions[obj_name]["ad_5"].append(float(ad_error < 0.05 * self.diameters[cur_label]))
                precisions[obj_name]["ad_10"].append(float(ad_error < 0.1 * self.diameters[cur_label]))
                # deg, cm
                precisions[obj_name]["rete_2"].append(float(r_error < 2 and t_error < 0.02))
                precisions[obj_name]["rete_5"].append(float(r_error < 5 and t_error < 0.05))
                precisions[obj_name]["rete_10"].append(float(r_error < 10 and t_error < 0.1))

                precisions[obj_name]["re_2"].append(float(r_error < 2))
                precisions[obj_name]["re_5"].append(float(r_error < 5))
                precisions[obj_name]["re_10"].append(float(r_error < 10))

                precisions[obj_name]["te_2"].append(float(t_error < 0.02))
                precisions[obj_name]["te_5"].append(float(t_error < 0.05))
                precisions[obj_name]["te_10"].append(float(t_error < 0.1))
                # px
                precisions[obj_name]["proj_2"].append(float(proj_2d_error < 2))
                precisions[obj_name]["proj_5"].append(float(proj_2d_error < 5))
                precisions[obj_name]["proj_10"].append(float(proj_2d_error < 10))

        # summarize
        obj_names = sorted(list(precisions.keys()))
        header = ["objects"] + obj_names + [f"Avg({len(obj_names)})"]
        big_tab = [header]
        for metric_name in metric_names:
            line = [metric_name]
            this_line_res = []
            for obj_name in obj_names:
                res = precisions[obj_name][metric_name]
                if len(res) > 0:
                    line.append(f"{100 * np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(0.0)
                    this_line_res.append(0.0)
            # mean
            if len(obj_names) > 0:
                line.append(f"{100 * np.mean(this_line_res):.2f}")
            big_tab.append(line)

        for error_name in ["re", "te"]:
            line = [error_name]
            this_line_res = []
            for obj_name in obj_names:
                res = errors[obj_name][error_name]
                if len(res) > 0:
                    line.append(f"{np.mean(res):.2f}")
                    this_line_res.append(np.mean(res))
                else:
                    line.append(float("nan"))
                    this_line_res.append(float("nan"))
            # mean
            if len(obj_names) > 0:
                line.append(f"{np.mean(this_line_res):.2f}")
            big_tab.append(line)
        ### log big table
        self._logger.info("precisions")
        res_log_tab_str = tabulate(
            big_tab,
            tablefmt="plain",
            # floatfmt=floatfmt
        )
        self._logger.info("\n{}".format(res_log_tab_str))
        errors_cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_errors.pkl")
        recalls_cache_path = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_precisions.pkl")
        self._logger.info(f"{errors_cache_path}")
        self._logger.info(f"{recalls_cache_path}")
        mmcv.dump(errors, errors_cache_path)
        mmcv.dump(precisions, recalls_cache_path)

        dump_tab_name = osp.join(self._output_dir, f"{method_name}_{self.dataset_name}_tab_precisions.txt")
        with open(dump_tab_name, "w") as f:
            f.write("{}\n".format(res_log_tab_str))
        if self._distributed:
            self._logger.warning("\n The current evaluation on multi-gpu is not correct, run with single-gpu instead.")
        return {}
