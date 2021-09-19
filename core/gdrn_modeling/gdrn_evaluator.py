# -*- coding: utf-8 -*-
"""inference on dataset; save results; evaluate with bop_toolkit (if gt is
available)"""
import contextlib
import copy
import datetime
import itertools
import logging
import os.path as osp
import random
import time
from collections import OrderedDict

import cv2
import mmcv
import numpy as np
import ref
import torch
from core.utils.my_comm import all_gather, get_world_size, is_main_process, synchronize
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator, DatasetEvaluators, inference_context
from detectron2.utils.logger import log_every_n_seconds
from lib.pysixd import inout, misc
from lib.pysixd.pose_error import te
from lib.utils.mask_utils import binary_mask_to_rle
from lib.utils.utils import dprint
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2
from torch.cuda.amp import autocast
from transforms3d.quaternions import quat2mat

from .engine_utils import batch_data, get_out_coor, get_out_mask
from .test_utils import _to_str, eval_cached_results, save_and_eval_results, to_list


class GDRN_Evaluator(DatasetEvaluator):
    """use bop toolkit to evaluate."""

    def __init__(self, cfg, dataset_name, distributed, output_dir, train_objs=None):
        self.cfg = cfg
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        # if test objs are just a subset of train objs
        self.train_objs = train_objs

        self._metadata = MetadataCatalog.get(dataset_name)
        self.data_ref = ref.__dict__[self._metadata.ref_key]
        self.obj_names = self._metadata.objs
        self.obj_ids = [self.data_ref.obj2id[obj_name] for obj_name in self.obj_names]
        # with contextlib.redirect_stdout(io.StringIO()):
        #     self._coco_api = COCO(self._metadata.json_file)
        self.model_paths = [
            osp.join(self.data_ref.model_dir, "obj_{:06d}.ply".format(obj_id)) for obj_id in self.obj_ids
        ]
        self.models_3d = [
            inout.load_ply(model_path, vertex_scale=self.data_ref.vertex_scale) for model_path in self.model_paths
        ]

        # eval cached
        if cfg.VAL.EVAL_CACHED or cfg.VAL.EVAL_PRINT_ONLY:
            eval_cached_results(self.cfg, self._output_dir, obj_ids=self.obj_ids)

    def reset(self):
        self._predictions = []

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
            outputs: stores time
        """
        cfg = self.cfg
        if cfg.TEST.USE_PNP:
            if cfg.TEST.PNP_TYPE.lower() == "ransac_pnp":
                return self.process_pnp_ransac(inputs, outputs, out_dict)
            elif cfg.TEST.PNP_TYPE.lower() == "net_iter_pnp":
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="iter")
            elif cfg.TEST.PNP_TYPE.lower() == "net_ransac_pnp":
                return self.process_net_and_pnp(inputs, outputs, out_dict, pnp_type="ransac")
            else:
                raise NotImplementedError

        out_rots = out_dict["rot"].detach().to(self._cpu_device).numpy()
        out_transes = out_dict["trans"].detach().to(self._cpu_device).numpy()

        out_i = -1
        for i, (_input, output) in enumerate(zip(inputs, outputs)):
            json_results = []
            start_process_time = time.perf_counter()
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1  # the index in the flattened output
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
                obj_id = self.data_ref.obj2id[cls_name]

                # get pose
                rot_est = out_rots[out_i]
                trans_est = out_transes[out_i]
                pose_est = np.hstack([rot_est, trans_est.reshape(3, 1)])

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )

            output["time"] += time.perf_counter() - start_process_time
            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

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
            json_results = []
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1
                bbox_center_i = _input["bbox_center"][inst_i]
                cx_i, cy_i = bbox_center_i
                scale_i = _input["scale"][inst_i]

                coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"][inst_i]
                im_W = _input["im_W"][inst_i]

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
                obj_id = self.data_ref.obj2id[cls_name]

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

                    if pnp_type == "ransac":
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
                    diff_t_est = te(t_est, trans_est_net)
                    if diff_t_est > 1:  # diff too large
                        self._logger.warning("translation error too large: {}".format(diff_t_est))
                        t_est = trans_est_net
                    pose_est = np.concatenate([rot_est, t_est.reshape((3, 1))], axis=-1)
                else:
                    self._logger.warning("num points: {}".format(len(img_points)))
                    pose_est_net = np.hstack([rot_est_net, trans_est_net.reshape(3, 1)])
                    pose_est = pose_est_net

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )

            output["time"] += time.perf_counter() - start_process_time

            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

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
            json_results = []
            for inst_i in range(len(_input["roi_img"])):
                out_i += 1
                bbox_center_i = _input["bbox_center"][inst_i]
                cx_i, cy_i = bbox_center_i
                scale_i = _input["scale"][inst_i]

                coord_2d_i = _input["roi_coord_2d"][inst_i].cpu().numpy().transpose(1, 2, 0)  # CHW->HWC
                im_H = _input["im_H"][inst_i]
                im_W = _input["im_W"][inst_i]

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
                obj_id = self.data_ref.obj2id[cls_name]

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

                json_results.extend(
                    self.pose_prediction_to_json(
                        pose_est, scene_id, im_id, obj_id=obj_id, score=score, pose_time=output["time"], K=K
                    )
                )

            output["time"] += time.perf_counter() - start_process_time
            # process time for this image
            for item in json_results:
                item["time"] = output["time"]
            self._predictions.extend(json_results)

    def evaluate(self):
        # bop toolkit eval in subprocess, no return value
        if self._distributed:
            synchronize()
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

        return self._eval_predictions()
        # return copy.deepcopy(self._eval_predictions())

    def _eval_predictions(self):
        """Evaluate self._predictions on 6d pose.

        Return results with the metrics of the tasks.
        """
        self._logger.info("Eval results with BOP toolkit ...")
        results_all = {"iter0": self._predictions}
        save_and_eval_results(self.cfg, results_all, self._output_dir, obj_ids=self.obj_ids)
        return {}

    def pose_from_upnp(self, mean_pts2d, covar, points_3d, K):
        import scipy
        from core.csrc.pvnet_ext_utils.extend_utils import uncertainty_pnp

        cov_invs = []
        for vi in range(covar.shape[0]):
            if covar[vi, 0, 0] < 1e-6 or np.sum(np.isnan(covar)[vi]) > 0:
                cov_invs.append(np.zeros([2, 2]).astype(np.float32))
                continue

            cov_inv = np.linalg.inv(scipy.linalg.sqrtm(covar[vi]))
            cov_invs.append(cov_inv)
        cov_invs = np.asarray(cov_invs)  # pn,2,2
        weights = cov_invs.reshape([-1, 4])
        weights = weights[:, (0, 1, 3)]
        pose_pred = uncertainty_pnp(mean_pts2d, weights, points_3d, K)
        return pose_pred

    def pose_from_upnp_v2(self, mean_pts2d, covar, points_3d, K):
        from core.csrc.pvnet_ext_utils.extend_utils import uncertainty_pnp_v2

        pose_pred = uncertainty_pnp_v2(mean_pts2d, covar, points_3d, K)
        return pose_pred

    def pose_prediction_to_json(self, pose_est, scene_id, im_id, obj_id, score=None, pose_time=-1, K=None):
        """
        Args:
            pose_est:
            scene_id (str): the scene id
            img_id (str): the image id
            label: used to get obj_id
            score: confidence
            pose_time:

        Returns:
            list[dict]: the results in BOP evaluation format
        """
        cfg = self.cfg
        results = []
        if score is None:  # TODO: add score key in test bbox json file
            score = 1.0
        rot = pose_est[:3, :3]
        trans = pose_est[:3, 3]
        # for standard bop datasets, scene_id and im_id can be obtained from file_name
        result = {
            "scene_id": scene_id,  # if not available, assume 0
            "im_id": im_id,
            "obj_id": obj_id,  # the obj_id in bop datasets
            "score": score,
            "R": to_list(rot),
            "t": to_list(1000 * trans),  # mm
            "time": pose_time,
        }
        results.append(result)
        return results


def gdrn_inference_on_dataset(cfg, model, data_loader, evaluator, amp_test=False):
    """Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.forward` accurately. The model
    will be used in eval mode.

    Args:
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator (DatasetEvaluator): the evaluator to run. Use `None` if you only want
            to benchmark, but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    total_process_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
                total_process_time = 0

            start_compute_time = time.perf_counter()
            #############################
            # process input
            batch = batch_data(cfg, inputs, phase="test")
            if evaluator.train_objs is not None:
                roi_labels = batch["roi_cls"].cpu().numpy().tolist()
                obj_names = [evaluator.obj_names[_l] for _l in roi_labels]
                if all(_obj not in evaluator.train_objs for _obj in obj_names):
                    continue

            with autocast(enabled=amp_test):
                out_dict = model(
                    batch["roi_img"],
                    roi_classes=batch["roi_cls"],
                    roi_cams=batch["roi_cam"],
                    roi_whs=batch["roi_wh"],
                    roi_centers=batch["roi_center"],
                    resize_ratios=batch["resize_ratio"],
                    roi_coord_2d=batch.get("roi_coord_2d", None),
                    roi_extents=batch.get("roi_extent", None),
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time
            # NOTE: added
            # TODO: add detection time here
            outputs = [{} for _ in range(len(inputs))]
            for _i in range(len(outputs)):
                outputs[_i]["time"] = cur_compute_time

            start_process_time = time.perf_counter()
            evaluator.process(inputs, outputs, out_dict)  # RANSAC/PnP
            cur_process_time = time.perf_counter() - start_process_time
            total_process_time += cur_process_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO, f"Inference done {idx+1}/{total}. {seconds_per_img:.4f} s / img. ETA={str(eta)}", n=5
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        f"Total inference time: {total_time_str} "
        f"({total_time / (total - num_warmup):.6f} s / img per device, on {num_devices} devices)"
    )
    # pure forward time
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )
    # post_process time
    total_process_time_str = str(datetime.timedelta(seconds=int(total_process_time)))
    logger.info(
        "Total inference post process time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_process_time_str, total_process_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()  # results is always None
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def save_result_of_dataset(cfg, model, data_loader, output_dir, dataset_name):
    """
    Run model (in eval mode) on the data_loader and save predictions
    Args:
        cfg: config
        model (nn.Module): a module which accepts an object from
            `data_loader` and returns some outputs. It will be temporarily set to `eval` mode.

            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
    Returns:
        The return value of `evaluator.evaluate()`
    """
    cpu_device = torch.device("cpu")
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} images".format(len(data_loader)))

    # NOTE: dataset name should be the same as TRAIN to get the correct meta
    _metadata = MetadataCatalog.get(dataset_name)
    data_ref = ref.__dict__[_metadata.ref_key]
    obj_names = _metadata.objs
    obj_ids = [data_ref.obj2id[obj_name] for obj_name in obj_names]

    result_name = "results.pkl"
    mmcv.mkdir_or_exist(output_dir)
    result_path = osp.join(output_dir, result_name)

    total = len(data_loader)  # inference data loader must have a fixed length
    results = OrderedDict()
    VIS = False

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            if VIS:
                images_ori = [_input["image"].clone() for _input in inputs]
            start_compute_time = time.perf_counter()
            outputs = model(inputs)  # NOTE: do model inference
            torch.cuda.synchronize()
            cur_compute_time = time.perf_counter() - start_compute_time
            total_compute_time += cur_compute_time

            # NOTE: process results
            for i in range(len(inputs)):
                _input = inputs[i]
                output = outputs[i]
                cur_results = {}
                instances = output["instances"]
                HAS_MASK = False
                if instances.has("pred_masks"):
                    HAS_MASK = True
                    pred_masks = instances.pred_masks  # (#objs, imH, imW)
                    pred_masks = pred_masks.detach().cpu().numpy()
                    # NOTE: time comsuming step
                    rles = [binary_mask_to_rle(pred_masks[_k]) for _k in range(len(pred_masks))]

                instances = instances.to(cpu_device)
                boxes = instances.pred_boxes.tensor.clone().detach().cpu().numpy()  # xyxy

                scores = instances.scores.tolist()
                labels = instances.pred_classes.detach().cpu().numpy()

                obj_ids = [data_ref.obj2id[obj_names[int(label)]] for label in labels]
                ego_quats = instances.pred_ego_quats.detach().cpu().numpy()
                ego_rots = [quat2mat(ego_quats[k]) for k in range(len(ego_quats))]
                transes = instances.pred_transes.detach().cpu().numpy()

                cur_results = {
                    "time": cur_compute_time / len(inputs),
                    "obj_ids": obj_ids,
                    "scores": scores,
                    "boxes": boxes,  # xyxy
                    "Rs": ego_rots,
                    "ts": transes,  # m
                }
                if HAS_MASK:
                    cur_results["masks"] = rles

                if VIS:
                    import cv2
                    from lib.vis_utils.image import vis_image_mask_bbox_cv2

                    image = (images_ori[i].detach().cpu().numpy().transpose(1, 2, 0) + 0.5).astype("uint8")
                    img_vis = vis_image_mask_bbox_cv2(
                        image, pred_masks, boxes, labels=[obj_names[int(label)] for label in labels]
                    )
                    cv2.imshow("img", img_vis.astype("uint8"))
                    cv2.waitKey()
                results[_input["scene_im_id"]] = cur_results

            if (idx + 1) % logging_interval == 0:
                duration = time.perf_counter() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(seconds=int(seconds_per_img * (total - num_warmup) - duration))
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(idx + 1, total, seconds_per_img, str(eta))
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.perf_counter() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    mmcv.dump(results, result_path)
    logger.info("Results saved to {}".format(result_path))
