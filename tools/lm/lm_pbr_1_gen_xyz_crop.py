from __future__ import division, print_function

import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys

import mmcv
import numpy as np
from tqdm import tqdm

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../..")
sys.path.insert(0, PROJ_ROOT)
from lib.meshrenderer.meshrenderer_phong import Renderer
from lib.vis_utils.image import grid_show
from lib.pysixd import misc
from lib.utils.mask_utils import mask2bbox_xyxy


idx2class = {
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

class2idx = {_name: _id for _id, _name in idx2class.items()}

classes = idx2class.values()
classes = sorted(classes)

# DEPTH_FACTOR = 1000.
IM_H = 480
IM_W = 640
near = 0.01
far = 6.5

data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train_pbr"))

cls_indexes = sorted(idx2class.keys())
cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
model_paths = [osp.join(lm_model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]
texture_paths = None

scenes = [i for i in range(0, 49 + 1)]
xyz_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train_pbr/xyz_crop"))

K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])


def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(show_emb)
    return show_emb


class XyzGen(object):
    def __init__(self, split="train", scene="all"):
        if split == "train":
            scene_ids = scenes
            data_root = data_dir
        else:
            raise ValueError(f"split {split} error")

        if scene == "all":
            sel_scene_ids = scene_ids
        else:
            assert int(scene) in scene_ids, f"{scene} not in {scene_ids}"
            sel_scene_ids = [int(scene)]
        print("split: ", split, "selected scene ids: ", sel_scene_ids)
        self.split = split
        self.scene = scene
        self.sel_scene_ids = sel_scene_ids
        self.data_root = data_root
        self.renderer = None

    def get_renderer(self):
        if self.renderer is None:
            self.renderer = Renderer(
                model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache"), vertex_scale=0.001
            )
        return self.renderer

    def main(self):
        split = self.split
        scene = self.scene  # "all" or a single scene
        sel_scene_ids = self.sel_scene_ids
        data_root = self.data_root

        for scene_id in tqdm(sel_scene_ids, postfix=f"{split}_{scene}"):
            print("split: {} scene: {}".format(split, scene_id))
            scene_root = osp.join(data_root, f"{scene_id:06d}")

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            # gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            # cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                int_im_id = int(str_im_id)

                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in idx2class:
                        continue

                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    # pose = np.hstack([R, t.reshape(3, 1)])

                    save_path = osp.join(
                        xyz_root,
                        f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl",
                    )
                    # if osp.exists(save_path) and osp.getsize(save_path) > 0:
                    #     continue

                    render_obj_id = cls_indexes.index(obj_id)  # 0-based
                    bgr_gl, depth_gl = self.get_renderer().render(render_obj_id, IM_W, IM_H, K, R, t, near, far)
                    mask = (depth_gl > 0).astype("uint8")

                    if mask.sum() == 0:  # NOTE: this should be ignored at training phase
                        print(
                            f"not visible, split {split} scene {scene_id}, im {int_im_id} obj {idx2class[obj_id]} {obj_id}"
                        )
                        print(f"{save_path}")
                        xyz_info = {
                            "xyz_crop": np.zeros((IM_H, IM_W, 3), dtype=np.float16),
                            "xyxy": [0, 0, IM_W - 1, IM_H - 1],
                        }
                        if VIS:
                            im_path = osp.join(
                                data_root,
                                f"{scene_id:06d}/rgb/{int_im_id:06d}.jpg",
                            )
                            im = mmcv.imread(im_path)

                            mask_path = osp.join(
                                data_root,
                                f"{scene_id:06d}/mask/{int_im_id:06d}_{anno_i:06d}.png",
                            )
                            mask_visib_path = osp.join(
                                data_root,
                                f"{scene_id:06d}/mask_visib/{int_im_id:06d}_{anno_i:06d}.png",
                            )
                            mask_gt = mmcv.imread(mask_path, "unchanged")
                            mask_visib_gt = mmcv.imread(mask_visib_path, "unchanged")

                            show_ims = [
                                bgr_gl[:, :, [2, 1, 0]],
                                im[:, :, [2, 1, 0]],
                                mask_gt,
                                mask_visib_gt,
                            ]
                            show_titles = [
                                "bgr_gl",
                                "im",
                                "mask_gt",
                                "mask_visib_gt",
                            ]
                            grid_show(show_ims, show_titles, row=2, col=2)
                            raise RuntimeError(f"split {split} scene {scene_id}, im {int_im_id}")
                    else:
                        x1, y1, x2, y2 = mask2bbox_xyxy(mask)
                        xyz_np = misc.calc_xyz_bp_fast(depth_gl, R, t, K)
                        xyz_crop = xyz_np[y1 : y2 + 1, x1 : x2 + 1]
                        xyz_info = {
                            "xyz_crop": xyz_crop.astype("float16"),  # save disk space w/o performance drop
                            "xyxy": [x1, y1, x2, y2],
                        }

                        if VIS:
                            print(f"xyz_crop min {xyz_crop.min()} max {xyz_crop.max()}")
                            show_ims = [
                                bgr_gl[:, :, [2, 1, 0]],
                                get_emb_show(xyz_np),
                                get_emb_show(xyz_crop),
                            ]
                            show_titles = ["bgr_gl", "xyz", "xyz_crop"]
                            grid_show(show_ims, show_titles, row=1, col=3)

                    if not args.no_save:
                        mmcv.mkdir_or_exist(osp.dirname(save_path))
                        mmcv.dump(xyz_info, save_path)
        if self.renderer is not None:
            self.renderer.close()


if __name__ == "__main__":
    import argparse
    import time

    import setproctitle

    parser = argparse.ArgumentParser(description="gen lm train_pbr xyz")
    parser.add_argument("--split", type=str, default="train", help="split")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    parser.add_argument("--vis", default=False, action="store_true", help="vis")
    parser.add_argument("--no-save", default=False, action="store_true", help="do not save results")
    args = parser.parse_args()

    height = IM_H
    width = IM_W

    VIS = args.vis

    T_begin = time.perf_counter()
    setproctitle.setproctitle(f"gen_xyz_lm_train_pbr_{args.split}_{args.scene}")
    xyz_gen = XyzGen(args.split, args.scene)
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("split", args.split, "scene", args.scene, "total time: ", T_end)
