import hashlib
import logging
import os
import os.path as osp
import sys

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)
import time
from collections import OrderedDict
import mmcv
import numpy as np
from tqdm import tqdm
from transforms3d.quaternions import mat2quat, quat2mat
import ref
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from lib.pysixd import inout, misc
from lib.utils.mask_utils import binary_mask_to_rle, cocosegm2mask
from lib.utils.utils import dprint, iprint, lazy_property


logger = logging.getLogger(__name__)
DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))


class YCBV_BOP_TEST_Dataset:
    """ycbv bop test."""

    def __init__(self, data_cfg):
        """
        Set with_depth and with_masks default to True,
        and decide whether to load them into dataloader/network later
        with_masks:
        """
        self.name = data_cfg["name"]
        self.data_cfg = data_cfg

        self.objs = data_cfg["objs"]  # selected objects
        # all classes are self.objs, but this enables us to evaluate on selected objs
        self.select_objs = data_cfg.get("select_objs", self.objs)

        self.ann_file = data_cfg["ann_file"]  # json file with scene_id and im_id items

        self.dataset_root = data_cfg["dataset_root"]  # BOP_DATASETS/ycbv/test
        self.models_root = data_cfg["models_root"]  # BOP_DATASETS/ycbv/models
        self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001

        self.with_masks = data_cfg["with_masks"]  # True (load masks but may not use it)
        self.with_depth = data_cfg["with_depth"]  # True (load depth path here, but may not use it)

        self.height = data_cfg["height"]  # 480
        self.width = data_cfg["width"]  # 640

        self.cache_dir = data_cfg.get("cache_dir", osp.join(PROJ_ROOT, ".cache"))  # .cache
        self.use_cache = data_cfg.get("use_cache", True)
        self.num_to_load = data_cfg["num_to_load"]  # -1
        self.filter_invalid = data_cfg["filter_invalid"]
        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.ycbv.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.obj2label = OrderedDict((obj, obj_id) for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def __call__(self):
        """Load light-weight instance annotations of all images into a list of
        dicts in Detectron2 format.

        Do not load heavy data into memory in this file, since we will
        load the annotations of all images into memory.
        """
        # cache the dataset_dicts to avoid loading masks from files
        hashed_file_name = hashlib.md5(
            (
                "".join([str(fn) for fn in self.objs])
                + "dataset_dicts_{}_{}_{}_{}_{}".format(
                    self.name, self.dataset_root, self.with_masks, self.with_depth, __name__
                )
            ).encode("utf-8")
        ).hexdigest()
        cache_path = osp.join(self.dataset_root, "dataset_dicts_{}_{}.pkl".format(self.name, hashed_file_name))

        if osp.exists(cache_path) and self.use_cache:
            logger.info("load cached dataset dicts from {}".format(cache_path))
            return mmcv.load(cache_path)

        t_start = time.perf_counter()

        logger.info("loading dataset dicts: {}".format(self.name))
        self.num_instances_without_valid_segmentation = 0
        self.num_instances_without_valid_box = 0
        dataset_dicts = []  # ######################################################
        im_id_global = 0

        if True:
            targets = mmcv.load(self.ann_file)
            scene_im_ids = [(item["scene_id"], item["im_id"]) for item in targets]
            scene_im_ids = sorted(list(set(scene_im_ids)))

            # load infos for each scene
            gt_dicts = {}
            gt_info_dicts = {}
            cam_dicts = {}
            for scene_id, im_id in scene_im_ids:
                scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")
                if scene_id not in gt_dicts:
                    gt_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_gt.json"))
                if scene_id not in gt_info_dicts:
                    gt_info_dicts[scene_id] = mmcv.load(
                        osp.join(scene_root, "scene_gt_info.json")
                    )  # bbox_obj, bbox_visib
                if scene_id not in cam_dicts:
                    cam_dicts[scene_id] = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for scene_id, im_id in tqdm(scene_im_ids):
                str_im_id = str(im_id)
                scene_root = osp.join(self.dataset_root, f"{scene_id:06d}")
                rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(im_id)
                assert osp.exists(rgb_path), rgb_path

                depth_path = osp.join(scene_root, "depth/{:06d}.png".format(im_id))

                scene_id = int(rgb_path.split("/")[-3])

                cam = np.array(cam_dicts[scene_id][str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam_dicts[scene_id][str_im_id]["depth_scale"]
                record = {
                    "dataset_name": self.name,
                    "file_name": osp.relpath(rgb_path, PROJ_ROOT),
                    "depth_file": osp.relpath(depth_path, PROJ_ROOT),
                    "depth_factor": depth_factor,
                    "height": self.height,
                    "width": self.width,
                    "image_id": im_id_global,  # unique image_id in the dataset, for coco evaluation
                    "scene_im_id": "{}/{}".format(scene_id, im_id),  # for evaluation
                    "cam": cam,
                    "img_type": "real",
                }
                im_id_global += 1
                insts = []
                for anno_i, anno in enumerate(gt_dicts[scene_id][str_im_id]):
                    obj_id = anno["obj_id"]
                    if ref.ycbv.id2obj[obj_id] not in self.select_objs:
                        continue
                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0
                    pose = np.hstack([R, t.reshape(3, 1)])
                    quat = mat2quat(R).astype("float32")

                    proj = (record["cam"] @ t.T).T
                    proj = proj[:2] / proj[2]

                    bbox_visib = gt_info_dicts[scene_id][str_im_id][anno_i]["bbox_visib"]
                    bbox_obj = gt_info_dicts[scene_id][str_im_id][anno_i]["bbox_obj"]
                    x1, y1, w, h = bbox_visib
                    if self.filter_invalid:
                        if h <= 1 or w <= 1:
                            self.num_instances_without_valid_box += 1
                            continue

                    mask_file = osp.join(scene_root, "mask/{:06d}_{:06d}.png".format(im_id, anno_i))
                    mask_visib_file = osp.join(scene_root, "mask_visib/{:06d}_{:06d}.png".format(im_id, anno_i))
                    assert osp.exists(mask_file), mask_file
                    assert osp.exists(mask_visib_file), mask_visib_file
                    # load mask visib  TODO: load both mask_visib and mask_full
                    mask_single = mmcv.imread(mask_visib_file, "unchanged")
                    area = mask_single.sum()
                    if area < 3:  # filter out too small or nearly invisible instances
                        self.num_instances_without_valid_segmentation += 1
                        continue
                    mask_rle = binary_mask_to_rle(mask_single, compressed=True)
                    inst = {
                        "category_id": cur_label,  # 0-based label
                        "bbox": bbox_visib,  # TODO: load both bbox_obj and bbox_visib
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "pose": pose,
                        "quat": quat,
                        "trans": t,
                        "centroid_2d": proj,  # absolute (cx, cy)
                        "segmentation": mask_rle,
                        "mask_full_file": mask_file,  # TODO: load as mask_full, rle
                    }

                    model_info = self.models_info[str(obj_id)]
                    inst["model_info"] = model_info
                    # TODO: using full mask and full xyz
                    for key in ["bbox3d_and_center"]:
                        inst[key] = self.models[cur_label][key]
                    insts.append(inst)
                if len(insts) == 0:  # filter im without anno
                    continue
                record["annotations"] = insts
                dataset_dicts.append(record)

        if self.num_instances_without_valid_segmentation > 0:
            logger.warning(
                "Filtered out {} instances without valid segmentation. "
                "There might be issues in your dataset generation process.".format(
                    self.num_instances_without_valid_segmentation
                )
            )
        if self.num_instances_without_valid_box > 0:
            logger.warning(
                "Filtered out {} instances without valid box. "
                "There might be issues in your dataset generation process.".format(self.num_instances_without_valid_box)
            )
        ##########################################################################
        if self.num_to_load > 0:
            self.num_to_load = min(int(self.num_to_load), len(dataset_dicts))
            dataset_dicts = dataset_dicts[: self.num_to_load]
        logger.info("loaded {} dataset dicts, using {}s".format(len(dataset_dicts), time.perf_counter() - t_start))

        mmcv.mkdir_or_exist(osp.dirname(cache_path))
        mmcv.dump(dataset_dicts, cache_path, protocol=4)
        logger.info("Dumped dataset_dicts to {}".format(cache_path))
        return dataset_dicts

    @lazy_property
    def models_info(self):
        models_info_path = osp.join(self.models_root, "models_info.json")
        assert osp.exists(models_info_path), models_info_path
        models_info = mmcv.load(models_info_path)  # key is str(obj_id)
        return models_info

    @lazy_property
    def models(self):
        """Load models into a list."""
        cache_path = osp.join(self.models_root, f"models_{self.name}.pkl")
        if osp.exists(cache_path) and self.use_cache:
            # dprint("{}: load cached object models from {}".format(self.name, cache_path))
            return mmcv.load(cache_path)

        models = []
        for obj_name in self.objs:
            model = inout.load_ply(
                osp.join(self.models_root, f"obj_{ref.ycbv.obj2id[obj_name]:06d}.ply"), vertex_scale=self.scale_to_meter
            )
            # NOTE: the bbox3d_and_center is not obtained from centered vertices
            # for BOP models, not a big problem since they had been centered
            model["bbox3d_and_center"] = misc.get_bbox3d_and_center(model["pts"])

            models.append(model)
        logger.info("cache models to {}".format(cache_path))
        mmcv.dump(models, cache_path, protocol=4)
        return models

    def image_aspect_ratio(self):
        return self.width / self.height  # 4/3


########### register datasets ############################################################


def get_ycbv_metadata(obj_names, ref_key):
    """task specific metadata."""
    data_ref = ref.__dict__[ref_key]

    cur_sym_infos = {}  # label based key
    loaded_models_info = data_ref.get_models_info()

    for i, obj_name in enumerate(obj_names):
        obj_id = data_ref.obj2id[obj_name]
        model_info = loaded_models_info[str(obj_id)]
        if "symmetries_discrete" in model_info or "symmetries_continuous" in model_info:
            sym_transforms = misc.get_symmetry_transformations(model_info, max_sym_disc_step=0.01)
            sym_info = np.array([sym["R"] for sym in sym_transforms], dtype=np.float32)
        else:
            sym_info = None
        cur_sym_infos[i] = sym_info

    meta = {"thing_classes": obj_names, "sym_infos": cur_sym_infos}
    return meta


################################################################################

SPLITS_YCBV = dict(
    ycbv_bop_test=dict(
        name="ycbv_bop_test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/test"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/models"),
        objs=ref.ycbv.objects,  # selected objects
        ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/test_targets_bop19.json"),
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=False,
        ref_key="ycbv",
    )
)


# single objs (num_class is from all objs)
for obj in ref.ycbv.objects:
    name = "ycbv_bop_{}_test".format(obj)
    select_objs = [obj]
    if name not in SPLITS_YCBV:
        SPLITS_YCBV[name] = dict(
            name=name,
            dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/test"),
            models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/models"),
            objs=ref.ycbv.objects,
            select_objs=select_objs,  # selected objects
            ann_file=osp.join(DATASETS_ROOT, "BOP_DATASETS/ycbv/test_targets_bop19.json"),
            scale_to_meter=0.001,
            with_masks=True,  # (load masks but may not use it)
            with_depth=True,  # (load depth path here, but may not use it)
            height=480,
            width=640,
            cache_dir=osp.join(PROJ_ROOT, ".cache"),
            use_cache=True,
            num_to_load=-1,
            filter_invalid=False,
            ref_key="ycbv",
        )


def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    if name in SPLITS_YCBV:
        used_cfg = SPLITS_YCBV[name]
    else:
        assert data_cfg is not None, f"dataset name {name} is not registered"
        used_cfg = data_cfg
    DatasetCatalog.register(name, YCBV_BOP_TEST_Dataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        id="ycbv",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        objs=used_cfg["objs"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_ycbv_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )


def get_available_datasets():
    return list(SPLITS_YCBV.keys())


#### tests ###############################################
def test_vis():
    dset_name = sys.argv[1]
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.objs

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_cv2(d["file_name"], format="BGR")
        depth = mmcv.imread(d["depth_file"], "unchanged") / 1000.0

        imH, imW = img.shape[:2]
        annos = d["annotations"]
        masks = [cocosegm2mask(anno["segmentation"], imH, imW) for anno in annos]
        bboxes = [anno["bbox"] for anno in annos]
        bbox_modes = [anno["bbox_mode"] for anno in annos]
        bboxes_xyxy = np.array(
            [BoxMode.convert(box, box_mode, BoxMode.XYXY_ABS) for box, box_mode in zip(bboxes, bbox_modes)]
        )
        kpts_3d_list = [anno["bbox3d_and_center"] for anno in annos]
        quats = [anno["quat"] for anno in annos]
        transes = [anno["trans"] for anno in annos]
        Rs = [quat2mat(quat) for quat in quats]
        # 0-based label
        cat_ids = [anno["category_id"] for anno in annos]
        K = d["cam"]
        kpts_2d = [misc.project_pts(kpt3d, K, R, t) for kpt3d, R, t in zip(kpts_3d_list, Rs, transes)]
        # # TODO: visualize pose and keypoints
        labels = [objs[cat_id] for cat_id in cat_ids]
        # img_vis = vis_image_bboxes_cv2(img, bboxes=bboxes_xyxy, labels=labels)
        img_vis = vis_image_mask_bbox_cv2(img, masks, bboxes=bboxes_xyxy, labels=labels)
        img_vis_kpts2d = img.copy()
        for anno_i in range(len(annos)):
            img_vis_kpts2d = misc.draw_projected_box3d(img_vis_kpts2d, kpts_2d[anno_i])
        grid_show(
            [img[:, :, [2, 1, 0]], img_vis[:, :, [2, 1, 0]], img_vis_kpts2d[:, :, [2, 1, 0]], depth],
            [f"img:{d['file_name']}", "vis_img", "img_vis_kpts2d", "depth"],
            row=2,
            col=2,
        )


if __name__ == "__main__":
    """Test the  dataset loader.

    Usage:
        python -m core.datasets.ycbv_bop_test dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from core.utils.data_utils import read_image_cv2
    from lib.vis_utils.image import vis_image_mask_bbox_cv2

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg(sys.argv[1])
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
