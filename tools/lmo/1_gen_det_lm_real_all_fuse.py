# modified based on:
#  https://github.com/DLR-RM/AugmentedAutoencoder/blob/master/detection_utils/generate_sixd_train.py
import numpy as np
import cv2
import os.path as osp
import sys
from tqdm import tqdm
import glob
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, ".."))
from lib.pysixd import misc
from lib.vis_utils.image import grid_show, vis_image_bboxes_cv2

np.random.seed(2333)


visualize = False

num_train_imgs = 50000
max_objects_in_scene = 10
num_bg_imgs = 15000
min_visib = 0.6
blackness_thres = 16
# vocpath = 'datasets/VOCdevkit/VOC2012/JPEGImages/*.jpg'
bg_path = "datasets/coco/train2017/*.jpg"
bg_img_paths = glob.glob(bg_path)
assert len(bg_img_paths) > 1000, len(bg_img_paths)

output_path = "datasets/BOP_DATASETS/lmo/lm_real_all_fuse_det_coco/"


id2obj = {
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


def main():
    """using lm_real images, using cut&paste to generate detection data for
    lmo."""
    sixd_train_path = "datasets/BOP_DATASETS/lm/test"
    # cad_path = 'datasets/BOP_DATASETS/tless/models_reconst'
    W, H = 640, 480
    num_objects = 15  # using all 15 objects

    mmcv.mkdir_or_exist(output_path)

    obj_infos = []
    obj_gts = []
    for obj_id in range(1, num_objects + 1):
        obj_infos.append(mmcv.load(osp.join(sixd_train_path, "{:06d}".format(obj_id), "scene_gt_info.json")))
        obj_gts.append(mmcv.load(osp.join(sixd_train_path, "{:06d}".format(obj_id), "scene_gt.json")))

    new_scene_gt = {}

    for i in tqdm(range(num_train_imgs)):
        new_scene_gt[i] = []
        new_train_img = np.zeros((H, W, 3), dtype=np.uint8)
        new_train_mask = np.zeros((H, W, 1), dtype=np.uint8)

        cur_num_objs = 0
        while True:
            rand_obj_id = np.random.randint(0, num_objects)  # 0-based
            rand_view_id = np.random.randint(0, len(obj_infos[rand_obj_id]))
            img_path = osp.join(
                sixd_train_path, "{:06d}".format(rand_obj_id + 1), "rgb", "{:06d}.png".format(rand_view_id)
            )

            rand_img = cv2.imread(img_path)

            # random rotate in-plane
            rot_angle = np.random.rand() * 360
            M = cv2.getRotationMatrix2D((int(rand_img.shape[1] / 2), int(rand_img.shape[0] / 2)), rot_angle, 1)
            rand_img = cv2.warpAffine(rand_img, M, (rand_img.shape[1], rand_img.shape[0]))

            # with ground truth masks
            mask_path = osp.join(
                sixd_train_path, "{:06d}".format(rand_obj_id + 1), "mask", "{:06d}_{:06d}.png".format(rand_view_id, 0)
            )
            orig_mask = mmcv.imread(mask_path, "unchanged").astype(np.float32)
            mask = cv2.warpAffine(orig_mask, M, (orig_mask.shape[1], orig_mask.shape[0]))
            mask = mask > 0
            rand_img[mask == False] = 0  # here == can not be replaced by is

            ys, xs = np.nonzero(mask)
            new_bb = misc.calc_2d_bbox_xywh(xs, ys, height=mask.shape[0], width=mask.shape[1], clip=True)

            # # tless specific
            # crop_x = np.array([20, 380]) + np.random.randint(-15, 15)
            # crop_y = np.array([20, 380]) + np.random.randint(-15, 15)

            # linemod specific
            crop_x = np.array([80, 560]) + np.random.randint(-20, 20)
            crop_y = np.array([0, 480])  # + np.random.randint(-20,20)

            mask = mask[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]
            rand_img = rand_img[crop_y[0] : crop_y[1], crop_x[0] : crop_x[1]]

            orig_H, orig_W = rand_img.shape[:2]
            s = 0.5 * np.random.rand() + 0.5  # [0.5, 1)
            new_H, new_W = int(s * orig_H), int(s * orig_W)
            scaled_img = cv2.resize(rand_img, (new_W, new_H), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(
                mask.astype(np.int32).reshape(orig_H, orig_W, 1), (new_W, new_H), interpolation=cv2.INTER_NEAREST
            )
            y_offset = np.random.randint(0, H - scaled_img.shape[0])
            x_offset = np.random.randint(0, W - scaled_img.shape[1])

            y1, y2 = y_offset, y_offset + scaled_img.shape[0]
            x1, x2 = x_offset, x_offset + scaled_img.shape[1]

            alpha_s = np.dstack((scaled_mask, scaled_mask, scaled_mask)) > 0
            alpha_l = 1.0 - alpha_s  # bg mask 3 channels
            old_train_mask = new_train_mask.copy()
            new_train_mask[y1:y2, x1:x2, 0] = (
                alpha_s[:, :, 0] * scaled_mask + alpha_l[:, :, 0] * new_train_mask[y1:y2, x1:x2, 0]
            )

            old_scene_pix = len(old_train_mask[y1:y2, x1:x2, 0] > 0)
            new_scene_pix = len(new_train_mask > 0)
            new_mask_pix = len(scaled_mask > 0)
            if (new_scene_pix - old_scene_pix) / float(new_mask_pix) < min_visib:
                new_train_mask = old_train_mask.copy()
                continue

            new_train_img[y1:y2, x1:x2, :] = alpha_s * scaled_img + alpha_l * new_train_img[y1:y2, x1:x2, :]

            x, y, w, h = np.round(
                (np.array(new_bb) + np.array([-crop_x[0], -crop_y[0], 0, 0])) * s + np.array([x_offset, y_offset, 0, 0])
            )
            # x,y,w,h = np.round(np.array(gt['obj_bb'])*s+np.array([x_offset,y_offset,0,0])).astype(np.int32)
            new_scene_gt[i].append({"obj_id": int(rand_obj_id + 1), "obj_bb": [float(x), float(y), float(w), float(h)]})
            cur_num_objs += 1
            if cur_num_objs >= max_objects_in_scene:
                break

        rand_bg_path = bg_img_paths[np.random.randint(0, len(bg_img_paths))]
        bg = cv2.resize(cv2.imread(rand_bg_path), (W, H))

        stacked_new_train_mask = np.dstack((new_train_mask, new_train_mask, new_train_mask))
        new_train_img[stacked_new_train_mask == 0] = bg[stacked_new_train_mask == 0]
        # new_train_img = augmenters.augment_image(new_train_img)

        if visualize:
            print("num objs: ", len(new_scene_gt[i]))
            bboxes = []
            labels = []
            for sc_gt in new_scene_gt[i]:
                x, y, w, h = sc_gt["obj_bb"]
                labels.append(id2obj[sc_gt["obj_id"]])
                bboxes.append([x, y, x + w - 1, y + h - 1])
            bboxes = np.array(bboxes)
            img_vis = vis_image_bboxes_cv2(new_train_img, bboxes, labels=labels)

            cv2.imshow("new_train_img", img_vis)
            cv2.imshow("new_train_mask", new_train_mask.astype(np.float32))
            if cv2.waitKey(0) == 27:
                cv2.destroyAllWindows()
                exit(0)  # esc to quit

        cv2.imwrite(osp.join(output_path, f"{i:06d}.png"), new_train_img)
        mmcv.dump(new_scene_gt[i], osp.join(output_path, f"{i:06d}.json"))

    mmcv.dump(new_scene_gt, osp.join(osp.dirname(output_path), "gt.json"))


if __name__ == "__main__":
    main()
