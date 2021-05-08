"""benchvise_test images of linemod dataset, 85% of the real lmo test set."""

import mmcv
import os.path as osp
import json

cur_dir = osp.dirname(osp.abspath(__file__))


IDX2CLASS = {
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
CLASSES = IDX2CLASS.values()
CLASSES = list(sorted(CLASSES))
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}

lm_data_root = "datasets/BOP_DATASETS/lm"
data_root = "datasets/BOP_DATASETS/lmo"
idx_path = osp.join(lm_data_root, "image_set/benchvise_test.txt")
assert osp.exists(idx_path), idx_path


def main():
    with open(idx_path, "r") as f:
        sel_test_indices = [int(line.strip("\r\n")) for line in f.readlines()]

    test_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    num_test_im = 0
    test_scenes = [2]
    for scene_id in test_scenes:
        print("scene_id", scene_id)
        BOP_gt_file = osp.join(data_root, f"test/{scene_id:06d}/scene_gt.json")
        assert osp.exists(BOP_gt_file), BOP_gt_file
        gt_dict = mmcv.load(BOP_gt_file)
        all_ids = [int(k) for k in gt_dict.keys()]
        print(len(all_ids))

        for idx in all_ids:
            if idx not in sel_test_indices:
                continue
            # scene_im_id_str = f'{int(scene_id):04d}/{int(idx):06d}'
            annos = gt_dict[str(idx)]
            obj_ids = [anno["obj_id"] for anno in annos]
            num_inst_dict = {}
            # stat num instances for each obj
            for obj_id in obj_ids:
                if obj_id not in num_inst_dict:
                    num_inst_dict[obj_id] = 1
                else:
                    num_inst_dict[obj_id] += 1
            for obj_id in num_inst_dict:
                target = {"im_id": idx, "inst_count": num_inst_dict[obj_id], "obj_id": obj_id, "scene_id": scene_id}
                test_targets.append(target)
            num_test_im += 1
    res_file = osp.join(cur_dir, "lmo_test_targets_benchvise_test.json")
    print(res_file)
    print(len(test_targets))
    print("num test images: ", num_test_im)  # 2949
    # mmcv.dump(test_targets, res_file)
    with open(res_file, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(item) for item in test_targets) + "]\n")
    print("done")


if __name__ == "__main__":
    main()
