import mmcv
import os.path as osp
import json

cur_dir = osp.dirname(osp.abspath(__file__))


IDX2CLASS = {
    1: "002_master_chef_can",  # [1.3360, -0.5000, 3.5105]
    2: "003_cracker_box",  # [0.5575, 1.7005, 4.8050]
    3: "004_sugar_box",  # [-0.9520, 1.4670, 4.3645]
    4: "005_tomato_soup_can",  # [-0.0240, -1.5270, 8.4035]
    5: "006_mustard_bottle",  # [1.2995, 2.4870, -11.8290]
    6: "007_tuna_fish_can",  # [-0.1565, 0.1150, 4.2625]
    7: "008_pudding_box",  # [1.1645, -4.2015, 3.1190]
    8: "009_gelatin_box",  # [1.4460, -0.5915, 3.6085]
    9: "010_potted_meat_can",  # [2.4195, 0.3075, 8.0715]
    10: "011_banana",  # [-18.6730, 12.1915, -1.4635]
    11: "019_pitcher_base",  # [5.3370, 5.8855, 25.6115]
    12: "021_bleach_cleanser",  # [4.9290, -2.4800, -13.2920]
    13: "024_bowl",  # [-0.2270, 0.7950, -2.9675]
    14: "025_mug",  # [-8.4675, -0.6995, -1.6145]
    15: "035_power_drill",  # [9.0710, 20.9360, -2.1190]
    16: "036_wood_block",  # [1.4265, -2.5305, 17.1890]
    17: "037_scissors",  # [7.0535, -28.1320, 0.0420]
    18: "040_large_marker",  # [0.0460, -2.1040, 0.3500]
    19: "051_large_clamp",  # [10.5180, -1.9640, -0.4745]
    20: "052_extra_large_clamp",  # [-0.3950, -10.4130, 0.1620]
    21: "061_foam_brick",  # [-0.0805, 0.0805, -8.2435]
}
CLASSES = IDX2CLASS.values()
CLASSES = list(sorted(CLASSES))
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}

data_root = "datasets/BOP_DATASETS/ycbv"


def main():
    test_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    test_scenes = [i for i in range(48, 59 + 1)]
    for scene_id in test_scenes:
        print("scene_id", scene_id)
        BOP_gt_file = osp.join(data_root, f"test/{scene_id:06d}/scene_gt.json")
        assert osp.exists(BOP_gt_file), BOP_gt_file
        gt_dict = mmcv.load(BOP_gt_file)
        all_ids = [int(k) for k in gt_dict.keys()]
        print(len(all_ids))
        for idx in all_ids:
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
    res_file = osp.join(cur_dir, "ycbv_test_targets_all.json")
    print(res_file)
    print(len(test_targets))  # 15526
    # mmcv.dump(test_targets, res_file)
    with open(res_file, "w") as f:
        f.write("[\n" + ",\n".join(json.dumps(item) for item in test_targets) + "]\n")
    print("done")


if __name__ == "__main__":
    main()
