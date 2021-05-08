import mmcv
import os.path as osp

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

data_root = "data/BOP_DATASETS/lm_full"


def main():

    image_set_dir = "training_range_BB8/"
    test_targets = []  # {"im_id": , "inst_count": , "obj_id": , "scene_id": }
    for cls_idx, cls_name in IDX2CLASS.items():
        print(cls_idx, cls_name)
        if cls_name == "camera":
            BB8_train_idx_file = osp.join(image_set_dir, "cam.txt")
        else:
            BB8_train_idx_file = osp.join(image_set_dir, f"{cls_name}.txt")
        with open(BB8_train_idx_file, "r") as f:
            BB8_train_ids = [int(line.strip("\r\n")) for line in f]

        BOP_gt_file = osp.join(data_root, f"test/{cls_idx:06d}/scene_gt.json")
        assert osp.exists(BOP_gt_file), BOP_gt_file
        gt_dict = mmcv.load(BOP_gt_file)
        all_ids = [int(k) for k in gt_dict.keys()]
        test_ids = [k for k in all_ids if k not in BB8_train_ids]
        print(len(test_ids))
        for idx in test_ids:
            target = {"im_id": idx, "inst_count": 1, "obj_id": cls_idx, "scene_id": cls_idx}
            test_targets.append(target)
    res_file = osp.join(cur_dir, "lm_test_targets_bb8.json")
    print(res_file)
    print(len(test_targets))  # 15526
    mmcv.dump(test_targets, res_file)
    print("done")


# NOTE: some images are missed in sixd
"""
1 ape 1050
2 benchvise 1031
3 bowl 1048
4 camera 1020
5 can 1016
6 cat 1002
7 cup 1053
8 driller 1009
9 duck 1065
10 eggbox 1065
11 glue 1036
12 holepuncher 1051
13 iron 979
14 lamp 1042
15 phone 1059
~/PoseEst/mylib/lib/pysixd/lm_test_targets_bb8.json
15526
"""
if __name__ == "__main__":
    main()
    # print('_________________________')
    # num = 0
    # for cls_idx, cls_name in IDX2CLASS.items():
    #     print(cls_name, cls_idx)
    #     with open(osp.join(data_root, 'image_set/{}_test.txt'.format(cls_name))) as f:
    #         ids = [int(line.strip('\r\n')) for line in f]
    #     print(len(ids))
    #     num += len(ids)
    # print(num)
