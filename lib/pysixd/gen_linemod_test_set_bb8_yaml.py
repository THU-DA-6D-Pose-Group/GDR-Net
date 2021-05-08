import mmcv
import os.path as osp
import ruamel.yaml as yaml
from ruamel.yaml.comments import CommentedSeq, CommentedMap

IDX2CLASS = {
    1: "ape",
    2: "benchvise",
    # 3: 'bowl',
    4: "camera",
    5: "can",
    6: "cat",
    # 7: 'cup',
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
CLASSES = sorted(CLASSES)
CLASS2IDX = {cls_name: idx for idx, cls_name in IDX2CLASS.items()}


def main():
    image_set_dir = "data/LINEMOD_6D/LM6d_converted/LM6d_render_v1/image_set/real/"
    test_set_dict = CommentedMap({cls_idx: CommentedSeq([]) for cls_idx in IDX2CLASS.keys()})
    for cls_idx, cls_name in IDX2CLASS.items():
        print(cls_idx, cls_name)
        index_file = osp.join(image_set_dir, "{}_test.txt".format(cls_name))
        with open(index_file, "r") as f:
            test_indices = [line.strip("\r\n") for line in f.readlines()]
        for test_idx in test_indices:
            sixd_idx = int(test_idx.split("/")[1]) - 1
            test_set_dict[cls_idx].append(sixd_idx)
    for cls_idx, indices in test_set_dict.items():
        indices.fa.set_flow_style()

    res_file = osp.join(osp.expanduser("~/Storage/SIXD_DATASETS/LM6d_origin/", "test_set_bb8_sixd.yml"))
    with open(res_file, "w") as f:
        yaml.dump(test_set_dict, f, Dumper=yaml.RoundTripDumper, width=10000)

    print("done")


# NOTE: some images are missed in sixd

if __name__ == "__main__":
    main()
