import logging
import os
import os.path as osp
import sys
import subprocess
import time

import mmcv
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

cur_dir = osp.abspath(osp.dirname(__file__))
sys.path.insert(0, osp.join(cur_dir, "../.."))
import ref
from lib.pysixd import misc


logger = logging.getLogger(__name__)


def _to_str(item):
    if isinstance(item, (list, tuple)):
        return " ".join(["{}".format(e) for e in item])
    else:
        return "{}".format(item)


def to_list(array):
    return array.flatten().tolist()


def save_and_eval_results(cfg, results_all, output_dir, obj_ids=None):
    save_root = output_dir  # eval_path
    split_type_str = f"-{cfg.VAL.SPLIT_TYPE}" if cfg.VAL.SPLIT_TYPE != "" else ""
    mmcv.mkdir_or_exist(save_root)
    header = "scene_id,im_id,obj_id,score,R,t,time"
    keys = header.split(",")
    result_names = []
    for name, result_list in results_all.items():
        method_name = f"{cfg.EXP_ID.replace('_', '-')}-{name}"
        result_name = f"{method_name}_{cfg.VAL.DATASET_NAME}-{cfg.VAL.SPLIT}{split_type_str}.csv"
        res_path = osp.join(save_root, result_name)
        result_names.append(result_name)
        with open(res_path, "w") as f:
            f.write(header + "\n")
            for line_i, result in enumerate(result_list):
                items = []
                for res_k in keys:
                    items.append(_to_str(result[res_k]))
                f.write(",".join(items) + "\n")
        logger.info("wrote results to: {}".format(res_path))

    result_names_str = ",".join(result_names)
    eval_cmd = [
        "python",
        cfg.VAL.SCRIPT_PATH,
        "--results_path={}".format(save_root),
        "--result_filenames={}".format(result_names_str),
        "--renderer_type={}".format(cfg.VAL.RENDERER_TYPE),
        "--error_types={}".format(cfg.VAL.ERROR_TYPES),
        "--eval_path={}".format(save_root),
        "--targets_filename={}".format(cfg.VAL.TARGETS_FILENAME),
        "--n_top={}".format(cfg.VAL.N_TOP),
    ]
    if cfg.VAL.SCORE_ONLY:
        eval_cmd += ["--score_only"]
    eval_time = time.perf_counter()
    if subprocess.call(eval_cmd) != 0:
        logger.warning("evaluation failed.")

    load_and_print_val_scores_tab(
        cfg, eval_root=save_root, result_names=result_names, error_types=cfg.VAL.ERROR_TYPES.split(","), obj_ids=obj_ids
    )
    logger.info("eval time: {}s".format(time.perf_counter() - eval_time))


def eval_cached_results(cfg, output_dir, obj_ids=None):
    logger.info("eval cached results")
    split_type_str = f"-{cfg.VAL.SPLIT_TYPE}" if cfg.VAL.SPLIT_TYPE != "" else ""
    save_root = output_dir  # eval_path
    assert osp.exists(save_root), save_root
    result_names = []
    names = ["iter{}".format(i) for i in range(cfg.TEST.get("ITER_NUM", 0) + 1)]
    exp_id = cfg.EXP_ID
    # print('exp_id', exp_id)
    for name in names:
        method_name = "{}-{}".format(exp_id.replace("_", "-"), name)
        result_name = f"{method_name}_{cfg.VAL.DATASET_NAME}-{cfg.VAL.SPLIT}{split_type_str}.csv"
        res_path = osp.join(save_root, result_name)
        if not osp.exists(res_path):
            if exp_id.endswith("_test"):
                method_name = "{}-{}".format(exp_id.replace("_test", "").replace("_", "-"), name)
                result_name = f"{method_name}_{cfg.VAL.DATASET_NAME}-{cfg.VAL.SPLIT}{split_type_str}.csv"
                res_path = osp.join(save_root, result_name)
        assert osp.exists(res_path), res_path
        result_names.append(result_name)
    try:
        if not cfg.VAL.EVAL_PRINT_ONLY:
            raise RuntimeError()
        load_and_print_val_scores_tab(
            cfg,
            eval_root=save_root,
            result_names=result_names,
            error_types=cfg.VAL.ERROR_TYPES.split(","),
            obj_ids=obj_ids,
        )
    except:
        result_names_str = ",".join(result_names)
        eval_cmd = [
            "python",
            cfg.VAL.SCRIPT_PATH,
            "--results_path={}".format(save_root),
            "--result_filenames={}".format(result_names_str),
            "--renderer_type={}".format(cfg.VAL.RENDERER_TYPE),
            "--error_types={}".format(cfg.VAL.ERROR_TYPES),
            "--eval_path={}".format(save_root),
            "--targets_filename={}".format(cfg.VAL.TARGETS_FILENAME),
            "--n_top={}".format(cfg.VAL.N_TOP),
        ]
        if cfg.VAL.SCORE_ONLY:
            eval_cmd += ["--score_only"]
        eval_time = time.perf_counter()
        if subprocess.call(eval_cmd) != 0:
            logger.warning("evaluation failed.")

        load_and_print_val_scores_tab(
            cfg,
            eval_root=save_root,
            result_names=result_names,
            error_types=cfg.VAL.ERROR_TYPES.split(","),
            obj_ids=obj_ids,
        )
        logger.info("eval time: {}s".format(time.perf_counter() - eval_time))
    exit(0)


def get_data_ref(dataset_name):
    ref_key_dict = {"lm": "lm_full", "lmo": "lmo_full", "ycbv": "ycbv", "ycbvposecnn": "ycbv", "tless": "tless"}
    ref_key = ref_key_dict[dataset_name]
    return ref.__dict__[ref_key]


def get_thr(score_path):
    # used for sorting score json files
    # scores_th:2.000_min-visib:0.100.json
    # rete: scores_th:10.000-10.000_min-visib:-1.000.json
    # NOTE: assume the same threshold (currently can deal with rete, rete_s)
    return float(score_path.split("/")[-1].replace("scores_th:", "").split("_")[0].split("-")[0])


def simplify_float_str(float_str):
    value = float(float_str)
    if value == int(value):
        return str(int(value))
    return float_str


def get_thr_str(score_path):
    # path/to/scores_th:2.000_min-visib:0.100.json  --> 2
    # rete: path/to/scores_th:10.000-10.000_min-visib:-1.000.json --> 10
    thr_str = score_path.split("/")[-1].split("_")[1]
    thr_str = thr_str.split(":")[-1]
    if "-" in thr_str:
        thr_str_split = thr_str.split("-")
        simple_str_list = [simplify_float_str(_thr) for _thr in thr_str_split]
        if len(set(simple_str_list)) == 1:
            res_thr_str = simple_str_list[0]
        else:
            res_thr_str = "-".join(simple_str_list)
    else:
        res_thr_str = simplify_float_str(thr_str)
    return res_thr_str


def is_auc_metric(error_type):
    if error_type in ["AUCadd", "AUCadi", "AUCad", "vsd", "mssd", "mspd"]:
        return True
    return False


def summary_scores(score_paths, error_type, val_dataset_name, print_all_objs=False, obj_ids=None):
    data_ref = get_data_ref(val_dataset_name)

    sorted_score_paths = sorted(score_paths.keys(), key=get_thr)

    max_thr_str = None
    obj_recalls_dict = {}
    if is_auc_metric(error_type):
        max_thr_str = get_thr_str(sorted_score_paths[-1])

    tabs_col2 = []
    for score_path in sorted_score_paths:
        score_dict = mmcv.load(score_path)
        if obj_ids is None:
            sel_obj_ids = [int(_id) for _id in score_dict["obj_recalls"].keys()]
        else:
            sel_obj_ids = obj_ids

        thr_str = get_thr_str(score_path)
        # logging the results with tabulate
        # tab_header = ["objects", "{}[{}](%)".format(error_type, thr_str)]
        tab_header = ["objects", "{}_{}".format(error_type, thr_str)]  # 2 columns, objs in col
        cur_tab_col2 = [tab_header]
        for _id, _recall in score_dict["obj_recalls"].items():
            obj_name = data_ref.id2obj[int(_id)]
            if int(_id) in sel_obj_ids:
                cur_tab_col2.append([obj_name, f"{_recall * 100:.2f}"])
                if max_thr_str is not None:  # for AUC metrics
                    if obj_name not in obj_recalls_dict:
                        obj_recalls_dict[obj_name] = []
                    obj_recalls_dict[obj_name].append(_recall)
            else:
                if print_all_objs:
                    cur_tab_col2.append([obj_name, "-"])

        # mean of selected objs
        num_objs = len(sel_obj_ids)
        if num_objs > 1:
            sel_obj_recalls = [_recall for _id, _recall in score_dict["obj_recalls"].items() if int(_id) in sel_obj_ids]
            mean_obj_recall = np.mean(sel_obj_recalls)
            cur_tab_col2.append(["Avg({})".format(num_objs), f"{mean_obj_recall * 100:.2f}"])

        cur_tab_col2 = np.array(cur_tab_col2)
        tabs_col2.append(cur_tab_col2)

    if len(tabs_col2) == 1:
        return tabs_col2[0]
    else:
        if max_thr_str is None:  # not AUC metrics, concat
            res_tab = np.concatenate([tabs_col2[0]] + [_tab[:, 1:2] for _tab in tabs_col2[1:]], axis=1)
        else:  # AUC metrics, mean
            auc_header = ["objects", "{}_{}".format(error_type, max_thr_str)]  # 2 columns, objs in col
            res_tab = [auc_header]
            obj_aucs = []
            for obj_name in tabs_col2[0][1:-1, 0].tolist():
                if obj_name in obj_recalls_dict:
                    cur_auc = np.mean(obj_recalls_dict[obj_name])
                    obj_aucs.append(cur_auc)
                    res_tab.append([obj_name, f"{cur_auc * 100:.2f}"])
            res_tab.append(["Avg({})".format(len(obj_aucs)), f"{np.mean(obj_aucs) * 100:.2f}"])
            res_tab = np.array(res_tab)
        return res_tab


def load_and_print_val_scores_tab(
    cfg, eval_root, result_names, error_types=["projS", "ad", "reteS"], obj_ids=None, print_all_objs=False
):

    vsd_deltas = {
        "hb": 15,
        "hbs": 15,
        "icbin": 15,
        "icmi": 15,
        "itodd": 5,
        "lm": 15,
        "lmo": 15,
        "ruapc": 15,
        "tless": 15,
        "tudl": 15,
        "tyol": 15,
        "ycbv": 15,
        "ycbvposecnn": 15,
    }
    ntop = cfg.VAL.N_TOP
    val_dataset_name = cfg.VAL.DATASET_NAME
    vsd_delta = vsd_deltas[val_dataset_name]
    data_ref = get_data_ref(val_dataset_name)

    vsd_taus = list(np.arange(0.05, 0.51, 0.05))
    # visib_gt_min = 0.1

    for result_name in tqdm(result_names):
        logger.info("=====================================================================")
        big_tab_row = []
        for error_type in error_types:
            result_name = result_name.replace(".csv", "")
            # logger.info(f"************{result_name} *** [{error_type}]*******************")
            if error_type == "vsd":
                error_signs = [
                    misc.get_error_signature(error_type, ntop, vsd_delta=vsd_delta, vsd_tau=vsd_tau)
                    for vsd_tau in vsd_taus
                ]
            else:
                error_signs = [misc.get_error_signature(error_type, ntop)]
            score_roots = [osp.join(eval_root, result_name, error_sign) for error_sign in error_signs]

            for score_root in score_roots:
                if osp.exists(score_root):
                    # get all score json files for this metric under this threshold
                    score_paths = {
                        osp.join(score_root, fn.name): None
                        for fn in os.scandir(score_root)
                        if ".json" in fn.name and "scores" in fn.name
                    }

                    tab_obj_col = summary_scores(
                        score_paths,
                        error_type,
                        val_dataset_name=val_dataset_name,
                        print_all_objs=print_all_objs,
                        obj_ids=obj_ids,
                    )
                    # print single metric with obj in col here
                    logger.info(f"************{result_name} *********************")
                    tab_obj_col_log_str = tabulate(
                        tab_obj_col,
                        tablefmt="plain",
                        # floatfmt=floatfmt
                    )
                    logger.info("\n{}".format(tab_obj_col_log_str))
                    #####
                    big_tab_row.append(tab_obj_col.T)  # objs in row

                else:
                    logger.warning("{} does not exist.".format(score_root))
                    raise RuntimeError("{} does not exist.".format(score_root))

        if len(big_tab_row) > 0:
            logger.info(f"************{result_name} *********************")
            if len(big_tab_row) == 1:
                res_log_tab = big_tab_row[0]
            else:
                res_log_tab = np.concatenate([big_tab_row[0]] + [_tab[1:, :] for _tab in big_tab_row[1:]], axis=0)
            res_log_tab_col = res_log_tab.T

            if len(res_log_tab) < len(res_log_tab_col):  # print the table with more rows later
                log_tabs = [res_log_tab, res_log_tab_col]
                suffixes = ["row", "col"]
            else:
                log_tabs = [res_log_tab_col, res_log_tab]
                suffixes = ["col", "row"]
            for log_tab_i, suffix in zip(log_tabs, suffixes):
                dump_tab_name = osp.join(eval_root, f"{result_name}_tab_obj_{suffix}.txt")
                log_tab_i_str = tabulate(
                    log_tab_i,
                    tablefmt="plain",
                    # floatfmt=floatfmt
                )
                logger.info("\n{}".format(log_tab_i_str))
                with open(dump_tab_name, "w") as f:
                    f.write("{}\n".format(log_tab_i_str))
    logger.info("{}".format(eval_root))


if __name__ == "__main__":
    import argparse
    from mmcv import Config, DictAction
    from lib.utils.setup_logger import setup_my_logger

    parser = argparse.ArgumentParser(description="wrapper functions to evaluate with bop toolkit")
    parser.add_argument(
        "--script-path",
        default="lib/pysixd/scripts/eval_pose_results_more.py",
        help="script path to run bop evaluation",
    )

    parser.add_argument("--result_dir", default="", help="result dir")
    # f"{method_name}_{cfg.VAL.DATASET_NAME}-{cfg.VAL.SPLIT}{split_type_str}.csv"
    parser.add_argument("--result_names", default="", help="result names: a.csv,b.csv,c.csv")

    parser.add_argument("--dataset", default="lmo", help="dataset name")
    parser.add_argument("--split", default="test", help="split")
    parser.add_argument("--split-type", default="bb8", help="split type")

    parser.add_argument("--targets_name", default="test_targets_bop19.json", help="targets filename")
    parser.add_argument("--obj_ids", default=None, help="obj ids to evaluate: 1,2,3,4")
    # "vsd,mssd,mspd"
    parser.add_argument("--n_top", default=1, type=int, help="top n to be evaluated, VIVO: -1, SISO: 1")
    parser.add_argument("--error_types", default="ad,reteS,reS,teS,projS", help="error types")
    parser.add_argument("--render_type", default="cpp", help="render type: python | cpp | egl")
    parser.add_argument("--score_only", default=False, action="store_true", help="score only")
    parser.add_argument("--print_only", default=False, action="store_true", help="print only")
    parser.add_argument(
        "--opts", nargs="+", action=DictAction, help="arguments in dict, modify config using command-line args"
    )
    args = parser.parse_args()

    if args.obj_ids is not None:
        obj_ids = [int(_e) for _e in args.obj_ids.split(",")]
    else:
        obj_ids = args.obj_ids
    result_dir = args.result_dir
    setup_my_logger(name="core")
    setup_my_logger(name="__main__")
    result_names_str = args.result_names
    if "," not in result_names_str:
        result_names = [result_names_str]
    else:
        result_names = result_names_str.split(",")

    cfg_dict = dict(
        VAL=dict(
            DATASET_NAME=args.dataset,
            SCRIPT_PATH=args.script_path,
            RESULTS_PATH=result_dir,
            TARGETS_FILENAME=args.targets_name,
            ERROR_TYPES=args.error_types,
            RENDERER_TYPE=args.render_type,  # cpp, python, egl
            SPLIT=args.split,
            SPLIT_TYPE=args.split_type,
            N_TOP=args.n_top,  # SISO: 1, VIVO: -1 (for LINEMOD, 1/-1 are the same)
            SCORE_ONLY=args.score_only,  # if the errors have been calculated
            EVAL_PRINT_ONLY=args.print_only,  # if the scores/recalls have been saved
        )
    )
    cfg = Config(cfg_dict)
    if args.opts is not None:
        cfg.merge_from_dict(args.opts)

    eval_time = time.perf_counter()
    if not args.print_only:
        eval_cmd = [
            "python",
            cfg.VAL.SCRIPT_PATH,
            "--results_path={}".format(result_dir),
            "--result_filenames={}".format(result_names_str),
            "--renderer_type={}".format(cfg.VAL.RENDERER_TYPE),
            "--error_types={}".format(cfg.VAL.ERROR_TYPES),
            "--eval_path={}".format(result_dir),
            "--targets_filename={}".format(cfg.VAL.TARGETS_FILENAME),
            "--n_top={}".format(cfg.VAL.N_TOP),
        ]
        if cfg.VAL.SCORE_ONLY:
            eval_cmd += ["--score_only"]
        if subprocess.call(eval_cmd) != 0:
            logger.warning("evaluation failed.")

    print("print scores")
    load_and_print_val_scores_tab(
        cfg,
        eval_root=result_dir,
        result_names=result_names,
        error_types=cfg.VAL.ERROR_TYPES.split(","),
        obj_ids=obj_ids,
    )
    logger.info("eval time: {}s".format(time.perf_counter() - eval_time))
