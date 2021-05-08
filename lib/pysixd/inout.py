# from pysixd toolkit
# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague
# modified
"""I/O functions."""
import json
import os
import os.path as osp
import struct
import sys

import imageio
import mmcv
import numpy as np
import png
import scipy.io as sio
import scipy.misc
import six

# import ruamel.yaml as yaml
import yaml

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(cur_dir, "../.."))
from lib.utils import logger
from lib.utils.is_binary_file import is_binary as _is_binary


def load_im(path):
    """Loads an image from a file.

    :param path: Path to the image file to load.
    :return: ndarray with the loaded image.
    """
    im = imageio.imread(path)
    return im


def save_im(path, im, jpg_quality=95):
    """Saves an image to a file.

    :param path: Path to the output image file.
    :param im: ndarray with the image to save.
    :param jpg_quality: Quality of the saved image (applies only to JPEG).
    """
    ext = os.path.splitext(path)[1][1:]
    if ext.lower() in ["jpg", "jpeg"]:
        imageio.imwrite(path, im, quality=jpg_quality)
    else:
        imageio.imwrite(path, im)


def load_depth(path):
    # NOTE: cv2 is more faster (about 2x)
    """Loads a depth image from a file.

    :param path: Path to the depth image file to load.
    :return: ndarray with the loaded depth image.
    """
    d = imageio.imread(path)
    return d.astype(np.float32)


def save_depth(path, im):
    """Saves a depth image (16-bit) to a PNG file.

    :param path: Path to the output depth image file.
    :param im: ndarray with the depth image to save.
    """
    if path.split(".")[-1].lower() != "png":
        raise ValueError("Only PNG format is currently supported.")

    im_uint16 = np.round(im).astype(np.uint16)

    # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
    # cv2 is far more faster
    w_depth = png.Writer(im.shape[1], im.shape[0], greyscale=True, bitdepth=16)
    with open(path, "wb") as f:
        w_depth.write(f, np.reshape(im_uint16, (-1, im.shape[1])))


def load_json(path, keys_to_int=False):
    """Loads content of a JSON file.

    :param path: Path to the JSON file.
    :return: Content of the loaded JSON file.
    """
    # Keys to integers.
    def convert_keys_to_int(x):
        return {int(k) if k.lstrip("-").isdigit() else k: v for k, v in x.items()}

    with open(path, "r") as f:
        if keys_to_int:
            content = json.load(f, object_hook=lambda x: convert_keys_to_int(x))
        else:
            content = json.load(f)

    return content


def save_json(path, content, sort=False):
    """Saves the provided content to a JSON file.

    :param path: Path to the output JSON file.
    :param content: Dictionary/list to save.
    """
    with open(path, "w") as f:

        if isinstance(content, dict):
            f.write("{\n")
            if sort:
                content_sorted = sorted(content.items(), key=lambda x: x[0])
            else:
                content_sorted = content.items()
            for elem_id, (k, v) in enumerate(content_sorted):
                f.write('  "{}": {}'.format(k, json.dumps(v, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("}")

        elif isinstance(content, list):
            f.write("[\n")
            for elem_id, elem in enumerate(content):
                f.write("  {}".format(json.dumps(elem, sort_keys=True)))
                if elem_id != len(content) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]")

        else:
            json.dump(content, f, sort_keys=True)


# Set representation of the floating point numbers in YAML files
def float_representer(dumper, value):
    text = "{0:.8f}".format(value)
    return dumper.represent_scalar(u"tag:yaml.org,2002:float", text)


yaml.add_representer(float, float_representer)


# ===================================
# LINEMOD_ORIG
def read_linemod_transform_dat(transform_dat_path):
    # for  LINEMOD ORIG
    transform_dat = np.loadtxt(transform_dat_path, skiprows=1)[:, 1]
    transform_dat = np.reshape(transform_dat, newshape=[3, 4])
    return transform_dat


def load_linemod_orig_model(model_dir):
    # for LINEMOD_ORIG/cls/
    orig_model_path = osp.join(model_dir, "mesh.ply")
    orig_old_model_path = osp.join(model_dir, "OLDmesh.ply")
    transform_dat_path = osp.join(model_dir, "transform.dat")
    if os.path.exists(orig_model_path):
        return ply_vtx(orig_model_path) / 1000.0
    else:
        transform = read_linemod_transform_dat(transform_dat_path)
        old_model = ply_vtx(orig_old_model_path) / 1000.0
        old_model = np.dot(old_model, transform[:, :3].T) + transform[:, 3]
        return old_model


def read_linemod_pose(rot_path, tra_path):
    """read linemod orig pose."""
    rot = np.loadtxt(rot_path, skiprows=1)
    tra = np.loadtxt(tra_path, skiprows=1) / 100.0
    return np.concatenate([rot, np.reshape(tra, newshape=[3, 1])], axis=-1)


# ======================================
def pose_from_meta_LM6d(meta_file, cls_idx):
    """assume single instance of cls_idx."""
    meta_data = sio.loadmat(meta_file)
    inner_id = np.where(np.squeeze(meta_data["cls_indexes"]) == cls_idx)
    if len(meta_data["poses"].shape) == 2:
        pose = meta_data["poses"]
    else:
        pose = np.squeeze(meta_data["poses"][:, :, inner_id])
    return pose


def load_emb_pkl_single(emb_path, cls_idx=None, width=640, height=480):
    assert osp.exists(emb_path), "not exist {}".format(emb_path)
    emb = None
    tmp = mmcv.load(emb_path)
    if isinstance(tmp, dict):
        emb = np.zeros((height, width, 3), dtype=np.float32)
        x1, y1, x2, y2 = tmp["bbox"]
        emb[y1 : y2 + 1, x1 : x2 + 1, :] = tmp["emb"]
    elif isinstance(tmp, list) and cls_idx is not None:
        """assume single instance of cls_idx."""
        for e in tmp:
            if e is not None and e["cls_idx"] == cls_idx:
                emb = np.zeros((height, width, 3), dtype=np.float32)
                x1, y1, x2, y2 = e["bbox"]
                emb[y1 : y2 + 1, x1 : x2 + 1, :] = e["emb"]
    return emb


def load_cam_params(path):
    """Loads camera parameters from a JSON file.

    :param path: Path to the JSON file.
    :return: Dictionary with the following items:
     - 'im_size': (width, height).
     - 'K': 3x3 intrinsic camera matrix.
     - 'depth_scale': Scale factor to convert the depth images to mm (optional).
    """
    c = load_json(path)

    cam = {
        "im_size": (c["width"], c["height"]),
        "K": np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]]),
    }

    if "depth_scale" in c.keys():
        cam["depth_scale"] = float(c["depth_scale"])

    return cam


def load_scene_camera(path):
    """Loads content of a JSON file with information about the scene camera.

    See docs/bop_datasets_format.md for details.

    :param path: Path to the JSON file.
    :return: Dictionary with the loaded content.
    """
    scene_camera = load_json(path, keys_to_int=True)

    for im_id in scene_camera.keys():
        if "cam_K" in scene_camera[im_id].keys():
            scene_camera[im_id]["cam_K"] = np.array(scene_camera[im_id]["cam_K"], np.float).reshape((3, 3))
        if "cam_R_w2c" in scene_camera[im_id].keys():
            scene_camera[im_id]["cam_R_w2c"] = np.array(scene_camera[im_id]["cam_R_w2c"], np.float).reshape((3, 3))
        if "cam_t_w2c" in scene_camera[im_id].keys():
            scene_camera[im_id]["cam_t_w2c"] = np.array(scene_camera[im_id]["cam_t_w2c"], np.float).reshape((3, 1))
    return scene_camera


def save_scene_camera(path, scene_camera):
    """Saves information about the scene camera to a JSON file.

    See docs/bop_datasets_format.md for details.

    :param path: Path to the output JSON file.
    :param scene_camera: Dictionary to save to the JSON file.
    """
    for im_id in sorted(scene_camera.keys()):
        im_camera = scene_camera[im_id]
        if "cam_K" in im_camera.keys():
            im_camera["cam_K"] = im_camera["cam_K"].flatten().tolist()
        if "cam_R_w2c" in im_camera.keys():
            im_camera["cam_R_w2c"] = im_camera["cam_R_w2c"].flatten().tolist()
        if "cam_t_w2c" in im_camera.keys():
            im_camera["cam_t_w2c"] = im_camera["cam_t_w2c"].flatten().tolist()
    save_json(path, scene_camera)


def load_scene_gt(path):
    """Loads content of a JSON file with ground-truth annotations.

    See docs/bop_datasets_format.md for details.

    :param path: Path to the JSON file.
    :return: Dictionary with the loaded content.
    """
    scene_gt = load_json(path, keys_to_int=True)

    for im_id, im_gt in scene_gt.items():
        for gt in im_gt:
            if "cam_R_m2c" in gt.keys():
                gt["cam_R_m2c"] = np.array(gt["cam_R_m2c"], np.float).reshape((3, 3))
            if "cam_t_m2c" in gt.keys():
                gt["cam_t_m2c"] = np.array(gt["cam_t_m2c"], np.float).reshape((3, 1))
    return scene_gt


def save_scene_gt(path, scene_gt):
    """Saves ground-truth annotations to a JSON file.

    See docs/bop_datasets_format.md for details.

    :param path: Path to the output JSON file.
    :param scene_gt: Dictionary to save to the JSON file.
    """
    for im_id in sorted(scene_gt.keys()):
        im_gts = scene_gt[im_id]
        for gt in im_gts:
            if "cam_R_m2c" in gt.keys():
                gt["cam_R_m2c"] = gt["cam_R_m2c"].flatten().tolist()
            if "cam_t_m2c" in gt.keys():
                gt["cam_t_m2c"] = gt["cam_t_m2c"].flatten().tolist()
            if "obj_bb" in gt.keys():
                gt["obj_bb"] = [int(x) for x in gt["obj_bb"]]
    save_json(path, scene_gt)


def load_bop_results(path, version="bop19"):
    """Loads 6D object pose estimates from a file.

    :param path: Path to a file with pose estimates.
    :param version: Version of the results.
    :return: List of loaded poses.
    """
    results = []

    # See docs/bop_challenge_2019.md for details.
    if version == "bop19":
        header = "scene_id,im_id,obj_id,score,R,t,time"
        with open(path, "r") as f:
            line_id = 0
            for line in f:
                line_id += 1
                if line_id == 1 and header in line:
                    continue
                else:
                    elems = line.split(",")
                    if len(elems) != 7:
                        raise ValueError("A line does not have 7 comma-sep. elements: {}".format(line))
                    # import pdb; pdb.set_trace();
                    result = {
                        "scene_id": int(elems[0]),
                        "im_id": int(elems[1]),
                        "obj_id": int(elems[2]),
                        "score": float(elems[3]),
                        "R": np.array(list(map(float, elems[4].split())), np.float).reshape((3, 3)),
                        "t": np.array(list(map(float, elems[5].split())), np.float).reshape((3, 1)),
                        "time": float(elems[6]),
                    }

                    results.append(result)
    else:
        raise ValueError("Unknown version of BOP results.")

    return results


def save_bop_results(path, results, version="bop19"):
    """Saves 6D object pose estimates to a file.

    :param path: Path to the output file.
    :param results: Dictionary with pose estimates.
    :param version: Version of the results.
    """
    # See docs/bop_challenge_2019.md for details.
    if version == "bop19":
        lines = ["scene_id,im_id,obj_id,score,R,t,time"]
        for res in results:
            if "time" in res:
                run_time = res["time"]
            else:
                run_time = -1

            lines.append(
                "{scene_id},{im_id},{obj_id},{score},{R},{t},{time}".format(
                    scene_id=res["scene_id"],
                    im_id=res["im_id"],
                    obj_id=res["obj_id"],
                    score=res["score"],
                    R=" ".join(map(str, res["R"].flatten().tolist())),
                    t=" ".join(map(str, res["t"].flatten().tolist())),
                    time=run_time,
                )
            )

        with open(path, "w") as f:
            f.write("\n".join(lines))

    else:
        raise ValueError("Unknown version of BOP results.")


def check_bop_results(path, version="bop19"):
    """Checks if the format of BOP results is correct.

    :param result_filenames: Path to a file with pose estimates.
    :param version: Version of the results.
    :return: True if the format is correct, False if it is not correct.
    """
    check_passed = True
    check_msg = "OK"
    try:
        results = load_bop_results(path, version)

        if version == "bop19":
            # Check if the time for all estimates from the same image are the same.
            times = {}
            for result in results:
                result_key = "{:06d}_{:06d}".format(result["scene_id"], result["im_id"])
                if result_key in times:
                    if abs(times[result_key] - result["time"]) > 0.001:
                        check_passed = False
                        check_msg = (
                            "The running time for scene {} and image {} is not the same for"
                            " all estimates.".format(result["scene_id"], result["im_id"])
                        )
                        logger.info(check_msg)
                        break
                else:
                    times[result_key] = result["time"]

    except Exception as e:
        check_passed = False
        check_msg = "Error when loading BOP results: {}".format(e)
        logger.info(check_msg)

    return check_passed, check_msg


def ply_vtx(path, vertex_scale=1.0):
    """
    discription: read all vertices from a ply file
    (borrow from https://github.com/paroj/linemod_dataset/blob/master/read.py)

    : param path: path to ply model file
    return: all 3D points of ply model: (N, 3)
    """
    with open(path) as f:
        assert f.readline().strip() == "ply"
        for line in f:
            line = line.strip()
            if "element vertex" in line:
                N_str = line.split()[-1]
                # print(N_str)
                N = int(N_str)
                break
        while f.readline().strip() != "end_header":
            continue
        pts = []
        for _ in range(N):
            pts.append(np.float32(f.readline().split()[:3]))
    return np.array(pts) * vertex_scale


def ply_vtx_expand(path, vertex_scale=1.0):
    """
    discription: read all vertices from a ply file and expand vertices using polygon info.
    (borrow from https://github.com/paroj/linemod_dataset/blob/master/read.py)

    : param path: path to ply model file
    return: all 3D points of ply model: (N, 3)
    """
    f = open(path)
    assert f.readline().strip() == "ply"
    while True:
        line = f.readline().strip()
        if line.startswith("element vertex"):
            N = int(line.split()[-1])
        if line.startswith("element face"):
            F = int(line.split()[-1])
        if line == "end_header":
            break
    pts = []
    for _ in range(N):
        pts.append(np.float32(f.readline().split()[:3]))
    ptsExpand = []
    for _ in range(F):
        line = f.readline()
        num, *ptsIdx = line.strip().split()
        for i in range(int(num)):
            for j in range(int(num)):
                if i < j:
                    pts_i = pts[int(ptsIdx[i])]
                    pts_j = pts[int(ptsIdx[j])]
                    pts_bais = 1 / 3.0 * (pts_j - pts_i)
                    ptsExpand.append(pts_i + pts_bais)
                    ptsExpand.append(pts_i + 2 * pts_bais)
    f.close()
    return np.array(pts + ptsExpand) * vertex_scale


# def load_ply_model(model_path):
#   from plyfile import PlyData
#   ply = PlyData.read(model_path)
#   data = ply.elements[0].data
#   x = data['x']
#   y = data['y']
#   z = data['z']
#   return np.stack([x, y, z], axis=-1)


def load_ply_model(model_path, vertex_scale=1.0):
    model = load_ply(model_path)
    return model["pts"] * vertex_scale


def load_ply(path, vertex_scale=1.0):
    # https://github.com/thodan/sixd_toolkit/blob/master/pysixd/inout.py
    # bop_toolkit
    """Loads a 3D mesh model from a PLY file.

    :param path: Path to a PLY file.
    :return: The loaded model given by a dictionary with items:
    -' pts' (nx3 ndarray),
    - 'normals' (nx3 ndarray), optional
    - 'colors' (nx3 ndarray), optional
    - 'faces' (mx3 ndarray), optional.
    - 'texture_uv' (nx2 ndarray), optional
    - 'texture_uv_face' (mx6 ndarray), optional
    - 'texture_file' (string), optional
    """
    if _is_binary(path):
        f = open(path, "rb")
    else:
        f = open(path, "r")

    # Only triangular faces are supported.
    face_n_corners = 3

    n_pts = 0
    n_faces = 0
    pt_props = []
    face_props = []
    is_binary = False
    header_vertex_section = False
    header_face_section = False
    texture_file = None

    # Read the header.
    while True:

        # Strip the newline character(s)
        line = f.readline()
        if isinstance(line, str):
            line = line.rstrip("\n").rstrip("\r")
        else:
            line = str(line, "utf-8").rstrip("\n").rstrip("\r")

        if line.startswith("comment TextureFile"):
            texture_file = line.split()[-1]
        elif line.startswith("element vertex"):
            n_pts = int(line.split()[-1])
            header_vertex_section = True
            header_face_section = False
        elif line.startswith("element face"):
            n_faces = int(line.split()[-1])
            header_vertex_section = False
            header_face_section = True
        elif line.startswith("element"):  # Some other element.
            header_vertex_section = False
            header_face_section = False
        elif line.startswith("property") and header_vertex_section:
            # (name of the property, data type)
            prop_name = line.split()[-1]
            if prop_name == "s":
                prop_name = "texture_u"
            if prop_name == "t":
                prop_name = "texture_v"
            prop_type = line.split()[-2]
            pt_props.append((prop_name, prop_type))
        elif line.startswith("property list") and header_face_section:
            elems = line.split()
            if elems[-1] == "vertex_indices" or elems[-1] == "vertex_index":
                # (name of the property, data type)
                face_props.append(("n_corners", elems[2]))
                for i in range(face_n_corners):
                    face_props.append(("ind_" + str(i), elems[3]))
            elif elems[-1] == "texcoord":
                # (name of the property, data type)
                face_props.append(("texcoord", elems[2]))
                for i in range(face_n_corners * 2):
                    face_props.append(("texcoord_ind_" + str(i), elems[3]))
            else:
                logger.warning("Warning: Not supported face property: " + elems[-1])
        elif line.startswith("format"):
            if "binary" in line:
                is_binary = True
        elif line.startswith("end_header"):
            break

    # Prepare data structures.
    model = {}
    if texture_file is not None:
        model["texture_file"] = texture_file
    model["pts"] = np.zeros((n_pts, 3), np.float)
    if n_faces > 0:
        model["faces"] = np.zeros((n_faces, face_n_corners), np.float)

    # print(pt_props)
    pt_props_names = [p[0] for p in pt_props]
    face_props_names = [p[0] for p in face_props]
    # print(pt_props_names)

    is_normal = False
    if {"nx", "ny", "nz"}.issubset(set(pt_props_names)):
        is_normal = True
        model["normals"] = np.zeros((n_pts, 3), np.float)

    is_color = False
    if {"red", "green", "blue"}.issubset(set(pt_props_names)):
        is_color = True
        model["colors"] = np.zeros((n_pts, 3), np.float)

    is_texture_pt = False
    if {"texture_u", "texture_v"}.issubset(set(pt_props_names)):
        is_texture_pt = True
        model["texture_uv"] = np.zeros((n_pts, 2), np.float)

    is_texture_face = False
    if {"texcoord"}.issubset(set(face_props_names)):
        is_texture_face = True
        model["texture_uv_face"] = np.zeros((n_faces, 6), np.float)

    # Formats for the binary case.
    formats = {"float": ("f", 4), "double": ("d", 8), "int": ("i", 4), "uchar": ("B", 1)}

    # Load vertices.
    for pt_id in range(n_pts):
        prop_vals = {}
        load_props = ["x", "y", "z", "nx", "ny", "nz", "red", "green", "blue", "texture_u", "texture_v"]
        if is_binary:
            for prop in pt_props:
                format = formats[prop[1]]
                read_data = f.read(format[1])
                val = struct.unpack(format[0], read_data)[0]
                if prop[0] in load_props:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(pt_props):
                if prop[0] in load_props:
                    prop_vals[prop[0]] = elems[prop_id]

        model["pts"][pt_id, 0] = float(prop_vals["x"])
        model["pts"][pt_id, 1] = float(prop_vals["y"])
        model["pts"][pt_id, 2] = float(prop_vals["z"])

        if is_normal:
            model["normals"][pt_id, 0] = float(prop_vals["nx"])
            model["normals"][pt_id, 1] = float(prop_vals["ny"])
            model["normals"][pt_id, 2] = float(prop_vals["nz"])

        if is_color:
            model["colors"][pt_id, 0] = float(prop_vals["red"])
            model["colors"][pt_id, 1] = float(prop_vals["green"])
            model["colors"][pt_id, 2] = float(prop_vals["blue"])

        if is_texture_pt:
            model["texture_uv"][pt_id, 0] = float(prop_vals["texture_u"])
            model["texture_uv"][pt_id, 1] = float(prop_vals["texture_v"])

    # Load faces.
    for face_id in range(n_faces):
        prop_vals = {}
        if is_binary:
            for prop in face_props:
                format = formats[prop[1]]
                val = struct.unpack(format[0], f.read(format[1]))[0]
                if prop[0] == "n_corners":
                    if val != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                        # print("Number of face corners: " + str(val))
                        # exit(-1)
                elif prop[0] == "texcoord":
                    if val != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = val
        else:
            elems = f.readline().rstrip("\n").rstrip("\r").split()
            for prop_id, prop in enumerate(face_props):
                if prop[0] == "n_corners":
                    if int(elems[prop_id]) != face_n_corners:
                        raise ValueError("Only triangular faces are supported.")
                elif prop[0] == "texcoord":
                    if int(elems[prop_id]) != face_n_corners * 2:
                        raise ValueError("Wrong number of UV face coordinates.")
                else:
                    prop_vals[prop[0]] = elems[prop_id]

        model["faces"][face_id, 0] = int(prop_vals["ind_0"])
        model["faces"][face_id, 1] = int(prop_vals["ind_1"])
        model["faces"][face_id, 2] = int(prop_vals["ind_2"])

        if is_texture_face:
            for i in range(6):
                model["texture_uv_face"][face_id, i] = float(prop_vals["texcoord_ind_{}".format(i)])

    f.close()
    model["pts"] *= vertex_scale

    return model


def save_ply(path, model, extra_header_comments=None):
    """Saves a 3D mesh model to a PLY file.

    :param path: Path to a PLY file.
    :param model: 3D model given by a dictionary with items:
     - 'pts' (nx3 ndarray)
     - 'normals' (nx3 ndarray, optional)
     - 'colors' (nx3 ndarray, optional)
     - 'faces' (mx3 ndarray, optional)
     - 'texture_uv' (nx2 ndarray, optional)
     - 'texture_uv_face' (mx6 ndarray, optional)
     - 'texture_file' (string, optional)
    :param extra_header_comments: Extra header comment (optional).
    """
    pts = model["pts"]
    pts_colors = model["colors"] if "colors" in model.keys() else None
    pts_normals = model["normals"] if "normals" in model.keys() else None
    faces = model["faces"] if "faces" in model.keys() else None
    texture_uv = model["texture_uv"] if "texture_uv" in model.keys() else None
    texture_uv_face = model["texture_uv_face"] if "texture_uv_face" in model.keys() else None
    texture_file = model["texture_file"] if "texture_file" in model.keys() else None

    save_ply2(
        path, pts, pts_colors, pts_normals, faces, texture_uv, texture_uv_face, texture_file, extra_header_comments
    )


def save_ply2(
    path,
    pts,
    pts_colors=None,
    pts_normals=None,
    faces=None,
    texture_uv=None,
    texture_uv_face=None,
    texture_file=None,
    extra_header_comments=None,
):
    """Saves a 3D mesh model to a PLY file.

    :param path: Path to the resulting PLY file.
    :param pts: nx3 ndarray with vertices.
    :param pts_colors: nx3 ndarray with vertex colors (optional).
    :param pts_normals: nx3 ndarray with vertex normals (optional).
    :param faces: mx3 ndarray with mesh faces (optional).
    :param texture_uv: nx2 ndarray with per-vertex UV texture coordinates
      (optional).
    :param texture_uv_face: mx6 ndarray with per-face UV texture coordinates
      (optional).
    :param texture_file: Path to a texture image -- relative to the resulting
      PLY file (optional).
    :param extra_header_comments: Extra header comment (optional).
    """
    if pts_colors is not None:
        pts_colors = np.array(pts_colors)
        assert len(pts) == len(pts_colors)

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, "w")
    f.write(
        "ply\n"
        "format ascii 1.0\n"
        # 'format binary_little_endian 1.0\n'
    )

    if texture_file is not None:
        f.write("comment TextureFile {}\n".format(texture_file))

    if extra_header_comments is not None:
        for comment in extra_header_comments:
            f.write("comment {}\n".format(comment))

    f.write(
        "element vertex " + str(valid_pts_count) + "\n" "property float x\n" "property float y\n" "property float z\n"
    )
    if pts_normals is not None:
        f.write("property float nx\n" "property float ny\n" "property float nz\n")
    if pts_colors is not None:
        f.write("property uchar red\n" "property uchar green\n" "property uchar blue\n")
    if texture_uv is not None:
        f.write("property float texture_u\n" "property float texture_v\n")
    if faces is not None:
        f.write("element face " + str(len(faces)) + "\n" "property list uchar int vertex_indices\n")
    if texture_uv_face is not None:
        f.write("property list uchar float texcoord\n")
    f.write("end_header\n")

    format_float = "{:.4f}"
    format_2float = " ".join((format_float for _ in range(2)))
    format_3float = " ".join((format_float for _ in range(3)))
    format_int = "{:d}"
    format_3int = " ".join((format_int for _ in range(3)))

    # Save vertices.
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(format_3float.format(*pts[pt_id].astype(float)))
            if pts_normals is not None:
                f.write(" ")
                f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
            if pts_colors is not None:
                f.write(" ")
                f.write(format_3int.format(*pts_colors[pt_id].astype(int)))
            if texture_uv is not None:
                f.write(" ")
                f.write(format_2float.format(*texture_uv[pt_id].astype(float)))
            f.write("\n")

    # Save faces.
    if faces is not None:
        for face_id, face in enumerate(faces):
            line = " ".join(map(str, map(int, [len(face)] + list(face.squeeze()))))
            if texture_uv_face is not None:
                uv = texture_uv_face[face_id]
                line += " " + " ".join(map(str, [len(uv)] + map(float, list(uv.squeeze()))))
            f.write(line)
            f.write("\n")

    f.close()


def save_ply_float_color(path, pts, pts_colors=np.array([]), pts_normals=np.array([]), faces=np.array([])):
    # https://github.com/thodan/sixd_toolkit/blob/master/pysixd/inout.py
    """Saves a 3D mesh model to a PLY file.

    :param path: Path to the resulting PLY file.
    :param pts: nx3 ndarray
    :param pts_colors: nx3 ndarray
    :param pts_normals: nx3 ndarray
    :param faces: mx3 ndarray
    """
    pts_colors = np.array(pts_colors)
    if pts_colors.size != 0:
        assert len(pts) == len(pts_colors)

    valid_pts_count = 0
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            valid_pts_count += 1

    f = open(path, "w")
    f.write(
        "ply\n"
        "format ascii 1.0\n"
        # 'format binary_little_endian 1.0\n'
        "element vertex " + str(valid_pts_count) + "\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
    )
    if pts_normals.size != 0:
        f.write("property float nx\n" "property float ny\n" "property float nz\n")
    if pts_colors.size != 0:
        f.write("property float red\n" "property float green\n" "property float blue\n")
    if faces.size != 0:
        f.write("element face " + str(len(faces)) + "\n" "property list uchar int vertex_indices\n")
    f.write("end_header\n")

    format_float = "{:.4f}"
    format_3float = " ".join((format_float for _ in range(3)))
    format_int = "{:d}"
    # format_3int = " ".join((format_int for _ in range(3)))
    for pt_id, pt in enumerate(pts):
        if not np.isnan(np.sum(pt)):
            f.write(format_3float.format(*pts[pt_id].astype(float)))
            if pts_normals.size != 0:
                f.write(" ")
                f.write(format_3float.format(*pts_normals[pt_id].astype(float)))
            if pts_colors.size != 0:
                f.write(" ")
                f.write(format_3float.format(*pts_colors[pt_id].astype(float)))
            f.write("\n")
    for face in faces:
        f.write(" ".join(map(str, map(int, [len(face)] + list(face.squeeze())))) + " ")
        f.write("\n")
    f.close()


def obj_vtx(filename):
    """borrow from glumpy:

    https://github.com/glumpy/glumpy/blob/master/glumpy/data/__init__.py.
    """
    V = []  # vertex
    for lineno, line in enumerate(open(filename)):
        if line[0] == "#":
            continue
        values = line.strip().split(" ")
        code = values[0]
        values = values[1:]
        # vertex (v)
        if code == "v":
            V.append([float(x) for x in values])
    # Building the vertices
    V = np.array(V, dtype=np.float32)
    return V


if __name__ == "__main__":
    # test load (binary/text) ply model
    train_model_dir = "data/BOP_DATASETS/lm_full/models"
    val_model_dir = "data/BOP_DATASETS/lm_full/models_eval"
    obj_id = 1
    model_train = load_ply(osp.join(train_model_dir, "obj_{:06d}.ply".format(obj_id)))
    print("train", model_train["pts"].shape, model_train["pts"].min(0), model_train["pts"].max(0))
    model_val = load_ply(osp.join(val_model_dir, "obj_{:06d}.ply".format(obj_id)))
    print("val", model_val["pts"].shape, model_val["pts"].min(0), model_val["pts"].max(0))
    """
    # test PlyData and load_ply
    model_dir = "data/LINEMOD_6D/models"
    cls_name = "ape"
    model_path = osp.join(model_dir, "{0}/{0}_sphere.ply".format(cls_name))
    model = load_ply(model_path)
    print(model.keys())
    model_indices = model["faces"].astype(np.int32)
    # print(model_indices)

    from plyfile import PlyData

    data = PlyData.read(model_path)
    print(data["vertex"])
    vertex_indices = data["face"]["vertex_indices"]
    # print(vertex_indices)
    indices = np.asarray(list(vertex_indices), np.uint32)
    # print(indices)

    print(np.array_equal(model_indices, indices))
    """
