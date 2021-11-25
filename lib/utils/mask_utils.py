# NOTE: use +1 convention for box
# w=x2-x1+1
# h=y2-y1+1
import os.path as osp
import sys
from itertools import groupby

import numpy as np
import numpy.random as npr
import pycocotools.mask as cocomask
from PIL import Image, ImageFile
from skimage.morphology import binary_dilation as _binary_dilation
from skimage.morphology import binary_erosion as _binary_erosion
from skimage.morphology import disk
import cv2


cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../.."))


def get_edge(mask, bw=1, out_channel=3):
    if len(mask.shape) > 2:
        channel = mask.shape[2]
    else:
        channel = 1
    if channel == 3:
        mask = mask[:, :, 0] != 0
    edges = np.zeros(mask.shape[:2])
    edges[:-bw, :] = np.logical_and(mask[:-bw, :] == 1, mask[bw:, :] == 0) + edges[:-bw, :]
    edges[bw:, :] = np.logical_and(mask[bw:, :] == 1, mask[:-bw, :] == 0) + edges[bw:, :]
    edges[:, :-bw] = np.logical_and(mask[:, :-bw] == 1, mask[:, bw:] == 0) + edges[:, :-bw]
    edges[:, bw:] = np.logical_and(mask[:, bw:] == 1, mask[:, :-bw] == 0) + edges[:, bw:]
    if out_channel == 3:
        edges = np.dstack((edges, edges, edges))
    return edges


def mask2bbox_xyxy(mask):
    """NOTE: the bottom right point is included"""
    ys, xs = np.nonzero(mask)[:2]
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    return [bb_tl[0], bb_tl[1], bb_br[0], bb_br[1]]


def mask2bbox_xywh(mask):
    ys, xs = np.nonzero(mask)[:2]
    bb_tl = [xs.min(), ys.min()]
    bb_br = [xs.max(), ys.max()]
    return [bb_tl[0], bb_tl[1], bb_br[0] - bb_tl[0] + 1, bb_br[1] - bb_tl[1] + 1]


def binary_mask_to_rle(mask, compressed=True):
    mask = mask.astype(np.uint8)
    if compressed:
        rle = cocomask.encode(np.asfortranarray(mask))
        rle["counts"] = rle["counts"].decode("ascii")
    else:
        rle = {"counts": [], "size": list(mask.shape)}
        counts = rle.get("counts")
        for i, (value, elements) in enumerate(groupby(mask.ravel(order="F"))):  # noqa: E501
            if i == 0 and value == 1:
                counts.append(0)
            counts.append(len(list(elements)))
    return rle


def binary_mask_to_polygons(mask):
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/visualizer.py#L108
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]
    return res, has_holes


def mask_has_holes(mask):
    # https://github.com/facebookresearch/detectron2/blob/master/detectron2/utils/visualizer.py#L100
    _, has_holes = binary_mask_to_polygons(mask)
    return has_holes


def rle2mask(rle, height, width):
    if "counts" in rle and isinstance(rle["counts"], list):
        # if compact RLE, ignore this conversion
        # Magic RLE format handling painfully discovered by looking at the
        # COCO API showAnns function.
        rle = cocomask.frPyObjects(rle, height, width)
    mask = cocomask.decode(rle)
    return mask


def segmToRLE(segm, h, w):
    """Convert segmentation which can be polygons, uncompressed RLE to RLE.

    :return: binary mask (numpy 2D array)
    """
    if isinstance(segm, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = cocomask.frPyObjects(segm, h, w)
        rle = cocomask.merge(rles)
    elif isinstance(segm["counts"], list):
        # uncompressed RLE
        rle = cocomask.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def cocosegm2mask(segm, h, w):
    rle = segmToRLE(segm, h, w)
    mask = rle2mask(rle, h, w)
    return mask


def mask_dilate(mask_origin, thickness=10):
    """from DeepIM.

    :param mask_origin: mask to be dilated
    :param thickness: the thickness of the margin
    :return:
    """
    mask_expand = np.copy(mask_origin)
    h, w = mask_origin.shape
    for up_down in [0, thickness]:
        for left_right in [0, thickness]:
            mask_expand[up_down : (h - thickness + up_down), left_right : (w - thickness + left_right)] += mask_origin[
                thickness - up_down : (h - up_down), thickness - left_right : (w - left_right)
            ]

    mask_expand[mask_expand >= 1] = 1
    return mask_expand


def random_mask_dilate(mask_origin, max_thickness=10):
    """from DeepIM.

    :param pairdb:
    :param config:
    :param phase:
    :param random_k:
    :return:
    """
    mask_expand = np.copy(mask_origin)
    h, w = mask_origin.shape
    for ud in [0, 1]:
        thickness = np.random.randint(max_thickness)
        for lr in [0, 1]:
            up_down = ud * thickness
            left_right = lr * thickness
            mask_expand[up_down : (h - thickness + up_down), left_right : (w - thickness + left_right)] += mask_origin[
                (thickness - up_down) : (h - up_down), (thickness - left_right) : (w - left_right)
            ]

    mask_expand[mask_expand >= 1] = 1
    return mask_expand


def binary_dilation(x, radius=3):
    """Return fast binary morphological dilation of an image.

    # https://github.com/zsdonghao/tensorlayer2/blob/master/tensorlayer/prepro.py
    see `skimage.morphology.binary_dilation
        <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_dilation>`__.
    Parameters
    -----------
    x : 2D array
        A binary image.
    radius : int
        For the radius of mask.
    Returns
    -------
    numpy.array
        A processed binary image.
    """
    mask = disk(radius)
    x = _binary_dilation(x, selem=mask)

    return x


def random_binary_dilation(x, radious=3):
    r = np.random.randint(radious)
    return binary_dilation(x, r)


def binary_erosion(x, radius=3):
    """Return binary morphological erosion of an image, see
    `skimage.morphology.binary_erosion.

        <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_erosion>`__.
    Parameters
    -----------
    x : 2D array
        A binary image.
    radius : int
        For the radius of mask.
    Returns
    -------
    numpy.array
        A processed binary image.
    """
    mask = disk(radius)
    x = _binary_erosion(x, selem=mask)
    return x


def random_binary_erosion(x, radious=3):
    r = npr.randint(radious)
    return binary_erosion(x, r)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from lib.vis_utils.image import grid_show

    poly = [
        [
            423.0,
            306.5,
            406.5,
            277.0,
            400.0,
            271.5,
            389.5,
            277.0,
            387.5,
            292.0,
            384.5,
            295.0,
            374.5,
            220.0,
            378.5,
            210.0,
            391.0,
            200.5,
            404.0,
            199.5,
            414.0,
            203.5,
            425.5,
            221.0,
            438.5,
            297.0,
            423.0,
            306.5,
        ],
        [100, 100, 200, 100, 200, 200, 100, 200],
    ]
    width = 640
    height = 480
    size = width, height
    a = cocosegm2mask(poly, height, width)
    a = a.astype("uint8")

    b = random_mask_dilate(a, max_thickness=10)
    b_binary_dilation = random_binary_dilation(a, 10)
    b_binary_erosion = random_binary_erosion(a, 10)
    c = a * 127 + b * 128
    c_1 = a * 127 + b_binary_dilation * 128
    c_2 = a * 127 + b_binary_erosion * 128

    show_ims = [c, c_1, c_2]
    show_titles = ["random_dilate", "random_binary_dilation", "random_binary_erosion"]
    grid_show(show_ims, show_titles, row=1, col=3)
