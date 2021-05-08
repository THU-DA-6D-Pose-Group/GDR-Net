import os
import sys
import os.path as osp

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mmcv import color_val
from mmcv.image import imread, imwrite

cur_dir = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(cur_dir, "../.."))

from lib.pysixd import misc as misc_6d
from lib.utils import logger, mask_utils
from lib.utils.fs import execute_only_once, mkdir_p
from lib.utils.mask_utils import mask2bbox_xyxy
from lib.utils.utils import dprint

from .colormap import colormap

plt.rcParams["pdf.fonttype"] = 42  # For editing in Adobe Illustrator

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def grid_show(ims, titles=None, row=1, col=3, dpi=200, save_path=None, title_fontsize=5, show=True):
    if row * col < len(ims):
        print("_____________row*col < len(ims)___________")
        col = int(np.ceil(len(ims) / row))
    if titles is not None:
        assert len(ims) == len(titles), "{} != {}".format(len(ims), len(titles))
    fig = plt.figure(dpi=dpi, figsize=plt.figaspect(row / float(col)))
    k = 0
    for i in range(row):
        for j in range(col):
            if k >= len(ims):
                break
            plt.subplot(row, col, k + 1)
            plt.axis("off")
            plt.imshow(ims[k])
            if titles is not None:
                # plt.title(titles[k], size=title_fontsize)
                plt.text(
                    0.5,
                    1.08,
                    titles[k],
                    horizontalalignment="center",
                    fontsize=title_fontsize,
                    transform=plt.gca().transAxes,
                )
            k += 1

    # plt.tight_layout()
    if show:
        plt.show()
    else:
        if save_path is not None:
            mkdir_p(osp.dirname(save_path))
            plt.savefig(save_path)
    return fig


def heatmap(input, min=None, max=None, to_255=False, to_rgb=False, colormap=cv2.COLORMAP_JET):
    """Returns a BGR heatmap representation."""
    if min is None:
        min = np.amin(input)
    if max is None:
        max = np.amax(input)
    rescaled = 255 * ((input - min) / (max - min + 0.001))

    final = cv2.applyColorMap(rescaled.astype(np.uint8), colormap)
    if to_rgb:
        final = final[:, :, [2, 1, 0]]
    if to_255:
        return final.astype(np.uint8)
    else:
        return final.astype(np.float32) / 255.0


def vis_bbox_opencv(img, bbox, thick=1, fmt="xywh", bbox_color="green"):
    """Visualizes a bounding box."""
    bbox = np.array(bbox + 0.5).astype(np.int)
    if fmt == "xywh":
        (x1, y1, w, h) = bbox
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x1 + w - 1), int(y1 + h - 1)
    else:
        x1, y1, x2, y2 = bbox
    _img = img.copy()
    bbox_color = color_val(bbox_color)
    cv2.rectangle(_img, (x1, y1), (x2, y2), bbox_color, thickness=thick)
    return _img


def vis_image_mask_cv2(img, mask, color=None):
    # import pycocotools.mask as cocomask
    if color is None:
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
    else:
        color_mask = np.array(color_val(color), dtype=np.uint8)
        # print(color_mask, type(color_mask))
    mask = mask.astype(np.bool)
    img_show = img.copy()
    img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
    return img_show


def vis_image_mask_bbox_cv2(
    img, masks, bboxes=None, labels=None, font_scale=0.5, text_color="green", font_thickness=2, box_thickness=1
):
    """
    bboxes: xyxy
    """
    # import pycocotools.mask as cocomask
    text_color = color_val(text_color)
    img_show = img.copy()
    for i, mask in enumerate(masks):
        color_mask = np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        mask = mask.astype(np.bool)
        img_show[mask] = img_show[mask] * 0.5 + color_mask * 0.5
        if bboxes is None:
            x1, y1, x2, y2 = mask2bbox_xyxy(mask)
        else:
            bbox = bboxes[i].astype(np.int32)
            x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(img_show, (x1, y1), (x2, y2), _GREEN, thickness=box_thickness)
        if labels is not None:
            label_text = labels[i]
            cv2.putText(
                img_show,
                label_text,
                (x1, max(y1 - 2, 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
            )
    return img_show


def vis_image_bboxes_cv2(
    img,
    bboxes,
    labels=None,
    font_scale=0.5,
    text_color="green",
    font_thickness=2,
    box_thickness=2,
    box_color=_GREEN,
    draw_center=False,
):
    """
    bboxes: xyxy
    """
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    text_color = color_val(text_color)
    box_color = tuple(int(_c) for _c in color_val(box_color))

    img_show = img.copy()
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox[:4]
        cv2.rectangle(img_show, (x1, y1), (x2, y2), box_color, thickness=box_thickness)
        if draw_center:
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            img_show = cv2.circle(img_show, center, radius=box_thickness, color=box_color, thickness=-1)
        if labels is not None:
            label_text = labels[i]
            cv2.putText(
                img_show, label_text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness
            )
    return img_show


def vis_image_mask_plt(im, mask, dpi=200, color=None, outfile=None, show=True):
    if color is None:
        color_list = colormap(rgb=True) / 255
        mask_color_id = 0
        color_mask = color_list[mask_color_id % len(color_list), 0:3]
    else:
        color_mask = color_val(color)
    # cmap = plt.get_cmap('rainbow')

    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    fig.add_axes(ax)
    ax.imshow(im[:, :, [2, 1, 0]])

    # show mask
    img = np.ones(im.shape)

    w_ratio = 0.4
    for c in range(3):
        color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
    for c in range(3):
        img[:, :, c] = color_mask[c]
    e = mask

    _, contour, hier = cv2.findContours(e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for c in contour:
        polygon = Polygon(c.reshape((-1, 2)), fill=True, facecolor=color_mask, edgecolor="w", linewidth=1.2, alpha=0.5)
        ax.add_patch(polygon)
    if outfile is not None:
        mkdir_p(os.path.dirname(outfile))
        fig.savefig(outfile, dpi=dpi)
        plt.close("all")
    if show:
        plt.show()


def save_image_plt(data, filename):
    sizes = np.shape(data)
    height = sizes[0]
    width = sizes[1]
    fig = plt.figure(frameon=False, figsize=(1, 1))
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(data)
    plt.savefig(filename, dpi=height)
    plt.close()


def save_heatmap(data, fn, cm="hot"):
    sizes = np.shape(data)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure(frameon=False)
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(data, cmap=cm, interpolation="nearest")
    plt.savefig(fn, dpi=height)
    plt.close()


def arrowed_spines(fig, ax):
    # used before plt.show()
    # get arrowed axes
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ["bottom", "right", "top", "left"]:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([])  # labels
    plt.yticks([])
    ax.xaxis.set_ticks_position("none")  # tick markers
    ax.yaxis.set_ticks_position("none")

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1.0 / 20.0 * (ymax - ymin)
    hl = 1.0 / 20.0 * (xmax - xmin)
    lw = 1.0  # axis line width
    ohg = 0.3  # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    ax.arrow(
        xmin,
        0,
        xmax - xmin,
        0.0,
        fc="k",
        ec="k",
        lw=lw,
        head_width=hw,
        head_length=hl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )

    ax.arrow(
        0,
        ymin,
        0.0,
        ymax - ymin,
        fc="k",
        ec="k",
        lw=lw,
        head_width=yhw,
        head_length=yhl,
        overhang=ohg,
        length_includes_head=True,
        clip_on=False,
    )


def imshow(img, win_name="", wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    cv2.waitKey(wait_time)


def imshow_bboxes(
    img, bboxes, colors="green", top_k=-1, thickness=1, show=True, win_name="", wait_time=0, out_file=None
):
    """Draw bboxes on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (list or ndarray): A list of ndarray of shape (k, 4).
        colors (list[str or tuple or Color]): A list of colors.
        top_k (int): Plot the first k bboxes only if set positive.
        thickness (int): Thickness of lines.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str, optional): The filename to write the image.
    """
    img = imread(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]
    if not isinstance(colors, list):
        colors = [colors for _ in range(len(bboxes))]
    colors = [color_val(c) for c in colors]
    assert len(bboxes) == len(colors)

    for i, _bboxes in enumerate(bboxes):
        _bboxes = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes[j, 0], _bboxes[j, 1])
            right_bottom = (_bboxes[j, 2], _bboxes[j, 3])
            cv2.rectangle(img, left_top, right_bottom, colors[i], thickness=thickness)

    if show:
        imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes(
    img,
    bboxes,
    labels,
    class_names=None,
    score_thr=0,
    bbox_color="green",
    text_color="green",
    thickness=1,
    font_scale=0.5,
    show=True,
    win_name="",
    wait_time=0,
    out_file=None,
    vis_tool="matplotlib",
):
    """Draw bboxes and class labels (with scores) on an image.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
    """
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[label] if class_names is not None else "cls {}".format(label)
        if len(bbox) > 4:
            label_text += "|{:.02f}".format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2), cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        if vis_tool == "matplotlib":
            fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
            tmp = fig.add_subplot(1, 1, 1)
            tmp.set_title("{}".format(win_name))
            plt.axis("off")
            plt.imshow(img[:, :, [2, 1, 0]])
            plt.show()
        else:  # use 'mmcv'
            imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)


def imshow_det_bboxes_poses(
    img,
    bboxes,
    labels,
    class_names=None,
    score_thr=0,
    bbox_color="green",
    text_color="green",
    thickness=1,
    font_scale=0.5,
    show=True,
    win_name="",
    wait_time=0,
    out_file=None,
    poses=None,
    corners_3d=None,
    dataste_name=None,
    renderer=None,
    K=None,
    vis_tool="matplotlib",
):
    """Draw bboxes and class labels (with scores) on an image. Render the
    contours of poses to image. (or the 3d bounding box)

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes. 0-based
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
        out_file (str or None): The filename to write the image.
        ------
        poses:
        corners_3d: dict of 3d corners(un-transformed), key is cls_name
        dataset_name: camera intrinsic parameter
        renderer:
        K: camera intrinsic
    """
    # logger.info('poses: {}'.format(poses))
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = imread(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)

    for bbox, label in zip(bboxes, labels):
        # pose
        if poses is not None:
            if poses[label]:
                pose = poses[label][0]  # TODO: handle multiple poses
                bgr, depth = renderer.render(label, pose[:, :3], pose[:, 3], r_type="mat")
                # img = img - bgr
                pose_mask = np.zeros(depth.shape)
                pose_mask[depth != 0] = 1
                edges_3 = mask_utils.get_edge(pose_mask, bw=3)
                edges_3[:, :, [0, 1]] = 0  # red
                img[edges_3 != 0] = 255
                cls_name = class_names[label]
                corners_2d, _ = misc_6d.points_to_2D(corners_3d[cls_name], pose[:, :3], pose[:, 3], K)
                img = misc_6d.draw_projected_box3d(img, corners_2d, thickness=thickness)

        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness=thickness)
        label_text = class_names[label] if class_names is not None else "cls {}".format(label)
        if len(bbox) > 4:
            label_text += "|{:.02f}".format(bbox[-1])
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2), cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)

    if show:
        if vis_tool == "matplotlib":
            fig = plt.figure(frameon=False, figsize=(8, 6), dpi=100)
            tmp = fig.add_subplot(1, 1, 1)
            tmp.set_title("{}".format(win_name))
            plt.axis("off")
            plt.imshow(img[:, :, [2, 1, 0]])
            plt.show()
        else:  # use 'mmcv'
            imshow(img, win_name, wait_time)
    if out_file is not None:
        imwrite(img, out_file)
    return img
