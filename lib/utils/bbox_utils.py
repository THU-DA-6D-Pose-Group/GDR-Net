import warnings
import numpy as np
import copy

"""Box manipulation functions. The internal Detectron box format is
[x1, y1, x2, y2] where (x1, y1) specify the top-left box corner and (x2, y2)
specify the bottom-right box corner. Boxes from external sources, e.g.,
datasets, may be in other formats (such as [x, y, w, h]) and require conversion.
This module uses a convention that may seem strange at first: the width of a box
is computed as x2 - x1 + 1 (likewise for height). The "+ 1" dates back to old
object detection days when the coordinates were integer pixel indices, rather
than floating point coordinates in a subpixel coordinate frame. A box with x2 =
x1 and y2 = y1 was taken to include a single pixel, having a width of 1, and
hence requiring the "+ 1". Now, most datasets will likely provide boxes with
floating point coordinates and the width should be more reasonably computed as
x2 - x1.
In practice, as long as a model is trained and tested with a consistent
convention either decision seems to be ok (at least in our experience on COCO).
Since we have a long history of training models with the "+ 1" convention, we
are reluctant to change it even if our modern tastes prefer not to use it.
"""
# NOTE: this file uses +1 convention for box
# i.e., the bottom right corner is included
# w = x2 - x1 + 1
# h = y2 - y1 + 1
def clip_xyxy_to_im(xyxy, height, width):
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        _xyxy = copy.deepcopy(xyxy)
        for i in range(4):
            if i == 0 or i == 2:
                _xyxy[i] = min(max(_xyxy[i], 0), width - 1)
            elif i == 1 or i == 3:
                _xyxy[i] = min(max(_xyxy[i], 0), height - 1)
            else:
                raise RuntimeError("bad i: {}".format(i))
        return _xyxy
    elif isinstance(xyxy, np.ndarray):
        _xyxy = xyxy.copy()
        if len(xyxy.shape) == 1:
            for i in range(4):
                if i == 0 or i == 2:
                    _xyxy[i] = np.clip(_xyxy[i], 0, width - 1)
                else:
                    _xyxy[i] = np.clip(_xyxy[i], 0, height - 1)
            return _xyxy
        else:
            # Multiple boxes given as a 2D ndarray
            for i in range(4):
                if i == 0 or i == 2:
                    _xyxy[:, i] = np.clip(_xyxy[:, i], 0, width - 1)
                else:
                    _xyxy[:, i] = np.clip(_xyxy[:, i], 0, height - 1)
            return _xyxy
    else:
        raise TypeError("Argument xyxy must be a list, tuple, or numpy array.")


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h) if isinstance(xyxy, tuple) else [x1, y1, w, h]
    elif isinstance(xyxy, np.ndarray):
        if len(xyxy.shape) == 1:
            return np.hstack((xyxy[0:2], xyxy[2:4] - xyxy[0:2] + 1))
        else:
            # Multiple boxes given as a 2D ndarray
            return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError("Argument xyxy must be a list, tuple, or numpy array.")


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0.0, xywh[2] - 1.0)
        y2 = y1 + np.maximum(0.0, xywh[3] - 1.0)
        return (x1, y1, x2, y2) if isinstance(xywh, tuple) else [x1, y1, x2, y2]
    elif isinstance(xywh, np.ndarray):
        if len(xywh.shape) == 1:
            return np.hstack((xywh[0:2], xywh[0:2] + np.maximum(0, xywh[2:4] - 1)))
        else:
            # Multiple boxes given as a 2D ndarray
            return np.hstack((xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1)))
    else:
        raise TypeError("Argument xywh must be a list, tuple, or numpy array.")


def boxes_area(boxes):
    """Compute the area of an array of boxes."""
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    areas = w * h

    neg_area_idx = np.where(areas < 0)[0]
    if neg_area_idx.size:
        warnings.warn("Negative areas founds: %d" % neg_area_idx.size, RuntimeWarning)
    # TODO proper warm up and learning rate may reduce the prob of assertion fail
    # assert np.all(areas >= 0), 'Negative areas founds'
    return areas, neg_area_idx


def bbox_center(bbox, fmt="xyxy"):
    """
    fmt: xyxy or xywh
    -------
    return:
        x_ctr, y_ctr
    """
    if fmt == "xyxy":
        x1, y1, x2, y2 = bbox
        return 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    elif fmt == "xywh":
        x1, y1, w, h = bbox
        return x1 + 0.5 * (w - 1), y1 + 0.5 * (h - 1)
