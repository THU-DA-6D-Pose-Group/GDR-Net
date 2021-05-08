from detectron2.utils.colormap import colormap
from detectron2.utils.visualizer import ColorMode, Visualizer

__all__ = ["MyVisualizer", "_RED", "_GREEN", "_BLUE", "_GREY"]

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)
_GREEN = (0, 1.0, 0)
_GREY = (0.5, 0.5, 0.5)
_BLUE = (0, 0, 1.0)
_KEYPOINT_THRESHOLD = 0.05


class MyVisualizer(Visualizer):
    def __init__(
        self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE, keypoint_type=None, keypoint_color=_RED
    ):
        """keypoint_type (2d): bbox3d_and_center | fps8_and_center | None."""
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.keypoint_type = keypoint_type
        self.keypoint_color = keypoint_color

    def draw_and_connect_keypoints(self, keypoints):
        """Draws keypoints of an instance and follows the rules for keypoint
        connections to draw lines between appropriate keypoints.

        This follows color heuristics for
        line color.
        Args:
            keypoints (Tensor): a tensor of shape (K, 2), where K is the number of keypoints
                and the last dimension corresponds to (x, y).
        Returns:
            output (VisImage): image object with visualizations.
        """
        if self.keypoint_type is None:
            return super().draw_and_connect_keypoints(keypoints)

        if True:  # from center to other points
            # assume keypoints (K, 2), the last row is object center
            cx, cy = keypoints[-1, :2]
            self.draw_circle((cx, cy), color=self.keypoint_color)

            colors = colormap(rgb=True, maximum=1)
            for idx in range(len(keypoints) - 1):
                # draw keypoint
                x, y = keypoints[idx]
                self.draw_circle((x, y), color=self.keypoint_color)  # default red (1,0,0)
                # connect keypoints
                line_color = colors[idx % len(colors)]
                self.draw_line([cx, x], [cy, y], color=line_color)
        else:
            # assume keypoints (K, 2), the last row is object center
            colors = colormap(rgb=True, maximum=1)
            for idx in range(len(keypoints) - 1):
                # draw keypoint
                x, y = keypoints[idx, :2]
                xe, ye = keypoints[idx + 1, :2]
                self.draw_circle((x, y), color=self.keypoint_color)  # default red (1,0,0)
                # connect keypoints
                line_color = colors[idx % len(colors)]
                # x_data, y_data
                self.draw_line([x, xe], [y, ye], color=line_color)
            x, y = keypoints[-1, :2]
            self.draw_circle((x, y), color=self.keypoint_color)  # default red (1,0,0)

        return self.output

    def draw_bbox3d_and_center(
        self,
        keypoints,
        top_color=_RED,
        middle_color=None,
        bottom_color=(0.5, 0.5, 0.5),
        linewidth=None,
        draw_points=False,
        draw_center=False,
    ):
        """1 -------- 0.

           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
          8: center
        Args:
            keypoints (Tensor): a tensor of shape (K, 2), where K is the number of keypoints
                and the last dimension corresponds to (x, y).
        Returns:
            output (VisImage): image object with visualizations.
        """
        assert keypoints.shape[-1] == 2, keypoints.shape
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        colors = colormap(rgb=True, maximum=1)

        # connect keypoints
        for idx, edge in enumerate([(4, 5), (5, 6), (6, 7), (7, 4)]):
            i, j = edge
            x1, y1 = keypoints[i, :2]
            x2, y2 = keypoints[j, :2]
            if bottom_color is None:
                _bottom_color = colors[idx % len(colors)]
            else:
                _bottom_color = bottom_color
            self.draw_line([x1, x2], [y1, y2], color=_bottom_color, linewidth=linewidth)

        for idx, edge in enumerate([(0, 4), (3, 7), (2, 6), (1, 5)]):
            i, j = edge
            x1, y1 = keypoints[i, :2]
            x2, y2 = keypoints[j, :2]
            if middle_color is None:
                edge_color = colors[idx % len(colors)]
            else:
                edge_color = middle_color
            self.draw_line([x1, x2], [y1, y2], color=edge_color, linewidth=linewidth)

        for edge in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            i, j = edge
            x1, y1 = keypoints[i, :2]
            x2, y2 = keypoints[j, :2]
            self.draw_line([x1, x2], [y1, y2], color=top_color, linewidth=linewidth)

        if draw_center:
            # assume keypoints (K, 2), the last row is object center
            cx, cy = keypoints[-1, :2]
            # draw center with edge color
            self.draw_circle((cx, cy), color=top_color)

        if draw_points:
            for idx in range(len(keypoints) - 1):
                # draw keypoint
                x, y = keypoints[idx, :2]
                point_color = colors[idx % len(colors)]
                self.draw_circle((x, y), color=point_color)
        return self.output

    def draw_axis3d_and_center(
        self,
        keypoints,
        up_color=_RED,
        right_color=_GREEN,
        front_color=_BLUE,
        center_color=_GREY,
        linewidth=None,
        draw_points=False,
        draw_center=False,
    ):
        """
            2
            |
            3 ---1
           /
          0
        Args:
            keypoints (Tensor): a tensor of shape (4, 2), where K is the number of keypoints
                and the last dimension corresponds to (x, y).
        Returns:
            output (VisImage): image object with visualizations.
        """
        assert keypoints.shape[-1] == 2, keypoints.shape
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        colors = colormap(rgb=True, maximum=1)

        # connect keypoints
        for idx, edge in enumerate([(3, 0), (3, 1), (3, 2)]):
            i, j = edge
            x1, y1 = keypoints[i, :2]
            x2, y2 = keypoints[j, :2]
            if j == 0:
                color = front_color
            elif j == 1:
                color = right_color
            else:
                color = up_color
            self.draw_line([x1, x2], [y1, y2], color=color, linewidth=linewidth)

        if draw_center:
            # assume keypoints (K, 2), the last row is object center
            cx, cy = keypoints[-1, :2]
            # draw center with center color
            self.draw_circle((cx, cy), color=center_color)

        if draw_points:
            for idx in range(len(keypoints) - 1):
                # draw keypoint
                x, y = keypoints[idx, :2]
                point_color = colors[idx % len(colors)]
                self.draw_circle((x, y), color=point_color)
        return self.output
