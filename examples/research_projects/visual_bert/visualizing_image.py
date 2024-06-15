"""
coding=utf-8
Copyright 2018, Antonio Mendoza Hao Tan, Mohit Bansal
Adapted From Facebook Inc, Detectron2

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.import copy
"""

import colorsys
import io

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from utils import img_tensorize


_SMALL_OBJ = 1000


class SingleImageViz:
    def __init__(
        self,
        img,
        scale=1.2,
        edgecolor="g",
        alpha=0.5,
        linestyle="-",
        saveas="test_out.jpg",
        rgb=True,
        pynb=False,
        id2obj=None,
        id2attr=None,
        pad=0.7,
    ):
        """
        img: an RGB image of shape (H, W, 3).
        """
        if isinstance(img, torch.Tensor):
            img = img.numpy().astype("np.uint8")
        if isinstance(img, str):
            img = img_tensorize(img)
        assert isinstance(img, np.ndarray)

        width, height = img.shape[1], img.shape[0]
        fig = mplfigure.Figure(frameon=False)
        dpi = fig.get_dpi()
        width_in = (width * scale + 1e-2) / dpi
        height_in = (height * scale + 1e-2) / dpi
        fig.set_size_inches(width_in, height_in)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        ax.set_xlim(0.0, width)
        ax.set_ylim(height)

        self.saveas = saveas
        self.rgb = rgb
        self.pynb = pynb
        self.img = img
        self.edgecolor = edgecolor
        self.alpha = 0.5
        self.linestyle = linestyle
        self.font_size = int(np.sqrt(min(height, width)) * scale // 3)
        self.width = width
        self.height = height
        self.scale = scale
        self.fig = fig
        self.ax = ax
        self.pad = pad
        self.id2obj = id2obj
        self.id2attr = id2attr
        self.canvas = FigureCanvasAgg(fig)

    def add_box(self, box, color=None):
        if color is None:
            color = self.edgecolor
        (x0, y0, x1, y1) = box
        width = x1 - x0
        height = y1 - y0
        self.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=color,
                linewidth=self.font_size // 3,
                alpha=self.alpha,
                linestyle=self.linestyle,
            )
        )

    def draw_boxes(self, boxes, obj_ids=None, obj_scores=None, attr_ids=None, attr_scores=None):
        if len(boxes.shape) > 2:
            boxes = boxes[0]
        if len(obj_ids.shape) > 1:
            obj_ids = obj_ids[0]
        if len(obj_scores.shape) > 1:
            obj_scores = obj_scores[0]
        if len(attr_ids.shape) > 1:
            attr_ids = attr_ids[0]
        if len(attr_scores.shape) > 1:
            attr_scores = attr_scores[0]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        if isinstance(boxes, list):
            boxes = np.array(boxes)
        assert isinstance(boxes, np.ndarray)
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        sorted_idxs = np.argsort(-areas).tolist()
        boxes = boxes[sorted_idxs] if boxes is not None else None
        obj_ids = obj_ids[sorted_idxs] if obj_ids is not None else None
        obj_scores = obj_scores[sorted_idxs] if obj_scores is not None else None
        attr_ids = attr_ids[sorted_idxs] if attr_ids is not None else None
        attr_scores = attr_scores[sorted_idxs] if attr_scores is not None else None

        assigned_colors = [self._random_color(maximum=1) for _ in range(len(boxes))]
        assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
        if obj_ids is not None:
            labels = self._create_text_labels_attr(obj_ids, obj_scores, attr_ids, attr_scores)
            for i in range(len(boxes)):
                color = assigned_colors[i]
                self.add_box(boxes[i], color)
                self.draw_labels(labels[i], boxes[i], color)

    def draw_labels(self, label, box, color):
        x0, y0, x1, y1 = box
        text_pos = (x0, y0)
        instance_area = (y1 - y0) * (x1 - x0)
        small = _SMALL_OBJ * self.scale
        if instance_area < small or y1 - y0 < 40 * self.scale:
            if y1 >= self.height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)

        height_ratio = (y1 - y0) / np.sqrt(self.height * self.width)
        lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
        font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
        font_size *= 0.75 * self.font_size

        self.draw_text(
            text=label,
            position=text_pos,
            color=lighter_color,
        )

    def draw_text(
        self,
        text,
        position,
        color="g",
        ha="left",
    ):
        rotation = 0
        font_size = self.font_size
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        bbox = {
            "facecolor": "black",
            "alpha": self.alpha,
            "pad": self.pad,
            "edgecolor": "none",
        }
        x, y = position
        self.ax.text(
            x,
            y,
            text,
            size=font_size * self.scale,
            family="sans-serif",
            bbox=bbox,
            verticalalignment="top",
            horizontalalignment=ha,
            color=color,
            zorder=10,
            rotation=rotation,
        )

    def save(self, saveas=None):
        if saveas is None:
            saveas = self.saveas
        if saveas.lower().endswith(".jpg") or saveas.lower().endswith(".png"):
            cv2.imwrite(
                saveas,
                self._get_buffer()[:, :, ::-1],
            )
        else:
            self.fig.savefig(saveas)

    def _create_text_labels_attr(self, classes, scores, attr_classes, attr_scores):
        labels = [self.id2obj[i] for i in classes]
        attr_labels = [self.id2attr[i] for i in attr_classes]
        labels = [
            f"{label} {score:.2f} {attr} {attr_score:.2f}"
            for label, score, attr, attr_score in zip(labels, scores, attr_labels, attr_scores)
        ]
        return labels

    def _create_text_labels(self, classes, scores):
        labels = [self.id2obj[i] for i in classes]
        if scores is not None:
            if labels is None:
                labels = ["{:.0f}%".format(s * 100) for s in scores]
            else:
                labels = ["{} {:.0f}%".format(li, s * 100) for li, s in zip(labels, scores)]
        return labels

    def _random_color(self, maximum=255):
        idx = np.random.randint(0, len(_COLORS))
        ret = _COLORS[idx] * maximum
        if not self.rgb:
            ret = ret[::-1]
        return ret

    def _get_buffer(self):
        if not self.pynb:
            s, (width, height) = self.canvas.print_to_buffer()
            if (width, height) != (self.width, self.height):
                img = cv2.resize(self.img, (width, height))
            else:
                img = self.img
        else:
            buf = io.BytesIO()  # works for cairo backend
            self.canvas.print_rgba(buf)
            width, height = self.width, self.height
            s = buf.getvalue()
            img = self.img

        buffer = np.frombuffer(s, dtype="uint8")
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)

        try:
            import numexpr as ne  # fuse them with numexpr

            visualized_image = ne.evaluate("img * (1 - alpha / 255.0) + rgb * (alpha / 255.0)")
        except ImportError:
            alpha = alpha.astype("float32") / 255.0
            visualized_image = img * (1 - alpha) + rgb * alpha

        return visualized_image.astype("uint8")

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color


# Color map
_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.857,
            0.857,
            0.857,
            1.000,
            1.000,
            1.000,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)
