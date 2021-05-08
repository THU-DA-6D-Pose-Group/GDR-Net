import datetime
import json
import logging
import os

import numpy as np
import torch

from detectron2.engine.train_loop import HookBase
from detectron2.utils.events import EventWriter, get_event_storage
from fvcore.common.file_io import PathManager


class MyPeriodicWriter(HookBase):
    """Write events to EventStorage periodically.

    It is executed every ``period`` iterations and after the last iteration.
    NOTE: modified to write more frequently at the beginning
    """

    def __init__(self, writers, period=20, verbose_iter=100):
        """
        Args:
            writers (list[EventWriter]): a list of EventWriter objects
            period (int):
        """
        self._writers = writers
        for w in writers:
            assert isinstance(w, EventWriter), w
        self._period = period
        self.verbose_iter = verbose_iter

    def after_step(self):
        if (
            (self.trainer.iter + 1) % self._period == 0
            or (self.trainer.iter == self.trainer.max_iter - 1)
            or (self.trainer.iter + 1) < self.verbose_iter
        ):
            for writer in self._writers:
                writer.write()

    def after_train(self):
        for writer in self._writers:
            writer.close()


class MyTensorboardXWriter(EventWriter):
    """Write all scalars to a tensorboard file."""

    def __init__(self, log_dir: str, window_size: int = 20, backend: str = "pytorch", **kwargs):
        """
        Args:
            log_dir (str): The directory to save the output events
            window_size (int): the scalars will be median-smoothed by this window size
            kwargs: other arguments passed to `torch.utils.tensorboard.SummaryWriter(...)`
        """
        self._window_size = window_size
        assert backend.lower() in ["pytorch", "tensorboardx"], backend

        if backend.lower() == "pytorch":
            from torch.utils.tensorboard import SummaryWriter
        elif backend.lower() == "tensorboardx":
            from tensorboardX import SummaryWriter
        else:
            raise ValueError(
                "Unknown TensorboardXWriter backend: {}, available backends are: pytorch or tensorboardX".format(
                    backend
                )
            )

        self.backend = backend
        self._writer = SummaryWriter(log_dir, **kwargs)

    def write(self):
        storage = get_event_storage()
        # NOTE: this is default median(20)
        # for k, v in storage.latest_with_smoothing_hint(self._window_size).items():
        #     self._writer.add_scalar(k, v, storage.iter)
        # for k, v in storage.latest().items():  # let tensorboard do the smoothing
        #     self._writer.add_scalar(k, v, storage.iter)
        for k, v in storage.histories().items():
            self._writer.add_scalar(k, v.median(self._window_size), storage.iter)
            # self._writer.add_scalar(k, v.latest(), storage.iter)

        # storage.put_{image,histogram} is only meant to be used by
        # tensorboard writer. So we access its internal fields directly from here.
        if len(storage._vis_data) >= 1:
            for img_name, img, step_num in storage._vis_data:
                # default format CHW (C=1 or 3), rgb,
                # can be either float[0,1] or uint8[0,255]
                self._writer.add_image(img_name, img, step_num)
            # Storage stores all image data and rely on this writer to clear them.
            # As a result it assumes only one writer will use its image data.
            # An alternative design is to let storage store limited recent
            # data (e.g. only the most recent image) that all writers can access.
            # In that case a writer may not see all image data if its period is long.
            storage.clear_images()

        if len(storage._histograms) >= 1:
            for params in storage._histograms:
                self._writer.add_histogram_raw(**params)
            storage.clear_histograms()

    def close(self):
        if hasattr(self, "_writer"):  # doesn't exist when the code fails at import
            self._writer.close()


class MyCommonMetricPrinter(EventWriter):
    """Print __common__ metrics to the terminal, including iteration time, ETA,
    memory, all losses, and the learning rate.

    To print something different, please implement a similar printer by
    yourself.
    """

    def __init__(self, max_iter):
        """
        Args:
            max_iter (int): the maximum number of iterations to train.
                Used to compute ETA.
        """
        self.logger = logging.getLogger(__name__)
        self._max_iter = max_iter

    def write(self):
        storage = get_event_storage()
        iteration = storage.iter

        data_time, time = None, None
        eta_median_string = "N/A"
        eta_gavg_string = "N/A"
        eta_avg_string = "N/A"
        try:
            # time = storage.history("time").global_avg()
            time = storage.history("time").avg(20)
            # eta_seconds = storage.history("time").median(1000) * (self._max_iter - iteration)
            # eta_median_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            # eta_seconds = storage.history("time").global_avg() * (self._max_iter - iteration)
            # eta_gavg_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            eta_seconds = storage.history("time").avg(20) * (self._max_iter - iteration)
            eta_avg_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        try:
            data_time = storage.history("data_time").avg(20)
        except KeyError:  # they may not exist in the first few iterations (due to warmup)
            pass

        try:
            lr = "{:.4e}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        try:
            total_grad_norm = storage.history("total_grad_norm").latest()
            if np.isnan(total_grad_norm):  # nan will throw warning in tensorboard
                total_grad_norm = "NaN"
            else:
                total_grad_norm = "{:.6f}".format(total_grad_norm)
        except KeyError:
            total_grad_norm = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None

        # NOTE: max_mem is parsed by grep in "dev/parse_results.sh"
        self.logger.info(
            """\
eta: {eta_avg} \
iter: {iter}/{max_iter}[{percent:.1f}%] {time}  {data_time} lr: {lr}  {memory} \
total_grad_norm: {total_grad_norm} \
{losses}
""".format(
                # eta_median=eta_median_string,
                # eta_gavg=eta_gavg_string,
                eta_avg=eta_avg_string,
                iter=iteration,
                max_iter=self._max_iter,
                percent=iteration / self._max_iter * 100,
                losses="  ".join(
                    [
                        "{}: {:.4e} ({:.4e})".format(k, v.median(20), v.global_avg())
                        for k, v in storage.histories().items()
                        if "loss" in k
                    ]
                ),  # print median(gavg) as maskrcnn-benchmark/vision
                time="time: {:.4f}".format(time) if time is not None else "",
                data_time="data_time: {:.4f}".format(data_time) if data_time is not None else "",
                lr=lr,
                memory="max_mem: {:.0f}M".format(max_mem_mb) if max_mem_mb is not None else "",
                total_grad_norm=total_grad_norm,
            )
        )


class MyJSONWriter(EventWriter):
    """Write scalars to a json file.

    It saves scalars as one json per line (instead of a big json) for easy parsing.

    Examples parsing such a json file:

    .. code-block:: none

        $ cat metrics.json | jq -s '.[0:2]'
        [
          {
            "data_time": 0.008433341979980469,
            "iteration": 20,
            "loss": 1.9228371381759644,
            "loss_box_reg": 0.050025828182697296,
            "loss_classifier": 0.5316952466964722,
            "loss_mask": 0.7236229181289673,
            "loss_rpn_box": 0.0856662318110466,
            "loss_rpn_cls": 0.48198649287223816,
            "lr": 0.007173333333333333,
            "time": 0.25401854515075684
          },
          {
            "data_time": 0.007216215133666992,
            "iteration": 40,
            "loss": 1.282649278640747,
            "loss_box_reg": 0.06222952902317047,
            "loss_classifier": 0.30682939291000366,
            "loss_mask": 0.6970193982124329,
            "loss_rpn_box": 0.038663312792778015,
            "loss_rpn_cls": 0.1471673548221588,
            "lr": 0.007706666666666667,
            "time": 0.2490077018737793
          }
        ]

        $ cat metrics.json | jq '.loss_mask'
        0.7126231789588928
        0.689423680305481
        0.6776131987571716
        ...
    """

    def __init__(self, json_file, window_size=20):
        """
        Args:
            json_file (str): path to the json file. New data will be appended if the file exists.
            window_size (int): the window size of median smoothing for the scalars whose
                `smoothing_hint` are True.
        """
        self._file_handle = PathManager.open(json_file, "a")
        self._window_size = window_size

    def write(self):
        storage = get_event_storage()
        to_save = {"iteration": storage.iter}
        to_save.update(storage.latest_with_smoothing_hint(self._window_size))
        self._file_handle.write(json.dumps(to_save, sort_keys=True) + "\n")
        self._file_handle.flush()
        try:
            os.fsync(self._file_handle.fileno())
        except AttributeError:
            pass

    def close(self):
        self._file_handle.close()
