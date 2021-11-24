#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
# modified by Gu Wang
import inspect
import os
import sys
import logging
from loguru import logger
import time
from collections import Counter
import warnings
from termcolor import colored
from functools import partial


class InterceptHandler(logging.Handler):
    # https://github.com/Delgan/loguru#entirely-compatible-with-standard-logging
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_intercept():
    logging.basicConfig(handlers=[InterceptHandler()], level=0)


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth. Default value: 0.

    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    # caller = inspect.getframeinfo(inspect.stack()[1][0])
    # import ipdb; ipdb.set_trace()
    for _ in range(depth):
        if frame.f_back is not None:
            frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """stream object that redirects writes to a logger instance."""

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools"), stream_logger=None):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self._logger = logger if stream_logger is None else stream_logger
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                if module_name in ["__main__"]:
                    log_depth = -1
                else:
                    log_depth = 2
                if isinstance(self._logger, logging.Logger):
                    self._logger.log(logging.getLevelName(self.level), line.rstrip())
                else:
                    self._logger.opt(depth=log_depth).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

    def flush(self):
        pass


def redirect_sys_output(
    log_level="INFO", caller_names=("apex", "pycocotools", "__main__"), stdout_logger=None, stderr_logger=None
):
    # logging.getLogger("STDOUT")
    # stderr_logger = None  # logging.getLogger("STDERR")
    sys.stdout = StreamToLoguru(log_level, caller_names=caller_names, stream_logger=stdout_logger)
    sys.stderr = StreamToLoguru(log_level, caller_names=caller_names, stream_logger=stderr_logger)


def setup_logger(
    output=None,
    distributed_rank=0,
    log_level="DEBUG",
    redirect_sys_out_callers=("apex", "pycocotools", "__main__"),
):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.

    Return:
        logger instance.
    """

    def formatter(record, is_file=False):
        level_name = record["level"].name
        level_name_map = {
            "INFO": "",  # "INF",
            "WARNING": "{}|".format(colored("WRN", "red", attrs=["blink"])),
            "ERROR": "{}|".format(colored("ERR", "red", attrs=["blink", "underline"])),
            "CRITICAL": "{}|".format(colored("ERR", "red", attrs=["blink", "underline"])),
            "DEBUG": "{}|".format(colored("DBG", "yellow", attrs=["blink"])),
        }

        level_abbr = level_name_map.get(level_name, f"<lvl>{level_name}</lvl>|")

        caller_name = record["name"]
        # print(record["file"].name, record["file"].path)
        # print(record)
        # print(get_caller_name(3))
        if caller_name.startswith("detectron2."):
            caller_abbr = caller_name.replace("detectron2.", "d2.")
        else:
            caller_abbr = caller_name
        if is_file:
            func_name = ":{function}"
        else:
            func_name = ""
        loguru_format = (
            "<green>{time:YYYYMMDD_HHmmss}</green>|"
            "%s"
            "<cyan>%s</cyan>%s@<cyan>{line}</cyan>: <lvl>{message}</lvl>"
            "\n{exception}"
        ) % (level_abbr, caller_abbr, func_name)
        return loguru_format

    logger.remove()  # Remove the pre-configured handler
    # only keep logger in rank0 process
    if distributed_rank == 0:
        logger.add(
            sys.stderr,
            format=formatter,
            level=log_level,
            enqueue=True,
        )

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logger.add(
            filename,
            format=partial(formatter, is_file=True),
            level=log_level,
            enqueue=True,
        )

    setup_intercept()

    # redirect stdout/stderr to loguru
    redirect_sys_output("INFO", caller_names=redirect_sys_out_callers)
