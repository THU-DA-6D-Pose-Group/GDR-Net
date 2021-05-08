# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# modified from detectron2.utils.logger
# support showing line number and debug mode color
import functools
import logging
import os
import os.path as osp
import sys
from collections import Counter
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from termcolor import colored
import datetime


def _get_time_str():
    # return datetime.now().strftime("%Y%m%d-%H%M%S")
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WRN", "red", attrs=["blink"])
        elif record.levelno == logging.DEBUG:
            prefix = colored("DBG", "yellow", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()  # so that calling setup_my_logger multiple times won't add many handlers
def setup_my_logger(output=None, distributed_rank=0, *, color=True, name="mylib", abbrev_name=None):
    """
    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger
        abbrev_name (str): an abbreviation of the module, to avoid long names in logs.
            Set to "" to not log the root module in logs.
            By default, will abbreviate:
                detectron2 --> d2
                mylib --> lib
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        if name == "mylib":
            abbrev_name = "lib"
        elif name == "detectron2":
            abbrev_name = "d2"
        else:
            abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s@%(lineno)d: %(message)s", datefmt="%m%d_%H%M%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s@%(lineno)d]: ", "green") + "%(message)s",
                datefmt="%m%d_%H%M%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
        else:
            formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = osp.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + ".rank{}".format(distributed_rank)
        PathManager.mkdirs(osp.dirname(filename))

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger


# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return PathManager.open(filename, "a")


"""
Below are some other convenient logging methods.
They are mainly adopted from
https://github.com/abseil/abseil-py/blob/master/absl/logging/__init__.py
"""


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if (
            osp.join("utils", "setup_logger.") not in code.co_filename
            and osp.join("utils", "logger.") not in code.co_filename
        ):
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "lib"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()


def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):
    """Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = (key,)
    assert len(key) > 0

    caller_module, caller_key = _find_caller()
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg,)

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


def log_every_n(lvl, msg, n=1, *, name=None):
    """Log once per n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()
    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        logging.getLogger(name or caller_module).log(lvl, msg)


def create_small_table(small_dict):
    """Create a small table using the keys of small_dict as headers. This is
    only suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate([values], headers=keys, tablefmt="pipe", floatfmt=".3f", stralign="center", numalign="center")
    return table
