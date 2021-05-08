from __future__ import division
import timeit
import time
from datetime import datetime, timedelta
from . import logger


def average_time_of_func(func, num_iter=10000, ms=False):
    """
    ms: if True, xxx ms/iter
    """
    duration = timeit.timeit(func, number=num_iter)
    avg_time = duration / num_iter
    if ms:
        avg_time *= 1000
        logger.info("{} {} ms/iter".format(func.__name__, avg_time))
    else:
        logger.info("{} {} s/iter".format(func.__name__, avg_time))
    return avg_time


def my_timeit(func, number=100000):
    tic = time.perf_counter()
    for i in range(number):
        func()
    return time.perf_counter() - tic


def get_time_str(fmt="%Y%m%d_%H%M%S"):
    return datetime.now().strftime(fmt)


# def get_time_str(fmt='%Y%m%d_%H%M%S'):
#     # from mmcv.runner import get_time_str
#     return time.strftime(fmt, time.localtime())  # defined in mmcv


def get_time_delta(sec):
    """Humanize timedelta given in seconds, modified from maskrcnn-
    benchmark."""
    if sec < 0:
        logger.warning("get_time_delta() obtains negative seconds!")
        return "{:.3g} seconds".format(sec)
    delta_time_str = str(timedelta(seconds=sec))
    return delta_time_str


class Timer(object):
    # modified from maskrcnn-benchmark
    def __init__(self):
        self.reset()

    @property
    def average_time(self):
        return self.total_time / self.calls if self.calls > 0 else 0.0

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.perf_counter()

    def toc(self, average=True):
        self.add(time.perf_counter() - self.start_time)
        if average:
            return self.average_time
        else:
            return self.diff

    def add(self, time_diff):
        self.diff = time_diff
        self.total_time += self.diff
        self.calls += 1

    def reset(self):
        self.total_time = 0.0
        self.calls = 0
        self.start_time = 0.0
        self.diff = 0.0

    def avg_time_str(self):
        time_str = get_time_delta(self.average_time)
        return time_str


def humanize_time_delta(sec):
    """Humanize timedelta given in seconds
    Args:
        sec (float): time difference in seconds. Must be positive.
    Returns:
        str - time difference as a readable string
    Example:
    .. code-block:: python
        print(humanize_time_delta(1))                                   # 1 second
        print(humanize_time_delta(60 + 1))                              # 1 minute 1 second
        print(humanize_time_delta(87.6))                                # 1 minute 27 seconds
        print(humanize_time_delta(0.01))                                # 0.01 seconds
        print(humanize_time_delta(60 * 60 + 1))                         # 1 hour 1 second
        print(humanize_time_delta(60 * 60 * 24 + 1))                    # 1 day 1 second
        print(humanize_time_delta(60 * 60 * 24 + 60 * 2 + 60*60*9 + 3)) # 1 day 9 hours 2 minutes 3 seconds
    """
    if sec < 0:
        logger.warning("humanize_time_delta() obtains negative seconds!")
        return "{:.3g} seconds".format(sec)
    if sec == 0:
        return "0 second"
    _time = datetime(2000, 1, 1) + timedelta(seconds=int(sec))
    units = ["day", "hour", "minute", "second"]
    vals = [int(sec // 86400), _time.hour, _time.minute, _time.second]
    if sec < 60:
        vals[-1] = sec

    def _format(v, u):
        return "{:.3g} {}{}".format(v, u, "s" if v > 1 else "")

    ans = []
    for v, u in zip(vals, units):
        if v > 0:
            ans.append(_format(v, u))
    return " ".join(ans)
