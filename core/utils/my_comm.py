import detectron2.utils.comm as comm
import torch
import torch.distributed as dist
import logging
import pickle


def reduce_dict(input_dict, average=True):
    return comm.reduce_dict(input_dict, average=average)


def all_gather(data, group=None):
    return comm.all_gather(data, group=group)


def synchronize():
    return comm.synchronize()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def shared_random_seed():
    return comm.shared_random_seed()


def get_world_size():
    return comm.get_world_size()


def get_rank():
    return comm.get_rank()


def get_local_rank():
    return comm.get_local_rank()


def get_local_size():
    return comm.get_local_size()


def is_main_process():
    return get_rank() == 0
