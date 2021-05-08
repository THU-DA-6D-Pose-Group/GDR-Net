import types
from bisect import bisect_right

import torch
from torch.optim import Optimizer
from math import pi, cos
from lib.utils import logger


def build_scheduler(lr_config, optimizer, epoch_length):
    """
    total_epochs = 80
    # learning policy
    lr_config = dict(
        policy='flat_and_anneal',  #
        warmup_method='linear',
        warmup_iters=800,
        warmup_factor=1.0 / 10,
        target_lr_factor=0.001,
        anneal_method='cosine',  # step, linear, poly, exp, cosine
        anneal_point=0.72,  # no use when method is step
        steps=[0.5, 0.75],
        step_gamma=0.1,
        poly_power=0.5,
        epochs=total_epochs)
    warmup init lr = base_lr * warmup_factor
    epoch_length: len(train_loader)
    """
    policy = lr_config["policy"]
    assert policy in ("flat_and_anneal", "linear", "step", "poly", "multistep", "warmup_multistep")
    total_iters = lr_config["epochs"] * epoch_length

    # update_mode = 'epoch' if lr_config.get('by_epoch', False) else 'batch'
    if policy == "flat_and_anneal":
        scheduler = flat_and_anneal_lr_scheduler(
            optimizer=optimizer,
            total_iters=total_iters,
            warmup_method=lr_config["warmup_method"],
            warmup_factor=lr_config["warmup_factor"],
            warmup_iters=lr_config["warmup_iters"],
            anneal_method=lr_config["anneal_method"],
            anneal_point=lr_config["anneal_point"],
            target_lr_factor=lr_config["target_lr_factor"],
            poly_power=lr_config["poly_power"],
            step_gamma=lr_config["step_gamma"],
            steps=lr_config["steps"],
        )
    elif policy == "warmup_multistep":
        # if update_mode == 'epoch':
        #     milestones = [epoch_length * _step for _step in lr_config['steps']]
        # else:
        milestones = [_step * total_iters for _step in lr_config["steps"]]
        scheduler = WarmupMultiStepLR(
            optimizer,
            milestones,
            gamma=lr_config["step_gamma"],
            warmup_factor=lr_config["warmup_factor"],
            warmup_iters=lr_config["warmup_iters"],
            warmup_method=lr_config["warmup_method"],
            last_epoch=-1,
        )
    elif policy == "linear":
        # if update_mode == "batch":
        #     count = epoch_length * lr_config["epochs"]
        # else:
        # count = lr_config["epochs"]
        count = total_iters

        beta = float(lr_config["from"])
        alpha = float(lr_config["to"] - beta) / count

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: it * alpha + beta)
    elif policy == "step":
        if len(lr_config["steps"]) != 1:
            raise ValueError("step policy only support 1 step. got {}".format(len(lr_config["steps"])))
        step_size = lr_config["steps"][0] * total_iters  # by batch/iter
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, lr_config["step_gamma"])
    elif policy == "poly":
        # if update_mode == "batch":
        #     count = epoch_length * lr_config["epochs"]
        # else:
        #     count = lr_config["epochs"]
        count = total_iters
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: (1 - float(it) / count) ** lr_config["poly_power"]
        )
    elif policy == "multistep":
        milestones = [_step * total_iters for _step in lr_config["steps"]]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, lr_config["step_gamma"])
    else:
        raise ValueError(
            "Unrecognized scheduler type {}, "
            "valid options: 'flat_and_anneal', 'linear', 'step', 'poly', 'multistep'".format(policy)
        )
    return scheduler


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor, warmup_method="linear"):
    # https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    """
    # in epoch 0:
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # in one epoch:
    optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
    # iter < warmup_iters: use this scheduler
    # iter >= warmup_iters: use whatever scheduler
    # if warmup is not only happen in epoch 0,
    # convert the other's to be based on iters other than epochs
    """
    if warmup_method not in ("constant", "linear"):
        raise ValueError("Only 'constant' or 'linear' warmup_method accepted" "got {}".format(warmup_method))

    def f(x):  # x is the iter in lr scheduler
        # the final lr is warmup_factor * base_lr
        if x >= warmup_iters:
            return 1
        if warmup_method == "linear":
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha
        elif warmup_method == "constant":
            return warmup_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=5,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not milestones == sorted(milestones):
            raise ValueError("Milestones should be a list of" " increasing integers. Got {}", milestones)

        if warmup_method not in ("constant", "linear"):
            raise ValueError("Only 'constant' or 'linear' warmup_method accepted" "got {}".format(warmup_method))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # to add warmup for other schedulers, we only need to implement warmup_factor
        # another way is to combine the warmup_lr_scheduler function and native scheduler,
        # so that we don't need to reimplement many schedulers
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr * warmup_factor * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def flat_and_anneal_lr_scheduler(
    optimizer,
    total_iters,
    warmup_iters=0,
    warmup_factor=0.1,
    warmup_method="linear",
    anneal_point=0.72,
    anneal_method="cosine",
    target_lr_factor=0,
    poly_power=1.0,
    step_gamma=0.1,
    steps=[2 / 3.0, 8 / 9.0],
):
    """https://github.com/fastai/fastai/blob/master/fastai/callbacks/flat_cos_a
    nneal.py.

    warmup_initial_lr = warmup_factor * base_lr
    target_lr = base_lr * target_lr_factor
    """
    if warmup_method not in ("constant", "linear"):
        raise ValueError("Only 'constant' or 'linear' warmup_method accepted," "got {}".format(warmup_method))

    if anneal_method not in ("cosine", "linear", "poly", "exp", "step", "none"):
        raise ValueError(
            "Only 'cosine', 'linear', 'poly', 'exp', 'step' or 'none' anneal_method accepted,"
            "got {}".format(anneal_method)
        )

    if anneal_method == "step":
        if any([_step < warmup_iters / total_iters or _step > 1 for _step in steps]):
            raise ValueError(
                "error in steps: {}. warmup_iters: {} total_iters: {}."
                "steps should be in ({},1)".format(steps, warmup_iters, total_iters, warmup_iters / total_iters)
            )
        if list(steps) != sorted(steps):
            raise ValueError("steps {} is not in ascending order.".format(steps))
        logger.warning("ignore anneal_point when using step anneal_method")
        anneal_start = steps[0] * total_iters
    else:
        if anneal_point > 1 or anneal_point < 0:
            raise ValueError("anneal_point should be in [0,1], got {}".format(anneal_point))
        anneal_start = anneal_point * total_iters

    def f(x):  # x is the iter in lr scheduler, return the lr_factor
        # the final lr is warmup_factor * base_lr
        if x < warmup_iters:
            if warmup_method == "linear":
                alpha = float(x) / warmup_iters
                return warmup_factor * (1 - alpha) + alpha
            elif warmup_method == "constant":
                return warmup_factor
        elif x >= anneal_start:
            if anneal_method == "step":
                # ignore anneal_point and target_lr_factor
                milestones = [_step * total_iters for _step in steps]
                lr_factor = step_gamma ** bisect_right(milestones, float(x))
            elif anneal_method == "cosine":
                # slow --> fast --> slow
                lr_factor = target_lr_factor + 0.5 * (1 - target_lr_factor) * (
                    1 + cos(pi * ((float(x) - anneal_start) / (total_iters - anneal_start)))
                )
            elif anneal_method == "linear":
                # (y-m) / (B-x) = (1-m) / (B-A)
                lr_factor = target_lr_factor + (1 - target_lr_factor) * (total_iters - float(x)) / (
                    total_iters - anneal_start
                )
            elif anneal_method == "poly":
                # slow --> fast if poly_power < 1
                # fast --> slow if poly_power > 1
                # when poly_power == 1.0, it is the same with linear
                lr_factor = (
                    target_lr_factor
                    + (1 - target_lr_factor) * ((total_iters - float(x)) / (total_iters - anneal_start)) ** poly_power
                )
            elif anneal_method == "exp":
                # fast --> slow
                # do not decay too much, especially if lr_end == 0, lr will be
                # 0 at anneal iter, so we should avoid that
                _target_lr_factor = max(target_lr_factor, 5e-3)
                lr_factor = _target_lr_factor ** ((float(x) - anneal_start) / (total_iters - anneal_start))
            else:
                lr_factor = 1
            return lr_factor
        else:  # warmup_iter <= x < anneal_start_iter
            return 1

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def update_learning_rate(optimizer, cur_lr, new_lr):
    # old way of update learning rate
    """Update learning rate."""
    if cur_lr == new_lr:
        return
    ratio = max((new_lr / max((cur_lr, 1e-10)), cur_lr / max((new_lr, 1e-10))))
    if ratio > 1.1:
        print("Changing learning rate {} -> {}".format(cur_lr, new_lr))
    # Update learning rate, note that different parameter may have different learning rate
    param_keys = []
    for ind, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = new_lr if ind == 0 else new_lr * 2  # bias params
        param_keys += param_group["params"]


def test_flat_and_anneal():
    from mmcv import Config
    import numpy as np

    model = resnet18()

    optimizer_cfg = dict(type="Adam", lr=1e-4, weight_decay=0)
    optimizer = obj_from_dict(optimizer_cfg, torch.optim, dict(params=model.parameters()))

    # learning policy
    total_epochs = 80
    epoch_len = 500
    total_iters = epoch_len * total_epochs
    # poly, step, linear, exp, cosine
    lr_cfg = Config(
        dict(
            anneal_method="cosine",
            warmup_method="linear",
            step_gamma=0.1,
            warmup_factor=0.1,
            warmup_iters=800,
            poly_power=5,
            target_lr_factor=0.0,
            steps=[0.5, 0.75, 0.9],
            anneal_point=0.72,
        )
    )

    # scheduler = build_scheduler(lr_config, optimizer, epoch_length)
    scheduler = flat_and_anneal_lr_scheduler(
        optimizer=optimizer,
        total_iters=total_iters,
        warmup_method=lr_cfg.warmup_method,
        warmup_factor=lr_cfg.warmup_factor,
        warmup_iters=lr_cfg.warmup_iters,
        anneal_method=lr_cfg.anneal_method,
        anneal_point=lr_cfg.anneal_point,
        target_lr_factor=lr_cfg.target_lr_factor,
        poly_power=lr_cfg.poly_power,
        step_gamma=lr_cfg.step_gamma,
        steps=lr_cfg.steps,
    )
    print("start lr: {}".format(scheduler.get_lr()))
    steps = []
    lrs = []

    epoch_lrs = []
    global_step = 0

    start_epoch = 20
    for epoch in range(start_epoch):
        for batch in range(epoch_len):
            scheduler.step()  # when no state_dict availble
            global_step += 1

    for epoch in range(start_epoch, total_epochs):
        # if global_step >= lr_config['warmup_iters']:
        #     scheduler.step(epoch)
        # print(type(scheduler.get_lr()[0]))
        # import pdb;pdb.set_trace()
        epoch_lrs.append([epoch, scheduler.get_lr()[0]])  # only get the first lr (maybe a group of lrs)
        for batch in range(epoch_len):
            # if global_step < lr_config['warmup_iters']:
            #     scheduler.step(global_step)
            cur_lr = scheduler.get_lr()[0]
            if global_step == 0 or (len(lrs) >= 1 and cur_lr != lrs[-1]):
                print("epoch {}, batch: {}, global_step:{} lr: {}".format(epoch, batch, global_step, cur_lr))
            steps.append(global_step)
            lrs.append(cur_lr)
            global_step += 1
            scheduler.step()  # usually after optimizer.step()
    # print(epoch_lrs)
    # import pdb;pdb.set_trace()
    epoch_lrs.append([total_epochs, scheduler.get_lr()[0]])

    epoch_lrs = np.asarray(epoch_lrs, dtype=np.float32)
    for i in range(len(epoch_lrs)):
        print("{:02d} {}".format(int(epoch_lrs[i][0]), epoch_lrs[i][1]))

    plt.figure(dpi=200)
    plt.suptitle("{}".format(dict(lr_cfg)), size=4)
    plt.subplot(1, 2, 1)
    plt.plot(steps, lrs)
    # plt.show()
    plt.subplot(1, 2, 2)
    # print(epoch_lrs.dtype)
    plt.plot(epoch_lrs[:, 0], epoch_lrs[:, 1])
    plt.show()


if __name__ == "__main__":
    from mmcv.runner import obj_from_dict
    import sys
    import os.path as osp
    import torch
    from torchvision.models import resnet18
    import matplotlib.pyplot as plt

    cur_dir = osp.dirname(osp.abspath(__file__))
    sys.path.insert(0, osp.join(cur_dir, "../.."))

    test_flat_and_anneal()
    exit(0)

    total_epochs = 24
    model = resnet18()
    optimizer_cfg = dict(type="Adam", lr=6.25e-5, weight_decay=0)
    # learning policy
    lr_config = dict(
        policy="warmup_multistep",
        gamma=0.1,
        warmup="linear",
        warmup_iters=500,
        warmup_ratio=1.0 / 3,
        step=[16, 22],
        epochs=total_epochs,
    )
    optimizer = obj_from_dict(optimizer_cfg, torch.optim, dict(params=model.parameters()))
    epoch_length = 1000
    scheduler = build_scheduler(lr_config, optimizer, epoch_length)
    print("start lr: {}".format(scheduler.get_lr()))
    steps = []
    lrs = []
    epoch_lrs = []
    global_step = 0
    for epoch in range(total_epochs):
        # if global_step >= lr_config['warmup_iters']:
        #     scheduler.step(epoch)
        epoch_lrs.append(scheduler.get_lr())
        for batch in range(epoch_length):
            # if global_step < lr_config['warmup_iters']:
            # scheduler.step(global_step)
            cur_lr = scheduler.get_lr()
            if global_step == 0 or cur_lr != lrs[-1]:
                print("epoch {}, batch: {}, global_step:{} lr: {}".format(epoch, batch, global_step, cur_lr))
            steps.append(global_step)
            lrs.append(cur_lr)
            global_step += 1
            scheduler.step()  # usually after optimizer.step()
    for i, lr in enumerate(epoch_lrs):
        print("{:02d} {}".format(i, lr))
    plt.subplot(1, 2, 1)
    plt.plot(steps, lrs)
    # plt.show()
    plt.subplot(1, 2, 2)
    plt.plot(list(range(len(epoch_lrs))), epoch_lrs)
    plt.show()
