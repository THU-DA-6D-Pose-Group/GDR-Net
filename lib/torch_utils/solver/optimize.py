import re
import torch
from torch.nn.utils import clip_grad
import mmcv
from mmcv.runner import obj_from_dict
from lib.utils import logger
from lib.utils.utils import msg


def _get_optimizer(params, optimizer_cfg, use_hvd=False):
    # cfg.optimizer = dict(type='RMSprop', lr=1e-4, weight_decay=0)
    # cfg.optimizer = dict(type='Ranger', lr=1e-4) # , N_sma_threshhold=5, betas=(.95, 0.999))  # 4, (0.90, 0.999)
    optim_type_str = optimizer_cfg.pop("type")
    if optim_type_str.lower() in ["rangerlars", "over9000"]:  # RangerLars
        optim_type_str = "lookahead_Ralamb"
    optim_split = optim_type_str.split("_")

    optim_type = optim_split[-1]
    logger.info(f"optimizer: {optim_type_str} {optim_split}")

    if optim_type == "Ranger":
        from lib.torch_utils.solver.ranger import Ranger

        optimizer_cls = Ranger
    elif optim_type == "Ralamb":
        from lib.torch_utils.solver.ralamb import Ralamb

        optimizer_cls = Ralamb
    elif optim_type == "RAdam":
        from lib.torch_utils.solver.radam import RAdam

        optimizer_cls = RAdam
    else:
        optimizer_cls = getattr(torch.optim, optim_type)
    opt_kwargs = {k: v for k, v in optimizer_cfg.items() if "lookahead" not in k}
    optimizer = optimizer_cls(params, **opt_kwargs)

    if len(optim_split) > 1 and not use_hvd:
        if optim_split[0].lower() == "lookahead":
            from lib.torch_utils.solver.lookahead import Lookahead

            # TODO: pass lookahead hyper-params
            optimizer = Lookahead(
                optimizer, alpha=optimizer_cfg.get("lookahead_alpha", 0.5), k=optimizer_cfg.get("lookahead_k", 6)
            )
    # logger.info(msg(type(optimizer)))
    return optimizer


def build_optimizer_on_params(params, optimizer_cfg, use_hvd=False):
    optimizer_cfg = optimizer_cfg.copy()
    return _get_optimizer(params, optimizer_cfg, use_hvd=use_hvd)


def build_optimizer(model, optimizer_cfg, cfg=None, use_hvd=False):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.
                cfg.optimizer
            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.
            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with 3 accepted fileds
                  (bias_lr_mult, bias_decay_mult, norm_decay_mult).
                  `bias_lr_mult` and `bias_decay_mult` will be multiplied to
                  the lr and weight decay respectively for all bias parameters
                  (except for the normalization layers), and
                  `norm_decay_mult` will be multiplied to the weight decay
                  for all weight and bias parameters of normalization layers.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.
    """
    if hasattr(model, "module"):
        model = model.module

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop("paramwise_options", None)
    # if no paramwise option is specified, just use the global setting
    if paramwise_options is None:
        if cfg is not None and "train" in cfg and cfg.train.get("slow_base", False):
            base_params = [p for p_n, p in model.named_parameters() if "emb_head" not in p_n]
            active_params = [p for p_n, p in model.named_parameters() if "emb_head" in p_n]
            params = [
                {"params": base_params, "lr": cfg.ref.slow_base_ratio * optimizer_cfg["lr"]},
                {"params": active_params},
            ]
            return _get_optimizer(params, optimizer_cfg, use_hvd=use_hvd)
        else:
            return _get_optimizer(model.parameters(), optimizer_cfg, use_hvd=use_hvd)
    else:
        assert isinstance(paramwise_options, dict)
        # get base lr and weight decay
        base_lr = optimizer_cfg["lr"]
        base_wd = optimizer_cfg.get("weight_decay", None)
        # weight_decay must be explicitly specified if mult is specified
        if "bias_decay_mult" in paramwise_options or "norm_decay_mult" in paramwise_options:
            assert base_wd is not None
        # get param-wise options
        bias_lr_mult = paramwise_options.get("bias_lr_mult", 1.0)
        bias_decay_mult = paramwise_options.get("bias_decay_mult", 1.0)
        norm_decay_mult = paramwise_options.get("norm_decay_mult", 1.0)
        # set param-wise lr and weight decay
        params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            param_group = {"params": [param]}
            # for norm layers, overwrite the weight decay of weight and bias
            # TODO: obtain the norm layer prefixes dynamically
            if re.search(r"(bn|gn)(\d+)?.(weight|bias)", name):
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * norm_decay_mult
            # for other layers, overwrite both lr and weight decay of bias
            elif name.endswith(".bias"):
                param_group["lr"] = base_lr * bias_lr_mult
                if base_wd is not None:
                    param_group["weight_decay"] = base_wd * bias_decay_mult

            # NOTE: add
            if cfg is not None and "train" in cfg and cfg.train.get("slow_base", False):
                if "emb_head" not in name:  # backbone parameters
                    param_group["lr"] = cfg.ref.slow_base_ratio * base_lr
            # otherwise use the global settings

            params.append(param_group)
        return _get_optimizer(params, optimizer_cfg, use_hvd=use_hvd)


def clip_grad_norm(params, max_norm=35, norm_type=2):
    """
    clip_grad_norm = {'max_norm': 35, 'norm_type': 2}
    slow down training
    """
    clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, params), max_norm=max_norm, norm_type=norm_type)


def clip_grad_value(params, clip_value=10):
    # slow down training
    clip_grad.clip_grad_value_(filter(lambda p: p.requires_grad, params), clip_value=clip_value)
