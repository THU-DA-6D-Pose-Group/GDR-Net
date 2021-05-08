import logging

logger = logging.getLogger(__name__)


def setup_for_distributed(is_master):

    """This function disables printing when not in master process."""
    import builtins as __builtin__

    builtin_print = __builtin__.print
    if not is_master:
        logger.setLevel("WARN")

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
