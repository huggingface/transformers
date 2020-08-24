# Integrations with other Python libraries

import os


try:
    import comet_ml  # noqa: F401

    _has_comet = True
except (ImportError):
    _has_comet = False


try:
    import wandb

    wandb.ensure_configured()
    if wandb.api.api_key is None:
        _has_wandb = False
        wandb.termwarn("W&B installed but not logged in.  Run `wandb login` or set the WANDB_API_KEY env variable.")
    else:
        _has_wandb = False if os.getenv("WANDB_DISABLED") else True
except (ImportError, AttributeError):
    _has_wandb = False

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # noqa: F401

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


def is_wandb_available():
    return _has_wandb


def is_comet_available():
    return _has_comet


def is_tensorboard_available():
    return _has_tensorboard
