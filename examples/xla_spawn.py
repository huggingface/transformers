"""
A simple launcher script for TPU training

Inspired by https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

::
    >>> python xla_spawn.py --num_cores=NUM_CORES_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

"""


import importlib
import sys
from argparse import REMAINDER, ArgumentParser

import torch_xla.distributed.xla_multiprocessing as xmp


def trim_suffix(s: str, suffix: str):
    return s if not s.endswith(suffix) or len(suffix) == 0 else s[: -len(suffix)]


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "PyTorch TPU distributed training launch "
            "helper utility that will spawn up "
            "multiple distributed processes"
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument("--num_cores", type=int, default=1, help="Number of TPU cores to use (1 or 8).")

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full module name to the single TPU training "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "training script"
        ),
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()

    # Import training_script as a module.
    mod_name = trim_suffix(args.training_script, ".py")
    mod = importlib.import_module(mod_name)

    # Patch sys.argv
    sys.argv = [args.training_script] + args.training_script_args

    # script.main() does not use a rank.
    def _mp_fn(rank):
        mod.main()

    xmp.spawn(_mp_fn, args=(), nprocs=args.num_cores)


if __name__ == "__main__":
    main()
