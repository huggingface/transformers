from argparse import ArgumentParser
import platform

from transformers.commands import BaseTransformersCLICommand
from transformers import __version__ as version


def info_command_factory(_):
    return InfoCommand()


class InfoCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("info")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        pt_version = "not installed"
        pt_cuda_available = "NA"
        try:
            import torch
            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()
        except ImportError:
            pass

        tf_version = "not installed"
        tf_cuda_available = "NA"
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            try:
                # deprecated in v2.1
                tf_cuda_available = tf.test.is_gpu_available()
            except AttributeError:
                # returns list of devices, convert to bool
                tf_cuda_available = bool(tf.config.list_physical_devices('GPU'))
        except ImportError:
            pass

        info = {
            "`transformers` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "PyTorch version (GPU?)": f"{pt_version} ({pt_cuda_available})",
            "Tensorflow version (GPU?)": f"{tf_version} ({tf_cuda_available})",
            "Using GPU in script?": "<fill in>",
            "Using distributed or parallel set-up in script?": "<fill in>"
        }

        print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        s = "## Environment info\n"
        s += "\n".join([f"- {prop}: {val}" for prop, val in d.items()])
        s += "\n"

        return s
