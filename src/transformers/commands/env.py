import platform
from argparse import ArgumentParser

from transformers import __version__ as version
from transformers import is_tf_available, is_torch_available
from transformers.commands import BaseTransformersCLICommand


def info_command_factory(_):
    return EnvironmentCommand()


class EnvironmentCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("env")
        download_parser.set_defaults(func=info_command_factory)

    def run(self):
        pt_version = "not installed"
        pt_cuda_available = "NA"
        if is_torch_available():
            import torch

            pt_version = torch.__version__
            pt_cuda_available = torch.cuda.is_available()

        tf_version = "not installed"
        tf_cuda_available = "NA"
        if is_tf_available():
            import tensorflow as tf

            tf_version = tf.__version__
            try:
                # deprecated in v2.1
                tf_cuda_available = tf.test.is_gpu_available()
            except AttributeError:
                # returns list of devices, convert to bool
                tf_cuda_available = bool(tf.config.list_physical_devices("GPU"))

        info = {
            "`transformers` version": version,
            "Platform": platform.platform(),
            "Python version": platform.python_version(),
            "PyTorch version (GPU?)": "{} ({})".format(pt_version, pt_cuda_available),
            "Tensorflow version (GPU?)": "{} ({})".format(tf_version, tf_cuda_available),
            "Using GPU in script?": "<fill in>",
            "Using distributed or parallel set-up in script?": "<fill in>",
        }

        print("\nCopy-and-paste the text below in your GitHub issue and FILL OUT the two last points.\n")
        print(self.format_dict(info))

        return info

    @staticmethod
    def format_dict(d):
        return "\n".join(["- {}: {}".format(prop, val) for prop, val in d.items()]) + "\n"
