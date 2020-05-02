import logging
from argparse import ArgumentParser, Namespace
from typing import Any, List, Optional

from transformers import Pipeline
from transformers.commands import BaseTransformersCLICommand
from transformers.pipelines import SUPPORTED_TASKS, pipeline


try:
    from model_archiver.model_packaging import package_model
    from model_archiver.model_packaging import ModelExportUtils
    from model_archiver.manifest_components.manifest import RuntimeType

    _torchserve_dependencies_installed = True
except (ImportError, AttributeError):
    _torchserve_dependencies_installed = False


logger = logging.getLogger("transformers-cli/torchserve")


class TorchServeCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        torchserve_parser = parser.add_parser(
            "torchserve",
            help="CLI interface to package and version models for serving with pytorch/serve.",
        )
        torchserve_parser.set_defaults(func=lambda args: TorchServeCommand(args))

    def __init__(self, args: Namespace):
        if not _torchserve_dependencies_installed:
            raise RuntimeError(
                "Using torchserve command requires torchserve and torch-model-archiver. "
                'Please install transformers with [torchserve]: pip install "transformers[torchserve]".'
                "Or install torchserve and torch-model-archiver separately."
            )
        self.args = args

    def run(self):
        raise NotImplementedError()
