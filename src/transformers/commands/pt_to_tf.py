# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser, Namespace

from ..utils import logging
from . import BaseTransformersCLICommand


MAX_ERROR = 5e-5  # larger error tolerance than in our internal tests, to avoid flaky user-facing errors


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(
        args.model_name,
        args.local_dir,
        args.max_error,
        args.new_weights,
        args.no_pr,
        args.push,
        args.extra_commit_description,
        args.override_model_class,
    )


class PTtoTFCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "pt-to-tf",
            help=(
                "CLI tool to run convert a transformers model from a PyTorch checkpoint to a TensorFlow checkpoint."
                " Can also be used to validate existing weights without opening PRs, with --no-pr."
            ),
        )
        train_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The model name, including owner/organization, as seen on the hub.",
        )
        train_parser.add_argument(
            "--local-dir",
            type=str,
            default="",
            help="Optional local directory of the model repository. Defaults to /tmp/{model_name}",
        )
        train_parser.add_argument(
            "--max-error",
            type=float,
            default=MAX_ERROR,
            help=(
                f"Maximum error tolerance. Defaults to {MAX_ERROR}. This flag should be avoided, use at your own risk."
            ),
        )
        train_parser.add_argument(
            "--new-weights",
            action="store_true",
            help="Optional flag to create new TensorFlow weights, even if they already exist.",
        )
        train_parser.add_argument(
            "--no-pr", action="store_true", help="Optional flag to NOT open a PR with converted weights."
        )
        train_parser.add_argument(
            "--push",
            action="store_true",
            help="Optional flag to push the weights directly to `main` (requires permissions)",
        )
        train_parser.add_argument(
            "--extra-commit-description",
            type=str,
            default="",
            help="Optional additional commit description to use when opening a PR (e.g. to tag the owner).",
        )
        train_parser.add_argument(
            "--override-model-class",
            type=str,
            default=None,
            help="If you think you know better than the auto-detector, you can specify the model class here. "
            "Can be either an AutoModel class or a specific model class like BertForSequenceClassification.",
        )
        train_parser.set_defaults(func=convert_command_factory)

    def __init__(
        self,
        model_name: str,
        local_dir: str,
        max_error: float,
        new_weights: bool,
        no_pr: bool,
        push: bool,
        extra_commit_description: str,
        override_model_class: str,
        *args,
    ):
        self._logger = logging.get_logger("transformers-cli/pt_to_tf")
        self._model_name = model_name
        self._local_dir = local_dir if local_dir else os.path.join("/tmp", model_name)
        self._max_error = max_error
        self._new_weights = new_weights
        self._no_pr = no_pr
        self._push = push
        self._extra_commit_description = extra_commit_description
        self._override_model_class = override_model_class

    def run(self):
        # TODO (joao): delete file in v4.47
        raise NotImplementedError(
            "\n\nConverting PyTorch weights to TensorFlow weights was removed in v4.43. "
            "Instead, we recommend that you convert PyTorch weights to Safetensors, an improved "
            "format that can be loaded by any framework, including TensorFlow. For more information, "
            "please see the Safetensors conversion guide: "
            "https://huggingface.co/docs/safetensors/en/convert-weights\n\n"
        )
