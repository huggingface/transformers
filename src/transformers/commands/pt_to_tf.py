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

import numpy as np
from datasets import load_dataset

from huggingface_hub import Repository, upload_file

from .. import AutoFeatureExtractor, AutoModel, AutoTokenizer, TFAutoModel, is_tf_available, is_torch_available
from ..utils import logging
from . import BaseTransformersCLICommand


if is_tf_available():
    import tensorflow as tf

    tf.config.experimental.enable_tensor_float_32_execution(False)

if is_torch_available():
    import torch


MAX_ERROR = 5e-5  # larger error tolerance than in our internal tests, to avoid flaky user-facing errors
TF_WEIGHTS_NAME = "tf_model.h5"


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model PyTorch checkpoint in a TensorFlow 2 checkpoint.

    Returns: ServeCommand
    """
    return PTtoTFCommand(args.model_name, args.local_dir, args.no_pr)


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
            "--no-pr", action="store_true", help="Optional flag to NOT open a PR with converted weights."
        )
        train_parser.set_defaults(func=convert_command_factory)

    def __init__(self, model_name: str, local_dir: str, no_pr: bool, *args):
        self._logger = logging.get_logger("transformers-cli/pt_to_tf")
        self._model_name = model_name
        self._local_dir = local_dir if local_dir else os.path.join("/tmp", model_name)
        self._no_pr = no_pr

    def get_text_inputs(self):
        tokenizer = AutoTokenizer.from_pretrained(self._local_dir)
        sample_text = ["Hi there!", "I am a batch with more than one row and different input lengths."]
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pt_input = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
        tf_input = tokenizer(sample_text, return_tensors="tf", padding=True, truncation=True)
        return pt_input, tf_input

    def get_audio_inputs(self):
        processor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        num_samples = 2
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        speech_samples = ds.sort("id").select(range(num_samples))[:num_samples]["audio"]
        raw_samples = [x["array"] for x in speech_samples]
        pt_input = processor(raw_samples, return_tensors="pt", padding=True)
        tf_input = processor(raw_samples, return_tensors="tf", padding=True)
        return pt_input, tf_input

    def get_image_inputs(self):
        feature_extractor = AutoFeatureExtractor.from_pretrained(self._local_dir)
        num_samples = 2
        ds = load_dataset("cifar10", "plain_text", split="test")[:num_samples]["img"]
        pt_input = feature_extractor(images=ds, return_tensors="pt")
        tf_input = feature_extractor(images=ds, return_tensors="tf")
        return pt_input, tf_input

    def run(self):
        # Fetch remote data
        # TODO: implement a solution to pull a specific PR/commit, so we can use this CLI to validate pushes.
        repo = Repository(local_dir=self._local_dir, clone_from=self._model_name)
        repo.git_pull()  # in case the repo already exists locally, but with an older commit

        # Load models and acquire a basic input for its modality.
        pt_model = AutoModel.from_pretrained(self._local_dir)
        main_input_name = pt_model.main_input_name
        if main_input_name == "input_ids":
            pt_input, tf_input = self.get_text_inputs()
        elif main_input_name == "pixel_values":
            pt_input, tf_input = self.get_image_inputs()
        elif main_input_name == "input_features":
            pt_input, tf_input = self.get_audio_inputs()
        else:
            raise ValueError(f"Can't detect the model modality (`main_input_name` = {main_input_name})")
        tf_from_pt_model = TFAutoModel.from_pretrained(self._local_dir, from_pt=True)

        # Extra input requirements, in addition to the input modality
        if hasattr(pt_model, "encoder") and hasattr(pt_model, "decoder"):
            decoder_input_ids = np.asarray([[1], [1]], dtype=int) * pt_model.config.decoder_start_token_id
            pt_input.update({"decoder_input_ids": torch.tensor(decoder_input_ids)})
            tf_input.update({"decoder_input_ids": tf.convert_to_tensor(decoder_input_ids)})

        # Confirms that cross loading PT weights into TF worked.
        pt_last_hidden_state = pt_model(**pt_input).last_hidden_state.detach().numpy()
        tf_from_pt_last_hidden_state = tf_from_pt_model(**tf_input).last_hidden_state.numpy()
        crossload_diff = np.max(np.abs(pt_last_hidden_state - tf_from_pt_last_hidden_state))
        if crossload_diff >= MAX_ERROR:
            raise ValueError(
                "The cross-loaded TF model has different last hidden states, something went wrong! (max difference ="
                f" {crossload_diff})"
            )

        # Save the weights in a TF format (if they don't exist) and confirms that the results are still good
        tf_weights_path = os.path.join(self._local_dir, TF_WEIGHTS_NAME)
        if not os.path.exists(tf_weights_path):
            tf_from_pt_model.save_weights(tf_weights_path)
        del tf_from_pt_model, pt_model  # will no longer be used, and may have a large memory footprint
        tf_model = TFAutoModel.from_pretrained(self._local_dir)
        tf_last_hidden_state = tf_model(**tf_input).last_hidden_state.numpy()
        converted_diff = np.max(np.abs(pt_last_hidden_state - tf_last_hidden_state))
        if converted_diff >= MAX_ERROR:
            raise ValueError(
                "The converted TF model has different last hidden states, something went wrong! (max difference ="
                f" {converted_diff})"
            )

        if not self._no_pr:
            # TODO: remove try/except when the upload to PR feature is released
            # (https://github.com/huggingface/huggingface_hub/pull/884)
            try:
                self._logger.warn("Uploading the weights into a new PR...")
                hub_pr_url = upload_file(
                    path_or_fileobj=tf_weights_path,
                    path_in_repo=TF_WEIGHTS_NAME,
                    repo_id=self._model_name,
                    create_pr=True,
                    pr_commit_summary="Add TF weights",
                    pr_commit_description=(
                        f"Validated by the `pt_to_tf` CLI. Max crossload hidden state difference={crossload_diff:.3e};"
                        f" Max converted hidden state difference={converted_diff:.3e}."
                    ),
                )
                self._logger.warn(f"PR open in {hub_pr_url}")
            except TypeError:
                self._logger.warn(
                    f"You can now open a PR in https://huggingface.co/{self._model_name}/discussions, manually"
                    f" uploading the file in {tf_weights_path}"
                )
