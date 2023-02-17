# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import copy
import inspect
import json
import os
import random
from pathlib import Path

from transformers import PreTrainedModel, TFPreTrainedModel
from transformers.utils import direct_transformers_import

from .pipelines.test_pipelines_audio_classification import AudioClassificationPipelineTests
from .pipelines.test_pipelines_feature_extraction import FeatureExtractionPipelineTests
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
from .pipelines.test_pipelines_image_classification import ImageClassificationPipelineTests
from .pipelines.test_pipelines_text_classification import TextClassificationPipelineTests
from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests


pipeline_test_mapping = {
    "audio-classification": {"test": AudioClassificationPipelineTests},
    "feature-extraction": {"test": FeatureExtractionPipelineTests},
    "fill-mask": {"test": FillMaskPipelineTests},
    "image-classification": {"test": ImageClassificationPipelineTests},
    "text-classification": {"test": TextClassificationPipelineTests},
    "text-generation": {"test": TextGenerationPipelineTests},
}

for task, task_info in pipeline_test_mapping.items():
    test = task_info["test"]
    task_info["mapping"] = {
        "pt": getattr(test, "model_mapping", None),
        "tf": getattr(test, "tf_model_mapping", None),
    }


TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(Path(__file__).parent.parent, "tests/utils/tiny_model_summary.json")
with open(TINY_MODEL_SUMMARY_FILE_PATH) as fp:
    tiny_model_summary = json.load(fp)


PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent, "src/transformers")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(PATH_TO_TRANSFORMERS)


class PipelineTesterMixin:
    model_tester = None
    pipieline_model_mapping = None
    supported_frameworks = ["pt", "tf"]

    @staticmethod
    def disable_irrelevant_tests():
        """Remove an irrelevant pipeline test (to a model test class) from the test methods

        This is done in a **black magic** way: During the explicit model test class declaration (e.g. `BertModelTest`),
        this method gets the local namespace dictionary seen by the frame `frame_local` at the class declaration moment,
        and modify the mapping dictionary frame_local["pipieline_model_mapping"]` by setting its values to `None` for
        some test method names.

        This method could only be called in a model test class body, outside any method, and after the attribute
        `pipieline_model_mapping` is defined.

        The advantage of this hack is:
            - we no longer need any skip statement for a pipeline test that is irrelevant to a model test class:
                - only the tests that are meant to be tested will be collected as tests to run
            - the test report won't contain such **skipped** tests (as they are not test methods anymore):
                - we should, and we only need, rely on the explicit mappings `pipieline_model_mapping`.
                - if we use skip statements, the report won't contain the test class names for these skipped tests, and
                  we don't get any useful information from the report. It's therefore better to make the report easy to
                  read instead of full of such usefulness lines.
        """
        # Get the frame in the caller, which should be in a model test class body outside any method
        frame = inspect.currentframe().f_back
        frame_locals = frame.f_locals
        if "__qualname__" not in frame_locals or not frame_locals["__qualname__"].endswith("ModelTest"):
            raise ValueError(
                f"This method is called in a place that is not in a model test class body: {frame.f_code.co_filename}::{frame.f_lineno}"
            )
        for task in pipeline_test_mapping.keys():
            _task = task.replace("-", "_")
            test_name = f"test_pipeline_{_task}"
            if task not in frame_locals["pipieline_model_mapping"]:
                frame_locals[test_name] = None

    def run_task_tests(self, task):
        """Run pipeline tests for a specific `task`

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
        """
        model_architectures = self.pipieline_model_mapping[task]
        if issubclass(model_architectures, (PreTrainedModel, TFPreTrainedModel)):
            model_architectures = (model_architectures,)
        if not isinstance(model_architectures, tuple):
            raise ValueError(f"`model_architectures` must be a tuple. Got {type(model_architectures)} instead.")

        for model_architecture in model_architectures:
            model_arch_name = model_architecture.__name__

            # Get the canonical name
            for _prefix in ["Flax", "TF"]:
                if model_arch_name.startswith(_prefix):
                    model_arch_name = model_arch_name[len(_prefix) :]
                    break

            tokenizer_names = []
            processor_names = []
            if model_arch_name in tiny_model_summary:
                tokenizer_names = tiny_model_summary[model_arch_name]["tokenizer_classes"]
                processor_names = tiny_model_summary[model_arch_name]["processor_classes"]
            # Adding `None` (if empty) so we can generate tests
            tokenizer_names = [None] if len(tokenizer_names) == 0 else tokenizer_names
            processor_names = [None] if len(processor_names) == 0 else processor_names

            repo_name = f"tiny-random-{model_arch_name}"

            self.run_model_pipeline_tests(task, repo_name, model_architecture, tokenizer_names, processor_names)

    def run_model_pipeline_tests(self, task, repo_name, model_architecture, tokenizer_names, processor_names):
        """Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class names

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            repo_name (`str`):
                A model repository id on the Hub.
            model_architecture (`type`):
                A subclass of `PretrainedModel` or `PretrainedModel`.
            tokenizer_names (`List[str]`):
                A list of names of a subclasses of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.
            processor_names (`List[str]`):
                A list of names of subclasses of `BaseImageProcessor` or `FeatureExtractionMixin`.
        """
        for tokenizer_name in tokenizer_names:
            for processor_name in processor_names:
                self.run_pipeline_test(task, repo_name, model_architecture, tokenizer_name, processor_name)

    def run_pipeline_test(self, task, repo_name, model_architecture, tokenizer_name, processor_name):
        """Run pipeline tests for a specific `task` with the give model class and tokenizer/processor class name

        The model will be loaded from a model repository on the Hub.

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
            repo_name (`str`):
                A model repository id on the Hub.
            model_architecture (`type`):
                A subclass of `PretrainedModel` or `PretrainedModel`.
            tokenizer_name (`str`):
                The name of a subclass of `PreTrainedTokenizerFast` or `PreTrainedTokenizer`.
            processor_name (`str`):
                The name of a subclass of `BaseImageProcessor` or `FeatureExtractionMixin`.
        """
        repo_id = f"hf-internal-testing/{repo_name}"

        tokenizer = None
        if tokenizer_name is not None:
            tokenizer_class = getattr(transformers_module, tokenizer_name)
            tokenizer = tokenizer_class.from_pretrained(repo_id)

        processor = None
        if processor_name is not None:
            processor_class = getattr(transformers_module, processor_name)
            # If the required packages (like `Pillow`) are not installed, this will fail.
            try:
                processor = processor_class.from_pretrained(repo_id)
            except Exception:
                self.skipTest(f"Test is skipped: Could not load the processor from `{repo_id}` with `processor_name`.")

        try:
            model = model_architecture.from_pretrained(repo_id)
        except Exception:
            self.skipTest(f"Test is skipped: Could not load the model from `{repo_id}` with `{model_architecture}`.")

        # validate
        validate_test_components(self, model, tokenizer, processor)

        if hasattr(model, "eval"):
            model = model.eval()

        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        task_test = pipeline_test_mapping[task]["test"]()

        pipeline, examples = task_test.get_test_pipeline(model, tokenizer, processor)
        if pipeline is None:
            # The test can disable itself, but it should be very marginal
            # Concerns: Wav2Vec2ForCTC without tokenizer test (FastTokenizer don't exist)
            self.skipTest("Test is skipped: Could not get the pipeline for testing.")

        task_test.run_pipeline_test(pipeline, examples)

        def run_batch_test(pipeline, examples):
            # Need to copy because `Conversation` are stateful
            if pipeline.tokenizer is not None and pipeline.tokenizer.pad_token_id is None:
                return  # No batching for this and it's OK

            # 10 examples with batch size 4 means there needs to be a unfinished batch
            # which is important for the unbatcher
            def data(n):
                for _ in range(n):
                    # Need to copy because Conversation object is mutated
                    yield copy.deepcopy(random.choice(examples))

            out = []
            for item in pipeline(data(10), batch_size=4):
                out.append(item)
            self.assertEqual(len(out), 10)

        run_batch_test(pipeline, examples)

    def test_pipeline_feature_extraction(self):
        self.run_task_tests(task="feature-extraction")

    def test_pipeline_audio_classification(self):
        self.run_task_tests(task="audio-classification")

    def test_pipeline_fill_mask(self):
        self.run_task_tests(task="fill-mask")

    def test_pipeline_image_classification(self):
        self.run_task_tests(task="image-classification")

    def test_pipeline_text_classification(self):
        self.run_task_tests(task="text-classification")

    def test_pipeline_text_generation(self):
        self.run_task_tests(task="text-generation")


def validate_test_components(test_case, model, tokenizer, processor):
    # TODO: Move this to tiny model creation script
    # head-specific (within a model type) necessary changes to the config
    # 1. for `BlenderbotForCausalLM`
    if model.__class__.__name__ == "BlenderbotForCausalLM":
        model.config.encoder_no_repeat_ngram_size = 0

    # TODO: Change the tiny model creation script: don't create models with problematic tokenizers
    # Avoid `IndexError` in embedding layers
    CONFIG_WITHOUT_VOCAB_SIZE = ["CanineConfig"]
    if tokenizer is not None:
        config_vocab_size = getattr(model.config, "vocab_size", None)
        # For CLIP-like models
        if config_vocab_size is None and hasattr(model.config, "text_config"):
            config_vocab_size = getattr(model.config.text_config, "vocab_size", None)
        if config_vocab_size is None and model.config.__class__.__name__ not in CONFIG_WITHOUT_VOCAB_SIZE:
            raise ValueError(
                "Could not determine `vocab_size` from model configuration while `tokenizer` is not `None`."
            )
        # TODO: Remove tiny models from the Hub which have problematic tokenizers (but still keep this block)
        if config_vocab_size is not None and len(tokenizer) > config_vocab_size:
            test_case.skipTest(
                f"Test is skipped: tokenizer (`{tokenizer.__class__.__name__}`) has {len(tokenizer)} tokens which is "
                f"greater than `config_vocab_size` ({config_vocab_size}). Something is wrong."
            )
