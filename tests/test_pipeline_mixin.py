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
import json
import os
import random
import unittest
from pathlib import Path

from transformers.testing_utils import (
    is_pipeline_test,
    require_decord,
    require_pytesseract,
    require_timm,
    require_torch,
    require_torch_or_tf,
    require_vision,
)
from transformers.utils import direct_transformers_import, logging

from .pipelines.test_pipelines_audio_classification import AudioClassificationPipelineTests
from .pipelines.test_pipelines_automatic_speech_recognition import AutomaticSpeechRecognitionPipelineTests
from .pipelines.test_pipelines_conversational import ConversationalPipelineTests
from .pipelines.test_pipelines_depth_estimation import DepthEstimationPipelineTests
from .pipelines.test_pipelines_document_question_answering import DocumentQuestionAnsweringPipelineTests
from .pipelines.test_pipelines_feature_extraction import FeatureExtractionPipelineTests
from .pipelines.test_pipelines_fill_mask import FillMaskPipelineTests
from .pipelines.test_pipelines_image_classification import ImageClassificationPipelineTests
from .pipelines.test_pipelines_image_segmentation import ImageSegmentationPipelineTests
from .pipelines.test_pipelines_image_to_text import ImageToTextPipelineTests
from .pipelines.test_pipelines_mask_generation import MaskGenerationPipelineTests
from .pipelines.test_pipelines_object_detection import ObjectDetectionPipelineTests
from .pipelines.test_pipelines_question_answering import QAPipelineTests
from .pipelines.test_pipelines_summarization import SummarizationPipelineTests
from .pipelines.test_pipelines_table_question_answering import TQAPipelineTests
from .pipelines.test_pipelines_text2text_generation import Text2TextGenerationPipelineTests
from .pipelines.test_pipelines_text_classification import TextClassificationPipelineTests
from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests
from .pipelines.test_pipelines_text_to_audio import TextToAudioPipelineTests
from .pipelines.test_pipelines_token_classification import TokenClassificationPipelineTests
from .pipelines.test_pipelines_translation import TranslationPipelineTests
from .pipelines.test_pipelines_video_classification import VideoClassificationPipelineTests
from .pipelines.test_pipelines_visual_question_answering import VisualQuestionAnsweringPipelineTests
from .pipelines.test_pipelines_zero_shot import ZeroShotClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_audio_classification import ZeroShotAudioClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_image_classification import ZeroShotImageClassificationPipelineTests
from .pipelines.test_pipelines_zero_shot_object_detection import ZeroShotObjectDetectionPipelineTests


pipeline_test_mapping = {
    "audio-classification": {"test": AudioClassificationPipelineTests},
    "automatic-speech-recognition": {"test": AutomaticSpeechRecognitionPipelineTests},
    "conversational": {"test": ConversationalPipelineTests},
    "depth-estimation": {"test": DepthEstimationPipelineTests},
    "document-question-answering": {"test": DocumentQuestionAnsweringPipelineTests},
    "feature-extraction": {"test": FeatureExtractionPipelineTests},
    "fill-mask": {"test": FillMaskPipelineTests},
    "image-classification": {"test": ImageClassificationPipelineTests},
    "image-segmentation": {"test": ImageSegmentationPipelineTests},
    "image-to-text": {"test": ImageToTextPipelineTests},
    "mask-generation": {"test": MaskGenerationPipelineTests},
    "object-detection": {"test": ObjectDetectionPipelineTests},
    "question-answering": {"test": QAPipelineTests},
    "summarization": {"test": SummarizationPipelineTests},
    "table-question-answering": {"test": TQAPipelineTests},
    "text2text-generation": {"test": Text2TextGenerationPipelineTests},
    "text-classification": {"test": TextClassificationPipelineTests},
    "text-generation": {"test": TextGenerationPipelineTests},
    "text-to-audio": {"test": TextToAudioPipelineTests},
    "token-classification": {"test": TokenClassificationPipelineTests},
    "translation": {"test": TranslationPipelineTests},
    "video-classification": {"test": VideoClassificationPipelineTests},
    "visual-question-answering": {"test": VisualQuestionAnsweringPipelineTests},
    "zero-shot": {"test": ZeroShotClassificationPipelineTests},
    "zero-shot-audio-classification": {"test": ZeroShotAudioClassificationPipelineTests},
    "zero-shot-image-classification": {"test": ZeroShotImageClassificationPipelineTests},
    "zero-shot-object-detection": {"test": ZeroShotObjectDetectionPipelineTests},
}

for task, task_info in pipeline_test_mapping.items():
    test = task_info["test"]
    task_info["mapping"] = {
        "pt": getattr(test, "model_mapping", None),
        "tf": getattr(test, "tf_model_mapping", None),
    }


# The default value `hf-internal-testing` is for running the pipeline testing against the tiny models on the Hub.
# For debugging purpose, we can specify a local path which is the `output_path` argument of a previous run of
# `utils/create_dummy_models.py`.
TRANSFORMERS_TINY_MODEL_PATH = os.environ.get("TRANSFORMERS_TINY_MODEL_PATH", "hf-internal-testing")
if TRANSFORMERS_TINY_MODEL_PATH == "hf-internal-testing":
    TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(Path(__file__).parent.parent, "tests/utils/tiny_model_summary.json")
else:
    TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(TRANSFORMERS_TINY_MODEL_PATH, "reports", "tiny_model_summary.json")
with open(TINY_MODEL_SUMMARY_FILE_PATH) as fp:
    tiny_model_summary = json.load(fp)


PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent, "src/transformers")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(PATH_TO_TRANSFORMERS)

logger = logging.get_logger(__name__)


class PipelineTesterMixin:
    model_tester = None
    pipeline_model_mapping = None
    supported_frameworks = ["pt", "tf"]

    def run_task_tests(self, task):
        """Run pipeline tests for a specific `task`

        Args:
            task (`str`):
                A task name. This should be a key in the mapping `pipeline_test_mapping`.
        """
        if task not in self.pipeline_model_mapping:
            self.skipTest(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: `{task}` is not in "
                f"`self.pipeline_model_mapping` for `{self.__class__.__name__}`."
            )

        model_architectures = self.pipeline_model_mapping[task]
        if not isinstance(model_architectures, tuple):
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
            commit = None
            if model_arch_name in tiny_model_summary:
                tokenizer_names = tiny_model_summary[model_arch_name]["tokenizer_classes"]
                processor_names = tiny_model_summary[model_arch_name]["processor_classes"]
                if "sha" in tiny_model_summary[model_arch_name]:
                    commit = tiny_model_summary[model_arch_name]["sha"]
            # Adding `None` (if empty) so we can generate tests
            tokenizer_names = [None] if len(tokenizer_names) == 0 else tokenizer_names
            processor_names = [None] if len(processor_names) == 0 else processor_names

            repo_name = f"tiny-random-{model_arch_name}"
            if TRANSFORMERS_TINY_MODEL_PATH != "hf-internal-testing":
                repo_name = model_arch_name

            self.run_model_pipeline_tests(
                task, repo_name, model_architecture, tokenizer_names, processor_names, commit
            )

    def run_model_pipeline_tests(self, task, repo_name, model_architecture, tokenizer_names, processor_names, commit):
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
        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        pipeline_test_class_name = pipeline_test_mapping[task]["test"].__name__

        for tokenizer_name in tokenizer_names:
            for processor_name in processor_names:
                if self.is_pipeline_test_to_skip(
                    pipeline_test_class_name,
                    model_architecture.config_class,
                    model_architecture,
                    tokenizer_name,
                    processor_name,
                ):
                    logger.warning(
                        f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: test is "
                        f"currently known to fail for: model `{model_architecture.__name__}` | tokenizer "
                        f"`{tokenizer_name}` | processor `{processor_name}`."
                    )
                    continue
                self.run_pipeline_test(task, repo_name, model_architecture, tokenizer_name, processor_name, commit)

    def run_pipeline_test(self, task, repo_name, model_architecture, tokenizer_name, processor_name, commit):
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
        repo_id = f"{TRANSFORMERS_TINY_MODEL_PATH}/{repo_name}"
        if TRANSFORMERS_TINY_MODEL_PATH != "hf-internal-testing":
            model_type = model_architecture.config_class.model_type
            repo_id = os.path.join(TRANSFORMERS_TINY_MODEL_PATH, model_type, repo_name)

        tokenizer = None
        if tokenizer_name is not None:
            tokenizer_class = getattr(transformers_module, tokenizer_name)
            tokenizer = tokenizer_class.from_pretrained(repo_id, revision=commit)

        processor = None
        if processor_name is not None:
            processor_class = getattr(transformers_module, processor_name)
            # If the required packages (like `Pillow` or `torchaudio`) are not installed, this will fail.
            try:
                processor = processor_class.from_pretrained(repo_id, revision=commit)
            except Exception:
                logger.warning(
                    f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not load the "
                    f"processor from `{repo_id}` with `{processor_name}`."
                )
                return

        # TODO: Maybe not upload such problematic tiny models to Hub.
        if tokenizer is None and processor is None:
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not find or load "
                f"any tokenizer / processor from `{repo_id}`."
            )
            return

        # TODO: We should check if a model file is on the Hub repo. instead.
        try:
            model = model_architecture.from_pretrained(repo_id, revision=commit)
        except Exception:
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not find or load "
                f"the model from `{repo_id}` with `{model_architecture}`."
            )
            return

        pipeline_test_class_name = pipeline_test_mapping[task]["test"].__name__
        if self.is_pipeline_test_to_skip_more(pipeline_test_class_name, model.config, model, tokenizer, processor):
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: test is "
                f"currently known to fail for: model `{model_architecture.__name__}` | tokenizer "
                f"`{tokenizer_name}` | processor `{processor_name}`."
            )
            return

        # validate
        validate_test_components(self, task, model, tokenizer, processor)

        if hasattr(model, "eval"):
            model = model.eval()

        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        task_test = pipeline_test_mapping[task]["test"]()

        pipeline, examples = task_test.get_test_pipeline(model, tokenizer, processor)
        if pipeline is None:
            # The test can disable itself, but it should be very marginal
            # Concerns: Wav2Vec2ForCTC without tokenizer test (FastTokenizer don't exist)
            logger.warning(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not get the "
                "pipeline for testing."
            )
            return

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

    @is_pipeline_test
    def test_pipeline_audio_classification(self):
        self.run_task_tests(task="audio-classification")

    @is_pipeline_test
    def test_pipeline_automatic_speech_recognition(self):
        self.run_task_tests(task="automatic-speech-recognition")

    @is_pipeline_test
    def test_pipeline_conversational(self):
        self.run_task_tests(task="conversational")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_depth_estimation(self):
        self.run_task_tests(task="depth-estimation")

    @is_pipeline_test
    @require_pytesseract
    @require_torch
    @require_vision
    def test_pipeline_document_question_answering(self):
        self.run_task_tests(task="document-question-answering")

    @is_pipeline_test
    def test_pipeline_feature_extraction(self):
        self.run_task_tests(task="feature-extraction")

    @is_pipeline_test
    def test_pipeline_fill_mask(self):
        self.run_task_tests(task="fill-mask")

    @is_pipeline_test
    @require_torch_or_tf
    @require_vision
    def test_pipeline_image_classification(self):
        self.run_task_tests(task="image-classification")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_image_segmentation(self):
        self.run_task_tests(task="image-segmentation")

    @is_pipeline_test
    @require_vision
    def test_pipeline_image_to_text(self):
        self.run_task_tests(task="image-to-text")

    @unittest.skip(reason="`run_pipeline_test` is currently not implemented.")
    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_mask_generation(self):
        self.run_task_tests(task="mask-generation")

    @is_pipeline_test
    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_object_detection(self):
        self.run_task_tests(task="object-detection")

    @is_pipeline_test
    def test_pipeline_question_answering(self):
        self.run_task_tests(task="question-answering")

    @is_pipeline_test
    def test_pipeline_summarization(self):
        self.run_task_tests(task="summarization")

    @is_pipeline_test
    def test_pipeline_table_question_answering(self):
        self.run_task_tests(task="table-question-answering")

    @is_pipeline_test
    def test_pipeline_text2text_generation(self):
        self.run_task_tests(task="text2text-generation")

    @is_pipeline_test
    def test_pipeline_text_classification(self):
        self.run_task_tests(task="text-classification")

    @is_pipeline_test
    @require_torch_or_tf
    def test_pipeline_text_generation(self):
        self.run_task_tests(task="text-generation")

    @is_pipeline_test
    @require_torch
    def test_pipeline_text_to_audio(self):
        self.run_task_tests(task="text-to-audio")

    @is_pipeline_test
    def test_pipeline_token_classification(self):
        self.run_task_tests(task="token-classification")

    @is_pipeline_test
    def test_pipeline_translation(self):
        self.run_task_tests(task="translation")

    @is_pipeline_test
    @require_torch_or_tf
    @require_vision
    @require_decord
    def test_pipeline_video_classification(self):
        self.run_task_tests(task="video-classification")

    @is_pipeline_test
    @require_torch
    @require_vision
    def test_pipeline_visual_question_answering(self):
        self.run_task_tests(task="visual-question-answering")

    @is_pipeline_test
    def test_pipeline_zero_shot(self):
        self.run_task_tests(task="zero-shot")

    @is_pipeline_test
    @require_torch
    def test_pipeline_zero_shot_audio_classification(self):
        self.run_task_tests(task="zero-shot-audio-classification")

    @is_pipeline_test
    @require_vision
    def test_pipeline_zero_shot_image_classification(self):
        self.run_task_tests(task="zero-shot-image-classification")

    @is_pipeline_test
    @require_vision
    @require_torch
    def test_pipeline_zero_shot_object_detection(self):
        self.run_task_tests(task="zero-shot-object-detection")

    # This contains the test cases to be skipped without model architecture being involved.
    def is_pipeline_test_to_skip(
        self, pipeline_test_casse_name, config_class, model_architecture, tokenizer_name, processor_name
    ):
        """Skip some tests based on the classes or their names without the instantiated objects.

        This is to avoid calling `from_pretrained` (so reducing the runtime) if we already know the tests will fail.
        """
        # No fix is required for this case.
        if (
            pipeline_test_casse_name == "DocumentQuestionAnsweringPipelineTests"
            and tokenizer_name is not None
            and not tokenizer_name.endswith("Fast")
        ):
            # `DocumentQuestionAnsweringPipelineTests` requires a fast tokenizer.
            return True

        return False

    def is_pipeline_test_to_skip_more(self, pipeline_test_casse_name, config, model, tokenizer, processor):  # noqa
        """Skip some more tests based on the information from the instantiated objects."""
        # No fix is required for this case.
        if (
            pipeline_test_casse_name == "QAPipelineTests"
            and tokenizer is not None
            and getattr(tokenizer, "pad_token", None) is None
            and not tokenizer.__class__.__name__.endswith("Fast")
        ):
            # `QAPipelineTests` doesn't work with a slow tokenizer that has no pad token.
            return True

        return False


def validate_test_components(test_case, task, model, tokenizer, processor):
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
