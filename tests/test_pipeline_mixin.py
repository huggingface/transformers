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
from transformers.utils import direct_transformers_import

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
from .pipelines.test_pipelines_object_detection import ObjectDetectionPipelineTests
from .pipelines.test_pipelines_question_answering import QAPipelineTests
from .pipelines.test_pipelines_summarization import SummarizationPipelineTests
from .pipelines.test_pipelines_table_question_answering import TQAPipelineTests
from .pipelines.test_pipelines_text2text_generation import Text2TextGenerationPipelineTests
from .pipelines.test_pipelines_text_classification import TextClassificationPipelineTests
from .pipelines.test_pipelines_text_generation import TextGenerationPipelineTests
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
    "object-detection": {"test": ObjectDetectionPipelineTests},
    "question-answering": {"test": QAPipelineTests},
    "summarization": {"test": SummarizationPipelineTests},
    "table-question-answering": {"test": TQAPipelineTests},
    "text2text-generation": {"test": Text2TextGenerationPipelineTests},
    "text-classification": {"test": TextClassificationPipelineTests},
    "text-generation": {"test": TextGenerationPipelineTests},
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


TINY_MODEL_SUMMARY_FILE_PATH = os.path.join(Path(__file__).parent.parent, "tests/utils/tiny_model_summary.json")
with open(TINY_MODEL_SUMMARY_FILE_PATH) as fp:
    tiny_model_summary = json.load(fp)


PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent, "src/transformers")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(PATH_TO_TRANSFORMERS)


@is_pipeline_test
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
        # Get an instance of the corresponding class `XXXPipelineTests` in order to use `get_test_pipeline` and
        # `run_pipeline_test`.
        pipeline_test_class_name = pipeline_test_mapping[task]["test"].__name__

        for tokenizer_name in tokenizer_names:
            for processor_name in processor_names:
                if is_test_to_skip(
                    pipeline_test_class_name,
                    model_architecture.config_class,
                    model_architecture,
                    tokenizer_name,
                    processor_name,
                ):
                    self.skipTest(
                        f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: test is "
                        f"currently known to fail for: model `{model_architecture.__name__}` | tokenizer "
                        f"`{tokenizer_name}` | processor `{processor_name}`."
                    )
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
            # If the required packages (like `Pillow` or `torchaudio`) are not installed, this will fail.
            try:
                processor = processor_class.from_pretrained(repo_id)
            except Exception:
                self.skipTest(
                    f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not load the "
                    f"processor from `{repo_id}` with `{processor_name}`."
                )

        # TODO: Maybe not upload such problematic tiny models to Hub.
        if tokenizer is None and processor is None:
            self.skipTest(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not find or load "
                f"any tokenizer / processor from `{repo_id}`."
            )

        # TODO: We should check if a model file is on the Hub repo. instead.
        try:
            model = model_architecture.from_pretrained(repo_id)
        except Exception:
            self.skipTest(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not find or load "
                f"the model from `{repo_id}` with `{model_architecture}`."
            )

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
            self.skipTest(
                f"{self.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: Could not get the "
                "pipeline for testing."
            )

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

    @require_torch
    def test_pipeline_audio_classification(self):
        self.run_task_tests(task="audio-classification")

    def test_pipeline_automatic_speech_recognition(self):
        self.run_task_tests(task="automatic-speech-recognition")

    def test_pipeline_conversational(self):
        self.run_task_tests(task="conversational")

    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_depth_estimation(self):
        self.run_task_tests(task="depth-estimation")

    @require_pytesseract
    @require_torch
    @require_vision
    def test_pipeline_document_question_answering(self):
        self.run_task_tests(task="document-question-answering")

    def test_pipeline_feature_extraction(self):
        self.run_task_tests(task="feature-extraction")

    def test_pipeline_fill_mask(self):
        self.run_task_tests(task="fill-mask")

    @require_torch_or_tf
    @require_vision
    def test_pipeline_image_classification(self):
        self.run_task_tests(task="image-classification")

    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_image_segmentation(self):
        self.run_task_tests(task="image-segmentation")

    @require_vision
    def test_pipeline_image_to_text(self):
        self.run_task_tests(task="image-to-text")

    @require_vision
    @require_timm
    @require_torch
    def test_pipeline_object_detection(self):
        self.run_task_tests(task="object-detection")

    def test_pipeline_question_answering(self):
        self.run_task_tests(task="question-answering")

    def test_pipeline_summarization(self):
        self.run_task_tests(task="summarization")

    def test_pipeline_table_question_answering(self):
        self.run_task_tests(task="table-question-answering")

    def test_pipeline_text2text_generation(self):
        self.run_task_tests(task="text2text-generation")

    def test_pipeline_text_classification(self):
        self.run_task_tests(task="text-classification")

    @require_torch_or_tf
    def test_pipeline_text_generation(self):
        self.run_task_tests(task="text-generation")

    def test_pipeline_token_classification(self):
        self.run_task_tests(task="token-classification")

    def test_pipeline_translation(self):
        self.run_task_tests(task="translation")

    @require_torch_or_tf
    @require_vision
    @require_decord
    def test_pipeline_video_classification(self):
        self.run_task_tests(task="video-classification")

    @require_torch
    @require_vision
    def test_pipeline_visual_question_answering(self):
        self.run_task_tests(task="visual-question-answering")

    def test_pipeline_zero_shot(self):
        self.run_task_tests(task="zero-shot")

    @require_torch
    def test_pipeline_zero_shot_audio_classification(self):
        self.run_task_tests(task="zero-shot-audio-classification")

    @require_vision
    def test_pipeline_zero_shot_image_classification(self):
        self.run_task_tests(task="zero-shot-image-classification")

    @require_vision
    @require_torch
    def test_pipeline_zero_shot_object_detection(self):
        self.run_task_tests(task="zero-shot-object-detection")


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
        # TODO: Remove tiny models from the Hub which have problematic tokenizers (but still keep this block)
        if config_vocab_size is not None and len(tokenizer) > config_vocab_size:
            test_case.skipTest(
                f"{test_case.__class__.__name__}::test_pipeline_{task.replace('-', '_')} is skipped: tokenizer "
                f"(`{tokenizer.__class__.__name__}`) has {len(tokenizer)} tokens which is greater than "
                f"`config_vocab_size` ({config_vocab_size}). Something is wrong."
            )


def is_test_to_skip(test_casse_name, config_class, model_architecture, tokenizer_name, processor_name):
    """Some tests are just not working"""

    to_skip = False

    if config_class.__name__ == "RoCBertConfig" and test_casse_name in [
        "FillMaskPipelineTests",
        "FeatureExtractionPipelineTests",
        "TextClassificationPipelineTests",
        "TokenClassificationPipelineTests",
    ]:
        # Get error: IndexError: index out of range in self.
        # `word_shape_file` and `word_pronunciation_file` should be shrunk during tiny model creation,
        # otherwise `IndexError` could occur in some embedding layers. Skip for now until this model has
        # more usage.
        to_skip = True
    elif config_class.__name__ in ["LayoutLMv3Config", "LiltConfig"]:
        # Get error: ValueError: Words must be of type `List[str]`. Previously, `LayoutLMv3` is not
        # used in pipeline tests as it could not find a checkpoint
        # TODO: check and fix if possible
        to_skip = True
    # config/model class we decide to skip
    elif config_class.__name__ in ["TapasConfig"]:
        # Get error: AssertionError: Table must be of type pd.DataFrame. Also, the tiny model has large
        # vocab size as the fast tokenizer could not be converted. Previous, `Tapas` is not used in
        # pipeline tests due to the same reason.
        # TODO: check and fix if possible
        to_skip = True

    # TODO: check and fix if possible
    if not to_skip and tokenizer_name is not None:
        if (
            test_casse_name == "QAPipelineTests"
            and not tokenizer_name.endswith("Fast")
            and config_class.__name__
            in [
                "FlaubertConfig",
                "GPTJConfig",
                "LongformerConfig",
                "MvpConfig",
                "OPTConfig",
                "ReformerConfig",
                "XLMConfig",
            ]
        ):
            # `QAPipelineTests` fails for a few models when the slower tokenizer are used.
            # (The slower tokenizers were never used for pipeline tests before the pipeline testing rework)
            # TODO: check (and possibly fix) the `QAPipelineTests` with slower tokenizer
            to_skip = True
        elif test_casse_name == "ZeroShotClassificationPipelineTests" and config_class.__name__ in [
            "CTRLConfig",
            "OpenAIGPTConfig",
        ]:
            # Get `tokenizer does not have a padding token` error for both fast/slow tokenizers.
            # `CTRLConfig` and `OpenAIGPTConfig` were never used in pipeline tests, either because of a missing
            # checkpoint or because a tiny config could not be created
            to_skip = True
        elif test_casse_name == "TranslationPipelineTests" and config_class.__name__ in [
            "M2M100Config",
            "PLBartConfig",
        ]:
            # Get `ValueError: Translation requires a `src_lang` and a `tgt_lang` for this model`.
            # `M2M100Config` and `PLBartConfig` were never used in pipeline tests: cannot create a simple tokenizer
            to_skip = True
        elif test_casse_name == "TextGenerationPipelineTests" and config_class.__name__ in [
            "ProphetNetConfig",
            "TransfoXLConfig",
        ]:
            # Get `ValueError: AttributeError: 'NoneType' object has no attribute 'new_ones'` or `AssertionError`.
            # `TransfoXLConfig` and `ProphetNetConfig` were never used in pipeline tests: cannot create a simple
            # tokenizer.
            to_skip = True
        elif test_casse_name == "FillMaskPipelineTests" and config_class.__name__ in [
            "FlaubertConfig",
            "XLMConfig",
        ]:
            # Get `ValueError: AttributeError: 'NoneType' object has no attribute 'new_ones'` or `AssertionError`.
            # `FlaubertConfig` and `TransfoXLConfig` were never used in pipeline tests: cannot create a simple
            # tokenizer
            to_skip = True
        elif test_casse_name == "TextGenerationPipelineTests" and model_architecture.__name__ in [
            "TFRoFormerForCausalLM"
        ]:
            # TODO: add `prepare_inputs_for_generation` for `TFRoFormerForCausalLM`
            to_skip = True
        elif test_casse_name == "QAPipelineTests" and model_architecture.__name__ in ["FNetForQuestionAnswering"]:
            # TODO: The change in `base.py` in the PR #21132 (https://github.com/huggingface/transformers/pull/21132)
            #       fails this test case. Skip for now - a fix for this along with the initial changes in PR #20426 is
            #       too much. Let `ydshieh` to fix it ASAP once #20426 is merged.
            to_skip = True
        elif config_class.__name__ == "LayoutLMv2Config" and test_casse_name in [
            "QAPipelineTests",
            "TextClassificationPipelineTests",
            "TokenClassificationPipelineTests",
            "ZeroShotClassificationPipelineTests",
        ]:
            # `LayoutLMv2Config` was never used in pipeline tests (`test_pt_LayoutLMv2Config_XXX`) due to lack of tiny
            # config. With new tiny model creation, it is available, but we need to fix the failed tests.
            to_skip = True
        elif test_casse_name == "DocumentQuestionAnsweringPipelineTests" and not tokenizer_name.endswith("Fast"):
            # This pipeline uses `sequence_ids()` which is only available for fast tokenizers.
            to_skip = True

    return to_skip
