# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import importlib
import logging
import os
import random
import sys
import tempfile
import unittest
from abc import abstractmethod
from pathlib import Path
from unittest import skipIf

import datasets
import numpy as np
import requests
from huggingface_hub import HfFolder, Repository, create_repo, delete_repo, set_access_token
from requests.exceptions import HTTPError

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    TextClassificationPipeline,
    TFAutoModelForSequenceClassification,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY, get_task
from transformers.pipelines.base import Pipeline, _pad
from transformers.testing_utils import (
    TOKEN,
    USER,
    CaptureLogger,
    RequestCounter,
    is_staging_test,
    nested_simplify,
    require_tensorflow_probability,
    require_tf,
    require_torch,
    require_torch_or_tf,
    slow,
)
from transformers.utils import is_tf_available, is_torch_available
from transformers.utils import logging as transformers_logging


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_pipeline import PairClassificationPipeline  # noqa E402


logger = logging.getLogger(__name__)


PATH_TO_TRANSFORMERS = os.path.join(Path(__file__).parent.parent.parent, "src/transformers")


# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
    submodule_search_locations=[PATH_TO_TRANSFORMERS],
)
transformers_module = spec.loader.load_module()


class ANY:
    def __init__(self, *_types):
        self._types = _types

    def __eq__(self, other):
        return isinstance(other, self._types)

    def __repr__(self):
        return f"ANY({', '.join(_type.__name__ for _type in self._types)})"


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
                f"Ignore {model.__class__.__name__}: `tokenizer` ({tokenizer.__class__.__name__}) has"
                f" {len(tokenizer)} tokens which is greater than `config_vocab_size`"
                f" ({config_vocab_size}). Something is wrong."
            )


class PipelineTestCaseMeta(type):
    def __new__(mcs, name, bases, dct):
        def gen_test(repo_name, model_architecture, tokenizer_name, processor_name):
            @skipIf(
                tokenizer_name is None and processor_name is None,
                f"Ignore {model_architecture.__name__}: no processor class is provided (tokenizer, image processor,"
                " feature extractor, etc)",
            )
            def test(self):
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
                        self.skipTest(f"Ignore {model_architecture.__name__}: could not load the model from {repo_id}")

                try:
                    model = model_architecture.from_pretrained(repo_id)
                except Exception:
                    self.skipTest(f"Ignore {model_architecture.__name__}: could not load the model from {repo_id}")

                # validate
                validate_test_components(self, model, tokenizer, processor)

                if hasattr(model, "eval"):
                    model = model.eval()

                pipeline, examples = self.get_test_pipeline(model, tokenizer, processor)
                if pipeline is None:
                    # The test can disable itself, but it should be very marginal
                    # Concerns: Wav2Vec2ForCTC without tokenizer test (FastTokenizer don't exist)
                    self.skipTest(f"Ignore {model_architecture.__name__}: could not create the pipeline")
                self.run_pipeline_test(pipeline, examples)

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

            return test

        # Download tiny model summary (used to avoid requesting from Hub too many times)
        url = "https://huggingface.co/datasets/hf-internal-testing/tiny-random-model-summary/raw/main/processor_classes.json"
        tiny_model_summary = requests.get(url).json()

        for prefix, key in [("pt", "model_mapping"), ("tf", "tf_model_mapping")]:
            mapping = dct.get(key, {})
            if mapping:
                for config_class, model_architectures in mapping.items():
                    if not isinstance(model_architectures, tuple):
                        model_architectures = (model_architectures,)

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
                        for tokenizer_name in tokenizer_names:
                            for processor_name in processor_names:
                                if is_test_to_skip(
                                    name, config_class, model_architecture, tokenizer_name, processor_name
                                ):
                                    continue
                                test_name = f"test_{prefix}_{config_class.__name__}_{model_architecture.__name__}_{tokenizer_name}_{processor_name}"
                                dct[test_name] = gen_test(
                                    repo_name, model_architecture, tokenizer_name, processor_name
                                )

        @abstractmethod
        def inner(self):
            raise NotImplementedError("Not implemented test")

        # Force these 2 methods to exist
        dct["test_small_model_pt"] = dct.get("test_small_model_pt", inner)
        dct["test_small_model_tf"] = dct.get("test_small_model_tf", inner)

        return type.__new__(mcs, name, bases, dct)


class CommonPipelineTest(unittest.TestCase):
    @require_torch
    def test_pipeline_iteration(self):
        from torch.utils.data import Dataset

        class MyDataset(Dataset):
            data = [
                "This is a test",
                "This restaurant is great",
                "This restaurant is awful",
            ]

            def __len__(self):
                return 3

            def __getitem__(self, i):
                return self.data[i]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )
        dataset = MyDataset()
        for output in text_classifier(dataset):
            self.assertEqual(output, {"label": ANY(str), "score": ANY(float)})

    @require_torch
    def test_check_task_auto_inference(self):
        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")

        self.assertIsInstance(pipe, TextClassificationPipeline)

    @require_torch
    def test_pipeline_batch_size_global(self):
        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")
        self.assertEqual(pipe._batch_size, None)
        self.assertEqual(pipe._num_workers, None)

        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert", batch_size=2, num_workers=1)
        self.assertEqual(pipe._batch_size, 2)
        self.assertEqual(pipe._num_workers, 1)

    @require_torch
    def test_pipeline_pathlike(self):
        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")
        with tempfile.TemporaryDirectory() as d:
            pipe.save_pretrained(d)
            path = Path(d)
            newpipe = pipeline(task="text-classification", model=path)
        self.assertIsInstance(newpipe, TextClassificationPipeline)

    @require_torch
    def test_pipeline_override(self):
        class MyPipeline(TextClassificationPipeline):
            pass

        text_classifier = pipeline(model="hf-internal-testing/tiny-random-distilbert", pipeline_class=MyPipeline)

        self.assertIsInstance(text_classifier, MyPipeline)

    def test_check_task(self):
        task = get_task("gpt2")
        self.assertEqual(task, "text-generation")

        with self.assertRaises(RuntimeError):
            # Wrong framework
            get_task("espnet/siddhana_slurp_entity_asr_train_asr_conformer_raw_en_word_valid.acc.ave_10best")

    @require_torch
    def test_iterator_data(self):
        def data(n: int):
            for _ in range(n):
                yield "This is a test"

        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert")

        results = []
        for out in pipe(data(10)):
            self.assertEqual(nested_simplify(out), {"label": "LABEL_0", "score": 0.504})
            results.append(out)
        self.assertEqual(len(results), 10)

        # When using multiple workers on streamable data it should still work
        # This will force using `num_workers=1` with a warning for now.
        results = []
        for out in pipe(data(10), num_workers=2):
            self.assertEqual(nested_simplify(out), {"label": "LABEL_0", "score": 0.504})
            results.append(out)
        self.assertEqual(len(results), 10)

    @require_tf
    def test_iterator_data_tf(self):
        def data(n: int):
            for _ in range(n):
                yield "This is a test"

        pipe = pipeline(model="hf-internal-testing/tiny-random-distilbert", framework="tf")
        out = pipe("This is a test")
        results = []
        for out in pipe(data(10)):
            self.assertEqual(nested_simplify(out), {"label": "LABEL_0", "score": 0.504})
            results.append(out)
        self.assertEqual(len(results), 10)

    @require_torch
    def test_unbatch_attentions_hidden_states(self):
        model = DistilBertForSequenceClassification.from_pretrained(
            "hf-internal-testing/tiny-random-distilbert", output_hidden_states=True, output_attentions=True
        )
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-distilbert")
        text_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

        # Used to throw an error because `hidden_states` are a tuple of tensors
        # instead of the expected tensor.
        outputs = text_classifier(["This is great !"] * 20, batch_size=32)
        self.assertEqual(len(outputs), 20)


class PipelineScikitCompatTest(unittest.TestCase):
    @require_torch
    def test_pipeline_predict_pt(self):
        data = ["This is a test"]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )

        expected_output = [{"label": ANY(str), "score": ANY(float)}]
        actual_output = text_classifier.predict(data)
        self.assertEqual(expected_output, actual_output)

    @require_tf
    def test_pipeline_predict_tf(self):
        data = ["This is a test"]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )

        expected_output = [{"label": ANY(str), "score": ANY(float)}]
        actual_output = text_classifier.predict(data)
        self.assertEqual(expected_output, actual_output)

    @require_torch
    def test_pipeline_transform_pt(self):
        data = ["This is a test"]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="pt"
        )

        expected_output = [{"label": ANY(str), "score": ANY(float)}]
        actual_output = text_classifier.transform(data)
        self.assertEqual(expected_output, actual_output)

    @require_tf
    def test_pipeline_transform_tf(self):
        data = ["This is a test"]

        text_classifier = pipeline(
            task="text-classification", model="hf-internal-testing/tiny-random-distilbert", framework="tf"
        )

        expected_output = [{"label": ANY(str), "score": ANY(float)}]
        actual_output = text_classifier.transform(data)
        self.assertEqual(expected_output, actual_output)


class PipelinePadTest(unittest.TestCase):
    @require_torch
    def test_pipeline_padding(self):
        import torch

        items = [
            {
                "label": "label1",
                "input_ids": torch.LongTensor([[1, 23, 24, 2]]),
                "attention_mask": torch.LongTensor([[0, 1, 1, 0]]),
            },
            {
                "label": "label2",
                "input_ids": torch.LongTensor([[1, 23, 24, 43, 44, 2]]),
                "attention_mask": torch.LongTensor([[0, 1, 1, 1, 1, 0]]),
            },
        ]

        self.assertEqual(_pad(items, "label", 0, "right"), ["label1", "label2"])
        self.assertTrue(
            torch.allclose(
                _pad(items, "input_ids", 10, "right"),
                torch.LongTensor([[1, 23, 24, 2, 10, 10], [1, 23, 24, 43, 44, 2]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                _pad(items, "input_ids", 10, "left"),
                torch.LongTensor([[10, 10, 1, 23, 24, 2], [1, 23, 24, 43, 44, 2]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                _pad(items, "attention_mask", 0, "right"), torch.LongTensor([[0, 1, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0]])
            )
        )

    @require_torch
    def test_pipeline_image_padding(self):
        import torch

        items = [
            {
                "label": "label1",
                "pixel_values": torch.zeros((1, 3, 10, 10)),
            },
            {
                "label": "label2",
                "pixel_values": torch.zeros((1, 3, 10, 10)),
            },
        ]

        self.assertEqual(_pad(items, "label", 0, "right"), ["label1", "label2"])
        self.assertTrue(
            torch.allclose(
                _pad(items, "pixel_values", 10, "right"),
                torch.zeros((2, 3, 10, 10)),
            )
        )

    @require_torch
    def test_pipeline_offset_mapping(self):
        import torch

        items = [
            {
                "offset_mappings": torch.zeros([1, 11, 2], dtype=torch.long),
            },
            {
                "offset_mappings": torch.zeros([1, 4, 2], dtype=torch.long),
            },
        ]

        self.assertTrue(
            torch.allclose(
                _pad(items, "offset_mappings", 0, "right"),
                torch.zeros((2, 11, 2), dtype=torch.long),
            ),
        )


class PipelineUtilsTest(unittest.TestCase):
    @require_torch
    def test_pipeline_dataset(self):
        from transformers.pipelines.pt_utils import PipelineDataset

        dummy_dataset = [0, 1, 2, 3]

        def add(number, extra=0):
            return number + extra

        dataset = PipelineDataset(dummy_dataset, add, {"extra": 2})
        self.assertEqual(len(dataset), 4)
        outputs = [dataset[i] for i in range(4)]
        self.assertEqual(outputs, [2, 3, 4, 5])

    @require_torch
    def test_pipeline_iterator(self):
        from transformers.pipelines.pt_utils import PipelineIterator

        dummy_dataset = [0, 1, 2, 3]

        def add(number, extra=0):
            return number + extra

        dataset = PipelineIterator(dummy_dataset, add, {"extra": 2})
        self.assertEqual(len(dataset), 4)

        outputs = [item for item in dataset]
        self.assertEqual(outputs, [2, 3, 4, 5])

    @require_torch
    def test_pipeline_iterator_no_len(self):
        from transformers.pipelines.pt_utils import PipelineIterator

        def dummy_dataset():
            for i in range(4):
                yield i

        def add(number, extra=0):
            return number + extra

        dataset = PipelineIterator(dummy_dataset(), add, {"extra": 2})
        with self.assertRaises(TypeError):
            len(dataset)

        outputs = [item for item in dataset]
        self.assertEqual(outputs, [2, 3, 4, 5])

    @require_torch
    def test_pipeline_batch_unbatch_iterator(self):
        from transformers.pipelines.pt_utils import PipelineIterator

        dummy_dataset = [{"id": [0, 1, 2]}, {"id": [3]}]

        def add(number, extra=0):
            return {"id": [i + extra for i in number["id"]]}

        dataset = PipelineIterator(dummy_dataset, add, {"extra": 2}, loader_batch_size=3)

        outputs = [item for item in dataset]
        self.assertEqual(outputs, [{"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}])

    @require_torch
    def test_pipeline_batch_unbatch_iterator_tensors(self):
        import torch

        from transformers.pipelines.pt_utils import PipelineIterator

        dummy_dataset = [{"id": torch.LongTensor([[10, 20], [0, 1], [0, 2]])}, {"id": torch.LongTensor([[3]])}]

        def add(number, extra=0):
            return {"id": number["id"] + extra}

        dataset = PipelineIterator(dummy_dataset, add, {"extra": 2}, loader_batch_size=3)

        outputs = [item for item in dataset]
        self.assertEqual(
            nested_simplify(outputs), [{"id": [[12, 22]]}, {"id": [[2, 3]]}, {"id": [[2, 4]]}, {"id": [[5]]}]
        )

    @require_torch
    def test_pipeline_chunk_iterator(self):
        from transformers.pipelines.pt_utils import PipelineChunkIterator

        def preprocess_chunk(n: int):
            for i in range(n):
                yield i

        dataset = [2, 3]

        dataset = PipelineChunkIterator(dataset, preprocess_chunk, {}, loader_batch_size=3)

        outputs = [item for item in dataset]

        self.assertEqual(outputs, [0, 1, 0, 1, 2])

    @require_torch
    def test_pipeline_pack_iterator(self):
        from transformers.pipelines.pt_utils import PipelinePackIterator

        def pack(item):
            return {"id": item["id"] + 1, "is_last": item["is_last"]}

        dataset = [
            {"id": 0, "is_last": False},
            {"id": 1, "is_last": True},
            {"id": 0, "is_last": False},
            {"id": 1, "is_last": False},
            {"id": 2, "is_last": True},
        ]

        dataset = PipelinePackIterator(dataset, pack, {})

        outputs = [item for item in dataset]
        self.assertEqual(
            outputs,
            [
                [
                    {"id": 1},
                    {"id": 2},
                ],
                [
                    {"id": 1},
                    {"id": 2},
                    {"id": 3},
                ],
            ],
        )

    @require_torch
    def test_pipeline_pack_unbatch_iterator(self):
        from transformers.pipelines.pt_utils import PipelinePackIterator

        dummy_dataset = [{"id": [0, 1, 2], "is_last": [False, True, False]}, {"id": [3], "is_last": [True]}]

        def add(number, extra=0):
            return {"id": [i + extra for i in number["id"]], "is_last": number["is_last"]}

        dataset = PipelinePackIterator(dummy_dataset, add, {"extra": 2}, loader_batch_size=3)

        outputs = [item for item in dataset]
        self.assertEqual(outputs, [[{"id": 2}, {"id": 3}], [{"id": 4}, {"id": 5}]])

        # is_false Across batch
        dummy_dataset = [{"id": [0, 1, 2], "is_last": [False, False, False]}, {"id": [3], "is_last": [True]}]

        def add(number, extra=0):
            return {"id": [i + extra for i in number["id"]], "is_last": number["is_last"]}

        dataset = PipelinePackIterator(dummy_dataset, add, {"extra": 2}, loader_batch_size=3)

        outputs = [item for item in dataset]
        self.assertEqual(outputs, [[{"id": 2}, {"id": 3}, {"id": 4}, {"id": 5}]])

    @slow
    @require_torch
    def test_load_default_pipelines_pt(self):
        import torch

        from transformers.pipelines import SUPPORTED_TASKS

        set_seed_fn = lambda: torch.manual_seed(0)  # noqa: E731
        for task in SUPPORTED_TASKS.keys():
            if task == "table-question-answering":
                # test table in seperate test due to more dependencies
                continue

            self.check_default_pipeline(task, "pt", set_seed_fn, self.check_models_equal_pt)

    @slow
    @require_tf
    def test_load_default_pipelines_tf(self):
        import tensorflow as tf

        from transformers.pipelines import SUPPORTED_TASKS

        set_seed_fn = lambda: tf.random.set_seed(0)  # noqa: E731
        for task in SUPPORTED_TASKS.keys():
            if task == "table-question-answering":
                # test table in seperate test due to more dependencies
                continue

            self.check_default_pipeline(task, "tf", set_seed_fn, self.check_models_equal_tf)

    @slow
    @require_torch
    def test_load_default_pipelines_pt_table_qa(self):
        import torch

        set_seed_fn = lambda: torch.manual_seed(0)  # noqa: E731
        self.check_default_pipeline("table-question-answering", "pt", set_seed_fn, self.check_models_equal_pt)

    @slow
    @require_tf
    @require_tensorflow_probability
    def test_load_default_pipelines_tf_table_qa(self):
        import tensorflow as tf

        set_seed_fn = lambda: tf.random.set_seed(0)  # noqa: E731
        self.check_default_pipeline("table-question-answering", "tf", set_seed_fn, self.check_models_equal_tf)

    def check_default_pipeline(self, task, framework, set_seed_fn, check_models_equal_fn):
        from transformers.pipelines import SUPPORTED_TASKS, pipeline

        task_dict = SUPPORTED_TASKS[task]
        # test to compare pipeline to manually loading the respective model
        model = None
        relevant_auto_classes = task_dict[framework]

        if len(relevant_auto_classes) == 0:
            # task has no default
            logger.debug(f"{task} in {framework} has no default")
            return

        # by default use first class
        auto_model_cls = relevant_auto_classes[0]

        # retrieve correct model ids
        if task == "translation":
            # special case for translation pipeline which has multiple languages
            model_ids = []
            revisions = []
            tasks = []
            for translation_pair in task_dict["default"].keys():
                model_id, revision = task_dict["default"][translation_pair]["model"][framework]

                model_ids.append(model_id)
                revisions.append(revision)
                tasks.append(task + f"_{'_to_'.join(translation_pair)}")
        else:
            # normal case - non-translation pipeline
            model_id, revision = task_dict["default"]["model"][framework]

            model_ids = [model_id]
            revisions = [revision]
            tasks = [task]

        # check for equality
        for model_id, revision, task in zip(model_ids, revisions, tasks):
            # load default model
            try:
                set_seed_fn()
                model = auto_model_cls.from_pretrained(model_id, revision=revision)
            except ValueError:
                # first auto class is possible not compatible with model, go to next model class
                auto_model_cls = relevant_auto_classes[1]
                set_seed_fn()
                model = auto_model_cls.from_pretrained(model_id, revision=revision)

            # load default pipeline
            set_seed_fn()
            default_pipeline = pipeline(task, framework=framework)

            # compare pipeline model with default model
            models_are_equal = check_models_equal_fn(default_pipeline.model, model)
            self.assertTrue(models_are_equal, f"{task} model doesn't match pipeline.")

            logger.debug(f"{task} in {framework} succeeded with {model_id}.")

    def check_models_equal_pt(self, model1, model2):
        models_are_equal = True
        for model1_p, model2_p in zip(model1.parameters(), model2.parameters()):
            if model1_p.data.ne(model2_p.data).sum() > 0:
                models_are_equal = False

        return models_are_equal

    def check_models_equal_tf(self, model1, model2):
        models_are_equal = True
        for model1_p, model2_p in zip(model1.weights, model2.weights):
            if np.abs(model1_p.numpy() - model2_p.numpy()).sum() > 1e-5:
                models_are_equal = False

        return models_are_equal


class CustomPipeline(Pipeline):
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, text, maybe_arg=2):
        input_ids = self.tokenizer(text, return_tensors="pt")
        return input_ids

    def _forward(self, model_inputs):
        outputs = self.model(**model_inputs)
        return outputs

    def postprocess(self, model_outputs):
        return model_outputs["logits"].softmax(-1).numpy()


class CustomPipelineTest(unittest.TestCase):
    def test_warning_logs(self):
        transformers_logging.set_verbosity_debug()
        logger_ = transformers_logging.get_logger("transformers.pipelines.base")

        alias = "text-classification"
        # Get the original task, so we can restore it at the end.
        # (otherwise the subsequential tests in `TextClassificationPipelineTests` will fail)
        _, original_task, _ = PIPELINE_REGISTRY.check_task(alias)

        try:
            with CaptureLogger(logger_) as cm:
                PIPELINE_REGISTRY.register_pipeline(alias, PairClassificationPipeline)
            self.assertIn(f"{alias} is already registered", cm.out)
        finally:
            # restore
            PIPELINE_REGISTRY.supported_tasks[alias] = original_task

    def test_register_pipeline(self):
        PIPELINE_REGISTRY.register_pipeline(
            "custom-text-classification",
            pipeline_class=PairClassificationPipeline,
            pt_model=AutoModelForSequenceClassification if is_torch_available() else None,
            tf_model=TFAutoModelForSequenceClassification if is_tf_available() else None,
            default={"pt": "hf-internal-testing/tiny-random-distilbert"},
            type="text",
        )
        assert "custom-text-classification" in PIPELINE_REGISTRY.get_supported_tasks()

        _, task_def, _ = PIPELINE_REGISTRY.check_task("custom-text-classification")
        self.assertEqual(task_def["pt"], (AutoModelForSequenceClassification,) if is_torch_available() else ())
        self.assertEqual(task_def["tf"], (TFAutoModelForSequenceClassification,) if is_tf_available() else ())
        self.assertEqual(task_def["type"], "text")
        self.assertEqual(task_def["impl"], PairClassificationPipeline)
        self.assertEqual(task_def["default"], {"model": {"pt": "hf-internal-testing/tiny-random-distilbert"}})

        # Clean registry for next tests.
        del PIPELINE_REGISTRY.supported_tasks["custom-text-classification"]

    @require_torch_or_tf
    def test_dynamic_pipeline(self):
        PIPELINE_REGISTRY.register_pipeline(
            "pair-classification",
            pipeline_class=PairClassificationPipeline,
            pt_model=AutoModelForSequenceClassification if is_torch_available() else None,
            tf_model=TFAutoModelForSequenceClassification if is_tf_available() else None,
        )

        classifier = pipeline("pair-classification", model="hf-internal-testing/tiny-random-bert")

        # Clean registry as we won't need the pipeline to be in it for the rest to work.
        del PIPELINE_REGISTRY.supported_tasks["pair-classification"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            classifier.save_pretrained(tmp_dir)
            # checks
            self.assertDictEqual(
                classifier.model.config.custom_pipelines,
                {
                    "pair-classification": {
                        "impl": "custom_pipeline.PairClassificationPipeline",
                        "pt": ("AutoModelForSequenceClassification",) if is_torch_available() else (),
                        "tf": ("TFAutoModelForSequenceClassification",) if is_tf_available() else (),
                    }
                },
            )
            # Fails if the user forget to pass along `trust_remote_code=True`
            with self.assertRaises(ValueError):
                _ = pipeline(model=tmp_dir)

            new_classifier = pipeline(model=tmp_dir, trust_remote_code=True)
            # Using trust_remote_code=False forces the traditional pipeline tag
            old_classifier = pipeline("text-classification", model=tmp_dir, trust_remote_code=False)
        # Can't make an isinstance check because the new_classifier is from the PairClassificationPipeline class of a
        # dynamic module
        self.assertEqual(new_classifier.__class__.__name__, "PairClassificationPipeline")
        self.assertEqual(new_classifier.task, "pair-classification")
        results = new_classifier("I hate you", second_text="I love you")
        self.assertDictEqual(
            nested_simplify(results),
            {"label": "LABEL_0", "score": 0.505, "logits": [-0.003, -0.024]},
        )

        self.assertEqual(old_classifier.__class__.__name__, "TextClassificationPipeline")
        self.assertEqual(old_classifier.task, "text-classification")
        results = old_classifier("I hate you", text_pair="I love you")
        self.assertListEqual(
            nested_simplify(results),
            [{"label": "LABEL_0", "score": 0.505}],
        )

    @require_torch_or_tf
    def test_cached_pipeline_has_minimum_calls_to_head(self):
        # Make sure we have cached the pipeline.
        _ = pipeline("text-classification", model="hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = pipeline("text-classification", model="hf-internal-testing/tiny-random-bert")
            self.assertEqual(counter.get_request_count, 0)
            self.assertEqual(counter.head_request_count, 1)
            self.assertEqual(counter.other_request_count, 0)

    @require_torch
    def test_chunk_pipeline_batching_single_file(self):
        # Make sure we have cached the pipeline.
        pipe = pipeline(model="hf-internal-testing/tiny-random-Wav2Vec2ForCTC")
        ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation").sort("id")
        audio = ds[40]["audio"]["array"]

        pipe = pipeline(model="hf-internal-testing/tiny-random-Wav2Vec2ForCTC")
        # For some reason scoping doesn't work if not using `self.`
        self.COUNT = 0
        forward = pipe.model.forward

        def new_forward(*args, **kwargs):
            self.COUNT += 1
            return forward(*args, **kwargs)

        pipe.model.forward = new_forward

        for out in pipe(audio, return_timestamps="char", chunk_length_s=3, stride_length_s=[1, 1], batch_size=1024):
            pass

        self.assertEqual(self.COUNT, 1)


@require_torch
@is_staging_test
class DynamicPipelineTester(unittest.TestCase):
    vocab_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "I", "love", "hate", "you"]

    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        set_access_token(TOKEN)
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-dynamic-pipeline")
        except HTTPError:
            pass

    def test_push_to_hub_dynamic_pipeline(self):
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

        PIPELINE_REGISTRY.register_pipeline(
            "pair-classification",
            pipeline_class=PairClassificationPipeline,
            pt_model=AutoModelForSequenceClassification,
        )

        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = BertForSequenceClassification(config).eval()

        with tempfile.TemporaryDirectory() as tmp_dir:
            create_repo(f"{USER}/test-dynamic-pipeline", token=self._token)
            repo = Repository(tmp_dir, clone_from=f"{USER}/test-dynamic-pipeline", token=self._token)

            vocab_file = os.path.join(tmp_dir, "vocab.txt")
            with open(vocab_file, "w", encoding="utf-8") as vocab_writer:
                vocab_writer.write("".join([x + "\n" for x in self.vocab_tokens]))
            tokenizer = BertTokenizer(vocab_file)

            classifier = pipeline("pair-classification", model=model, tokenizer=tokenizer)

            # Clean registry as we won't need the pipeline to be in it for the rest to work.
            del PIPELINE_REGISTRY.supported_tasks["pair-classification"]

            classifier.save_pretrained(tmp_dir)
            # checks
            self.assertDictEqual(
                classifier.model.config.custom_pipelines,
                {
                    "pair-classification": {
                        "impl": "custom_pipeline.PairClassificationPipeline",
                        "pt": ("AutoModelForSequenceClassification",),
                        "tf": (),
                    }
                },
            )

            repo.push_to_hub()

        # Fails if the user forget to pass along `trust_remote_code=True`
        with self.assertRaises(ValueError):
            _ = pipeline(model=f"{USER}/test-dynamic-pipeline")

        new_classifier = pipeline(model=f"{USER}/test-dynamic-pipeline", trust_remote_code=True)
        # Can't make an isinstance check because the new_classifier is from the PairClassificationPipeline class of a
        # dynamic module
        self.assertEqual(new_classifier.__class__.__name__, "PairClassificationPipeline")

        results = classifier("I hate you", second_text="I love you")
        new_results = new_classifier("I hate you", second_text="I love you")
        self.assertDictEqual(nested_simplify(results), nested_simplify(new_results))

        # Using trust_remote_code=False forces the traditional pipeline tag
        old_classifier = pipeline(
            "text-classification", model=f"{USER}/test-dynamic-pipeline", trust_remote_code=False
        )
        self.assertEqual(old_classifier.__class__.__name__, "TextClassificationPipeline")
        self.assertEqual(old_classifier.task, "text-classification")
        new_results = old_classifier("I hate you", text_pair="I love you")
        self.assertListEqual(
            nested_simplify([{"label": results["label"], "score": results["score"]}]), nested_simplify(new_results)
        )
