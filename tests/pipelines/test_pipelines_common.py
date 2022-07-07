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
import random
import string
import unittest
from abc import abstractmethod
from functools import lru_cache
from unittest import skipIf

import numpy as np

from transformers import (
    FEATURE_EXTRACTOR_MAPPING,
    TOKENIZER_MAPPING,
    AutoFeatureExtractor,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
    IBertConfig,
    RobertaConfig,
    TextClassificationPipeline,
    pipeline,
)
from transformers.pipelines import PIPELINE_REGISTRY, get_task
from transformers.pipelines.base import Pipeline, _pad
from transformers.testing_utils import (
    CaptureLogger,
    is_pipeline_test,
    nested_simplify,
    require_scatter,
    require_tensorflow_probability,
    require_tf,
    require_torch,
    slow,
)
from transformers.utils import logging as transformers_logging


logger = logging.getLogger(__name__)


def get_checkpoint_from_architecture(architecture):
    try:
        module = importlib.import_module(architecture.__module__)
    except ImportError:
        logger.error(f"Ignoring architecture {architecture}")
        return

    if hasattr(module, "_CHECKPOINT_FOR_DOC"):
        return module._CHECKPOINT_FOR_DOC
    else:
        logger.warning(f"Can't retrieve checkpoint from {architecture.__name__}")


def get_tiny_config_from_class(configuration_class):
    if "OpenAIGPT" in configuration_class.__name__:
        # This is the only file that is inconsistent with the naming scheme.
        # Will rename this file if we decide this is the way to go
        return

    model_type = configuration_class.model_type
    camel_case_model_name = configuration_class.__name__.split("Config")[0]

    try:
        model_slug = model_type.replace("-", "_")
        module = importlib.import_module(f".test_modeling_{model_slug}", package=f"tests.models.{model_slug}")
        model_tester_class = getattr(module, f"{camel_case_model_name}ModelTester", None)
    except (ImportError, AttributeError):
        logger.error(f"No model tester class for {configuration_class.__name__}")
        return

    if model_tester_class is None:
        logger.warning(f"No model tester class for {configuration_class.__name__}")
        return

    model_tester = model_tester_class(parent=None)

    if hasattr(model_tester, "get_pipeline_config"):
        config = model_tester.get_pipeline_config()
    elif hasattr(model_tester, "get_config"):
        config = model_tester.get_config()
    else:
        config = None
        logger.warning(f"Model tester {model_tester_class.__name__} has no `get_config()`.")

    return config


@lru_cache(maxsize=100)
def get_tiny_tokenizer_from_checkpoint(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.vocab_size < 300:
        # Wav2Vec2ForCTC for instance
        # ByT5Tokenizer
        # all are already small enough and have no Fast version that can
        # be retrained
        return tokenizer
    logger.info("Training new from iterator ...")
    vocabulary = string.ascii_letters + string.digits + " "
    tokenizer = tokenizer.train_new_from_iterator(vocabulary, vocab_size=len(vocabulary), show_progress=False)
    logger.info("Trained.")
    return tokenizer


def get_tiny_feature_extractor_from_checkpoint(checkpoint, tiny_config, feature_extractor_class):
    try:
        feature_extractor = AutoFeatureExtractor.from_pretrained(checkpoint)
    except Exception:
        try:
            if feature_extractor_class is not None:
                feature_extractor = feature_extractor_class()
            else:
                feature_extractor = None
        except Exception:
            feature_extractor = None
    if hasattr(tiny_config, "image_size") and feature_extractor:
        feature_extractor = feature_extractor.__class__(size=tiny_config.image_size, crop_size=tiny_config.image_size)

    # Speech2TextModel specific.
    if hasattr(tiny_config, "input_feat_per_channel") and feature_extractor:
        feature_extractor = feature_extractor.__class__(
            feature_size=tiny_config.input_feat_per_channel, num_mel_bins=tiny_config.input_feat_per_channel
        )
    return feature_extractor


class ANY:
    def __init__(self, *_types):
        self._types = _types

    def __eq__(self, other):
        return isinstance(other, self._types)

    def __repr__(self):
        return f"ANY({', '.join(_type.__name__ for _type in self._types)})"


class PipelineTestCaseMeta(type):
    def __new__(mcs, name, bases, dct):
        def gen_test(ModelClass, checkpoint, tiny_config, tokenizer_class, feature_extractor_class):
            @skipIf(tiny_config is None, "TinyConfig does not exist")
            @skipIf(checkpoint is None, "checkpoint does not exist")
            def test(self):
                if ModelClass.__name__.endswith("ForCausalLM"):
                    tiny_config.is_encoder_decoder = False
                    if hasattr(tiny_config, "encoder_no_repeat_ngram_size"):
                        # specific for blenderbot which supports both decoder-only
                        # encoder/decoder but the test config  only reflects
                        # encoder/decoder arch
                        tiny_config.encoder_no_repeat_ngram_size = 0
                if ModelClass.__name__.endswith("WithLMHead"):
                    tiny_config.is_decoder = True
                try:
                    model = ModelClass(tiny_config)
                except ImportError as e:
                    self.skipTest(
                        f"Cannot run with {tiny_config} as the model requires a library that isn't installed: {e}"
                    )
                if hasattr(model, "eval"):
                    model = model.eval()
                if tokenizer_class is not None:
                    try:
                        tokenizer = get_tiny_tokenizer_from_checkpoint(checkpoint)
                        # XLNet actually defines it as -1.
                        if isinstance(model.config, (RobertaConfig, IBertConfig)):
                            tokenizer.model_max_length = model.config.max_position_embeddings - 2
                        elif (
                            hasattr(model.config, "max_position_embeddings")
                            and model.config.max_position_embeddings > 0
                        ):
                            tokenizer.model_max_length = model.config.max_position_embeddings
                    # Rust Panic exception are NOT Exception subclass
                    # Some test tokenizer contain broken vocabs or custom PreTokenizer, so we
                    # provide some default tokenizer and hope for the best.
                    except:  # noqa: E722
                        self.skipTest(f"Ignoring {ModelClass}, cannot create a simple tokenizer")
                else:
                    tokenizer = None
                feature_extractor = get_tiny_feature_extractor_from_checkpoint(
                    checkpoint, tiny_config, feature_extractor_class
                )

                if tokenizer is None and feature_extractor is None:
                    self.skipTest(
                        f"Ignoring {ModelClass}, cannot create a tokenizer or feature_extractor (PerceiverConfig with"
                        " no FastTokenizer ?)"
                    )
                pipeline, examples = self.get_test_pipeline(model, tokenizer, feature_extractor)
                if pipeline is None:
                    # The test can disable itself, but it should be very marginal
                    # Concerns: Wav2Vec2ForCTC without tokenizer test (FastTokenizer don't exist)
                    return
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

        for prefix, key in [("pt", "model_mapping"), ("tf", "tf_model_mapping")]:
            mapping = dct.get(key, {})
            if mapping:
                for configuration, model_architectures in mapping.items():
                    if not isinstance(model_architectures, tuple):
                        model_architectures = (model_architectures,)

                    for model_architecture in model_architectures:
                        checkpoint = get_checkpoint_from_architecture(model_architecture)
                        tiny_config = get_tiny_config_from_class(configuration)
                        tokenizer_classes = TOKENIZER_MAPPING.get(configuration, [])
                        feature_extractor_class = FEATURE_EXTRACTOR_MAPPING.get(configuration, None)
                        feature_extractor_name = (
                            feature_extractor_class.__name__ if feature_extractor_class else "nofeature_extractor"
                        )
                        if not tokenizer_classes:
                            # We need to test even if there are no tokenizers.
                            tokenizer_classes = [None]
                        else:
                            # Remove the non defined tokenizers
                            # ByT5 and Perceiver are bytes-level and don't define
                            # FastTokenizer, we can just ignore those.
                            tokenizer_classes = [
                                tokenizer_class for tokenizer_class in tokenizer_classes if tokenizer_class is not None
                            ]

                        for tokenizer_class in tokenizer_classes:
                            if tokenizer_class is not None:
                                tokenizer_name = tokenizer_class.__name__
                            else:
                                tokenizer_name = "notokenizer"

                            test_name = f"test_{prefix}_{configuration.__name__}_{model_architecture.__name__}_{tokenizer_name}_{feature_extractor_name}"

                            if tokenizer_class is not None or feature_extractor_class is not None:
                                dct[test_name] = gen_test(
                                    model_architecture,
                                    checkpoint,
                                    tiny_config,
                                    tokenizer_class,
                                    feature_extractor_class,
                                )

        @abstractmethod
        def inner(self):
            raise NotImplementedError("Not implemented test")

        # Force these 2 methods to exist
        dct["test_small_model_pt"] = dct.get("test_small_model_pt", inner)
        dct["test_small_model_tf"] = dct.get("test_small_model_tf", inner)

        return type.__new__(mcs, name, bases, dct)


@is_pipeline_test
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


@is_pipeline_test
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


@is_pipeline_test
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
    @require_scatter
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


@is_pipeline_test
class PipelineRegistryTest(unittest.TestCase):
    def test_warning_logs(self):
        transformers_logging.set_verbosity_debug()
        logger_ = transformers_logging.get_logger("transformers.pipelines.base")

        alias = "text-classification"
        # Get the original task, so we can restore it at the end.
        # (otherwise the subsequential tests in `TextClassificationPipelineTests` will fail)
        original_task, original_task_options = PIPELINE_REGISTRY.check_task(alias)

        try:
            with CaptureLogger(logger_) as cm:
                PIPELINE_REGISTRY.register_pipeline(alias, {})
            self.assertIn(f"{alias} is already registered", cm.out)
        finally:
            # restore
            PIPELINE_REGISTRY.register_pipeline(alias, original_task)

    @require_torch
    def test_register_pipeline(self):
        custom_text_classification = {
            "impl": CustomPipeline,
            "tf": (),
            "pt": (AutoModelForSequenceClassification,),
            "default": {"model": {"pt": "hf-internal-testing/tiny-random-distilbert"}},
            "type": "text",
        }
        PIPELINE_REGISTRY.register_pipeline("custom-text-classification", custom_text_classification)
        assert "custom-text-classification" in PIPELINE_REGISTRY.get_supported_tasks()

        task_def, _ = PIPELINE_REGISTRY.check_task("custom-text-classification")
        self.assertEqual(task_def, custom_text_classification)
        self.assertEqual(task_def["type"], "text")
        self.assertEqual(task_def["impl"], CustomPipeline)
