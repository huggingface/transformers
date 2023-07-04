# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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

import glob
import json
import os
import os.path
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder, delete_repo
from huggingface_hub.file_download import http_get
from pytest import mark
from requests.exceptions import HTTPError

from transformers import (
    AutoConfig,
    AutoModel,
    PretrainedConfig,
    is_torch_available,
    logging,
)
from transformers.testing_utils import (
    TOKEN,
    USER,
    CaptureLogger,
    TestCasePlus,
    is_staging_test,
    require_accelerate,
    require_safetensors,
    require_torch,
    require_torch_gpu,
    require_torch_multi_gpu,
    require_usr_bin_time,
    slow,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig, NoSuperInitConfig  # noqa E402


if is_torch_available():
    import torch
    from test_module.custom_modeling import CustomModel, NoSuperInitModel
    from torch import nn

    from transformers import (
        BERT_PRETRAINED_MODEL_ARCHIVE_LIST,
        AutoModelForCausalLM,
        AutoTokenizer,
        BertConfig,
        BertModel,
        CLIPTextModel,
        PreTrainedModel,
        T5Config,
        T5ForConditionalGeneration,
    )
    from transformers.modeling_utils import shard_checkpoint

    # Fake pretrained models for tests
    class BaseModel(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = nn.Linear(5, 5)
            self.linear_2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear_2(self.linear(x))

    class BaseModelWithTiedWeights(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = nn.Linear(5, 5)
            self.linear_2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear_2(self.linear(x))

        def tie_weights(self):
            self.linear_2.weight = self.linear.weight

    class ModelWithHead(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            # linear is a common name between Base and Head on purpose.
            self.linear = nn.Linear(5, 5)
            self.linear2 = nn.Linear(5, 5)

        def forward(self, x):
            return self.linear2(self.linear(self.base(x)))

    class ModelWithHeadAndTiedWeights(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            self.decoder = nn.Linear(5, 5)

        def forward(self, x):
            return self.decoder(self.base(x))

        def tie_weights(self):
            self.decoder.weight = self.base.linear.weight


TINY_T5 = "patrickvonplaten/t5-tiny-random"
TINY_BERT_FOR_TOKEN_CLASSIFICATION = "hf-internal-testing/tiny-bert-for-token-classification"


def check_models_equal(model1, model2):
    models_are_equal = True
    for model1_p, model2_p in zip(model1.parameters(), model2.parameters()):
        if model1_p.data.ne(model2_p.data).sum() > 0:
            models_are_equal = False

    return models_are_equal


@require_torch
class ModelUtilsTest(TestCasePlus):
    @slow
    def test_model_from_pretrained(self):
        for model_name in BERT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            config = BertConfig.from_pretrained(model_name)
            self.assertIsNotNone(config)
            self.assertIsInstance(config, PretrainedConfig)

            model = BertModel.from_pretrained(model_name)
            model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
            self.assertIsNotNone(model)
            self.assertIsInstance(model, PreTrainedModel)

            self.assertEqual(len(loading_info["missing_keys"]), 0)
            self.assertEqual(len(loading_info["unexpected_keys"]), 8)
            self.assertEqual(len(loading_info["mismatched_keys"]), 0)
            self.assertEqual(len(loading_info["error_msgs"]), 0)

            config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

            # Not sure this is the intended behavior. TODO fix Lysandre & Thom
            config.name_or_path = model_name

            model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
            self.assertEqual(model.config.output_hidden_states, True)
            self.assertEqual(model.config, config)

    def test_model_from_pretrained_subfolder(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder))

            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)

            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_subfolder_sharded(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder), max_shard_size="10KB")

            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)

            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_hub_subfolder(self):
        subfolder = "bert"
        model_id = "hf-internal-testing/tiny-random-bert-subfolder"
        with self.assertRaises(OSError):
            _ = BertModel.from_pretrained(model_id)

        model = BertModel.from_pretrained(model_id, subfolder=subfolder)

        self.assertIsNotNone(model)

    def test_model_from_pretrained_hub_subfolder_sharded(self):
        subfolder = "bert"
        model_id = "hf-internal-testing/tiny-random-bert-sharded-subfolder"
        with self.assertRaises(OSError):
            _ = BertModel.from_pretrained(model_id)

        model = BertModel.from_pretrained(model_id, subfolder=subfolder)

        self.assertIsNotNone(model)

    def test_model_from_pretrained_with_different_pretrained_model_name(self):
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
        self.assertIsNotNone(model)

        logger = logging.get_logger("transformers.configuration_utils")
        with CaptureLogger(logger) as cl:
            BertModel.from_pretrained(TINY_T5)
        self.assertTrue("You are using a model of type t5 to instantiate a model of type bert" in cl.out)

    def test_model_from_config_torch_dtype(self):
        # test that the model can be instantiated with dtype of user's choice - as long as it's a
        # float dtype. To make it happen config.torch_dtype needs to be set before instantiating the
        # model from the config object.

        config = T5Config.from_pretrained(TINY_T5)
        model = AutoModel.from_config(config)
        # XXX: isn't supported
        # model = T5ForConditionalGeneration.from_config(config)
        self.assertEqual(model.dtype, torch.float32)

        model = AutoModel.from_config(config, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # torch.set_default_dtype() supports only float dtypes, so will fail with non-float type
        with self.assertRaises(ValueError):
            model = AutoModel.from_config(config, torch_dtype=torch.int64)

    def test_model_from_pretrained_torch_dtype(self):
        # test that the model can be instantiated with dtype of either
        # 1. explicit from_pretrained's torch_dtype argument
        # 2. via autodiscovery by looking at model weights (torch_dtype="auto")
        # so if a model.half() was saved, we want it to be instantiated as such.
        #
        # test an explicit model class, but also AutoModel separately as the latter goes through a different code path
        model_path = self.get_auto_remove_tmp_dir()

        # baseline - we know TINY_T5 is fp32 model
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5)
        self.assertEqual(model.dtype, torch.float32)

        def remove_torch_dtype(model_path):
            file = f"{model_path}/config.json"
            with open(file, "r", encoding="utf-8") as f:
                s = json.load(f)
            s.pop("torch_dtype")
            with open(file, "w", encoding="utf-8") as f:
                json.dump(s, f)

        # test the default fp32 save_pretrained => from_pretrained cycle
        model.save_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.assertEqual(model.dtype, torch.float32)
        # 1. test torch_dtype="auto" via `config.torch_dtype`
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float32)
        # 2. test torch_dtype="auto" via auto-derivation
        # now remove the torch_dtype entry from config.json and try "auto" again which should
        # perform auto-derivation from weights
        remove_torch_dtype(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float32)

        # test forced loading in fp16 (even though the weights are in fp32)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # test fp16 save_pretrained, loaded with auto-detection
        model = model.half()
        model.save_pretrained(model_path)
        # 1. test torch_dtype="auto" via `config.torch_dtype`
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.config.torch_dtype, torch.float16)
        self.assertEqual(model.dtype, torch.float16)
        # tests `config.torch_dtype` saving
        with open(f"{model_path}/config.json") as f:
            config_dict = json.load(f)
        self.assertEqual(config_dict["torch_dtype"], "float16")
        # 2. test torch_dtype="auto" via auto-derivation
        # now same with using config info
        remove_torch_dtype(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float16)

        # 3. now retest that AutoModel behaves the same wrt torch_dtype="auto" as T5ForConditionalGeneration
        model = AutoModel.from_pretrained(model_path, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float16)

        # test fp16 save_pretrained, loaded with the explicit fp16
        model = T5ForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # test AutoModel separately as it goes through a different path
        # test auto-detection - as currently TINY_T5 doesn't have torch_dtype entry
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype="auto")
        # test that the config object didn't get polluted with torch_dtype="auto"
        # there was a bug that after this call we ended up with config.torch_dtype=="auto"
        self.assertNotEqual(model.config.torch_dtype, "auto")
        # now test the outcome
        self.assertEqual(model.dtype, torch.float32)
        model = AutoModel.from_pretrained(TINY_T5, torch_dtype=torch.float16)
        self.assertEqual(model.dtype, torch.float16)

        # test model whose first param is not of a floating type, but int
        model = AutoModel.from_pretrained(TINY_BERT_FOR_TOKEN_CLASSIFICATION, torch_dtype="auto")
        self.assertEqual(model.dtype, torch.float32)

    def test_no_super_init_config_and_model(self):
        config = NoSuperInitConfig(attribute=32)
        model = NoSuperInitModel(config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            new_model = NoSuperInitModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_shard_checkpoint(self):
        # This is the model we will use, total size 340,000 bytes.
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 200, bias=False),  # size 80,000
            torch.nn.Linear(200, 200, bias=False),  # size 160,000
            torch.nn.Linear(200, 100, bias=False),  # size 80,000
            torch.nn.Linear(100, 50, bias=False),  # size 20,000
        )
        state_dict = model.state_dict()

        with self.subTest("No shard when max size is bigger than model size"):
            shards, index = shard_checkpoint(state_dict)
            self.assertIsNone(index)
            self.assertDictEqual(shards, {WEIGHTS_NAME: state_dict})

        with self.subTest("Test sharding, no weights bigger than max size"):
            shards, index = shard_checkpoint(state_dict, max_shard_size="300kB")
            # Split is first two layers then last two.
            self.assertDictEqual(
                index,
                {
                    "metadata": {"total_size": 340000},
                    "weight_map": {
                        "0.weight": "pytorch_model-00001-of-00002.bin",
                        "1.weight": "pytorch_model-00001-of-00002.bin",
                        "2.weight": "pytorch_model-00002-of-00002.bin",
                        "3.weight": "pytorch_model-00002-of-00002.bin",
                    },
                },
            )

            shard1 = {"0.weight": state_dict["0.weight"], "1.weight": state_dict["1.weight"]}
            shard2 = {"2.weight": state_dict["2.weight"], "3.weight": state_dict["3.weight"]}
            self.assertDictEqual(
                shards, {"pytorch_model-00001-of-00002.bin": shard1, "pytorch_model-00002-of-00002.bin": shard2}
            )

        with self.subTest("Test sharding with weights bigger than max size"):
            shards, index = shard_checkpoint(state_dict, max_shard_size="100kB")
            # Split is first layer, second layer then last 2.
            self.assertDictEqual(
                index,
                {
                    "metadata": {"total_size": 340000},
                    "weight_map": {
                        "0.weight": "pytorch_model-00001-of-00003.bin",
                        "1.weight": "pytorch_model-00002-of-00003.bin",
                        "2.weight": "pytorch_model-00003-of-00003.bin",
                        "3.weight": "pytorch_model-00003-of-00003.bin",
                    },
                },
            )

            shard1 = {"0.weight": state_dict["0.weight"]}
            shard2 = {"1.weight": state_dict["1.weight"]}
            shard3 = {"2.weight": state_dict["2.weight"], "3.weight": state_dict["3.weight"]}
            self.assertDictEqual(
                shards,
                {
                    "pytorch_model-00001-of-00003.bin": shard1,
                    "pytorch_model-00002-of-00003.bin": shard2,
                    "pytorch_model-00003-of-00003.bin": shard3,
                },
            )

    def test_checkpoint_sharding_local(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
            for max_size in ["50kB", "50kiB", "100kB", "100kiB", "200kB", "200kiB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size)

                # Get each shard file and its size
                shard_to_size = {}
                for shard in os.listdir(tmp_dir):
                    if shard.endswith(".bin"):
                        shard_file = os.path.join(tmp_dir, shard)
                        shard_to_size[shard_file] = os.path.getsize(shard_file)

                index_file = os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)
                # Check there is an index but no regular weight file
                self.assertTrue(os.path.isfile(index_file))
                self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

                # Check a file is bigger than max_size only when it has a single weight
                for shard_file, size in shard_to_size.items():
                    if max_size.endswith("kiB"):
                        max_size_int = int(max_size[:-3]) * 2**10
                    else:
                        max_size_int = int(max_size[:-2]) * 10**3
                    # Note: pickle adds some junk so the weight of the file can end up being slightly bigger than
                    # the size asked for (since we count parameters)
                    if size >= max_size_int + 50000:
                        state_dict = torch.load(shard_file)
                        self.assertEqual(len(state_dict), 1)

                # Check the index and the shard files found match
                with open(index_file, "r", encoding="utf-8") as f:
                    index = json.loads(f.read())

                all_shards = set(index["weight_map"].values())
                shards_found = {f for f in os.listdir(tmp_dir) if f.endswith(".bin")}
                self.assertSetEqual(all_shards, shards_found)

                # Finally, check the model can be reloaded
                new_model = BertModel.from_pretrained(tmp_dir)
                for p1, p2 in zip(model.parameters(), new_model.parameters()):
                    self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_sharding_from_hub(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
        # the model above is the same as the model below, just a sharded version.
        ref_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        for p1, p2 in zip(model.parameters(), ref_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2")

            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB")

            weights_index_name = ".".join(WEIGHTS_INDEX_NAME.split(".")[:-1] + ["v2"] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))

            for i in range(1, 5):
                weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + [f"v2-0000{i}-of-00005"] + ["bin"])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_checkpoint_variant_local_safe(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=True)

            weights_name = ".".join(SAFE_WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["safetensors"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_checkpoint_variant_local_sharded_safe(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB", safe_serialization=True)

            weights_index_name = ".".join(SAFE_WEIGHTS_INDEX_NAME.split(".")[:-1] + ["v2"] + ["json"])
            weights_index_file = os.path.join(tmp_dir, weights_index_name)
            self.assertTrue(os.path.isfile(weights_index_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))

            for i in range(1, 5):
                weights_name = ".".join(SAFE_WEIGHTS_NAME.split(".")[:-1] + [f"v2-0000{i}-of-00005"] + ["safetensors"])
                weights_name_file = os.path.join(tmp_dir, weights_name)
                self.assertTrue(os.path.isfile(weights_name_file))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_hub(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    def test_checkpoint_variant_hub_sharded(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_sharded_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    def test_checkpoint_variant_save_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            model.save_pretrained(tmp_dir, variant="v2")
            # saving will create a variant checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            model.save_pretrained(tmp_dir)
            # saving shouldn't delete variant checkpoints
            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            # there should be a normal checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

        self.assertIsNotNone(model)

    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_functional(self):
        # test that we can use `from_pretrained(..., low_cpu_mem_usage=True)` with normal and
        # sharded models

        mnames = [
            "hf-internal-testing/tiny-random-bert-sharded",
            "hf-internal-testing/tiny-random-bert",
        ]
        for mname in mnames:
            _ = BertModel.from_pretrained(mname, low_cpu_mem_usage=True)

    @require_usr_bin_time
    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_measured(self):
        # test that `from_pretrained(..., low_cpu_mem_usage=True)` uses less cpu memory than default

        mname = "bert-base-cased"

        preamble = "from transformers import AutoModel"
        one_liner_str = f'{preamble}; AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=False)'
        max_rss_normal = self.python_one_liner_max_rss(one_liner_str)
        # print(f"{max_rss_normal=}")

        one_liner_str = f'{preamble};  AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=True)'
        max_rss_low_mem = self.python_one_liner_max_rss(one_liner_str)
        # print(f"{max_rss_low_mem=}")

        diff_bytes = max_rss_normal - max_rss_low_mem
        diff_percent = diff_bytes / max_rss_low_mem
        # print(f"{diff_bytes=}, {diff_percent=}")
        # ideally we would compare that the diff is close to ~1x checkpoint size in bytes, but
        # measuring cpu memory on linux is very tricky and inconsistent, so instead let's check that
        # it's at least 15% less cpu memory consumed

        self.assertGreater(
            diff_percent,
            0.15,
            "should use less CPU memory for low_cpu_mem_usage=True, "
            f"but got max_rss_normal={max_rss_normal} and max_rss_low_mem={max_rss_low_mem}",
        )

        # if you want to compare things manually, let's first look at the size of the model in bytes
        # model = BertModel.from_pretrained(mname, low_cpu_mem_usage=False)
        # total_numel = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        # total_bytes = total_numel * 4  # 420MB
        # Now the diff_bytes should be very close to total_bytes, but the reports are inconsistent.
        # The easiest way to test this is to switch the model and torch.load to do all the work on
        # gpu - that way one can measure exactly the total and peak memory used. Perhaps once we add
        # functionality to load models directly on gpu, this test can be rewritten to use torch's
        # cuda memory tracking and then we should be able to do a much more precise test.

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_multi_gpu
    @slow
    def test_model_parallelism_gpt2(self):
        device_map = {"transformer.wte": 0, "transformer.wpe": 0, "lm_head": 0, "transformer.ln_f": 1}
        for i in range(12):
            device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1

        model = AutoModelForCausalLM.from_pretrained("gpt2", device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        output = model.generate(inputs["input_ids"].to(0))

        text_output = tokenizer.decode(output[0].tolist())
        self.assertEqual(text_output, "Hello, my name is John. I'm a writer, and I'm a writer. I'm")

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_gpu
    def test_from_pretrained_disk_offload_task_model(self):
        model = AutoModel.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        device_map = {
            "transformer.wte": 0,
            "transformer.wpe": 0,
            "transformer.h.0": "cpu",
            "transformer.h.1": "cpu",
            "transformer.h.2": "cpu",
            "transformer.h.3": "disk",
            "transformer.h.4": "disk",
            "transformer.ln_f": 0,
            "lm_head": 0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = torch.tensor([[1, 2, 3]]).to(0)

            model.save_pretrained(tmp_dir)
            new_model = AutoModelForCausalLM.from_pretrained(tmp_dir).to(0)
            outputs1 = new_model.to(0)(inputs)

            offload_folder = os.path.join(tmp_dir, "offload")
            new_model_with_offload = AutoModelForCausalLM.from_pretrained(
                tmp_dir, device_map=device_map, offload_folder=offload_folder
            )
            outputs2 = new_model_with_offload(inputs)

            self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

            # With state dict temp offload
            offload_folder = os.path.join(tmp_dir, "offload")
            new_model_with_offload = AutoModelForCausalLM.from_pretrained(
                tmp_dir,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
            outputs2 = new_model_with_offload(inputs)

            self.assertTrue(torch.allclose(outputs1.logits.cpu(), outputs2.logits.cpu()))

    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_load_from_one_file(self):
        try:
            tmp_file = tempfile.mktemp()
            with open(tmp_file, "wb") as f:
                http_get(
                    "https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/pytorch_model.bin", f
                )

            config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
            _ = BertModel.from_pretrained(tmp_file, config=config)
        finally:
            os.remove(tmp_file)

    def test_legacy_load_from_url(self):
        # This test is for deprecated behavior and can be removed in v5
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        _ = BertModel.from_pretrained(
            "https://huggingface.co/hf-internal-testing/tiny-random-bert/resolve/main/pytorch_model.bin", config=config
        )

    @require_safetensors
    def test_use_safetensors(self):
        # test nice error message if no safetensor files available
        with self.assertRaises(OSError) as env_error:
            AutoModel.from_pretrained("hf-internal-testing/tiny-random-RobertaModel", use_safetensors=True)

        self.assertTrue(
            "model.safetensors or model.safetensors.index.json and thus cannot be loaded with `safetensors`"
            in str(env_error.exception)
        )

        # test that error if only safetensors is available
        with self.assertRaises(OSError) as env_error:
            BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors", use_safetensors=False)

        self.assertTrue("does not appear to have a file named pytorch_model.bin" in str(env_error.exception))

        # test that only safetensors if both available and use_safetensors=False
        with tempfile.TemporaryDirectory() as tmp_dir:
            CLIPTextModel.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all",
                subfolder="text_encoder",
                use_safetensors=False,
                cache_dir=tmp_dir,
            )

            all_downloaded_files = glob.glob(os.path.join(tmp_dir, "*", "snapshots", "*", "*", "*"))
            self.assertTrue(any(f.endswith("bin") for f in all_downloaded_files))
            self.assertFalse(any(f.endswith("safetensors") for f in all_downloaded_files))

        # test that no safetensors if both available and use_safetensors=True
        with tempfile.TemporaryDirectory() as tmp_dir:
            CLIPTextModel.from_pretrained(
                "hf-internal-testing/diffusers-stable-diffusion-tiny-all",
                subfolder="text_encoder",
                use_safetensors=True,
                cache_dir=tmp_dir,
            )

            all_downloaded_files = glob.glob(os.path.join(tmp_dir, "*", "snapshots", "*", "*", "*"))
            self.assertTrue(any(f.endswith("safetensors") for f in all_downloaded_files))
            self.assertFalse(any(f.endswith("bin") for f in all_downloaded_files))

    @require_safetensors
    def test_safetensors_save_and_load(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            # No pytorch_model.bin file, only a model.safetensors
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

            new_model = BertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_load_from_hub(self):
        safetensors_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-safetensors")
        pytorch_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Check models are equal
        for p1, p2 in zip(safetensors_model.parameters(), pytorch_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_save_and_load_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="100kB")
            # No pytorch_model.bin index file, only a model.safetensors index
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_INDEX_NAME)))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_INDEX_NAME)))
            # No regular weights file
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME)))

            new_model = BertModel.from_pretrained(tmp_dir)

            # Check models are equal
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

    @require_safetensors
    def test_safetensors_load_from_hub_sharded(self):
        safetensors_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded-safetensors")
        pytorch_model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")

        # Check models are equal
        for p1, p2 in zip(safetensors_model.parameters(), pytorch_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_base_model_to_head_model_load(self):
        base_model = BaseModel(PretrainedConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_model.save_pretrained(tmp_dir)

            # Can load a base model in a model with head
            model = ModelWithHead.from_pretrained(tmp_dir)
            for p1, p2 in zip(model.base.parameters(), base_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

            # It doesn't work if the state dict has a mix of keys of the head and base without prefix though.
            base_state_dict = base_model.state_dict()
            head_state_dict = model.state_dict()
            base_state_dict["linear2.weight"] = head_state_dict["linear2.weight"]
            base_state_dict["linear2.bias"] = head_state_dict["linear2.bias"]
            torch.save(base_state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))

            with self.assertRaisesRegex(
                ValueError, "The state dictionary of the model you are trying to load is corrupted."
            ):
                _ = ModelWithHead.from_pretrained(tmp_dir)

    def test_tied_weights_reload(self):
        # Base
        model = BaseModelWithTiedWeights(PretrainedConfig())
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            new_model = BaseModelWithTiedWeights.from_pretrained(tmp_dir)
            self.assertIs(new_model.linear.weight, new_model.linear_2.weight)

            state_dict = model.state_dict()
            # Remove tied weight from state_dict -> model should load with no complain of missing keys
            del state_dict["linear_2.weight"]
            torch.save(state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))
            new_model, load_info = BaseModelWithTiedWeights.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertListEqual(load_info["missing_keys"], [])
            self.assertIs(new_model.linear.weight, new_model.linear_2.weight)

            # With head
            model.save_pretrained(tmp_dir)
            new_model, load_info = ModelWithHeadAndTiedWeights.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIs(new_model.base.linear.weight, new_model.decoder.weight)
            # Should only complain about the missing bias
            self.assertListEqual(load_info["missing_keys"], ["decoder.bias"])

    def test_unexpected_keys_warnings(self):
        model = ModelWithHead(PretrainedConfig())
        logger = logging.get_logger("transformers.modeling_utils")
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)

            # Loading the model with a new class, we don't get a warning for unexpected weights, just an info
            with CaptureLogger(logger) as cl:
                _, loading_info = BaseModel.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertNotIn("were not used when initializing ModelWithHead", cl.out)
            self.assertEqual(
                set(loading_info["unexpected_keys"]),
                {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias"},
            )

            # Loading the model with the same class, we do get a warning for unexpected weights
            state_dict = model.state_dict()
            state_dict["added_key"] = state_dict["linear.weight"]
            torch.save(state_dict, os.path.join(tmp_dir, WEIGHTS_NAME))
            with CaptureLogger(logger) as cl:
                _, loading_info = ModelWithHead.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIn("were not used when initializing ModelWithHead: ['added_key']", cl.out)
            self.assertEqual(loading_info["unexpected_keys"], ["added_key"])

    def test_warn_if_padding_and_no_attention_mask(self):
        logger = logging.get_logger("transformers.modeling_utils")

        with self.subTest("Ensure no warnings when pad_token_id is None."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config_no_pad_token = PretrainedConfig()
                config_no_pad_token.pad_token_id = None
                model = ModelWithHead(config_no_pad_token)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure no warnings when there is an attention_mask."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure no warnings when there are no pad_token_ids in the input_ids."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[1, 345, 232, 328, 740, 140, 1695, 69, 6078, 2341, 25]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure a warning is shown when the input_ids start with a pad_token_id."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure a warning is shown when the input_ids end with a pad_token_id."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[432, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure that the warning is shown at most once."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertEqual(cl.out.count("We strongly recommend passing in an `attention_mask`"), 1)

        with self.subTest("Ensure a different warning is shown when the pad_token_id is equal to the bos_token_id."):
            logger.warning_once.cache_clear()
            with CaptureLogger(logger) as cl:
                config = PretrainedConfig()
                config.pad_token_id = 0
                config.bos_token_id = config.pad_token_id
                model = ModelWithHead(config)
                input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("You may ignore this warning if your `pad_token_id`", cl.out)

    @require_torch_gpu
    @slow
    def test_pretrained_low_mem_new_config(self):
        # Checking for 1 model(the same one which was described in the issue) .
        model_ids = ["gpt2"]

        for model_id in model_ids:
            model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_id)
            model_config.n_layer = 48
            model_config.n_head = 25
            model_config.n_embd = 1600
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id,
                config=model_config,
                ignore_mismatched_sizes=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            model_ref = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id)

            self.assertEqual(model.__class__.__name__, model_ref.__class__.__name__)


@require_torch
@is_staging_test
class ModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @classmethod
    def tearDownClass(cls):
        try:
            delete_repo(token=cls._token, repo_id="test-model")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="valid_org/test-model-org")
        except HTTPError:
            pass

        try:
            delete_repo(token=cls._token, repo_id="test-dynamic-model")
        except HTTPError:
            pass

    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = BertModel(config)
        model.push_to_hub("test-model", use_auth_token=self._token)

        new_model = BertModel.from_pretrained(f"{USER}/test-model")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=self._token, repo_id="test-model")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id="test-model", push_to_hub=True, use_auth_token=self._token)

        new_model = BertModel.from_pretrained(f"{USER}/test-model")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_in_organization(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = BertModel(config)
        model.push_to_hub("valid_org/test-model-org", use_auth_token=self._token)

        new_model = BertModel.from_pretrained("valid_org/test-model-org")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=self._token, repo_id="valid_org/test-model-org")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(
                tmp_dir, push_to_hub=True, use_auth_token=self._token, repo_id="valid_org/test-model-org"
            )

        new_model = BertModel.from_pretrained("valid_org/test-model-org")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_dynamic_model(self):
        CustomConfig.register_for_auto_class()
        CustomModel.register_for_auto_class()

        config = CustomConfig(hidden_size=32)
        model = CustomModel(config)

        model.push_to_hub("test-dynamic-model", use_auth_token=self._token)
        # checks
        self.assertDictEqual(
            config.auto_map,
            {"AutoConfig": "custom_configuration.CustomConfig", "AutoModel": "custom_modeling.CustomModel"},
        )

        new_model = AutoModel.from_pretrained(f"{USER}/test-dynamic-model", trust_remote_code=True)
        # Can't make an isinstance check because the new_model is from the CustomModel class of a dynamic module
        self.assertEqual(new_model.__class__.__name__, "CustomModel")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        config = AutoConfig.from_pretrained(f"{USER}/test-dynamic-model", trust_remote_code=True)
        new_model = AutoModel.from_config(config, trust_remote_code=True)
        self.assertEqual(new_model.__class__.__name__, "CustomModel")
