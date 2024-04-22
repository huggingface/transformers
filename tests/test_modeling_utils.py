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
import copy
import gc
import glob
import json
import os
import os.path
import sys
import tempfile
import threading
import unittest
import unittest.mock as mock
import uuid
from pathlib import Path

import requests
from huggingface_hub import HfApi, HfFolder, delete_repo
from pytest import mark
from requests.exceptions import HTTPError

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    OwlViTForObjectDetection,
    PretrainedConfig,
    is_torch_available,
    logging,
)
from transformers.testing_utils import (
    TOKEN,
    USER,
    CaptureLogger,
    LoggingLevel,
    TestCasePlus,
    is_staging_test,
    require_accelerate,
    require_flax,
    require_safetensors,
    require_tf,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    require_torch_multi_accelerator,
    require_usr_bin_time,
    slow,
    torch_device,
)
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from transformers.utils.import_utils import (
    is_flash_attn_2_available,
    is_flax_available,
    is_tf_available,
    is_torch_sdpa_available,
    is_torchdynamo_available,
)


sys.path.append(str(Path(__file__).parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig, NoSuperInitConfig  # noqa E402


if is_torch_available():
    import torch
    from safetensors.torch import save_file as safe_save_file
    from test_module.custom_modeling import CustomModel, NoSuperInitModel
    from torch import nn

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BertConfig,
        BertModel,
        CLIPTextModel,
        PreTrainedModel,
        T5Config,
        T5ForConditionalGeneration,
    )
    from transformers.modeling_attn_mask_utils import (
        AttentionMaskConverter,
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
        _prepare_4d_causal_attention_mask,
    )
    from transformers.modeling_utils import _find_disjoint, _find_identical, shard_checkpoint

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

    class Prepare4dCausalAttentionMaskModel(nn.Module):
        def forward(self, inputs_embeds):
            batch_size, seq_length, _ = inputs_embeds.shape
            past_key_values_length = 4
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
            return attention_mask

    class Create4dCausalAttentionMaskModel(nn.Module):
        def forward(self, inputs_embeds):
            batch_size, seq_length, _ = inputs_embeds.shape
            past_key_values_length = 4
            attention_mask = _create_4d_causal_attention_mask(
                (batch_size, seq_length),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            return attention_mask

    class Prepare4dAttentionMaskModel(nn.Module):
        def forward(self, mask, inputs_embeds):
            attention_mask = _prepare_4d_attention_mask(mask, dtype=inputs_embeds.dtype)
            return attention_mask


if is_flax_available():
    from transformers import FlaxBertModel

if is_tf_available():
    from transformers import TFBertModel


TINY_T5 = "patrickvonplaten/t5-tiny-random"
TINY_BERT_FOR_TOKEN_CLASSIFICATION = "hf-internal-testing/tiny-bert-for-token-classification"
TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"


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
        model_name = "google-bert/bert-base-uncased"
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

    def test_model_manually_shared_disjointed_tensors_optimum(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        # Let's fuse qkv
        attn = model.encoder.layer[0].attention.self
        q = attn.query.weight
        k = attn.key.weight
        v = attn.value.weight
        # Force some shared storage
        qkv = torch.stack([q, k, v], dim=0)
        attn.query.weight = torch.nn.Parameter(qkv[0])
        attn.key.weight = torch.nn.Parameter(qkv[1])
        attn.value.weight = torch.nn.Parameter(qkv[2])
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model_loaded = BertModel.from_pretrained(tmp_dir)

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
        with LoggingLevel(logging.WARNING):
            with CaptureLogger(logger) as cl:
                BertModel.from_pretrained(TINY_T5)
        self.assertTrue("You are using a model of type t5 to instantiate a model of type bert" in cl.out)

    @require_accelerate
    def test_model_from_pretrained_with_none_quantization_config(self):
        # Needs a device_map for to enter the low_cpu_mem branch. We also load AutoModelForSequenceClassification
        # deliberately to enter the missing keys branch.
        model = AutoModelForSequenceClassification.from_pretrained(
            TINY_MISTRAL, device_map="auto", quantization_config=None
        )
        self.assertIsNotNone(model)

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

    def test_model_from_pretrained_attn_implementation(self):
        # test that the model can be instantiated with attn_implementation of either
        # 1. explicit from_pretrained's attn_implementation argument
        # 2. explicit from_pretrained's attn_implementation argument with a config argument
        attn_implementation_available = ["eager"]
        if is_torch_sdpa_available():
            attn_implementation_available.append("sdpa")

        if is_flash_attn_2_available():
            attn_implementation_available.append("flash_attention_2")

        mistral_attention_classes = {
            "eager": "MistralAttention",
            "sdpa": "MistralSdpaAttention",
            "flash_attention_2": "MistralFlashAttention2",
        }
        for requested_attn_implementation in attn_implementation_available:
            model = AutoModelForCausalLM.from_pretrained(
                TINY_MISTRAL, attn_implementation=requested_attn_implementation
            )
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)
            for module in model.modules():
                if "Attention" in module.__class__.__name__:
                    self.assertEqual(
                        module.__class__.__name__, mistral_attention_classes[requested_attn_implementation]
                    )

            config = AutoConfig.from_pretrained(TINY_MISTRAL)
            model = AutoModelForCausalLM.from_pretrained(
                TINY_MISTRAL, config=config, attn_implementation=requested_attn_implementation
            )
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)
            for module in model.modules():
                if "Attention" in module.__class__.__name__:
                    self.assertEqual(
                        module.__class__.__name__, mistral_attention_classes[requested_attn_implementation]
                    )

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

    def test_checkpoint_sharding_local_bin(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            # We use the same folder for various sizes to make sure a new save erases the old checkpoint.
            for max_size in ["50kB", "50kiB", "100kB", "100kiB", "200kB", "200kiB"]:
                model.save_pretrained(tmp_dir, max_shard_size=max_size, safe_serialization=False)

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

    def test_checkpoint_variant_local_bin(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=False)

            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            weights_file = os.path.join(tmp_dir, weights_name)
            self.assertTrue(os.path.isfile(weights_file))
            self.assertFalse(os.path.isfile(os.path.join(tmp_dir, WEIGHTS_NAME)))

            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(tmp_dir)

            new_model = BertModel.from_pretrained(tmp_dir, variant="v2")

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.allclose(p1, p2))

    def test_checkpoint_variant_local_sharded_bin(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, variant="v2", max_shard_size="50kB", safe_serialization=False)

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

    def test_checkpoint_variant_save_load_bin(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant", cache_dir=tmp_dir, variant="v2"
            )
            weights_name = ".".join(WEIGHTS_NAME.split(".")[:-1] + ["v2"] + ["bin"])

            model.save_pretrained(tmp_dir, variant="v2", safe_serialization=False)
            # saving will create a variant checkpoint
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, weights_name)))

            model.save_pretrained(tmp_dir, safe_serialization=False)
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

        mname = "google-bert/bert-base-cased"

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
    @require_torch_multi_accelerator
    @slow
    def test_model_parallelism_gpt2(self):
        device_map = {"transformer.wte": 0, "transformer.wpe": 0, "lm_head": 0, "transformer.ln_f": 1}
        for i in range(12):
            device_map[f"transformer.h.{i}"] = 0 if i <= 5 else 1

        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        inputs = tokenizer("Hello, my name is", return_tensors="pt")
        output = model.generate(inputs["input_ids"].to(f"{torch_device}:0"))

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

    @require_accelerate
    @mark.accelerate_tests
    @require_torch_gpu
    def test_from_pretrained_disk_offload_derived_to_base_model(self):
        derived_model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")

        device_map = {
            "wte": 0,
            "wpe": 0,
            "h.0": "cpu",
            "h.1": "cpu",
            "h.2": "cpu",
            "h.3": "disk",
            "h.4": "disk",
            "ln_f": 0,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            inputs = torch.tensor([[1, 2, 3]]).to(0)
            derived_model.save_pretrained(tmp_dir, use_safetensors=True)
            base_model = AutoModel.from_pretrained(tmp_dir)
            outputs1 = base_model.to(0)(inputs)

            # with disk offload
            offload_folder = os.path.join(tmp_dir, "offload")
            base_model_with_offload = AutoModel.from_pretrained(
                tmp_dir, device_map=device_map, offload_folder=offload_folder
            )
            outputs2 = base_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1[0].cpu(), outputs2[0].cpu()))

            # With state dict temp offload
            new_model_with_offload = AutoModel.from_pretrained(
                tmp_dir,
                device_map=device_map,
                offload_folder=offload_folder,
                offload_state_dict=True,
            )
            outputs2 = new_model_with_offload(inputs)
            self.assertTrue(torch.allclose(outputs1[0].cpu(), outputs2[0].cpu()))

    @slow
    @require_torch
    def test_from_pretrained_non_contiguous_checkpoint(self):
        # See: https://github.com/huggingface/transformers/pull/28414
        # Tiny models on the Hub have contiguous weights, contrarily to google/owlvit
        model = OwlViTForObjectDetection.from_pretrained("fxmarty/owlvit-tiny-non-contiguous-weight")
        self.assertTrue(model.owlvit.visual_projection.weight.is_contiguous())

        model = OwlViTForObjectDetection.from_pretrained(
            "fxmarty/owlvit-tiny-non-contiguous-weight", device_map="auto"
        )
        self.assertTrue(model.owlvit.visual_projection.weight.is_contiguous())

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=False)
            model.save_pretrained(tmp_dir, safe_serialization=True)

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

    @require_safetensors
    def test_use_safetensors(self):
        # Should not raise anymore
        AutoModel.from_pretrained("hf-internal-testing/tiny-random-RobertaModel", use_safetensors=True)

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
            base_model.save_pretrained(tmp_dir, safe_serialization=False)

            # Can load a base model in a model with head
            model = ModelWithHead.from_pretrained(tmp_dir)
            for p1, p2 in zip(model.base.parameters(), base_model.parameters()):
                self.assertTrue(torch.allclose(p1, p2))

            # It doesn't work if the state dict has a mix of keys of the head and base without prefix though.
            base_state_dict = base_model.state_dict()
            head_state_dict = model.state_dict()
            base_state_dict["linear2.weight"] = head_state_dict["linear2.weight"]
            base_state_dict["linear2.bias"] = head_state_dict["linear2.bias"]
            safe_save_file(base_state_dict, os.path.join(tmp_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})

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
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    _, loading_info = BaseModel.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertNotIn("were not used when initializing ModelWithHead", cl.out)
            self.assertEqual(
                set(loading_info["unexpected_keys"]),
                {"linear.weight", "linear.bias", "linear2.weight", "linear2.bias"},
            )

            # Loading the model with the same class, we do get a warning for unexpected weights
            state_dict = model.state_dict()
            state_dict["added_key"] = copy.deepcopy(state_dict["linear.weight"])
            safe_save_file(state_dict, os.path.join(tmp_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    _, loading_info = ModelWithHead.from_pretrained(tmp_dir, output_loading_info=True)
            self.assertIn("were not used when initializing ModelWithHead: ['added_key']", cl.out)
            self.assertEqual(loading_info["unexpected_keys"], ["added_key"])

    def test_warn_if_padding_and_no_attention_mask(self):
        logger = logging.get_logger("transformers.modeling_utils")

        with self.subTest("Ensure no warnings when pad_token_id is None."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config_no_pad_token = PretrainedConfig()
                    config_no_pad_token.pad_token_id = None
                    model = ModelWithHead(config_no_pad_token)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure no warnings when there is an attention_mask."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
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
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[1, 345, 232, 328, 740, 140, 1695, 69, 6078, 2341, 25]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertNotIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure a warning is shown when the input_ids start with a pad_token_id."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure a warning is shown when the input_ids end with a pad_token_id."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[432, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("We strongly recommend passing in an `attention_mask`", cl.out)

        with self.subTest("Ensure that the warning is shown at most once."):
            logger.warning_once.cache_clear()
            with LoggingLevel(logging.WARNING):
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
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    config = PretrainedConfig()
                    config.pad_token_id = 0
                    config.bos_token_id = config.pad_token_id
                    model = ModelWithHead(config)
                    input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 0, 0]])
                    model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)
            self.assertIn("You may ignore this warning if your `pad_token_id`", cl.out)

        if not is_torchdynamo_available():
            return
        with self.subTest("Ensure that the warning code is skipped when compiling with torchdynamo."):
            logger.warning_once.cache_clear()
            from torch._dynamo import config, testing

            config = PretrainedConfig()
            config.pad_token_id = 0
            model = ModelWithHead(config)
            input_ids = torch.tensor([[0, 345, 232, 328, 740, 140, 1695, 69, 6078, 432, 5232]])

            def f(input_ids):
                model.warn_if_padding_and_no_attention_mask(input_ids, attention_mask=None)

            compile_counter = testing.CompileCounter()
            opt_fn = torch.compile(f, dynamic=True, backend=compile_counter)
            opt_fn(input_ids)
            self.assertEqual(compile_counter.frame_count, 0)

    @require_torch_accelerator
    @slow
    def test_pretrained_low_mem_new_config(self):
        # Checking for 1 model(the same one which was described in the issue) .
        model_ids = ["openai-community/gpt2"]

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

    def test_generation_config_is_loaded_with_model(self):
        # Note: `joaogante/tiny-random-gpt2-with-generation-config` has a `generation_config.json` containing a dummy
        # `transformers_version` field set to `foo`. If loading the file fails, this test also fails.

        # 1. Load without further parameters
        model = AutoModelForCausalLM.from_pretrained(
            "joaogante/tiny-random-gpt2-with-generation-config", use_safetensors=False
        )
        self.assertEqual(model.generation_config.transformers_version, "foo")

        # 2. Load with `device_map`
        model = AutoModelForCausalLM.from_pretrained(
            "joaogante/tiny-random-gpt2-with-generation-config", device_map="auto", use_safetensors=False
        )
        self.assertEqual(model.generation_config.transformers_version, "foo")

    @require_safetensors
    def test_safetensors_torch_from_torch(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_safetensors
    @require_flax
    def test_safetensors_torch_from_flax(self):
        hub_model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")
        model = FlaxBertModel.from_pretrained("hf-internal-testing/tiny-bert-flax-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(hub_model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_tf
    @require_safetensors
    def test_safetensors_torch_from_tf(self):
        hub_model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")
        model = TFBertModel.from_pretrained("hf-internal-testing/tiny-bert-tf-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(hub_model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    @require_safetensors
    def test_safetensors_torch_from_torch_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="100kB")
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_modifying_model_config_causes_warning_saving_generation_config(self):
        model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        model.config.top_k = 1
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertLogs("transformers.modeling_utils", level="WARNING") as logs:
                model.save_pretrained(tmp_dir)
            self.assertEqual(len(logs.output), 1)
            self.assertIn("Your generation config was originally created from the model config", logs.output[0])

    @require_safetensors
    def test_model_from_pretrained_from_mlx(self):
        from safetensors import safe_open

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-mistral-mlx")
        self.assertIsNotNone(model)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            with safe_open(os.path.join(tmp_dir, "model.safetensors"), framework="pt") as f:
                metadata = f.metadata()
                self.assertEqual(metadata.get("format"), "pt")
            new_model = AutoModelForCausalLM.from_pretrained(tmp_dir)

        input_ids = torch.randint(100, 1000, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
            outputs_from_saved = new_model(input_ids)
            self.assertTrue(torch.allclose(outputs_from_saved["logits"], outputs["logits"]))


@slow
@require_torch
class ModelOnTheFlyConversionTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.user = "huggingface-hub-ci"
        cls.token = os.getenv("HUGGINGFACE_PRODUCTION_USER_TOKEN", None)

        if cls.token is None:
            raise ValueError("Cannot run tests as secret isn't setup.")

        cls.api = HfApi(token=cls.token)

    def setUp(self) -> None:
        self.repo_name = f"{self.user}/test-model-on-the-fly-{uuid.uuid4()}"

    def tearDown(self) -> None:
        self.api.delete_repo(self.repo_name)

    def test_safetensors_on_the_fly_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_conversion_private(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, private=True)
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name, token=self.token)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_conversion_gated(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.put(
            f"https://huggingface.co/api/models/{self.repo_name}/settings", json={"gated": "auto"}, headers=headers
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, max_shard_size="200kb")
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion_private(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(
            self.repo_name, token=self.token, safe_serialization=False, max_shard_size="200kb", private=True
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion_gated(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, max_shard_size="200kb", safe_serialization=False)
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.put(
            f"https://huggingface.co/api/models/{self.repo_name}/settings", json={"gated": "auto"}, headers=headers
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.parameters(), converted_model.parameters()):
                self.assertTrue(torch.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    @unittest.skip("Edge case, should work once the Space is updated`")
    def test_safetensors_on_the_fly_wrong_user_opened_pr(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, private=True)
        BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        # This should have opened a PR with the user's account
        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

        # We now switch the repo visibility to public
        self.api.update_repo_visibility(self.repo_name, private=False)

        # We once again call from_pretrained, which should call the bot to open a PR
        BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)

            bot_opened_pr = None
            bot_opened_pr_title = None

            for discussion in discussions:
                if discussion.author == "SFconvertbot":
                    bot_opened_pr = True
                    bot_opened_pr_title = discussion.title

            self.assertTrue(bot_opened_pr)
            self.assertEqual(bot_opened_pr_title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_specific_revision(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # Push a model on a given revision
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, revision="new-branch")

        # Try to convert the model on that revision should raise
        with self.assertRaises(EnvironmentError):
            BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token, revision="new-branch")

    def test_absence_of_safetensors_triggers_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # Download the model that doesn't have safetensors
        BertModel.from_pretrained(self.repo_name, token=self.token)

        for thread in threading.enumerate():
            if thread.name == "Thread-autoconversion":
                thread.join(timeout=10)

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)

            bot_opened_pr = None
            bot_opened_pr_title = None

            for discussion in discussions:
                if discussion.author == "SFconvertbot":
                    bot_opened_pr = True
                    bot_opened_pr_title = discussion.title

            self.assertTrue(bot_opened_pr)
            self.assertEqual(bot_opened_pr_title, "Adding `safetensors` variant of this model")

    @mock.patch("transformers.safetensors_conversion.spawn_conversion")
    def test_absence_of_safetensors_triggers_conversion_failed(self, spawn_conversion_mock):
        spawn_conversion_mock.side_effect = HTTPError()

        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # The auto conversion is mocked to always raise; ensure that it doesn't raise in the main thread
        BertModel.from_pretrained(self.repo_name, token=self.token)


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

        try:
            delete_repo(token=cls._token, repo_id="test-dynamic-model-with-tags")
        except HTTPError:
            pass

    @unittest.skip("This test is flaky")
    def test_push_to_hub(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = BertModel(config)
        model.push_to_hub("test-model", token=self._token)

        new_model = BertModel.from_pretrained(f"{USER}/test-model")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=self._token, repo_id="test-model")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, repo_id="test-model", push_to_hub=True, token=self._token)

        new_model = BertModel.from_pretrained(f"{USER}/test-model")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_with_description(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = BertModel(config)
        COMMIT_DESCRIPTION = """
The commit description supports markdown synthax see:
```python
>>> form transformers import AutoConfig
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
```
"""
        commit_details = model.push_to_hub(
            "test-model", use_auth_token=self._token, create_pr=True, commit_description=COMMIT_DESCRIPTION
        )
        self.assertEqual(commit_details.commit_description, COMMIT_DESCRIPTION)

    @unittest.skip("This test is flaky")
    def test_push_to_hub_in_organization(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        model = BertModel(config)
        model.push_to_hub("valid_org/test-model-org", token=self._token)

        new_model = BertModel.from_pretrained("valid_org/test-model-org")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        # Reset repo
        delete_repo(token=self._token, repo_id="valid_org/test-model-org")

        # Push to hub via save_pretrained
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, push_to_hub=True, token=self._token, repo_id="valid_org/test-model-org")

        new_model = BertModel.from_pretrained("valid_org/test-model-org")
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_push_to_hub_dynamic_model(self):
        CustomConfig.register_for_auto_class()
        CustomModel.register_for_auto_class()

        config = CustomConfig(hidden_size=32)
        model = CustomModel(config)

        model.push_to_hub("test-dynamic-model", token=self._token)
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

    def test_push_to_hub_with_tags(self):
        from huggingface_hub import ModelCard

        new_tags = ["tag-1", "tag-2"]

        CustomConfig.register_for_auto_class()
        CustomModel.register_for_auto_class()

        config = CustomConfig(hidden_size=32)
        model = CustomModel(config)

        self.assertTrue(model.model_tags is None)

        model.add_model_tags(new_tags)

        self.assertTrue(model.model_tags == new_tags)

        model.push_to_hub("test-dynamic-model-with-tags", token=self._token)

        loaded_model_card = ModelCard.load(f"{USER}/test-dynamic-model-with-tags")
        self.assertEqual(loaded_model_card.data.tags, new_tags)


@require_torch
class AttentionMaskTester(unittest.TestCase):
    def check_non_causal(self, bsz, q_len, kv_len, mask_2d, mask_4d):
        mask_indices = (mask_2d != 1)[:, None].broadcast_to((bsz, q_len, kv_len))
        mask_4d_values = mask_4d[:, 0][mask_indices]
        is_inf = mask_4d_values == -float("inf")
        is_min = mask_4d_values == torch.finfo(mask_4d.dtype).min
        assert torch.logical_or(is_inf, is_min).all()

    def check_to_4d(self, mask_converter, q_len, kv_len, additional_mask=None, bsz=3):
        mask_2d = torch.ones((bsz, kv_len), device=torch_device, dtype=torch.long)

        if additional_mask is not None:
            for bsz_idx, seq_idx in additional_mask:
                mask_2d[bsz_idx, seq_idx] = 0

        mask_4d = mask_converter.to_4d(mask_2d, query_length=q_len, key_value_length=kv_len, dtype=torch.float32)

        assert mask_4d.shape == (bsz, 1, q_len, kv_len)

        # make sure there are no overflows
        assert mask_4d.min() != float("-inf")

        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)

            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
            if 0 in mask_2d:
                # at least causal mask + maybe more
                assert (mask_4d != 0).sum().cpu().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif not mask_converter.is_causal and context is None:
            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == 0
            if 0 in mask_2d:
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif mask_converter.is_causal and context is not None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = (q_len * (q_len - 1) // 2) + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked

            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
            if 0 in mask_2d:
                # at least causal mask + maybe more
                assert (mask_4d != 0).sum().cpu().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)

    def check_to_causal(self, mask_converter, q_len, kv_len, bsz=3):
        mask_4d = mask_converter.to_causal_4d(
            bsz, query_length=q_len, key_value_length=kv_len, device=torch_device, dtype=torch.float32
        )

        if q_len == 1 and mask_converter.sliding_window is None:
            # no causal mask if q_len is 1
            assert mask_4d is None
            return

        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)

            assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked
        elif not mask_converter.is_causal and context is None:
            assert (mask_4d != 0).sum().cpu().item() == 0
        elif mask_converter.is_causal and context is not None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = (q_len * (q_len - 1) // 2) + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked

            assert (mask_4d != 0).sum().cpu().item() == num_tokens_masked

    def compute_num_context_mask(self, kv_len, context, q_len):
        # This function computes the # of attention tokens that are added for
        # the sliding window
        c_mask_len = kv_len - context - 1
        num_mask_triangle = c_mask_len * (c_mask_len + 1) // 2
        cut_mask_len = max(c_mask_len - q_len, 0)
        num_cut_mask = cut_mask_len * (cut_mask_len + 1) // 2
        return num_mask_triangle - num_cut_mask

    def test_2d_to_4d_causal(self):
        mask_converter = AttentionMaskConverter(is_causal=True)

        # auto-regressive use case
        self.check_to_4d(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_4d(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=1, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=3, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

        # check that the mask does not overflow on causal masked tokens
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 0), (1, 0), (1, 1)])

    def test_2d_to_4d(self):
        mask_converter = AttentionMaskConverter(is_causal=False)

        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_2d_to_4d_causal_sliding(self):
        mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=5)

        # auto-regressive use case
        self.check_to_4d(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_4d(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=1, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=3, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_causal_mask(self):
        mask_converter = AttentionMaskConverter(is_causal=True)

        # auto-regressive use case
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)

    def test_causal_mask_sliding(self):
        mask_converter = AttentionMaskConverter(is_causal=True, sliding_window=3)

        # auto-regressive use case
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)

    def test_torch_compile_fullgraph(self):
        model = Prepare4dCausalAttentionMaskModel()

        inputs_embeds = torch.rand([1, 3, 32])
        res_non_compiled = model(inputs_embeds)

        compiled_model = torch.compile(model, fullgraph=True)

        res_compiled = compiled_model(inputs_embeds)

        self.assertTrue(torch.equal(res_non_compiled, res_compiled))

        model = Create4dCausalAttentionMaskModel()

        inputs_embeds = torch.rand(2, 4, 16)
        res_non_compiled = model(inputs_embeds)

        compiled_model = torch.compile(model, fullgraph=True)
        res_compiled = compiled_model(inputs_embeds)

        self.assertTrue(torch.equal(res_non_compiled, res_compiled))

        model = Prepare4dAttentionMaskModel()

        mask = torch.ones(2, 4)
        mask[0, :2] = 0
        inputs_embeds = torch.rand(2, 4, 16)

        res_non_compiled = model(mask, inputs_embeds)

        compiled_model = torch.compile(model, fullgraph=True)
        res_compiled = compiled_model(mask, inputs_embeds)

        self.assertTrue(torch.equal(res_non_compiled, res_compiled))

    @require_torch
    @slow
    def test_unmask_unattended_left_padding(self):
        attention_mask = torch.Tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]]).to(torch.int64)

        expanded_mask = torch.Tensor(
            [
                [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[0, 0, 0], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(torch.int64)

        reference_output = torch.Tensor(
            [
                [[[1, 1, 1], [1, 1, 1], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[1, 1, 1], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(torch.int64)

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=1)

        self.assertTrue(torch.equal(result, reference_output))

        attention_mask = torch.Tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)
        min_inf = torch.finfo(torch.float32).min
        reference_output = torch.Tensor(
            [
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [min_inf, min_inf, 0, min_inf, min_inf],
                        [min_inf, min_inf, 0, 0, min_inf],
                        [min_inf, min_inf, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, min_inf, min_inf, min_inf, min_inf],
                        [0, 0, min_inf, min_inf, min_inf],
                        [0, 0, 0, min_inf, min_inf],
                        [0, 0, 0, 0, min_inf],
                        [0, 0, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 0, 0],
                        [min_inf, 0, min_inf, min_inf, min_inf],
                        [min_inf, 0, 0, min_inf, min_inf],
                        [min_inf, 0, 0, 0, min_inf],
                        [min_inf, 0, 0, 0, 0],
                    ]
                ],
            ]
        )

        self.assertTrue(torch.equal(reference_output, result))

    @require_torch
    @slow
    def test_unmask_unattended_right_padding(self):
        attention_mask = torch.Tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(torch.equal(expanded_mask, result))

    @require_torch
    @slow
    def test_unmask_unattended_random_mask(self):
        attention_mask = torch.Tensor([[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]]).to(torch.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=torch.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(torch.equal(expanded_mask, result))


@require_torch
class TestAttentionImplementation(unittest.TestCase):
    def test_error_no_sdpa_available(self):
        with self.assertRaises(ValueError) as cm:
            _ = AutoModel.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel", attn_implementation="sdpa")

        self.assertTrue(
            "does not support an attention implementation through torch.nn.functional.scaled_dot_product_attention"
            in str(cm.exception)
        )

        _ = AutoModel.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel")

    def test_error_no_flash_available(self):
        with self.assertRaises(ValueError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-tiny-model-private/tiny-random-MCTCTModel", attn_implementation="flash_attention_2"
            )

        self.assertTrue("does not support Flash Attention 2.0" in str(cm.exception))

    def test_error_no_flash_available_with_config(self):
        with self.assertRaises(ValueError) as cm:
            config = AutoConfig.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel")

            _ = AutoModel.from_pretrained(
                "hf-tiny-model-private/tiny-random-MCTCTModel", config=config, attn_implementation="flash_attention_2"
            )

        self.assertTrue("does not support Flash Attention 2.0" in str(cm.exception))

    def test_error_wrong_attn_implementation(self):
        with self.assertRaises(ValueError) as cm:
            _ = AutoModel.from_pretrained("hf-tiny-model-private/tiny-random-MCTCTModel", attn_implementation="foo")

        self.assertTrue('The only possible arguments are `attn_implementation="eager"' in str(cm.exception))

    def test_not_available_flash(self):
        if is_flash_attn_2_available():
            self.skipTest("Please uninstall flash-attn package to run test_not_available_flash")

        with self.assertRaises(ImportError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-internal-testing/tiny-random-GPTBigCodeModel", attn_implementation="flash_attention_2"
            )

        self.assertTrue("the package flash_attn seems to be not installed" in str(cm.exception))

    def test_not_available_flash_with_config(self):
        if is_flash_attn_2_available():
            self.skipTest("Please uninstall flash-attn package to run test_not_available_flash")

        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-GPTBigCodeModel")

        with self.assertRaises(ImportError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-internal-testing/tiny-random-GPTBigCodeModel",
                config=config,
                attn_implementation="flash_attention_2",
            )

        self.assertTrue("the package flash_attn seems to be not installed" in str(cm.exception))

    def test_not_available_sdpa(self):
        if is_torch_sdpa_available():
            self.skipTest("This test requires torch<=2.0")

        with self.assertRaises(ImportError) as cm:
            _ = AutoModel.from_pretrained(
                "hf-internal-testing/tiny-random-GPTBigCodeModel", attn_implementation="sdpa"
            )

        self.assertTrue("PyTorch SDPA requirements in Transformers are not met" in str(cm.exception))


@require_torch_gpu
class Mask4DTestBase(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def get_test_data(self):
        texts = ["the cat sat", "the cat had", "the cat is"]
        encoded = [self.tokenizer.encode(t) for t in texts]
        input_0 = torch.tensor(encoded, device=torch_device)
        # tensor([[   1,  278, 6635, 3290],
        # [   1,  278, 6635,  750],
        # [   1,  278, 6635,  338]], device='cuda:0')

        position_ids_0 = torch.tensor([[0, 1, 2, 3]] * 3, device=torch_device, dtype=torch.int64)

        # Combining common prefix with the unique ending tokens:
        input_1 = torch.cat([input_0[0][:-1], input_0[:, -1]]).unsqueeze(0)
        # tensor([[   1,  278, 6635, 3290,  750,  338]], device='cuda:0')

        # Creating a 4D mask where each of the last 3 tokens do not attend to each other.
        mask_1 = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0],
                        [1, 1, 1, 0, 1, 0],
                        [1, 1, 1, 0, 0, 1],
                    ]
                ]
            ],
            device="cuda:0",
            dtype=torch.int64,
        )

        # Creating a position_ids tensor. note the repeating figures in the end.
        position_ids_1 = torch.tensor([[0, 1, 2, 3, 3, 3]], device=torch_device, dtype=torch.int64)

        return input_0, position_ids_0, input_1, mask_1, position_ids_1


@require_torch_gpu
class Mask4DTestFP32(Mask4DTestBase):
    def setUp(self):
        model_name = "JackFram/llama-68m"  # small Llama-like model from FlexFlow
        self.model_dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.model_dtype).to(torch_device)

    def test_attention(self):
        """comparing outputs of attention layer"""
        # Input 0: one row per sentence; Input 1: same data, but stacked into a single row with custom attention
        input_0, position_ids_0, input_1, mask_1, position_ids_1 = self.get_test_data()
        causal_mask_1 = (1 - mask_1).to(self.model_dtype) * torch.finfo(self.model_dtype).min

        hid_0 = self.model.model.embed_tokens(input_0)
        outs_0 = self.model.model.layers[0].self_attn.forward(hid_0, position_ids=position_ids_0)[0]
        # outs_0.shape == torch.Size([3, 4, 768])

        hid_1 = self.model.model.embed_tokens(input_1)
        outs_1 = self.model.model.layers[0].self_attn.forward(
            hid_1, attention_mask=causal_mask_1, position_ids=position_ids_1
        )[0]
        # outs_1.shape == torch.Size([1, 6, 768])

        outs_0_last_tokens = outs_0[:, -1, :]  # last tokens in each batch line
        outs_1_last_tokens = outs_1[0, -3:, :]  # last three tokens
        torch.testing.assert_close(outs_0_last_tokens, outs_1_last_tokens)

    def test_causal_model_logits(self):
        """comparing logits outputs of whole inner model"""
        # Input 0: one row per sentence; Input 1: same data, but stacked into a single row with custom attention
        input_0, position_ids_0, input_1, mask_1, position_ids_1 = self.get_test_data()

        logits_0 = self.model.forward(input_0, position_ids=position_ids_0).logits
        logits_1 = self.model.forward(input_1, attention_mask=mask_1.bool(), position_ids=position_ids_1).logits

        logits_0_last_tokens = logits_0[:, -1, :]  # last tokens in each batch line
        logits_1_last_tokens = logits_1[0, -3:, :]  # last three tokens
        torch.testing.assert_close(logits_0_last_tokens, logits_1_last_tokens)


@require_torch_gpu
class Mask4DTestFP16(Mask4DTestBase):
    test_attention = Mask4DTestFP32.test_attention

    def setUp(self):
        model_name = "JackFram/llama-68m"  # small Llama-like model from FlexFlow
        self.model_dtype = torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.model_dtype).to(torch_device)

    def test_causal_model_logits(self):
        """comparing logits outputs of whole inner model"""
        # Input 0: one row per sentence; Input 1: same data, but stacked into a single row with custom attention
        input_0, position_ids_0, input_1, mask_1, position_ids_1 = self.get_test_data()

        logits_0 = self.model.forward(input_0, position_ids=position_ids_0).logits
        logits_1 = self.model.forward(input_1, attention_mask=mask_1.bool(), position_ids=position_ids_1).logits

        logits_0_last_tokens = logits_0[:, -1, :]  # last tokens in each batch line
        logits_1_last_tokens = logits_1[0, -3:, :]  # last three tokens

        indices_0 = logits_0_last_tokens.sort(descending=True).indices
        indices_1 = logits_1_last_tokens.sort(descending=True).indices

        # checking logits, but note relaxed tolerances for FP16
        torch.testing.assert_close(logits_0_last_tokens, logits_1_last_tokens, atol=0.02, rtol=0.001)

        # checking tokens order for the top tokens
        for token_ids_0, token_ids_1 in zip(indices_0, indices_1):
            self.assertTrue(torch.equal(token_ids_0[:128], token_ids_1[:128]))


@slow
@require_torch_gpu
class Mask4DTestHard(unittest.TestCase):
    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    def setUp(self):
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.model_dtype = torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.model_dtype).to(torch_device)

    def get_test_data(self):
        template = "my favorite {}"
        items = ("pet is a", "artist plays a", "name is L")  # same number of tokens in each item

        batch_0 = [template.format(x) for x in items]  # 3 separate lines
        batch_1 = template.format(" ".join(items))  # 1 line with options concatenated

        input_0 = self.tokenizer(batch_0, return_tensors="pt").input_ids.to(torch_device)
        input_1 = self.tokenizer(batch_1, return_tensors="pt").input_ids.to(torch_device)

        mask_1 = torch.tensor(
            [
                [
                    [
                        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                        [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                    ]
                ]
            ],
            device=torch_device,
            dtype=torch.int64,
        )

        position_ids_0 = torch.arange(input_0.shape[1]).tile(input_0.shape[0], 1).to(torch_device)
        # equivalent: position_ids_1 = torch.tensor([[0, 1, 2, 3, 4, 5, 3, 4, 5, 3, 4, 5]]).to(device)
        position_ids_1 = (mask_1.sum(dim=-1) - 1).reshape(1, -1)  # same but nicer

        return input_0, position_ids_0, input_1, mask_1, position_ids_1

    def test_stacked_causal_mask(self):
        # Input 0: one row per sentence; Input 1: same data, but stacked into a single row with custom attention
        input_0, position_ids_0, input_1, mask_1, position_ids_1 = self.get_test_data()

        # regular batch
        logits_0 = self.model.forward(input_0, position_ids=position_ids_0).logits
        logits_0_last = logits_0[:, -1, :]  # last tokens in each batch line
        decoded_0 = [self.tokenizer.decode(t) for t in logits_0_last.argmax(dim=-1)]

        # single forward run with 4D custom mask
        logits_1 = self.model.forward(input_1, attention_mask=mask_1.bool(), position_ids=position_ids_1).logits
        logits_1_last = logits_1[0, torch.where(position_ids_1 == position_ids_1.max())[1], :]  # last three tokens
        decoded_1 = [self.tokenizer.decode(t) for t in logits_1_last.argmax(dim=-1)]

        self.assertEqual(decoded_0, decoded_1)

    def test_partial_stacked_causal_mask(self):
        # Same as the test above, but the input is passed in two groups. It tests that we can pass partial 4D attention
        # masks

        # Input 0: one row per sentence; Input 1: same data, but stacked into a single row with custom attention
        input_0, position_ids_0, input_1, mask_1, position_ids_1 = self.get_test_data()

        # regular batch
        logits_0 = self.model.forward(input_0, position_ids=position_ids_0).logits
        logits_0_last = logits_0[:, -1, :]  # last tokens in each batch line
        decoded_0 = [self.tokenizer.decode(t) for t in logits_0_last.argmax(dim=-1)]

        # 2 forward runs with custom 4D masks
        part_a = 3  # split point

        input_1a = input_1[:, :part_a]
        position_ids_1a = position_ids_1[:, :part_a]
        mask_1a = mask_1[:, :, :part_a, :part_a]

        outs_1a = self.model.forward(input_1a, attention_mask=mask_1a.bool(), position_ids=position_ids_1a)
        past_key_values_a = outs_1a["past_key_values"]

        input_1b = input_1[:, part_a:]
        position_ids_1b = position_ids_1[:, part_a:]
        mask_1b = mask_1[:, :, part_a:, :]

        outs_1b = self.model.forward(
            input_1b, attention_mask=mask_1b.bool(), position_ids=position_ids_1b, past_key_values=past_key_values_a
        )

        decoded_1b = [
            self.tokenizer.decode(t)
            for t in outs_1b.logits.argmax(-1)[0, torch.where(position_ids_1 == position_ids_1.max())[1] - part_a]
        ]

        self.assertEqual(decoded_0, decoded_1b)


@require_torch
class TestTensorSharing(TestCasePlus):
    def test_disjoint(self):
        main = torch.zeros(10)
        a = main[:5]
        b = main[5:]
        state_dict = {"a": a, "b": b}

        shared_names, disjoint_names = _find_disjoint([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [])
        self.assertEqual(disjoint_names, ["a", "b"])

        a = main[::2]
        b = main[1::2]
        state_dict = {"a": a, "b": b}

        shared_names, disjoint_names = _find_disjoint([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [{"a", "b"}])
        self.assertEqual(disjoint_names, [])

    def test_identical(self):
        a = torch.zeros(10)
        b = a
        state_dict = {"a": a, "b": b}

        shared_names, identical_names = _find_identical([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [])
        self.assertEqual(identical_names, [{"a", "b"}])

        b = a[:5]
        state_dict = {"a": a, "b": b}

        shared_names, identical_names = _find_identical([{"a", "b"}], state_dict)
        self.assertEqual(shared_names, [{"a", "b"}])
        self.assertEqual(identical_names, [])
