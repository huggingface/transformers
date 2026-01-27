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
import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
import warnings
from pathlib import Path

import httpx

from transformers import (
    AutoConfig,
    BertConfig,
    Florence2Config,
    GPT2Config,
    Sam2Config,
    Sam3Config,
    Sam3TrackerConfig,
    logging,
)
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.edgetam.configuration_edgetam import EdgeTamConfig
from transformers.testing_utils import (
    TOKEN,
    CaptureLogger,
    LoggingLevel,
    TemporaryHubRepo,
    is_staging_test,
    require_torch,
)


sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

from test_module.custom_configuration import CustomConfig  # noqa E402


config_common_kwargs = {
    "return_dict": False,
    "output_hidden_states": True,
    "output_attentions": True,
    "dtype": "float16",
    "chunk_size_feed_forward": 5,
    "architectures": ["BertModel"],
    "id2label": {0: "label"},
    "label2id": {"label": "0"},
    "problem_type": "regression",
}


@is_staging_test
class ConfigPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN

    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                config.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_config = BertConfig.from_pretrained(tmp_repo.repo_id)
            for k, v in config.to_dict().items():
                if k != "transformers_version":
                    self.assertEqual(v, getattr(new_config, k))

    def test_push_to_hub_dynamic_config(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            CustomConfig.register_for_auto_class()
            config = CustomConfig(attribute=42)

            config.push_to_hub(tmp_repo.repo_id, token=self._token)

            # This has added the proper auto_map field to the config
            self.assertDictEqual(config.auto_map, {"AutoConfig": "custom_configuration.CustomConfig"})

            new_config = AutoConfig.from_pretrained(tmp_repo.repo_id, trust_remote_code=True)
            # Can't make an isinstance check because the new_config is from the FakeConfig class of a dynamic module
            self.assertEqual(new_config.__class__.__name__, "CustomConfig")
            self.assertEqual(new_config.attribute, 42)


class ConfigTestUtils(unittest.TestCase):
    def test_config_from_string(self):
        c = GPT2Config()

        # attempt to modify each of int/float/bool/str config records and verify they were updated
        n_embd = c.n_embd + 1  # int
        resid_pdrop = c.resid_pdrop + 1.0  # float
        scale_attn_weights = not c.scale_attn_weights  # bool
        summary_type = c.summary_type + "foo"  # str
        c.update_from_string(
            f"n_embd={n_embd},resid_pdrop={resid_pdrop},scale_attn_weights={scale_attn_weights},summary_type={summary_type}"
        )
        self.assertEqual(n_embd, c.n_embd, "mismatch for key: n_embd")
        self.assertEqual(resid_pdrop, c.resid_pdrop, "mismatch for key: resid_pdrop")
        self.assertEqual(scale_attn_weights, c.scale_attn_weights, "mismatch for key: scale_attn_weights")
        self.assertEqual(summary_type, c.summary_type, "mismatch for key: summary_type")

    def test_config_common_kwargs_is_complete(self):
        base_config = PreTrainedConfig()
        missing_keys = [key for key in base_config.__dict__ if key not in config_common_kwargs]
        # If this part of the test fails, you have arguments to add in config_common_kwargs above.
        self.assertListEqual(
            missing_keys,
            [
                "_output_attentions",
                "is_encoder_decoder",
                "_name_or_path",
                "_commit_hash",
                "_attn_implementation_internal",
                "_experts_implementation_internal",
                "transformers_version",
            ],
        )
        keys_with_defaults = [key for key, value in config_common_kwargs.items() if value == getattr(base_config, key)]
        if len(keys_with_defaults) > 0:
            raise ValueError(
                "The following keys are set with the default values in"
                " `test_configuration_common.config_common_kwargs` pick another value for them:"
                f" {', '.join(keys_with_defaults)}."
            )

    def test_nested_config_load_from_dict(self):
        config = AutoConfig.from_pretrained(
            "hf-internal-testing/tiny-random-CLIPModel", text_config={"num_hidden_layers": 2}
        )
        self.assertNotIsInstance(config.text_config, dict)
        self.assertEqual(config.text_config.__class__.__name__, "CLIPTextConfig")

    def test_from_pretrained_subfolder(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert-subfolder")
        self.assertIsNotNone(config)

        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert-subfolder", subfolder="bert")
        self.assertIsNotNone(config)

    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = httpx.HTTPStatusError(
            "failed", request=mock.Mock(), response=mock.Mock()
        )
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("httpx.Client.request", return_value=response_mock) as mock_head:
            _ = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()

    def test_local_versioning(self):
        configuration = AutoConfig.from_pretrained("google-bert/bert-base-cased")
        configuration.configuration_files = ["config.4.0.0.json"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            configuration.save_pretrained(tmp_dir)
            configuration.hidden_size = 2
            json.dump(configuration.to_dict(), open(os.path.join(tmp_dir, "config.4.0.0.json"), "w"))

            # This should pick the new configuration file as the version of Transformers is > 4.0.0
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 2)

            # Will need to be adjusted if we reach v42 and this test is still here.
            # Should pick the old configuration file as the version of Transformers is < 4.42.0
            configuration.configuration_files = ["config.42.0.0.json"]
            configuration.hidden_size = 768
            configuration.save_pretrained(tmp_dir)
            shutil.move(os.path.join(tmp_dir, "config.4.0.0.json"), os.path.join(tmp_dir, "config.42.0.0.json"))
            new_configuration = AutoConfig.from_pretrained(tmp_dir)
            self.assertEqual(new_configuration.hidden_size, 768)

    def test_repo_versioning_before(self):
        # This repo has two configuration files, one for v4.0.0 and above with a different hidden size.
        repo = "hf-internal-testing/test-two-configs"

        import transformers as new_transformers

        new_transformers.configuration_utils.__version__ = "v4.0.0"
        new_configuration, kwargs = new_transformers.models.auto.AutoConfig.from_pretrained(
            repo, return_unused_kwargs=True
        )
        self.assertEqual(new_configuration.hidden_size, 2)
        # This checks `_configuration_file` ia not kept in the kwargs by mistake.
        self.assertDictEqual(kwargs, {})

        # Testing an older version by monkey-patching the version in the module it's used.
        import transformers as old_transformers

        old_transformers.configuration_utils.__version__ = "v3.0.0"
        old_configuration = old_transformers.models.auto.AutoConfig.from_pretrained(repo)
        self.assertEqual(old_configuration.hidden_size, 768)

    def test_saving_config_with_custom_generation_kwargs_raises_error(self):
        config = BertConfig()
        config.min_length = 3  # `min_length = 3` is a non-default generation kwarg
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ValueError):
                config.save_pretrained(tmp_dir)

    def test_get_generation_parameters(self):
        config = BertConfig()
        self.assertFalse(len(config._get_generation_parameters()) > 0)
        config.min_length = 3
        self.assertTrue(len(config._get_generation_parameters()) > 0)
        config.min_length = 0
        self.assertTrue(len(config._get_generation_parameters()) > 0)

    def test_loading_config_do_not_raise_future_warnings(self):
        """Regression test for https://github.com/huggingface/transformers/issues/31002."""
        # Loading config should not raise a FutureWarning. It was the case before.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PreTrainedConfig.from_pretrained("bert-base-uncased")

    def test_get_text_config(self):
        """Tests the `get_text_config` method."""
        # 1. model with only text input -> returns the original config instance
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM")
        self.assertEqual(config.get_text_config(), config)
        self.assertEqual(config.get_text_config(decoder=True), config)

        # 2. composite model (VLM) -> returns the text component
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-LlavaForConditionalGeneration")
        self.assertEqual(config.get_text_config(), config.text_config)
        self.assertEqual(config.get_text_config(decoder=True), config.text_config)

        # 3. ! corner case! : composite model whose sub-config is an old composite model (should behave as above)
        config = Florence2Config()
        self.assertEqual(config.get_text_config(), config.text_config)
        self.assertEqual(config.get_text_config(decoder=True), config.text_config)

        # 4. old composite model -> may remove components based on the `decoder` or `encoder` argument
        config = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-bart")
        self.assertEqual(config.get_text_config(), config)
        # both encoder_layers and decoder_layers exist
        self.assertTrue(getattr(config, "encoder_layers", None) is not None)
        self.assertTrue(getattr(config, "decoder_layers", None) is not None)
        decoder_config = config.get_text_config(decoder=True)
        self.assertNotEqual(decoder_config, config)
        self.assertEqual(decoder_config.num_hidden_layers, config.decoder_layers)
        self.assertTrue(getattr(decoder_config, "encoder_layers", None) is None)  # encoder_layers is removed
        encoder_config = config.get_text_config(encoder=True)
        self.assertNotEqual(encoder_config, config)
        self.assertEqual(encoder_config.num_hidden_layers, config.encoder_layers)
        self.assertTrue(getattr(encoder_config, "decoder_layers", None) is None)  # decoder_layers is removed

    @require_torch
    def test_bc_torch_dtype(self):
        import torch

        config = PreTrainedConfig(dtype="bfloat16")
        self.assertEqual(config.dtype, torch.bfloat16)

        config = PreTrainedConfig(torch_dtype="bfloat16")
        self.assertEqual(config.dtype, torch.bfloat16)

        # Check that if we pass both, `dtype` is used
        config = PreTrainedConfig(dtype="bfloat16", torch_dtype="float32")
        self.assertEqual(config.dtype, torch.bfloat16)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)

            config = PreTrainedConfig.from_pretrained(tmpdirname)
            self.assertEqual(config.dtype, torch.bfloat16)

            config = PreTrainedConfig.from_pretrained(tmpdirname, dtype="float32")
            self.assertEqual(config.dtype, "float32")

            config = PreTrainedConfig.from_pretrained(tmpdirname, torch_dtype="float32")
            self.assertEqual(config.dtype, "float32")

    def test_unserializable_json_is_encoded(self):
        class NewConfig(PreTrainedConfig):
            def __init__(
                self,
                inf_positive: float = float("inf"),
                inf_negative: float = float("-inf"),
                nan: float = float("nan"),
                **kwargs,
            ):
                self.inf_positive = inf_positive
                self.inf_negative = inf_negative
                self.nan = nan

                super().__init__(**kwargs)

        new_config = NewConfig()

        # All floats should remain as floats when being accessed in the config
        self.assertIsInstance(new_config.inf_positive, float)
        self.assertIsInstance(new_config.inf_negative, float)
        self.assertIsInstance(new_config.nan, float)

        with tempfile.TemporaryDirectory() as tmpdirname:
            new_config.save_pretrained(tmpdirname)
            config_file = Path(tmpdirname) / "config.json"
            config_contents = json.loads(config_file.read_text())
            new_config_instance = NewConfig.from_pretrained(tmpdirname)

        # In the serialized JSON file, the non-JSON compatible floats should be updated
        self.assertDictEqual(config_contents["inf_positive"], {"__float__": "Infinity"})
        self.assertDictEqual(config_contents["inf_negative"], {"__float__": "-Infinity"})
        self.assertDictEqual(config_contents["nan"], {"__float__": "NaN"})

        with tempfile.TemporaryDirectory() as tmpdirname:
            new_config.save_pretrained(tmpdirname)

        # When reloading the config, it should have correct float values
        self.assertIsInstance(new_config_instance.inf_positive, float)
        self.assertIsInstance(new_config_instance.inf_negative, float)
        self.assertIsInstance(new_config_instance.nan, float)

    def test_compatible_model_types_suppresses_warning(self):
        """Test that compatible_model_types suppresses the model type mismatch warning."""

        # Create a config class that declares compatible_model_types
        class CompatibleConfig(PreTrainedConfig):
            model_type = "compatible_model"
            compatible_model_types = ("other_model",)

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        # Create a config class without compatible_model_types
        class IncompatibleConfig(PreTrainedConfig):
            model_type = "incompatible_model"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        # Create a config class to save with a specific model_type
        class OtherModelConfig(PreTrainedConfig):
            model_type = "other_model"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a config with model_type "other_model"
            config = OtherModelConfig()
            config.save_pretrained(tmpdir)

            logger = logging.get_logger("transformers.configuration_utils")

            # Loading with CompatibleConfig should NOT produce a warning
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    _ = CompatibleConfig.from_pretrained(tmpdir)
            self.assertNotIn("You are using a model of type", cl.out)

            # Loading with IncompatibleConfig SHOULD produce a warning
            with LoggingLevel(logging.WARNING):
                with CaptureLogger(logger) as cl:
                    _ = IncompatibleConfig.from_pretrained(tmpdir)
            self.assertIn(
                "You are using a model of type other_model to instantiate a model of type incompatible_model", cl.out
            )

    def test_sam2_sam3_edgetam_compatible_model_types(self):
        """Test that SAM2, SAM3, EdgeTam, and Sam3Tracker configs have correct compatible_model_types."""
        self.assertEqual(Sam2Config.compatible_model_types, ("sam2_video",))

        # Sam3Config should be compatible with sam3_video (overrides inherited sam2_video)
        self.assertEqual(Sam3Config.compatible_model_types, ("sam3_video",))

        # Sam3TrackerConfig should be compatible with sam3_video
        self.assertEqual(Sam3TrackerConfig.compatible_model_types, ("sam3_video",))

        # EdgeTamConfig should be compatible with edgetam_video (overrides inherited sam2_video)
        self.assertEqual(EdgeTamConfig.compatible_model_types, ("edgetam_video",))
