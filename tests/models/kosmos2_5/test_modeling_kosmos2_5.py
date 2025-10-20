# coding=utf-8
# Copyright 2024 Microsoft Research and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch KOSMOS-2.5 model."""

import copy
import inspect
import tempfile
import unittest

import numpy as np
import pytest
import requests
from parameterized import parameterized

from transformers import AutoProcessor, Kosmos2_5Config
from transformers.models.kosmos2_5.configuration_kosmos2_5 import (
    Kosmos2_5TextConfig,
    Kosmos2_5VisionConfig,
)
from transformers.testing_utils import (
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Kosmos2_5ForConditionalGeneration, Kosmos2_5Model


if is_vision_available():
    from PIL import Image


class Kosmos2_5VisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=6,
        image_size=32,
        patch_size=4,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0,
        attention_dropout=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_embed_hidden_size = patch_size * patch_size * num_channels
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.scope = scope

        # in ViT, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        flattened_patches = floats_tensor([self.batch_size, self.seq_length, self.patch_embed_hidden_size + 2])
        config = self.get_config()

        return config, flattened_patches

    def get_config(self):
        return Kosmos2_5VisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            patch_embed_hidden_size=self.patch_embed_hidden_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, flattened_patches = config_and_inputs
        inputs_dict = {"flattened_patches": flattened_patches}
        return config, inputs_dict


class Kosmos2_5TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=6,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        ffn_dim=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        dropout=0,
        attention_dropout=0,
        max_position_embeddings=512,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask

    def get_config(self):
        return Kosmos2_5TextConfig(
            vocab_size=self.vocab_size,
            embed_dim=self.hidden_size,
            ffn_dim=self.ffn_dim,
            layers=self.num_hidden_layers,
            attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


class Kosmos2_5ModelTester:
    def __init__(
        self,
        parent,
        text_kwargs=None,
        vision_kwargs=None,
        latent_query_num=3,
        is_training=True,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.parent = parent
        self.text_model_tester = Kosmos2_5TextModelTester(parent, **text_kwargs)
        self.vision_model_tester = Kosmos2_5VisionModelTester(parent, **vision_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length
        self.latent_query_num = latent_query_num
        self.is_training = is_training

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, flattened_patches = self.vision_model_tester.prepare_config_and_inputs()

        # build `image_embeds_position_mask`
        image_embeds_position_mask = torch.zeros_like(input_ids)
        image_embeds_position_mask[:, 1 : 1 + self.latent_query_num :] = 1

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            image_embeds_position_mask,
            flattened_patches,
        )

    def get_config(self):
        return Kosmos2_5Config(
            self.text_model_tester.get_config().to_dict(),
            self.vision_model_tester.get_config().to_dict(),
            latent_query_num=self.latent_query_num,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
        image_embeds_position_mask,
        flattened_patches,
    ):
        model = Kosmos2_5Model(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(input_ids, flattened_patches, image_embeds_position_mask, attention_mask)
        self.parent.assertEqual(
            result.last_hidden_state.shape,
            (
                self.text_model_tester.batch_size,
                self.text_model_tester.seq_length,
                self.text_model_tester.hidden_size,
            ),
        )
        self.parent.assertEqual(
            result.image_embeds.shape,
            (
                self.text_model_tester.batch_size,
                self.latent_query_num,
                self.text_model_tester.hidden_size,
            ),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            image_embeds_position_mask,
            flattened_patches,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_embeds_position_mask": image_embeds_position_mask,
            "flattened_patches": flattened_patches,
        }
        return config, inputs_dict


@require_torch
class Kosmos2_5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Kosmos2_5Model, Kosmos2_5ForConditionalGeneration) if is_torch_available() else ()
    all_generative_model_classes = (Kosmos2_5ForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": Kosmos2_5Model,
            "image-to-text": Kosmos2_5ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )

    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    # TODO: `image-to-text` pipeline for this model needs Processor.
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_casse_name,
        config_class,
        model_architecture,
        tokenizer_name,
        processor_name,
    ):
        return pipeline_test_casse_name == "ImageToTextPipelineTests"

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = copy.deepcopy(inputs_dict)

        if return_labels:
            if model_class.__name__ == "Kosmos2_5ForConditionalGeneration":
                inputs_dict["labels"] = torch.zeros(
                    (
                        self.model_tester.text_model_tester.batch_size,
                        self.model_tester.text_model_tester.seq_length,
                    ),
                    dtype=torch.long,
                    device=torch_device,
                )

        if model_class.__name__ in [
            "Kosmos2_5Model",
            "Kosmos2_5ForConditionalGeneration",
        ]:
            bs, _ = inputs_dict["input_ids"].shape
            seqlen = self.model_tester.text_model_tester.seq_length
            inputs_dict["input_ids"] = torch.arange(seqlen, device=torch_device).unsqueeze(0).expand(bs, seqlen)
            inputs_dict["input_ids"] = inputs_dict["input_ids"] % self.model_tester.text_model_tester.vocab_size
            inputs_dict["attention_mask"] = torch.ones((bs, seqlen), device=torch_device)
            inputs_dict["image_embeds_position_mask"] = torch.zeros((bs, seqlen), device=torch_device)
            inputs_dict["image_embeds_position_mask"][:, : self.model_tester.latent_query_num] = 1
        return inputs_dict

    def setUp(self):
        self.model_tester = Kosmos2_5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Kosmos2_5Config, hidden_size=37)

    @unittest.skip("KOSMOS-2.5 doesn't support padding")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("KOSMOS-2.5 doesn't support padding")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    @unittest.skip(
        "Kosmos-2.5 doesn't support assisted generation due to the need to extend `image_embeds_position_mask` length."
    )
    def test_assisted_decoding_matches_greedy_search(self):
        pass

    @pytest.mark.generate
    @unittest.skip(
        "Kosmos-2.5 doesn't support assisted generation due to the need to extend `image_embeds_position_mask` length."
    )
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(
        "Kosmos-2.5 doesn't support assisted generation due to the need to extend `image_embeds_position_mask` length."
    )
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["input_ids"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_save_without_tied_weights(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.tie_word_embeddings = False
        for model_class in self.all_model_classes:
            model = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)

                model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                # Checking the state dicts are correct
                reloaded_state = model_reloaded.state_dict()
                for k, v in model.state_dict().items():
                    self.assertIn(k, reloaded_state, f"Key {k} is missing from reloaded")
                    torch.testing.assert_close(
                        v,
                        reloaded_state[k],
                        msg=lambda x: f"{model_class.__name__}: Tensor {k}: {x}",
                    )
                # Checking there was no complain of missing weights
                self.assertEqual(infos["missing_keys"], [])

    # overwrite from common in order to use `self.model_tester.text_model_tester.num_hidden_layers`
    def test_hidden_states_output(self):
        def check_hidden_states_output(inputs_dict, config, model_class):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            with torch.no_grad():
                outputs = model(**self._prepare_for_class(inputs_dict, model_class))

            hidden_states = outputs.hidden_states

            expected_num_layers = getattr(
                self.model_tester,
                "expected_num_hidden_layers",
                self.model_tester.text_model_tester.num_hidden_layers + 1,
            )
            self.assertEqual(len(hidden_states), expected_num_layers)

            seq_length = self.model_tester.text_model_tester.seq_length

            self.assertListEqual(
                list(hidden_states[0].shape[-2:]),
                [seq_length, self.model_tester.text_model_tester.hidden_size],
            )

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            inputs_dict["output_hidden_states"] = True
            check_hidden_states_output(inputs_dict, config, model_class)

            # check that output_hidden_states also work using config
            del inputs_dict["output_hidden_states"]
            config.output_hidden_states = True

            check_hidden_states_output(inputs_dict, config, model_class)

    @slow
    def test_model_from_pretrained(self):
        model_name = "microsoft/kosmos-2.5"
        model = Kosmos2_5Model.from_pretrained(model_name)
        self.assertIsNotNone(model)

    @unittest.skip(reason="Does not work on the tiny model as we keep hitting edge cases.")
    def test_model_parallelism(self):
        pass

    # TODO: ydshieh
    @require_torch_gpu
    @slow
    @unittest.skip(reason="_update_causal_mask is not implemented yet which fails this test")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    # TODO: ydshieh
    @unittest.skip(reason="doesn't support padding yet")
    def test_eager_matches_sdpa_inference_1_bfloat16(self):
        pass

    # TODO: ydshieh
    @unittest.skip(reason=" the model hasn't been added to auto class")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("This test is currently not well designed for multimodal model (float type as an input).")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("This test is currently not well designed for multimodal model (float type as an input).")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Kosmos 2.5 is multimodel and has specific input shapes.")
    def test_flash_attn_2_generate_reuse_cache(self):
        pass

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    @unittest.skip(
        "KOSMOS-2.5 doesn't support inputs embeds. The test isn't skipped by checking input args because KOSMOS-2 has `generate()` overwritten",
    )
    def test_generate_from_inputs_embeds(self):
        pass

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # Overwrite -- Kosmos-2.5 needs to prepare `image_embeds_position_mask`, and it must be padded accordingly
        _, inputs_dict = self.prepare_config_and_inputs_for_generate()
        input_ids = inputs_dict["input_ids"]

        def _prepare_image_embeds_position_mask(input_ids, pad_size):
            image_embeds_position_mask = torch.zeros(
                input_ids.shape[0], input_ids.shape[1] + pad_size, device=torch_device, dtype=input_ids.dtype
            )
            image_embeds_position_mask[:, (pad_size + 1) : pad_size + 1 + self.model_tester.latent_query_num] = 1
            return image_embeds_position_mask

        # `image_embeds_position_mask` is randomly generated in `prepare_config_and_inputs_for_generate`, and it must
        # match its padded version for the test to be valid -- we need to pass both
        unpadded_custom_inputs = {"image_embeds_position_mask": _prepare_image_embeds_position_mask(input_ids, 0)}
        padded_custom_inputs = {"image_embeds_position_mask": _prepare_image_embeds_position_mask(input_ids, 32)}
        super().test_left_padding_compatibility(
            unpadded_custom_inputs=unpadded_custom_inputs, padded_custom_inputs=padded_custom_inputs
        )


@require_vision
@require_torch
@slow
class Kosmos2_5ModelIntegrationTest(unittest.TestCase):
    # This variable is used to determine which CUDA device are we using for our runners (A10 or T4)
    # Depending on the hardware we get different logits / generations
    cuda_compute_capability_major_version = None

    @classmethod
    def setUpClass(cls):
        if is_torch_available() and torch.cuda.is_available():
            # 8 is for A100 / A10 and 7 for T4
            cls.cuda_compute_capability_major_version = torch.cuda.get_device_capability()[0]

    def run_example(self, prompt, image, model, processor):
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(torch_device) if v is not None else None for k, v in inputs.items()}
        inputs["flattened_patches"] = inputs["flattened_patches"].to(model.dtype)

        generation_outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
        )
        generated_ids = generation_outputs
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_ids, generated_text

    def test_eager(self):
        url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        image = Image.open(requests.get(url, stream=True).raw)

        dtype = torch.bfloat16
        repo = "microsoft/kosmos-2.5"
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo, device_map=torch_device, dtype=dtype, attn_implementation="eager"
        )
        processor = AutoProcessor.from_pretrained(repo)
        prompt = "<ocr>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        EXPECTED_TEXT = {
            7: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_651></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_642></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_683></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n"
            ],
            8: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_650></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_644></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_687></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

        prompt = "<md>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)

        EXPECTED_TEXT = {
            7: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
            8: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

    def test_sdpa(self):
        url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        image = Image.open(requests.get(url, stream=True).raw)

        dtype = torch.bfloat16
        repo = "microsoft/kosmos-2.5"
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo, device_map=torch_device, dtype=dtype, attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(repo)
        prompt = "<ocr>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        EXPECTED_TEXT = {
            7: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_651></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_642></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_683></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n",
            ],
            8: [
                "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_611></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_810><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_648></bbox>1\n<bbox><x_79><y_614><x_468><y_651></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_609><x_812><y_642></bbox>0\n<bbox><x_50><y_658><x_69><y_693></bbox>1\n<bbox><x_79><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_814><y_683></bbox>0\n<bbox><x_31><y_742><x_820><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_781><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_872></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_836><y_1108></bbox>Card Payment 50,000\n"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

        prompt = "<md>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)

        EXPECTED_TEXT = {
            7: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
            8: [
                "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n\nCard Payment 50,000"
            ],
        }

        self.assertListEqual(generated_text, EXPECTED_TEXT[self.cuda_compute_capability_major_version])

    @require_flash_attn
    @require_torch_gpu
    @pytest.mark.flash_attn_test
    @slow
    def test_FA2(self):
        url = "https://huggingface.co/microsoft/kosmos-2.5/resolve/main/receipt_00008.png"
        image = Image.open(requests.get(url, stream=True).raw)

        dtype = torch.bfloat16
        repo = "microsoft/kosmos-2.5"
        model = Kosmos2_5ForConditionalGeneration.from_pretrained(
            repo,
            device_map=torch_device,
            dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(repo)
        prompt = "<ocr>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        EXPECTED_TEXT = [
            "<bbox><x_53><y_573><x_69><y_606></bbox>1\n<bbox><x_79><y_573><x_464><y_612></bbox>[REG] BLACK SAKURA\n<bbox><x_690><y_569><x_812><y_606></bbox>45,455\n<bbox><x_53><y_614><x_69><y_650></bbox>1\n<bbox><x_79><y_614><x_468><y_650></bbox>COOKIE DOH SAUCES\n<bbox><x_788><y_610><x_813><y_644></bbox>0\n<bbox><x_50><y_658><x_65><y_693></bbox>1\n<bbox><x_76><y_658><x_358><y_693></bbox>NATA DE COCO\n<bbox><x_790><y_652><x_815><y_687></bbox>0\n<bbox><x_31><y_742><x_822><y_781></bbox>Sub Total 45,455\n<bbox><x_27><y_780><x_822><y_827></bbox>PB1 (10%) 4,545\n<bbox><x_27><y_826><x_824><y_874></bbox>Rounding 0\n<bbox><x_24><y_872><x_827><y_921></bbox>Total 50,000\n<bbox><x_17><y_1056><x_835><y_1108></bbox>Card Payment 50,000\n"
        ]

        self.assertListEqual(generated_text, EXPECTED_TEXT)

        prompt = "<md>"
        generated_ids, generated_text = self.run_example(prompt, image, model, processor)
        # A10 gives the 1st one, but A100 gives the 2nd one
        EXPECTED_TEXT = [
            "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n\n<table>\n<thead>\n<tr>\n<th>\nSub Total\n</th>\n<th>\n45,455\n</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td>\nPB1 (10%)\n</td>\n<td>\n4,545\n</td>\n</tr>\n<tr>\n<td>\nRounding\n</td>\n<td>\n0\n</td>\n</tr>\n<tr>\n<td>\n<strong>\nTotal\n</strong>\n</td>\n<td>\n<strong>\n50,000\n</strong>\n</td>\n</tr>\n</tbody>\n</table>\n\nCard Payment 50,000",
            "- **1 \\[REG\\] BLACK SAKURA** 45,455\n- **1 COOKIE DOH SAUCES** 0\n- **1 NATA DE COCO** 0\n- **Sub Total** 45,455\n- **PB1 (10%)** 4,545\n- **Rounding** 0\n- **Total** **50,000**\n",
        ]
        self.assertIn(generated_text[0], EXPECTED_TEXT)
