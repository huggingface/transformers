# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch PaliGemma model."""

import gc
import unittest

import requests
from parameterized import parameterized

from transformers import (
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    require_read_token,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class PaliGemmaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=98,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        projection_dim=32,
        text_config={
            "model_type": "gemma",
            "seq_length": 128,
            "is_training": True,
            # "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 37,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 30,
            "patch_size": 2,
            "num_image_tokens": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        use_cache=False,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.projection_dim = projection_dim

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

    def get_config(self):
        return PaliGemmaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            projection_dim=self.projection_dim,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(1).to(torch_device)
        # setting the 4 first tokens to be image
        input_ids[:, :4] = config.image_token_index
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
            "token_type_ids": torch.zeros_like(input_ids),
        }
        return config, inputs_dict


@require_torch
class PaliGemmaForConditionalGenerationModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Model tester for `PaliGemmaForConditionalGeneration`.
    """

    all_model_classes = (PaliGemmaForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (PaliGemmaForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_torchscript = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = PaliGemmaVisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=PaliGemmaConfig, has_text_modality=False)

    @parameterized.expand([(True,), (False,)])
    def test_greedy_generation(self, use_cache: bool):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            out = model.generate(**inputs_dict, min_new_tokens=20, max_new_tokens=20, use_cache=use_cache)
            self.assertTrue(out.shape[1] == inputs_dict["input_ids"].shape[1] + 20)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_model_parallelism(self):
        pass

    @require_torch_sdpa
    @slow
    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    def test_eager_matches_sdpa_inference(self, torch_dtype: str):
        self.skipTest(
            "Due to custom causal mask, there is a slightly too big difference between eager and sdpa in bfloat16."
        )

    @unittest.skip(
        reason="PaliGemmma's SigLip encoder uses the same initialization scheme as the Flax original implementation"
    )
    def test_initialization(self):
        pass

    # TODO extend valid outputs to include this test @Molbap
    @unittest.skip(reason="PaliGemma has currently one output format.")
    def test_model_outputs_equivalence(self):
        pass

    # TODO fix the loss = nan in the testing configuration chosen @Molbap
    @unittest.skip(reason="Edge case giving loss nan values in testing configuration.")
    def test_determinism(self):
        pass

    @unittest.skip(reason="PaliGemma does not use feedforward chunking.")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="PaliGemma does not support low_cpu_mem_usage.")
    def test_save_load_low_cpu_mem_usage(self):
        pass

    @unittest.skip(reason="PaliGemma does not support low_cpu_mem_usage.")
    def test_save_load_low_cpu_mem_usage_checkpoints(self):
        pass

    @unittest.skip(reason="PaliGemma does not support low_cpu_mem_usage.")
    def test_save_load_low_cpu_mem_usage_no_safetensors(self):
        pass


@slow
@require_torch
@require_read_token
class PaliGemmaForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = PaliGemmaProcessor.from_pretrained("google/paligemma-3b-pt-224")

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_read_token
    def test_small_model_integration_test(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        prompt = ""
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt")
        EXPECTED_INPUT_IDS = torch.tensor([[257152] * 256 + [2, 108]])
        self.assertTrue(torch.equal(inputs["input_ids"], EXPECTED_INPUT_IDS))

        output = model.generate(**inputs, max_new_tokens=20)
        EXPECTED_DECODED_TEXT = "\ncow on the beach"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_read_token
    def test_small_model_integration_test_paligemma_VQA(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)
        prompt = "answer en Where is the cow standing?"
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "answer en Where is the cow standing?\nbeach"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_read_token
    def test_small_model_integration_test_paligemma_empty_prompt(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        prompt = ""
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(text=prompt, images=raw_image, return_tensors="pt").to(torch.float16)

        output = model.generate(**inputs, max_new_tokens=900, do_sample=False)
        EXPECTED_DECODED_TEXT = "\ncow on the beach"  # fmt: skip

        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    @slow
    @require_read_token
    def test_small_model_integration_test_paligemma_batched(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"

        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = self.processor(text=prompts, images=[image1, image2], return_tensors="pt", padding=True)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ["answer en Where is the cow standing?\nbeach", "\ncow on the beach"]  # fmt: skip

        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_torch
    @require_read_token
    def test_small_model_integration_test_paligemma_batched_bf16(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="bfloat16", torch_dtype=torch.bfloat16
        ).to(torch_device)
        # The first batch is longer in terms of text, the second will be padded.
        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = (
            self.processor(text=prompts, images=[image1, image2], return_tensors="pt", padding=True)
            .to(torch.bfloat16)
            .to(torch_device)
        )
        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ["answer en Where is the cow standing?\nbeach", "\ncow on the beach"]  # fmt: skip
        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_torch
    @require_read_token
    def test_small_model_integration_test_paligemma_batched_f16(self):
        # Let' s make sure we test the preprocessing to replace what is used
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="float16", torch_dtype=torch.float16
        ).to(torch_device)
        # The first batch is longer in terms of text, the second will be padded.
        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = (
            self.processor(text=prompts, images=[image1, image2], return_tensors="pt", padding=True)
            .to(torch.float16)
            .to(torch_device)
        )

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = ["answer en Where is the cow standing?\nbeach", "\ncow on the beach"]  # fmt: skip
        self.assertEqual(self.processor.batch_decode(output, skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_torch
    @require_read_token
    def test_integration_detection_bug(self):
        # this is a reproducer of https://github.com/huggingface/transformers/issues/31425 where not enough context
        # impacted negatively segmentation generations.
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="bfloat16", torch_dtype=torch.bfloat16
        ).to(torch_device)
        prompt = ("detect shoe",)

        image = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/shoe.png",
                stream=True,
            ).raw
        )

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(torch.bfloat16).to(torch_device)

        output = model.generate(**inputs, max_new_tokens=20)

        EXPECTED_DECODED_TEXT = "detect shoe\n<loc0051><loc0309><loc0708><loc0646> shoe"  # fmt: skip
        self.assertEqual(self.processor.decode(output[0], skip_special_tokens=True), EXPECTED_DECODED_TEXT)

    @slow
    @require_read_token
    def test_paligemma_index_error_bug(self):
        # This is a reproducer of https://github.com/huggingface/transformers/pull/28032 and makes sure it does not happen anymore
        # Please refer to that PR, or specifically https://github.com/huggingface/transformers/pull/28032#issuecomment-1860650043 for
        # more details
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

        # Simulate a super long prompt
        prompt = "\n" * 200
        image_file = (
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )

        raw_image = Image.open(requests.get(image_file, stream=True).raw)
        inputs = self.processor(
            text=prompt,
            images=raw_image,
            return_tensors="pt",
        ).to(torch.float16)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)

    @slow
    @require_torch
    @require_read_token
    def test_paligemma_finetuning_with_suffixes_bf16(self):
        # this is a supplementary test to ensure paligemma fine-tuning that relies on token_type_ids is robust to future changes
        model_id = "google/paligemma-3b-pt-224"
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, revision="bfloat16", torch_dtype=torch.bfloat16
        ).to(torch_device)
        # The first batch is longer in terms of text, the second will be padded.
        prompts = [
            "answer en Where is the cow standing?",
            "",
        ]

        suffixes = ["beach", "cow standing on the beach"]
        image1 = Image.open(
            requests.get(
                "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png",
                stream=True,
            ).raw
        )
        image2 = image1

        inputs = (
            self.processor(text=prompts, suffix=suffixes, images=[image1, image2], return_tensors="pt", padding=True)
            .to(torch.bfloat16)
            .to(torch_device)
        )

        expected_labels = torch.tensor(
            [266 * [-100] + [54901, 1], 262 * [-100] + [14706, 9980, 611, 573, 8318, 1]]
        ).to(torch_device)

        assert torch.equal(inputs["labels"], expected_labels)

        expected_token_type_ids = torch.tensor([266 * [0] + 2 * [1], 262 * [0] + 6 * [1]]).to(torch_device)

        assert torch.equal(inputs["token_type_ids"], expected_token_type_ids)

        output = model(**inputs)

        # check that loss does not error out
        _ = output.loss
