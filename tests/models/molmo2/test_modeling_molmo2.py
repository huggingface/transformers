# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Molmo2 model."""

import copy
import unittest
from unittest.mock import patch

from parameterized import parameterized

from transformers import (
    Molmo2Config,
    Molmo2ForConditionalGeneration,
    Molmo2Model,
    Molmo2Processor,
    is_torch_available,
    is_vision_available,
)
from transformers.models.molmo2.configuration_molmo2 import (
    Molmo2AdapterConfig,
    Molmo2TextConfig,
    Molmo2VisionConfig,
)
from transformers.testing_utils import (
    Expectations,
    require_torch,
    require_torch_large_accelerator,
    require_vision,
    slow,
    torch_device,
)
from transformers.video_utils import load_video

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


if is_vision_available():
    from transformers.image_utils import load_image


class Molmo2VisionText2TextModelTester(VLMModelTester):
    base_model_class = Molmo2Model
    config_class = Molmo2Config
    text_config_class = Molmo2TextConfig
    vision_config_class = Molmo2VisionConfig
    conditional_generation_class = Molmo2ForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_size", 378)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("num_image_tokens", 32)
        kwargs.setdefault("seq_length", 7 + kwargs["num_image_tokens"])
        kwargs.setdefault("hidden_size", 32)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("num_attention_heads", 4)
        kwargs.setdefault("num_key_value_heads", 2)
        kwargs.setdefault("head_dim", 128)
        kwargs.setdefault("num_hidden_layers", 2)
        kwargs.setdefault("hidden_act", "silu")
        kwargs.setdefault("max_position_embeddings", 512)
        kwargs.setdefault("bos_token_id", 0)
        kwargs.setdefault("eos_token_id", 1)
        kwargs.setdefault("pad_token_id", 2)
        kwargs.setdefault("image_start_token_id", 3)
        kwargs.setdefault("image_end_token_id", 4)
        kwargs.setdefault("image_patch_id", 5)
        # Alias so base helpers (special-token clearing, mismatch tests) protect image patch tokens.
        kwargs.setdefault("image_token_id", kwargs["image_patch_id"])
        super().__init__(parent, **kwargs)

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor(
            [
                batch_size,
                num_patches,
                self.patch_size * self.patch_size * self.num_channels,
            ]
        )

    def place_image_tokens(self, input_ids, config):
        input_ids = input_ids.clone()
        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.image_patch_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_patch_id
        return input_ids

    def get_additional_inputs(self, config, input_ids, pixel_values, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else input_ids.shape[0]
        num_patches = (self.image_size // self.patch_size) ** 2
        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == self.image_patch_id] = 1
        pooled = torch.randint(
            0,
            num_patches,
            (batch_size, self.num_image_tokens, 4),
            device=torch_device,
        )
        sample_offsets = (torch.arange(batch_size, device=torch_device) * num_patches).view(-1, 1, 1)
        image_token_pooling = (pooled + sample_offsets).view(-1, 4)
        return {
            "image_token_pooling": image_token_pooling,
            "image_grids": torch.tensor([[4, 4, 4, 4]] * batch_size, device=torch_device),
            "image_num_crops": torch.ones(batch_size, dtype=torch.long, device=torch_device),
            "mm_token_type_ids": mm_token_type_ids,
        }

    def get_config(self):
        text_config = Molmo2TextConfig(
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            hidden_act=self.hidden_act,
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            vocab_size=self.vocab_size,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            rope_theta=10000.0,
            tie_word_embeddings=self.tie_word_embeddings,
            rms_norm_eps=1e-6,
        )
        vision_config = Molmo2VisionConfig(
            hidden_size=32,
            intermediate_size=37,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            hidden_act="gelu_pytorch_tanh",
            layer_norm_eps=1e-6,
            image_size=[self.image_size, self.image_size],
            patch_size=self.patch_size,
            attention_dropout=0.0,
        )
        adapter_config = Molmo2AdapterConfig(
            vision_feature_layer=[1],
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            intermediate_size=37,
            text_hidden_size=32,
            hidden_act="silu",
        )
        return Molmo2Config(
            text_config=text_config,
            vision_config=vision_config,
            adapter_config=adapter_config,
            image_start_token_id=self.image_start_token_id,
            image_end_token_id=self.image_end_token_id,
            image_patch_id=self.image_patch_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )


@require_torch
class Molmo2ModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `Molmo2ForConditionalGeneration`.
    """

    model_tester_class = Molmo2VisionText2TextModelTester
    additional_model_inputs = ["mm_token_type_ids"]

    @unittest.skip(
        "The test slices every input tensor in half along dim 0, but `image_token_pooling` has no batch "
        "dimension: pooled patches of all images are concatenated along one axis."
    )
    def test_generate_compile_model_forward_fullgraph(self):
        pass

    def _video_features_prepare_config_and_inputs(self):
        # The generic helper only renames `pixel_values`; Molmo2's `get_video_features` also needs the
        # pooling index tensor under its video name and a `[num_frames, rows, cols]` grid per video.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        batch_size = inputs_dict["image_grids"].shape[0]
        inputs_dict = {
            "pixel_values_videos": inputs_dict["pixel_values"],
            "video_token_pooling": inputs_dict["image_token_pooling"],
            "video_grids": torch.tensor([[2, 4, 4]] * batch_size, device=torch_device),
        }
        return config, inputs_dict

    @unittest.skip(
        reason="Molmo2 always builds an attention mask in the forward (block-sequence mask for images, like "
        "gemma3/mllama), and SDPA cannot dispatch masked attention to the flash kernel"
    )
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    def flash_attn_from_config(self, attn_implementation: str, test_fwd_in_train: bool = True):
        super().flash_attn_from_config(attn_implementation, test_fwd_in_train=False)

    def flash_attn_inference_equivalence(
        self, attn_implementation: str, padding_side: str, atol: float = 4e-2, rtol: float = 4e-2
    ):
        # The common test keeps one sample with `[:1]`, which cannot slice the flat `image_token_pooling`,
        # so the check runs on text-only inputs.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        input_ids = inputs_dict["input_ids"].clone()
        input_ids[input_ids == config.image_patch_id] = self.model_tester.pad_token_id
        text_inputs = {"input_ids": input_ids, "attention_mask": inputs_dict["attention_mask"]}
        with patch.object(
            self.model_tester,
            "prepare_config_and_inputs_for_common",
            side_effect=lambda: (copy.deepcopy(config), copy.deepcopy(text_inputs)),
        ):
            super().flash_attn_inference_equivalence(attn_implementation, padding_side, atol=atol, rtol=rtol)

    @unittest.skip(
        reason="Multimodal special tokens live in the extra-vocab rows beyond `vocab_size`; standard resize is ill-defined"
    )
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(
        reason="Multimodal special tokens live in the extra-vocab rows beyond `vocab_size`; standard resize is ill-defined"
    )
    def test_resize_embeddings_untied(self):
        pass

    @parameterized.expand([("greedy", 1), ("beam_search", 2)])
    def test_generate_from_inputs_embeds_textonly(self, _, num_beams):
        """Pure-LLM path: drop the image inputs so generation runs from plain text `inputs_embeds`."""
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            config.is_decoder = True
            model = model_class(config).to(torch_device).eval()
            input_ids = inputs_dict.pop("input_ids")
            for key in (
                "pixel_values",
                "image_token_pooling",
                "image_grids",
                "image_num_crops",
                "pixel_values_videos",
                "video_token_pooling",
                "video_grids",
            ):
                inputs_dict.pop(key, None)
            gen_kwargs = {
                "return_dict_in_generate": True,
                "output_scores": True,
                "num_beams": num_beams,
                "do_sample": False,
                "max_new_tokens": 5,
                "min_new_tokens": 5,
                "use_cache": True,
            }
            inputs_embeds = model.get_input_embeddings()(input_ids)
            out = model.generate(inputs_embeds=inputs_embeds, **gen_kwargs, **inputs_dict)
            # inputs_embeds-only generation returns only the newly generated tokens
            self.assertEqual(out.sequences.shape[0], input_ids.shape[0])
            self.assertEqual(out.sequences.shape[1], 5)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs handle single-batch image inputs correctly.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            num_image_tokens = self.model_tester.num_image_tokens
            curr_input_dict["input_ids"] = curr_input_dict["input_ids"][:1, ...]
            curr_input_dict["attention_mask"] = curr_input_dict["attention_mask"][:1, ...]
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:1, ...]
            curr_input_dict["image_token_pooling"] = curr_input_dict["image_token_pooling"][:num_image_tokens]
            curr_input_dict["image_grids"] = curr_input_dict["image_grids"][:1, ...]
            curr_input_dict["image_num_crops"] = curr_input_dict["image_num_crops"][:1, ...]
            _ = model(**curr_input_dict)

    # Image features get cached in KV cache like other VLMs; no need to skip.

    def test_retain_grad_hidden_states_attentions(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.output_hidden_states = True
        config.output_attentions = self.has_attentions
        self._set_subconfig_attributes(config, "output_hidden_states", True)
        self._set_subconfig_attributes(config, "output_attentions", self.has_attentions)

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device)
            outputs = model(**inputs_dict)

            output = outputs[0]
            hidden_states = outputs.hidden_states[0]
            hidden_states.retain_grad()

            if self.has_attentions:
                attentions = outputs.attentions[0]
                attentions.retain_grad()

            output.flatten()[0].backward(retain_graph=True)

            self.assertIsNotNone(hidden_states.grad)
            if self.has_attentions:
                self.assertIsNotNone(attentions.grad)


IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"


@slow
@require_torch
@require_vision
class Molmo2_4BIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "allenai/Molmo2-4B"

    def setUp(self):
        super().setUp()
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = load_image(url_to_local_path(IMAGE_URL))
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]

    def build_inputs(self):
        return self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def test_preprocessing(self):
        inputs = self.build_inputs()
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["image_token_pooling"].shape, torch.Size([955, 4]))
        self.assertEqual(inputs["image_grids"].shape, torch.Size([1, 4]))
        self.assertEqual(inputs["image_num_crops"].tolist(), [7])
        self.assertEqual(inputs["mm_token_type_ids"].sum().item(), 982)
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        # 4B uses the Qwen tokenizer; `<|im_end|>` (151645) is the leading BOS.
        self.assertEqual(inputs["input_ids"][0, 0].item(), 151645)

        expected_pixel_slices = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [-0.07450979948043823, -0.05098038911819458, 0.019607901573181152],
                        [-0.7019608020782471, -0.6784313917160034, -0.6078431606292725],
                        [-0.8745098114013672, -0.8823529481887817, -0.843137264251709],
                    ],
                    dtype=torch.float32,
                ),
            }
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3].float().cpu(),
            expected_pixel_slices.get_expectation(),
            atol=1e-2,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = Expectations(
            {
                ("cuda", (8, 0)): [-10.407500, -5.903657, -10.977587, -10.325406, -16.847645, -14.505170, -11.184648, -9.696571, -11.637183, -9.205433],
                ("cuda", (8, 6)): [-10.407710, -5.903650, -10.977592, -10.325156, -16.847317, -14.505305, -11.184698, -9.697040, -11.636929, -9.205041],
            }
        )  # fmt: skip
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            torch.tensor(expected_last_logits.get_expectation(), dtype=torch.float32),
            atol=3e-1,
            rtol=5e-2,
        )
        self.assertEqual(logits[0, -1].argmax().item(), 641)

    def test_generation(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "In this captivating image, a large, chubby cat",
                ("cuda", (8, 6)): "In this captivating image, a large, chubby cat",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())


@slow
@require_torch
@require_vision
class Molmo2_O7BIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "allenai/Molmo2-O-7B"

    def setUp(self):
        super().setUp()
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = load_image(url_to_local_path(IMAGE_URL))
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]

    def build_inputs(self):
        return self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def test_preprocessing(self):
        inputs = self.build_inputs()
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["image_token_pooling"].shape, torch.Size([955, 4]))
        self.assertEqual(inputs["image_grids"].shape, torch.Size([1, 4]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        # O-7B uses the OLMo tokenizer; `<|endoftext|>` (100257) is the leading BOS.
        self.assertEqual(inputs["input_ids"][0, 0].item(), 100257)

        expected_pixel_slices = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [-0.07450979948043823, -0.05098038911819458, 0.019607901573181152],
                        [-0.7019608020782471, -0.6784313917160034, -0.6078431606292725],
                        [-0.8745098114013672, -0.8823529481887817, -0.843137264251709],
                    ],
                    dtype=torch.float32,
                ),
            }
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3].float().cpu(),
            expected_pixel_slices.get_expectation(),
            atol=1e-2,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = Expectations(
            {
                ("cuda", (8, 0)): [-13.0625, -5.9375, -11.75, -11.0, -12.6875, -16.25, -10.375, -12.3125, -12.6875, -10.625],
                ("cuda", (8, 6)): [-13.0625, -5.875, -11.6875, -11.0, -12.6875, -16.25, -10.3125, -12.25, -12.6875, -10.625],
            }
        )  # fmt: skip
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            torch.tensor(expected_last_logits.get_expectation(), dtype=torch.float32),
            atol=3e-1,
            rtol=5e-2,
        )
        self.assertEqual(logits[0, -1].argmax().item(), 644)

    def test_generation(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "In this captivating image, a small, chubby cat",
                ("cuda", (8, 6)): "In this captivating image, a small, chubby cat",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())


@slow
@require_torch
@require_vision
class Molmo2_8BIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "allenai/Molmo2-8B"

    def setUp(self):
        super().setUp()
        self.processor = Molmo2Processor.from_pretrained(self.model_id)
        self.image = load_image(url_to_local_path(IMAGE_URL))
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image", "image": self.image},
                ],
            }
        ]

    def build_inputs(self):
        return self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def test_preprocessing(self):
        inputs = self.build_inputs()
        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["image_token_pooling"].shape, torch.Size([955, 4]))
        self.assertEqual(inputs["image_grids"].shape, torch.Size([1, 4]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        # 8B uses the Qwen tokenizer; `<|im_end|>` (151645) is the leading BOS.
        self.assertEqual(inputs["input_ids"][0, 0].item(), 151645)

        expected_pixel_slices = Expectations(
            {
                (None, None): torch.tensor(
                    [
                        [-0.07450979948043823, -0.05098038911819458, 0.019607901573181152],
                        [-0.7019608020782471, -0.6784313917160034, -0.6078431606292725],
                        [-0.8745098114013672, -0.8823529481887817, -0.843137264251709],
                    ],
                    dtype=torch.float32,
                ),
            }
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3].float().cpu(),
            expected_pixel_slices.get_expectation(),
            atol=1e-2,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = Expectations(
            {
                ("cuda", (8, 0)): [-15.875, -7.875, -15.5625, -15.0, -16.5, -18.25, -14.4375, -15.8125, -15.4375, -12.4375],
                ("cuda", (8, 6)): [-15.875, -7.875, -15.625, -15.0, -16.5, -18.25, -14.5, -15.75, -15.4375, -12.5],
            }
        )  # fmt: skip
        torch.testing.assert_close(
            logits[0, -1, :10].cpu().float(),
            torch.tensor(expected_last_logits.get_expectation(), dtype=torch.float32),
            atol=3e-1,
            rtol=5e-2,
        )
        self.assertEqual(logits[0, -1].argmax().item(), 641)

    def test_generation(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "In this captivating image, a snow leopard is captured",
                ("cuda", (8, 6)): "In this captivating image, a snow leopard is captured",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())

    @require_torch_large_accelerator(memory=30)
    def test_generation_video_qa(self):
        """Test video question answering for Molmo2-8B."""
        video_url = "https://storage.googleapis.com/oe-training-public/demo_videos/many_penguins.mp4"
        video, metadata = load_video(url_to_local_path(video_url))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which animal appears in the video?"},
                    {"type": "video", "video": video},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            processor_kwargs={"video_metadata": [metadata]},
        )

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        device_inputs = inputs.to(torch_device)

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=64, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "Penguins appear in the video.",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())
