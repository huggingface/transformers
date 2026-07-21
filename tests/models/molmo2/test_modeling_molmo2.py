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
    cleanup,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.video_utils import load_video

from ...test_modeling_common import (
    _config_zero_init,
    floats_tensor,
)
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers.models.molmo2.modeling_molmo2 import token_type_ids_mask_function

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
        kwargs.setdefault("image_col_id", 6)
        # Alias so base helpers (special-token clearing, mismatch tests) protect image patch tokens.
        kwargs.setdefault("image_token_id", kwargs["image_patch_id"])
        super().__init__(parent, **kwargs)

    def create_pixel_values(self):
        num_patches = (self.image_size // self.patch_size) ** 2
        return floats_tensor(
            [
                self.batch_size,
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

    def create_attention_mask(self, input_ids):
        # Molmo2 expects a standard 2D padding mask of ones, not the base's tril matrix.
        return torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

    def get_additional_inputs(self, config, input_ids, pixel_values):
        batch_size = input_ids.shape[0]
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
            layer_norm_eps=1e-6,
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
            image_default_input_size=[self.image_size, self.image_size],
            image_patch_size=self.patch_size,
            image_num_pos=(self.image_size // self.patch_size) ** 2,
            attention_dropout=0.0,
            residual_dropout=0.0,
        )
        adapter_config = Molmo2AdapterConfig(
            vit_layers=[-1],
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
            image_col_id=self.image_col_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )


@require_torch
class Molmo2ModelTest(VLMModelTest, unittest.TestCase):
    """
    Model tester for `Molmo2ForConditionalGeneration`.
    """

    model_tester_class = Molmo2VisionText2TextModelTester
    pipeline_model_mapping = (
        {
            "image-to-text": Molmo2ForConditionalGeneration,
            "image-text-to-text": Molmo2ForConditionalGeneration,
        }
        if is_torch_available()
        else {}
    )
    test_torchscript = False
    test_pruning = False
    test_head_masking = False
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config, inputs_dict = config_and_inputs
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            model.eval()
            with torch.no_grad():
                _ = model(**inputs_dict)

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        num_image_tokens = self.model_tester.num_image_tokens
        full_pooling = self.model_tester.prepare_config_and_inputs_for_common()[1]["image_token_pooling"]
        inputs_dict["image_token_pooling"] = full_pooling[: batch_size * num_image_tokens]
        return config, inputs_dict

    def test_expand_inputs_for_generation_expands_visual_token_pooling(self):
        config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=2)
        model = Molmo2ForConditionalGeneration(config).to(torch_device)

        expand_size = 3
        input_ids = inputs_dict["input_ids"]
        image_token_pooling = inputs_dict["image_token_pooling"]
        expanded_input_ids, expanded_kwargs = model._expand_inputs_for_generation(
            expand_size=expand_size,
            input_ids=input_ids,
            image_token_pooling=image_token_pooling,
        )

        image_token_counts = (input_ids == config.image_token_id).sum(dim=-1).tolist()
        expected_pooling = []
        offset = 0
        for count in image_token_counts:
            image_token_pooling_slice = image_token_pooling[offset : offset + count]
            offset += count
            expected_pooling.extend(image_token_pooling_slice for _ in range(expand_size))
        expected_pooling = torch.cat(expected_pooling, dim=0)

        self.assertEqual(expanded_input_ids.shape[0], input_ids.shape[0] * expand_size)
        self.assertTrue(torch.equal(expanded_kwargs["image_token_pooling"], expected_pooling))

    # overwrite inputs_embeds tests because we need to delete "pixel_values" for VLMs
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)

            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_token_pooling"]
            del inputs["image_grids"]
            del inputs["image_num_crops"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    def _video_features_prepare_config_and_inputs(self):
        # The generic helper only renames `pixel_values`; Molmo2's `get_video_features` also needs the
        # pooling index tensor under its video name.
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        inputs_dict = {
            "pixel_values_videos": inputs_dict["pixel_values"],
            "video_token_pooling": inputs_dict["image_token_pooling"],
        }
        return config, inputs_dict

    # overwrite inputs_embeds tests because we need to delete "pixel_values" for VLMs
    def test_inputs_embeds_matches_input_ids(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = self._prepare_for_class(inputs_dict, model_class)
            input_ids = inputs["input_ids"]
            del inputs["input_ids"]
            del inputs["pixel_values"]
            del inputs["image_token_pooling"]
            del inputs["image_grids"]
            del inputs["image_num_crops"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            self.assertTrue(torch.allclose(out_embeds, out_ids))

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip(
        reason="This architecture does not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass

    # `test_tied_weights_keys` is inherited: Molmo2 sets `_tied_weights_keys = None` (it ties no weights),
    # so the base test passes without a skip (molbap: prefer the flag over skipping).

    # Resize is intentionally not supported: `embed_tokens` holds `vocab_size + additional_vocab_size` rows
    # and the multimodal special tokens (image_patch/start/end/col) have FIXED absolute ids in the extra-vocab
    # range (>= vocab_size). Standard resize is keyed on `config.vocab_size`, so it would drop or shift the
    # rows those fixed ids point at. (Kept as explicit skips rather than `test_resize_embeddings = False` so
    # the related `test_resize_embeddings_untied_no_reinit_on_post_init`, which does pass, still runs.)
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

    # `test_model_outputs_equivalence` is inherited and now passes: the earlier "shape mismatch" was the
    # image-merge bug (placeholder mask not expanded); with that fixed there is nothing to skip.

    @unittest.skip(
        reason="Supported only for text-only inputs (otherwise dynamic control flows for multimodal inputs)"
    )
    def test_generate_compile_model_forward(self):
        pass

    # `test_generate_from_inputs_embeds_0_greedy` is inherited and passes: Molmo2 now merges image features
    # into a provided `inputs_embeds` (instead of forbidding it), so the multimodal greedy path runs.
    @unittest.skip(
        reason="Multimodal beam search from inputs_embeds would need the flat-concatenated image crops and "
        "their pooling offsets expanded by beam width; greedy multimodal and text-only beam both work."
    )
    def test_generate_from_inputs_embeds_1_beam_search(self):
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

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_bidirectional_image_attention(self):
        """
        Image patch tokens attend bidirectionally while text tokens stay causal.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Molmo2Model._from_config(config, attn_implementation="eager").to(torch_device).eval()
        with torch.no_grad():
            outputs = model(**inputs_dict, output_attentions=True)

        token_types = inputs_dict["mm_token_type_ids"][0]
        image_positions = (token_types == 1).nonzero().flatten()
        text_positions = (token_types == 0).nonzero().flatten()
        self.assertGreater(len(image_positions), 1)
        self.assertGreater(len(text_positions), 1)

        attention = outputs.attentions[0][0]  # [num_heads, seq_len, seq_len]
        # an image token sees a later image token
        self.assertTrue((attention[:, image_positions[0], image_positions[-1]] > 0).all())
        # a text token never sees a later text token
        self.assertTrue((attention[:, text_positions[0], text_positions[-1]] == 0).all())

    def test_token_type_ids_mask_function_beyond_prompt(self):
        """
        Positions past `mm_token_type_ids` (static cache, assisted decoding) are masked as text.
        """
        token_type_ids = torch.tensor([[0, 1, 1, 0]])
        inner_mask = token_type_ids_mask_function(token_type_ids)

        indices = torch.arange(8)
        q_idx, kv_idx = torch.meshgrid(indices, indices, indexing="ij")
        mask = inner_mask(torch.zeros_like(q_idx), 0, q_idx, kv_idx)

        expected = torch.zeros(8, 8, dtype=torch.bool)
        expected[1:3, 1:3] = True
        self.assertTrue(torch.equal(mask, expected))

    def test_expand_inputs_for_generation_repeats_visual_pooling(self):
        """
        Beam expansion repeats each sample's flat pooling block `expand_size` times in batch order.
        """
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        model = Molmo2ForConditionalGeneration._from_config(config).to(torch_device).eval()

        input_ids = torch.ones((2, 4), dtype=torch.long, device=torch_device)
        input_ids[0, :3] = config.image_token_id  # sample 0: 3 patches
        input_ids[1, :2] = config.image_token_id  # sample 1: 2 patches
        token_pooling = torch.arange(5 * 4, device=torch_device).reshape(5, 4)

        expanded_ids, model_kwargs = model._expand_inputs_for_generation(
            expand_size=2, input_ids=input_ids, image_token_pooling=token_pooling
        )

        expected = torch.cat([token_pooling[:3], token_pooling[:3], token_pooling[3:], token_pooling[3:]], dim=0)
        self.assertTrue(torch.equal(model_kwargs["image_token_pooling"], expected))
        self.assertTrue(torch.equal(expanded_ids, input_ids.repeat_interleave(2, dim=0)))

        # a pooling tensor inconsistent with the per-sample patch counts raises instead of passing through
        with self.assertRaises(RuntimeError):
            model._expand_inputs_for_generation(
                expand_size=2, input_ids=input_ids, image_token_pooling=token_pooling[:4]
            )

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
class Molmo2IntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-4B"

    def setUp(self):
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

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

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

        for key in (
            "input_ids",
            "pixel_values",
            "image_token_pooling",
            "image_grids",
            "image_num_crops",
            "mm_token_type_ids",
        ):
            self.assertIn(key, inputs)

        self.assertEqual(inputs["pixel_values"].shape, torch.Size([7, 729, 588]))
        self.assertEqual(inputs["image_token_pooling"].shape, torch.Size([955, 4]))
        self.assertEqual(inputs["image_grids"].shape, torch.Size([1, 4]))
        self.assertEqual(inputs["input_ids"].shape[0], 1)
        # 4B uses the Qwen tokenizer; `<|im_end|>` (151645) is the leading BOS.
        self.assertEqual(inputs["input_ids"][0, 0].item(), 151645)

        expected_pixel_slice = torch.tensor(
            [
                [-0.07450979948043823, -0.05098038911819458, 0.019607901573181152],
                [-0.7019608020782471, -0.6784313917160034, -0.6078431606292725],
                [-0.8745098114013672, -0.8823529481887817, -0.843137264251709],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3].float().cpu(),
            expected_pixel_slice,
            atol=1e-2,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = Expectations(
            {
                ("cuda", (8, 0)): [-10.407500, -5.903657, -10.977587, -10.325406, -16.847645, -14.505170, -11.184648, -9.696571, -11.637183, -9.205433],
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
            torch_dtype=torch.float32,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "In this captivating image, a large, chubby cat",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())


@slow
@require_torch
@require_vision
class Molmo2O7BIntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-O-7B"

    def setUp(self):
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

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

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

        expected_pixel_slice = torch.tensor(
            [
                [-0.07450979948043823, -0.05098038911819458, 0.019607901573181152],
                [-0.7019608020782471, -0.6784313917160034, -0.6078431606292725],
                [-0.8745098114013672, -0.8823529481887817, -0.843137264251709],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3].float().cpu(),
            expected_pixel_slice,
            atol=1e-2,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = Expectations(
            {
                ("cuda", (8, 0)): [-13.0625, -5.9375, -11.75, -11.0, -12.6875, -16.25, -10.375, -12.3125, -12.6875, -10.625],
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
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "In this captivating image, a small, chubby cat",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())


@slow
@require_torch
@require_vision
class Molmo2_8BIntegrationTest(unittest.TestCase):
    model_id = "allenai/Molmo2-8B"

    def setUp(self):
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

    def tearDown(self):
        super().tearDown()
        cleanup(torch_device, gc_collect=True)

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

        expected_pixel_slice = torch.tensor(
            [
                [-0.07450979948043823, -0.05098038911819458, 0.019607901573181152],
                [-0.7019608020782471, -0.6784313917160034, -0.6078431606292725],
                [-0.8745098114013672, -0.8823529481887817, -0.843137264251709],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            inputs["pixel_values"][0, :3, :3].float().cpu(),
            expected_pixel_slice,
            atol=1e-2,
            rtol=1e-4,
        )

    def test_forward_logits(self):
        inputs = self.build_inputs()

        model = Molmo2ForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**device_inputs)

        logits = outputs.logits
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], device_inputs["input_ids"].shape[1])

        expected_last_logits = Expectations(
            {
                ("cuda", (8, 0)): [-15.875, -7.875, -15.5625, -15.0, -16.5, -18.25, -14.4375, -15.8125, -15.4375, -12.4375],
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
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = model.generate(**device_inputs, max_new_tokens=10, do_sample=False)

        input_len = device_inputs["input_ids"].shape[1]
        generated_text = self.processor.batch_decode(generated_ids[:, input_len:], skip_special_tokens=True)[0]
        expected_texts = Expectations(
            {
                ("cuda", (8, 0)): "In this captivating image, a snow leopard is captured",
            }
        )  # fmt: skip
        self.assertEqual(generated_text.strip(), expected_texts.get_expectation())

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
            torch_dtype=torch.bfloat16,
            device_map=torch_device,
        )
        model.eval()

        device_inputs = {k: v.to(torch_device) if hasattr(v, "to") else v for k, v in inputs.items()}

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
