# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch PerceptionLM model."""

import unittest

from huggingface_hub import hf_hub_download

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PerceptionLMConfig,
    PerceptionLMForConditionalGeneration,
    PerceptionLMModel,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_torch,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class PerceptionLMVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        image_token_id=0,
        video_token_id=2,
        seq_length=7,
        tie_word_embeddings=True,
        projector_pooling_ratio=1,
        text_config={
            "model_type": "llama",
            "seq_length": 7,
            "is_training": True,
            "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "architecture": "vit_pe_core_large_patch14_336",
            "model_args": {
                "embed_dim": 64,
                "img_size": (14, 14),
                "depth": 2,
                "global_pool": "",
                "use_post_transformer_norm": False,
                "init_values": 0.1,
                "ref_feat_shape": (1, 1),
            },
        },
    ):
        self.parent = parent
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.text_config = text_config
        self.vision_config = vision_config
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training
        self.tie_word_embeddings = tie_word_embeddings

        self.batch_size = 3
        self.num_tiles = 1
        self.num_frames = 1
        self.num_channels = 3
        self.image_size = self.vision_config["model_args"]["img_size"][0]
        self.num_image_tokens = (self.vision_config["model_args"]["img_size"][0] // 14) ** 2
        self.num_video_tokens = (self.vision_config["model_args"]["img_size"][0] // 14) ** 2
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = self.seq_length

    def get_config(self):
        return PerceptionLMConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            vision_use_cls_token=True,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.num_tiles,
                self.num_channels,
                self.vision_config["model_args"]["img_size"][0],
                self.vision_config["model_args"]["img_size"][1],
            ]
        )
        pixel_values_videos = floats_tensor(
            [
                self.batch_size,
                self.num_frames,
                self.num_channels,
                self.vision_config["model_args"]["img_size"][0],
                self.vision_config["model_args"]["img_size"][1],
            ]
        )
        config = self.get_config()

        return config, pixel_values, pixel_values_videos

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_values_videos = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        input_ids[input_ids == config.image_token_id] = self.pad_token_id
        input_ids[input_ids == config.video_token_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_id
        input_ids[:, self.num_image_tokens : self.num_video_tokens + self.num_image_tokens] = config.video_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "pixel_values_videos": pixel_values_videos,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class PerceptionLMForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `PerceptionLMForConditionalGeneration`.
    """

    all_model_classes = (
        (
            PerceptionLMModel,
            PerceptionLMForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )

    _is_composite = True

    def setUp(self):
        self.model_tester = PerceptionLMVisionText2TextModelTester(self)
        common_properties = [
            "image_token_id",
            "video_token_id",
        ]
        self.config_tester = ConfigTester(
            self,
            config_class=PerceptionLMConfig,
            has_text_modality=False,
            common_properties=common_properties,
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
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
            del inputs["pixel_values_videos"]

            wte = model.get_input_embeddings()
            inputs["inputs_embeds"] = wte(input_ids)

            with torch.no_grad():
                model(**inputs)

    # overwrite inputs_embeds tests because we need to delete "pixel values" for LVLMs
    # while some other models require pixel_values to be present
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
            del inputs["pixel_values_videos"]

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images doesn't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            if model_class == PerceptionLMModel:
                continue
            model = model_class(config).to(torch_device)
            model.eval()
            _ = model(**input_dict)  # successful forward with no modifications

            # remove one image but leave the image token in text
            input_dict["pixel_values"] = input_dict["pixel_values"][-1:, ...]
            with self.assertRaises(ValueError):
                _ = model(**input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = input_dict["input_ids"][:1]
            pixel_values = input_dict["pixel_values"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values)

    def test_training(self):
        self.all_model_classes = (PerceptionLMForConditionalGeneration,) if is_torch_available() else ()
        super().test_training()

    def test_training_gradient_checkpointing(self):
        self.all_model_classes = (PerceptionLMForConditionalGeneration,) if is_torch_available() else ()
        super().test_training_gradient_checkpointing()

    def test_training_gradient_checkpointing_use_reentrant_false(self):
        self.all_model_classes = (PerceptionLMForConditionalGeneration,) if is_torch_available() else ()
        super().test_training_gradient_checkpointing_use_reentrant_false()

    def test_training_gradient_checkpointing_use_reentrant_true(self):
        self.all_model_classes = (PerceptionLMForConditionalGeneration,) if is_torch_available() else ()
        super().test_training_gradient_checkpointing_use_reentrant_true()

    @unittest.skip(
        reason="PE/TIMM's attention implementation is self configured and won't raise ValueError on global attention implementation."
    )
    def test_flash_attn_2_can_dispatch_composite_models(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("ViT PE / TimmWrapperModel cannot be tested with meta device")
    def test_can_be_initialized_on_meta(self):
        pass

    @unittest.skip("Specifying both inputs_embeds and pixel_values are not supported for PerceptionLM")
    def test_generate_from_inputs_embeds_0_greedy(self):
        pass

    @unittest.skip("Specifying both inputs_embeds and pixel_values are not supported for PerceptionLM")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @unittest.skip("Specifying both inputs_embeds and pixel_values are not supported for PerceptionLM")
    def test_generate_from_inputs_embeds_with_static_cache(self):
        pass

    ## Skip flash attention releated tests below
    ## correct configuration:
    ## from_pretrained(model_id, attn_implementation={"text_config": "flash_attention_2", "vision_config": "eager"}
    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_fa2_generate(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_from_config(self):
        pass

    @unittest.skip("SDPA test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_sdpa_generate_with_dynamic_cache(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_inference_equivalence_right_padding(self):
        pass

    @unittest.skip("SDPA test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_eager_matches_sdpa_generate(self):
        pass

    @unittest.skip("Flash attn test is not configured correctly as we need to configure vision/timm model to 'eager'.")
    def test_flash_attn_2_inference_equivalence(self):
        pass

    @unittest.skip(
        "PerceptionLMForConditionalGeneration does not have language_model, vision_tower, multi_modal_projector."
    )
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip("Cannot set `output_attentions` for timm models.")
    def test_generate_compilation_all_outputs(self):
        pass


TEST_MODEL_PATH = "facebook/Perception-LM-1B"


@require_torch
@require_bitsandbytes
@slow
class PerceptionLMForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.processor = AutoProcessor.from_pretrained(TEST_MODEL_PATH)
        self.image_file = hf_hub_download(
            repo_id="shumingh/perception_lm_test_images",
            filename="14496_0.PNG",
            repo_type="dataset",
        )
        self.video_file = hf_hub_download(
            repo_id="shumingh/perception_lm_test_videos",
            filename="GUWR5TyiY-M_000012_000022.mp4",
            repo_type="dataset",
        )
        self.conversation1 = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_file},
                    {"type": "text", "text": "Describe the bar plot in the image."},
                ],
            }
        ]
        self.conversation2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "url": self.video_file,
                    },
                    {"type": "text", "text": "Can you describe the video in detail?"},
                ],
            }
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_small_model_integration_test(self):
        model = PerceptionLMForConditionalGeneration.from_pretrained(
            TEST_MODEL_PATH, quantization_config=BitsAndBytesConfig(load_in_4bit=True), cache_dir="./"
        )

        inputs = self.processor.apply_chat_template(
            [self.conversation1],
            num_frames=32,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(torch_device)

        generate_ids = model.generate(**inputs, max_new_tokens=18)
        input_length = inputs["input_ids"].shape[1]
        generate_ids_without_inputs = generate_ids[:, input_length:]

        EXPECTED_DECODED_TEXT = "The bar plot displays the values of four categories: step, horror, mood, and lumber"  # fmt: skip

        self.assertEqual(
            self.processor.decode(generate_ids_without_inputs[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_small_model_integration_test_batched(self):
        model = PerceptionLMForConditionalGeneration.from_pretrained(
            TEST_MODEL_PATH, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        processor = AutoProcessor.from_pretrained(TEST_MODEL_PATH)
        inputs = processor.apply_chat_template(
            [self.conversation1, self.conversation2],
            num_frames=32,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            padding_side="left",
        ).to(torch_device)

        generate_ids = model.generate(**inputs, max_new_tokens=18)
        input_length = inputs["input_ids"].shape[1]
        generate_ids_without_inputs = generate_ids[:, input_length:]

        EXPECTED_DECODED_TEXT = ['The bar plot displays the values of four categories: step, horror, mood, and lumber', 'The video shows a group of people in green shirts and white shorts performing a jump rope routine']  # fmt: skip

        self.assertEqual(
            processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )

    def test_generation_no_images(self):
        # model_id = "facebook/Perception-LM-1B"
        model = PerceptionLMForConditionalGeneration.from_pretrained(
            TEST_MODEL_PATH, quantization_config=BitsAndBytesConfig(load_in_4bit=True)
        )
        processor = AutoProcessor.from_pretrained(TEST_MODEL_PATH)

        # Prepare inputs with no images
        inputs = processor(text="Hello, I am", return_tensors="pt").to(torch_device)

        # Make sure that `generate` works
        _ = model.generate(**inputs, max_new_tokens=20)
