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
"""Testing suite for the PyTorch Magma model."""

import unittest

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import (
    AutoProcessor,
    MagmaConfig,
    MagmaForCausalLM,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    cleanup,
    require_bitsandbytes,
    require_torch,
    slow,
    torch_device,
)
from transformers.configuration_utils import PretrainedConfig
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class MagmaVisionText2TextModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        image_token_index=1,
        projector_hidden_act="gelu",
        seq_length=7,
        vision_feature_layer=-1,
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
            "num_key_value_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-5,
            "mm_hidden_size": 768,
            "vision_backbone": "convnexttiny",
            "vision_feature_layer": "clip_vis_dense",            
            "img_anyres_strategy": "crop",
            "mm_projector_type": "mlp2x_gelu",
            "mm_use_row_seperator": False,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = PretrainedConfig(**vision_config)
        self.pad_token_id = text_config["pad_token_id"]
        self.num_image_tokens = 1
        self.input_id_length = seq_length + self.num_image_tokens

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = 3
        self.crop_height = 1
        self.crop_width = 1
        self.image_size = self.vision_config.image_size
        self.seq_length = seq_length + self.crop_height * self.crop_width * (self.image_size // 32) ** 2 \
            + (self.crop_height * (self.image_size // 32) if self.vision_config.mm_use_row_seperator else 0)

    def get_config(self):
        return MagmaConfig(
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_id=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.crop_height*self.crop_width,
                self.vision_config.num_channels,
                self.vision_config.image_size,
                self.vision_config.image_size,
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.input_id_length], config.text_config.vocab_size - 2) + 2
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)

        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = config.image_token_index

        labels = torch.zeros((self.batch_size, self.seq_length), dtype=torch.long, device=torch_device)
        labels[:, : self.num_image_tokens] == self.ignore_index

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor([[self.crop_height, self.crop_width]] * self.batch_size).unsqueeze(1),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return config, inputs_dict

    def create_and_check_magma_model_fp16_forward(
        self, config, input_ids, pixel_values, attention_mask, image_sizes
    ):
        model = MagmaForCausalLM(config=config)
        model.to(torch_device)
        model.half()
        model.eval()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_sizes=image_sizes,
            pixel_values=pixel_values.to(torch.bfloat16),
            return_dict=True,
        )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())

    def create_and_check_magma_model_fp16_autocast_forward(
        self, config, input_ids, pixel_values, attention_mask, image_sizes
    ):
        config.torch_dtype = torch.float16
        model = MagmaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                pixel_values=pixel_values.to(torch.bfloat16),
                return_dict=True,
            )["logits"]
        self.parent.assertFalse(torch.isnan(logits).any().item())


@require_torch
class MagmaModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `MagmaForCausalLM`.
    """

    all_model_classes = (MagmaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"image-text-to-text": MagmaForCausalLM} if is_torch_available() else {}
    )
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = MagmaVisionText2TextModelTester(self)
        common_properties = ["image_token_index"]
        self.config_tester = ConfigTester(self, config_class=MagmaConfig, has_text_modality=False, common_properties=common_properties)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                # LLaVa Onevision has SigLIP backbone which init weights differently from CLIP
                if "image_newline" in name or "vision_tower" in name:
                    continue
                elif param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

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
            del inputs['image_sizes']

            inputs_embeds = model.get_input_embeddings()(input_ids)

            with torch.no_grad():
                out_ids = model(input_ids=input_ids, **inputs)[0]
                out_embeds = model(inputs_embeds=inputs_embeds, **inputs)[0]
            torch.testing.assert_close(out_embeds, out_ids)

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, SiglipVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, SiglipVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecture seem to not compute gradients properly when using GC, SiglipVisionModel does not support standalone training"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("FlashAttention only support fp16 and bf16 data type")
    def test_flash_attn_2_fp32_ln(self):
        pass

    @unittest.skip(
        "VLMs need lots of steps to prepare images/mask correctly to get pad-free inputs. Can be tested as part of LLM test"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("Magma has dynamic control flow in unpadding")
    def test_generate_compile_model_forward(self):
        pass

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="`image_attention_mask` has a specific shape")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip(reason="Cannot unpad inputs for all modalities so easily")
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(reason="Dynamo error")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip(reason="Test requires mpi4py, which is not necessarily installed")
    def test_resize_embeddings_untied_with_deepspeed(self):
        pass

    @unittest.skip(reason="Test requires mpi4py, which is not necessarily installed")
    def test_resize_embeddings_untied_with_deepspeed_multi_gpu(self):
        pass

    @unittest.skip(reason="Test requires mpi4py, which is not necessarily installed")
    def test_resize_tokens_embeddings_with_deepspeed(self):
        pass

    @unittest.skip(reason="Test requires mpi4py, which is not necessarily installed")
    def test_resize_tokens_embeddings_with_deepspeed_multi_gpu(self):
        pass

    @unittest.skip("Magma is too big for the common tests")
    def test_model_is_small(self):
        pass

    @unittest.skip(reason="Unsupported")
    def test_generate_from_inputs_embeds_0_greedy(self):
        pass

    @unittest.skip(reason="Unsupported")
    def test_generate_from_inputs_embeds_1_beam_search(self):
        pass

    @unittest.skip("Magma uses convnext as backbone, which is not supported by flash attention 2")
    def test_flash_attn_2_can_dispatch_composite_models(self):
        pass

    @unittest.skip("MagmaForCausalLM does not support an instance of `Cache` as `past_key_values`. Need to revisit later.")
    def test_generate_continue_from_inputs_embeds(self):
        pass

    @unittest.skip("MagmaForCausalLM does not support an instance of `Cache` as `past_key_values`. Need to revisit later.")
    def test_generate_continue_from_past_key_values(self):
        pass

    @unittest.skip(reason="CPU offload is not yet supported")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip("Model parallel not supported")
    def test_model_parallel_beam_search(self):
        pass

    @unittest.skip("Model parallel not supported")
    def test_model_parallelism(self):
        pass

@require_torch
class MagmaIntegrationTest(unittest.TestCase):
    checkpoint_path = "microsoft/Magma-8B"
    def setUp(self):
        image_url = "https://microsoft.github.io/Magma/static/images/logo.png"
        local_image_path = "/tmp/magma_logo.jpg"
        with open(local_image_path, "wb") as f:
            f.write(requests.get(image_url).content)
        self.image = Image.open(local_image_path).convert("RGB")

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_text_only_generation(self):
        dtype = torch.bfloat16
        model = MagmaForCausalLM.from_pretrained(
            self.checkpoint_path, torch_dtype=dtype, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint_path)

        convs = [
            {"role": "system", "content": "You are agent that can see, talk and act."},            
            {"role": "user", "content": "What is 1+1?"},
        ]
        prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=None, text=prompt, return_tensors="pt")
        inputs = inputs.to("cuda").to(dtype)

        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
            "use_cache": True,
            "num_beams": 1,
        } 

        output = model.generate(
            **inputs,
            **generation_args,
        )
        generate_ids = output[:, inputs["input_ids"].shape[-1] :]
        response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

        EXPECTED_RESPONSE = "1+1 is equal to 2."

        self.assertEqual(response, EXPECTED_RESPONSE)

    def test_vision_text_generation(self):
        dtype = torch.bfloat16
        model = MagmaForCausalLM.from_pretrained(
            self.checkpoint_path, torch_dtype=dtype, device_map=torch_device
        )
        processor = AutoProcessor.from_pretrained(self.checkpoint_path)

        convs = [
            {"role": "system", "content": "You are agent that can see, talk and act."},            
            {"role": "user", "content": "<image_start><image><image_end>\nWhat is the letter on the robot?"},
        ]
        prompt = processor.tokenizer.apply_chat_template(convs, tokenize=False, add_generation_prompt=True)
        inputs = processor(images=[self.image], text=prompt, return_tensors="pt")
        inputs = inputs.to("cuda").to(dtype)

        generation_args = { 
            "max_new_tokens": 500, 
            "temperature": 0.0, 
            "do_sample": False, 
            "use_cache": True,
            "num_beams": 1,
        } 

        output = model.generate(
            **inputs,
            **generation_args,
        )
        generate_ids = output[:, inputs["input_ids"].shape[-1] :]
        response = processor.decode(generate_ids[0], skip_special_tokens=True).strip()

        EXPECTED_RESPONSE = "The letter on the robot is \"M\"."

        self.assertEqual(response, EXPECTED_RESPONSE)