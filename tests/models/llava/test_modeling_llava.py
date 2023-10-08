# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch LLaMA model. """


import inspect
import unittest
from typing import Optional, Union

import requests
from PIL import Image

from transformers import LlamaConfig, LlavaConfig, LlavaVisionConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        LlamaModel,
        LlavaForCausalLM,
        LlavaProcessor,
    )


class LlamaModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)
        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        return config, input_ids


class LlavaVisionTester:
    def __init__(
        self,
        parent,
        d_model: int = 32,
        emb_pdrop: int = 0,
        embedding_fraction: float = 1.0,
        expansion_ratio: int = 4,
        freeze_mm_mlp_adapter: bool = False,
        init_device: str = "cpu",
        learned_pos_emb: bool = True,
        logit_scale: Optional[Union[float, str]] = None,
        max_seq_len: int = 7,
        mm_hidden_size: int = 30,
        mm_use_im_start_end: bool = True,
        mm_vision_select_layer: int = -2,
        mm_vision_tower: str = "openai/clip-vit-large-patch14",
        model_type: str = "llava_mpt",
        n_heads: int = 4,
        n_layers: int = 2,
        no_bias: bool = True,
        norm_type: str = "low_precision_layernorm",
        resid_pdrop: int = 0,
        sep_image_conv_front: bool = False,
        tune_mm_mlp_adapter: bool = False,
        use_cache: bool = True,
        use_mm_proj: bool = True,
        verbose: int = 0,
        vocab_size: int = 99,
        **kwargs,
    ):
        self.freeze_mm_mlp_adapter = freeze_mm_mlp_adapter
        self.mm_hidden_size = mm_hidden_size
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_vision_select_layer = mm_vision_select_layer
        self.sep_image_conv_front = sep_image_conv_front
        self.tune_mm_mlp_adapter = tune_mm_mlp_adapter
        self.use_mm_proj = use_mm_proj
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.verbose = verbose
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        # self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        self.image_size = 30
        self.batch_size = 13
        self.num_channels = 3
        self.projector = "None"

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([self.num_channels, self.image_size, self.image_size])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return LlavaVisionConfig(
            vocab_size=self.vocab_size,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            d_model=self.d_model,
            max_seq_len=self.max_seq_len,
            mm_hidden_size=self.mm_hidden_size,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            projector=self.projector,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        return config, pixel_values


class LlavaModelTester:
    def __init__(
        self,
        parent,
        llama_kwargs=None,
        llava_kwargs=None,
        num_hidden_layers=2,
    ):
        self.is_training = True

        if llama_kwargs is None:
            llama_kwargs = {}
        if llava_kwargs is None:
            llava_kwargs = {}

        self.num_hidden_layers = num_hidden_layers
        self.parent = parent
        self.batch_size = 13
        self.llama_model_tester = LlamaModelTester(parent, **llama_kwargs)
        self.llava_vision_tester = LlavaVisionTester(parent, **llava_kwargs)

    def prepare_config_and_inputs(self):
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.llama_model_tester.prepare_config_and_inputs()
        _, pixel_values = self.llava_vision_tester.prepare_config_and_inputs()

        config = self.get_config()

        return (
            config,
            input_ids,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        return LlavaConfig.from_llava_configs(
            llama_config=self.llama_model_tester.get_config(),
            llava_vision_config=self.llava_vision_tester.get_config(),
        )

    def create_and_check_model(
        self, config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LlamaModel(config=config.llama_model)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_causal_lm(
        self, config, input_ids, pixel_values, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels
    ):
        model = LlavaForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids)
        self.parent.assertEqual(
            result.logits.shape,
            (
                self.llama_model_tester.batch_size,
                self.llama_model_tester.seq_length,
                self.llama_model_tester.vocab_size,
            ),
        )

    @unittest.skip(reason="Llava does not have input/output embeddings")
    def create_and_check_decoder_model_past_large_inputs(
        self,
        config,
        input_ids,
        pixel_values,
    ):
        pass

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            pixel_values,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "labels": token_labels,
        }

        return config, inputs_dict


@require_torch
class LlavaModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (LlavaForCausalLM,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "text-generation": LlavaForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = False

    @unittest.skip
    def test_correct_missing_keys(self):
        pass

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="model creats its own input_embeds")
    def test_inputs_embeds(self):
        pass

    def setUp(self):
        self.model_tester = LlavaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlavaConfig)

    def test_for_causal_lm(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_causal_lm(*config_and_inputs)

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[-2:-1], expected_arg_names)

    def test_get_text_features(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        inputs_dict = {
            "input_ids": torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(torch_device),
            "attention_mask": torch.LongTensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]).to(torch_device),
        }
        model = LlavaForCausalLM(config).to(torch_device)
        model.eval()
        text_features = model(**inputs_dict)
        self.assertEqual(text_features[0].shape, (1, 10, self.model_tester.llama_model_tester.vocab_size))

    # won't work without input ids
    def test_get_image_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = LlavaForCausalLM(config).to(torch_device)
        model.eval()
        image_features = model.generate(**inputs_dict)
        self.assertEqual(image_features.shape, (self.model_tester.llama_model_tester.batch_size, 20))

    # override from common to deal with nested configurations (`vision_config`, `text_config` and `qformer_config`)
    def test_initialization(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        configs_no_init = _config_zero_init(config)
        for key in ["llama_config", "llava_vision_config"]:
            setattr(configs_no_init, key, _config_zero_init(getattr(configs_no_init, key)))
        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    @unittest.skip("LLaMA buffers include complex numbers, which breaks this test")
    def test_save_load_fast_init_from_base(self):
        pass


@require_torch
class LlavaIntegrationTest(unittest.TestCase):
    @slow
    def test_model_7b_logits(self):
        url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "What is unusual about this image?"

        model = LlavaForCausalLM.from_pretrained(
            "shauray/Llava-Llama-2-7B-hf",
            torch_dtype=torch.float16,
            device_map=torch_device,
        ).to(torch_device)

        processor = LlavaProcessor.from_pretrained(
            "shauray/Llava-Llama-2-7B-hf", torch_dtype=torch.float16, device=torch_device
        )
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, torch.float16)

        # verify logits
        with torch.no_grad():
            logits = model(**inputs).logits

        expected_slice = torch.tensor(
            [
                [-1.5898e-03, -2.4231e-01, 4.6484e-01],
                [-7.8398e00, -4.0273e00, -2.0098e00],
                [-4.6172e00, -2.3887e00, 4.2432e-01],
            ],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(logits[0, :3, :3].float(), expected_slice, atol=1e-3))

        # verify generation
        outputs = model.generate(**inputs, max_length=128, temperature=0.1)
        generated_text = processor.decode(outputs[0, inputs["input_ids"].shape[1] :]).strip()
        # fmt: off
        expected_outputs = [1, 200, 29871, 13, 5618, 338, 22910, 1048, 445, 1967, 29973, 13, 797, 278, 1967, 29892, 263, 767, 338, 13407, 373, 278, 1250, 310, 263, 13328, 1559, 29892, 13977, 292, 22095, 29889, 910, 338, 385, 22910, 11126, 408, 372, 338, 451, 3619, 363, 2305, 304, 13977, 22095, 1550, 13407, 373, 263, 1559, 29892, 7148, 297, 263, 19587, 4272, 11952, 29889, 450, 767, 338, 884, 591, 4362, 263, 13328, 528, 2728, 29892, 607, 12778, 304, 278, 22910, 5469, 310, 278, 9088, 29889, 13, 13, 13, 5618, 338, 278, 767, 2599, 29973, 13, 1576, 767, 338, 13977, 292, 22095, 1550, 13407, 373, 278, 1250, 310, 263, 13328, 1559, 29889, 940, 338, 773, 385, 13977, 292, 7613, 304, 13977, 22095, 29892, 607, 338, 451, 263, 15662, 6354, 297, 263, 19587, 4272]
        # fmt: on
        self.assertEqual(outputs[0].tolist(), expected_outputs)
        self.assertEqual(
            generated_text,
            """In the image, a man is standing on the back of a yellow car, ironing clothes. This is an unusual sight as it is not common for people to iron clothes while standing on a car, especially in a busy city street. The man is also wearing a yellow shirt, which adds to the unusual nature of the scene.
            What is the man doing
            The man is ironing clothes while standing on the back of a yellow car. He is using an ironing board to iron clothes, which is not a typical activity in a busy city""",
        )
