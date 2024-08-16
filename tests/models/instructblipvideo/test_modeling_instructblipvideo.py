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
"""Testing suite for the PyTorch InstructBlipVideo model."""

import inspect
import tempfile
import unittest

import numpy as np
from huggingface_hub import hf_hub_download
from parameterized import parameterized

from transformers import (
    CONFIG_MAPPING,
    InstructBlipVideoConfig,
    InstructBlipVideoProcessor,
    InstructBlipVideoQFormerConfig,
    InstructBlipVideoVisionConfig,
)
from transformers.testing_utils import (
    require_accelerate,
    require_bitsandbytes,
    require_torch,
    require_vision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    floats_tensor,
    ids_tensor,
    random_attention_mask,
)


if is_torch_available():
    import torch
    from torch import nn

    from transformers import InstructBlipVideoForConditionalGeneration, InstructBlipVideoVisionModel


if is_vision_available():
    pass


class InstructBlipVideoVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
        frames=4,
        patch_size=2,
        num_channels=3,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=1e-10,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.image_size = image_size
        self.frames = frames
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope

        # in case of a vision transformer, the seq length equals the number of patches + 1 (we add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches + 1

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor(
            [self.batch_size * self.frames, self.num_channels, self.image_size, self.image_size]
        )
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return InstructBlipVideoVisionConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        )

    def create_and_check_model(self, config, pixel_values):
        model = InstructBlipVideoVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(
            result.last_hidden_state.shape, (self.batch_size * self.frames, num_patches + 1, self.hidden_size)
        )
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size * self.frames, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class InstructBlipVideoVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as InstructBlipVideo's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (InstructBlipVideoVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = InstructBlipVideoVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=InstructBlipVideoVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="InstructBlipVideo's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="InstructBlipVideo's vision encoder is an nn.Embeddings layer")
    def test_model_get_set_embeddings(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skip(
        reason="InstructBlipVideoVisionModel is an internal building block, doesn't support standalone training"
    )
    def test_training(self):
        pass

    @unittest.skip(
        reason="InstructBlipVideoVisionModel is an internal building block, doesn't support standalone training"
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

    @unittest.skip(reason="InstructBlipVideoVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="InstructBlipVideoVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/instructblip-vicuna-7b"
        model = InstructBlipVideoVisionModel.from_pretrained(model_name)
        self.assertIsNotNone(model)


class InstructBlipVideoQFormerModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        projection_dim=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=512,
        initializer_range=0.02,
        bos_token_id=0,
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
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        qformer_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])
            qformer_attention_mask = ids_tensor([self.batch_size, self.seq_length], vocab_size=2)

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, input_ids, input_mask, qformer_input_ids, qformer_attention_mask

    def get_config(self):
        return InstructBlipVideoQFormerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
        )


# this class is based on `OPTModelTester` found in tests/models/opt/test_modeling_opt.py
class InstructBlipVideoTextModelDecoderOnlyTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=256,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        embed_dim=16,
        num_labels=3,
        word_embed_proj_dim=16,
        type_sequence_label_size=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.embed_dim = embed_dim
        self.num_labels = num_labels
        self.type_sequence_label_size = type_sequence_label_size
        self.word_embed_proj_dim = word_embed_proj_dim
        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(3)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        attention_mask = input_ids.ne(self.pad_token_id)

        return config, input_ids, attention_mask

    def get_config(self):
        return CONFIG_MAPPING["opt"](
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            embed_dim=self.embed_dim,
            is_encoder_decoder=False,
            word_embed_proj_dim=self.word_embed_proj_dim,
        )


# this model tester uses a decoder-only language model (OPT)
class InstructBlipVideoForConditionalGenerationDecoderOnlyModelTester:
    def __init__(
        self, parent, vision_kwargs=None, qformer_kwargs=None, text_kwargs=None, is_training=True, num_query_tokens=10
    ):
        if vision_kwargs is None:
            vision_kwargs = {}
        if qformer_kwargs is None:
            qformer_kwargs = {}
        if text_kwargs is None:
            text_kwargs = {}

        self.parent = parent
        self.vision_model_tester = InstructBlipVideoVisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = InstructBlipVideoQFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = InstructBlipVideoTextModelDecoderOnlyTester(parent, **text_kwargs)
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test
        self.seq_length = self.text_model_tester.seq_length  # need seq_length for common tests
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        _, _, _, qformer_input_ids, qformer_attention_mask = self.qformer_model_tester.prepare_config_and_inputs()
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        frames = self.vision_model_tester.frames
        _, c, h, w = pixel_values.shape
        pixel_values = pixel_values.reshape(-1, frames, c, h, w)

        config = self.get_config()

        return config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values

    def get_config(self):
        return InstructBlipVideoConfig.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values
    ):
        model = InstructBlipVideoForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                qformer_input_ids=qformer_input_ids,
                qformer_attention_mask=qformer_attention_mask,
            )

        expected_seq_length = (
            self.num_query_tokens * self.vision_model_tester.frames
        ) + self.text_model_tester.seq_length
        self.parent.assertEqual(
            result.logits.shape,
            (self.vision_model_tester.batch_size, expected_seq_length, self.text_model_tester.vocab_size),
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values = config_and_inputs
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "qformer_input_ids": qformer_input_ids,
            "qformer_attention_mask": qformer_attention_mask,
            "labels": input_ids,
        }
        return config, inputs_dict


@require_torch
class InstructBlipVideoForConditionalGenerationDecoderOnlyTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (InstructBlipVideoForConditionalGeneration,) if is_torch_available() else ()
    all_generative_model_classes = (InstructBlipVideoForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = InstructBlipVideoForConditionalGenerationDecoderOnlyModelTester(self)

    @parameterized.expand([(True,), (False,)])
    def test_greedy_generation(self, use_cache: bool):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            model.to(torch_device)
            model.eval()
            model.config.text_config.architectures = ["OptForCausalLM"]

            out = model.generate(**inputs_dict, min_new_tokens=20, max_new_tokens=20, use_cache=use_cache)
            self.assertTrue(out.shape[1] == 21)  # BLIP is special, therefore 21

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="InstructBlipVideoForConditionalGeneration doesn't support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Tied weights are tested in individual model tests")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="InstructBlipVideoModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="There's no base InstructBlipVideoModel")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base InstructBlipVideoModel")
    def test_save_load_fast_init_to_base(self):
        pass

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_load_vision_qformer_text_config(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # Save InstructBlipVideoConfig and check if we can load InstructBlipVideoVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = InstructBlipVideoVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save InstructBlipVideoConfig and check if we can load InstructBlipVideoQFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = InstructBlipVideoQFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        model_name = "Salesforce/instructblip-vicuna-7b"
        model = InstructBlipVideoForConditionalGeneration.from_pretrained(model_name)
        self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_video():
    video_file = hf_hub_download(
        repo_id="raushan-testing-hf/videos-test", filename="video_demo.npy", repo_type="dataset"
    )
    video = np.load(video_file)[::2]  # sample every 2nd frame to get 4 frames total
    return video


@require_vision
@require_torch
@require_bitsandbytes
@require_accelerate
@slow
class InstructBlipVideoModelIntegrationTest(unittest.TestCase):
    def test_inference_vicuna_7b(self):
        processor = InstructBlipVideoProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipVideoForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b", load_in_8bit=True, low_cpu_mem_usage=True
        )

        clip = prepare_video()
        prompt = "Explain what is happening in this short video."
        inputs = processor(images=clip, text=prompt, return_tensors="pt").to(torch_device, torch.float16)

        # verify generation
        outputs = model.generate(**inputs, max_new_tokens=30)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        self.assertEqual(
            generated_text,
            "a baby girl wearing glasses is reading a book on the bed 1080p",
        )
