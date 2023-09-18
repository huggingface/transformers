# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch InstructBLIP model. """


import inspect
import tempfile
import unittest

import numpy as np
import requests

from transformers import (
    CONFIG_MAPPING,
    InstructBlipConfig,
    InstructBlipProcessor,
    InstructBlipQFormerConfig,
    InstructBlipVisionConfig,
)
from transformers.testing_utils import require_bitsandbytes, require_torch, require_vision, slow, torch_device
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

    from transformers import InstructBlipForConditionalGeneration, InstructBlipVisionModel
    from transformers.models.instructblip.modeling_instructblip import INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST


if is_vision_available():
    from PIL import Image


class InstructBlipVisionModelTester:
    def __init__(
        self,
        parent,
        batch_size=12,
        image_size=30,
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
        pixel_values = floats_tensor([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return InstructBlipVisionConfig(
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
        model = InstructBlipVisionModel(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(pixel_values)
        # expected sequence length = num_patches + 1 (we add 1 for the [CLS] token)
        image_size = (self.image_size, self.image_size)
        patch_size = (self.patch_size, self.patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, num_patches + 1, self.hidden_size))
        self.parent.assertEqual(result.pooler_output.shape, (self.batch_size, self.hidden_size))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


@require_torch
class InstructBlipVisionModelTest(ModelTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as InstructBLIP's vision encoder does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (InstructBlipVisionModel,) if is_torch_available() else ()
    fx_compatible = False
    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = InstructBlipVisionModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=InstructBlipVisionConfig, has_text_modality=False, hidden_size=37
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="InstructBLIP's vision encoder does not use inputs_embeds")
    def test_inputs_embeds(self):
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

    @unittest.skip(reason="InstructBlipVisionModel is an internal building block, doesn't support standalone training")
    def test_training(self):
        pass

    @unittest.skip(reason="InstructBlipVisionModel is an internal building block, doesn't support standalone training")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(reason="InstructBlipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="InstructBlipVisionModel has no base class and is not available in MODEL_MAPPING")
    def test_save_load_fast_init_to_base(self):
        pass

    @slow
    def test_model_from_pretrained(self):
        for model_name in INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
            model = InstructBlipVisionModel.from_pretrained(model_name)
            self.assertIsNotNone(model)


class InstructBlipQFormerModelTester:
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
        return InstructBlipQFormerConfig(
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
class InstructBlipTextModelDecoderOnlyTester:
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
        max_position_embeddings=20,
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
class InstructBlipForConditionalGenerationDecoderOnlyModelTester:
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
        self.vision_model_tester = InstructBlipVisionModelTester(parent, **vision_kwargs)
        self.qformer_model_tester = InstructBlipQFormerModelTester(parent, **qformer_kwargs)
        self.text_model_tester = InstructBlipTextModelDecoderOnlyTester(parent, **text_kwargs)
        self.is_training = is_training
        self.num_query_tokens = num_query_tokens

    def prepare_config_and_inputs(self):
        _, pixel_values = self.vision_model_tester.prepare_config_and_inputs()
        _, _, _, qformer_input_ids, qformer_attention_mask = self.qformer_model_tester.prepare_config_and_inputs()
        _, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values

    def get_config(self):
        return InstructBlipConfig.from_vision_qformer_text_configs(
            vision_config=self.vision_model_tester.get_config(),
            qformer_config=self.qformer_model_tester.get_config(),
            text_config=self.text_model_tester.get_config(),
            num_query_tokens=self.num_query_tokens,
        )

    def create_and_check_for_conditional_generation(
        self, config, input_ids, attention_mask, qformer_input_ids, qformer_attention_mask, pixel_values
    ):
        model = InstructBlipForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            result = model(
                pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                qformer_input_ids=qformer_input_ids,
                qformer_attention_mask=qformer_attention_mask,
            )

        expected_seq_length = self.num_query_tokens + self.text_model_tester.seq_length
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
class InstructBlipForConditionalGenerationDecoderOnlyTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (InstructBlipForConditionalGeneration,) if is_torch_available() else ()
    fx_compatible = False
    test_head_masking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    test_torchscript = False

    def setUp(self):
        self.model_tester = InstructBlipForConditionalGenerationDecoderOnlyModelTester(self)

    def test_for_conditional_generation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_conditional_generation(*config_and_inputs)

    @unittest.skip(reason="Hidden_states is tested in individual model tests")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="InstructBlipForConditionalGeneration doesn't support inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="Tied weights are tested in individual model tests")
    def test_tied_weights_keys(self):
        pass

    @unittest.skip(reason="Retain_grad is tested in individual model tests")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="InstructBlipModel does not have input/output embeddings")
    def test_model_common_attributes(self):
        pass

    @unittest.skip(reason="There's no base InstructBlipModel")
    def test_save_load_fast_init_from_base(self):
        pass

    @unittest.skip(reason="There's no base InstructBlipModel")
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

        # Save InstructBlipConfig and check if we can load InstructBlipVisionConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            vision_config = InstructBlipVisionConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.vision_config.to_dict(), vision_config.to_dict())

        # Save InstructBlipConfig and check if we can load InstructBlipQFormerConfig from it
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config.save_pretrained(tmp_dir_name)
            qformer_config = InstructBlipQFormerConfig.from_pretrained(tmp_dir_name)
            self.assertDictEqual(config.qformer_config.to_dict(), qformer_config.to_dict())

    @slow
    def test_model_from_pretrained(self):
        for model_name in INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST:
            model = InstructBlipForConditionalGeneration.from_pretrained(model_name)
            self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
def prepare_img():
    url = "https://huggingface.co/hf-internal-testing/blip-test-image/resolve/main/demo.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@require_vision
@require_torch
@slow
class InstructBlipModelIntegrationTest(unittest.TestCase):
    @require_bitsandbytes
    def test_inference_vicuna_7b(self):
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b", load_in_8bit=True
        )

        url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "What is unusual about this image?"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device, torch.float16)

        # verify logits
        with torch.no_grad():
            logits = model(**inputs).logits

        expected_slice = torch.tensor(
            [[-3.4902, -12.5078, 8.4141], [-5.1211, -12.1328, 7.8281], [-4.0312, -13.5938, 9.1172]],
            device=torch_device,
        )
        self.assertTrue(torch.allclose(logits[0, :3, :3].float(), expected_slice, atol=1e-3))

        # verify generation
        outputs = model.generate(**inputs, max_new_tokens=30)
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

        # fmt: off
        expected_outputs = [2, 450, 22910, 9565, 310, 445, 1967, 338, 393, 263, 767, 338, 13977, 292, 22095, 373, 278, 1250, 310, 263, 13328, 20134, 29963, 1550, 19500, 1623, 263, 19587, 4272, 11952, 29889]
        # fmt: on
        self.assertEqual(outputs[0].tolist(), expected_outputs)
        self.assertEqual(
            generated_text,
            "The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV while driving down a busy city street.",
        )

    def test_inference_flant5_xl(self):
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-flan-t5-xl",
            torch_dtype=torch.bfloat16,
        ).to(torch_device)

        url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        prompt = "What is unusual about this image?"
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(torch_device)

        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)

        outputs = model.generate(
            **inputs,
            do_sample=False,
            num_beams=5,
            max_length=256,
            min_length=1,
            top_p=0.9,
            repetition_penalty=1.5,
            length_penalty=1.0,
            temperature=1,
        )
        generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        # fmt: off
        expected_outputs = [0, 37, 1023, 9850, 7, 3, 9, 388, 3575, 53, 4954, 30, 8, 223, 13, 3, 9, 4459, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 37, 388, 19, 5119, 3, 9, 4459, 8677, 28, 3, 9, 2756, 4459, 6177, 6, 11, 3, 88, 19, 338, 46, 3575, 53, 1476, 12, 743, 112, 2491, 5, 37, 1023, 19, 7225, 788, 12, 8, 685, 24, 34, 1267, 3, 9, 388, 3575, 53, 4954, 30, 8, 223, 13, 3, 9, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 94, 19, 487, 24, 8, 388, 19, 1119, 12, 1097, 540, 57, 692, 112, 10428, 30, 8, 223, 13, 8, 4049, 6, 68, 34, 19, 92, 487, 24, 3, 88, 19, 1119, 12, 1097, 97, 57, 692, 112, 10428, 30, 8, 223, 13, 8, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 3, 13865, 13, 8, 1053, 21, 8, 388, 31, 7, 2874, 6, 34, 19, 964, 24, 3, 88, 19, 1119, 12, 1097, 97, 57, 692, 112, 10428, 30, 8, 223, 13, 8, 4049, 16, 8, 2214, 13, 3, 9, 3164, 690, 2815, 5, 1]
        # fmt: on
        self.assertEqual(outputs[0].tolist(), expected_outputs)
        self.assertEqual(
            generated_text,
            "The image depicts a man ironing clothes on the back of a yellow van in the middle of a busy city street. The man is wearing a yellow shirt with a bright yellow tie, and he is using an ironing board to complete his task. The image is unusual due to the fact that it shows a man ironing clothes on the back of a van in the middle of a busy city street. It is possible that the man is trying to save money by doing his laundry on the back of the van, but it is also possible that he is trying to save time by doing his laundry on the back of the van in the middle of a busy city street. Regardless of the reason for the man's actions, it is clear that he is trying to save time by doing his laundry on the back of the van in the middle of a busy city street.",
        )
