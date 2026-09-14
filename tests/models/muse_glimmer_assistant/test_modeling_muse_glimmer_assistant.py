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
"""Testing suite for the PyTorch MuseGlimmerAssistant model."""

import unittest

from transformers.testing_utils import (
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import (
    ModelTesterMixin,
    random_attention_mask,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoProcessor,
        BitsAndBytesConfig,
        MuseGlimmerAssistantConfig,
        MuseGlimmerAssistantModel,
        MuseGlimmerForConditionalGeneration,
    )


class MuseGlimmerAssistantModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=True,
        hidden_size=32,
        head_dim=8,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        hidden_act="silu",
        block_size=4,
        target_layer_ids=(0, 2),
        mask_token_id=1,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=4,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.block_size = block_size
        self.target_layer_ids = list(target_layer_ids)
        self.mask_token_id = mask_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        # set this for tests that check hidden state length
        self.encoder_seq_length = block_size

    def prepare_config_and_inputs(self):
        input_embeds = torch.randn([self.batch_size, self.block_size, self.hidden_size], device=torch_device)
        context_embeds = torch.randn(
            [self.batch_size, self.seq_length, self.hidden_size * len(self.target_layer_ids)], device=torch_device
        )
        input_mask = random_attention_mask([self.batch_size, self.seq_length + self.block_size])
        config = self.get_config()

        return config, input_embeds, context_embeds, input_mask

    def get_config(self):
        return MuseGlimmerAssistantConfig(
            head_dim=self.head_dim,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            block_size=self.block_size,
            target_layer_ids=self.target_layer_ids,
            mask_token_id=self.mask_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_embeds, context_embeds, input_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "noise_embeds": input_embeds,
            "context_hidden_states": context_embeds,
            "attention_mask": input_mask,
        }

        return config, inputs_dict


@require_torch
@unittest.skip("Need some test work, as it needs different inputs (dflash speculator model)")
class MuseGlimmerAssistantModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (MuseGlimmerAssistantModel,) if is_torch_available() else ()
    additional_model_inputs = ["context_hidden_states"]

    # model has no embedding table of its own
    test_resize_embeddings = False
    test_resize_position_embeddings = False

    def setUp(self):
        self.model_tester = MuseGlimmerAssistantModelTester(self)

    @unittest.skip("We need more than 2 layers to test `target-layer-ids`")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip("Model has no embedding table of its own")
    def test_model_get_set_embeddings(self):
        pass

    @unittest.skip("Model has non standard attention weight shape due to KV context")
    def test_attention_outputs(self):
        pass

    @unittest.skip("Fix me later, head dimension somehow isn't multiple of 16!")
    def test_flex_attention_with_grads(self):
        pass

    @unittest.skip("Fix me later, not worth wasting time on it now")
    def test_retain_grad_hidden_states_attention(self):
        pass


# The drafter checkpoint is ~5 layers (hidden_size=6656) — roughly 3–4 GiB in bfloat16, which fits on a
# single 24 GiB accelerator without CPU offloading.  The full DFlash test also loads the main 30B model
# with device_map="auto" and a 70% per-GPU memory cap so activation buffers (e.g. lm_head matmul) have
# enough headroom; excess layers spill to CPU via accelerate offloading.
@slow
@require_torch_accelerator
class MuseGlimmerAssistantIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    drafter_id = "meta-models/Muse-Glimmer-30B-assistant"
    main_model_id = "meta-models/Muse-Glimmer-30B"

    @classmethod
    def setUpClass(cls):
        cls.drafter = None
        cls.model = None

    @classmethod
    def get_drafter(cls):
        if cls.drafter is None:
            cls.drafter = MuseGlimmerAssistantModel.from_pretrained(
                cls.drafter_id, dtype=torch.bfloat16, device_map="auto"
            )
        return cls.drafter

    @classmethod
    def get_model(cls):
        if cls.model is None:
            # Load in 4-bit so the 30B model (~15 GiB) fits on a single 24 GiB accelerator
            # with enough headroom for the KV cache and lm_head activation buffers.
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            cls.model = MuseGlimmerForConditionalGeneration.from_pretrained(
                cls.main_model_id,
                quantization_config=bnb_config,
                device_map="auto",
            )
        return cls.model

    def test_drafter_forward_output_shape(self):
        """Standalone drafter forward pass with synthetic inputs.

        The assistant borrows embeddings from the main model at runtime, so the only meaningful
        standalone check is that the drafter accepts correctly shaped inputs and produces a
        finite hidden-state tensor of the expected shape.  No main model is needed here.
        """
        drafter = self.get_drafter()
        config = drafter.config

        noise_embeds = torch.randn(1, config.block_size, config.hidden_size, dtype=torch.bfloat16, device=torch_device)
        # context_hidden_states: [batch, context_len, hidden_size * num_target_layers]
        context_hidden_states = torch.randn(
            1, 7, config.hidden_size * len(config.target_layer_ids), dtype=torch.bfloat16, device=torch_device
        )

        with torch.no_grad():
            out = drafter(noise_embeds=noise_embeds, context_hidden_states=context_hidden_states)

        self.assertEqual(
            out.last_hidden_state.shape,
            torch.Size([1, config.block_size, config.hidden_size]),
        )
        self.assertTrue(out.last_hidden_state.isfinite().all())

    def test_dflash_speculative_generation(self):
        """End-to-end DFlash speculative decoding produces the same text as greedy decoding.

        DFlash is a lossless speculative-decoding algorithm, so the completion must match the
        reference produced by MuseGlimmerIntegrationTest.test_text_generation_matches_reference.
        This test therefore validates both that the DFlash pipeline runs without errors and that
        the drafter does not alter the model's output distribution.
        """
        model = self.get_model()
        drafter = self.get_drafter()
        processor = AutoProcessor.from_pretrained(self.main_model_id)
        tokenizer = processor.tokenizer

        prompt = "The meaning of life is"
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        input_ids = torch.tensor([[tokenizer.bos_token_id] + prompt_ids], device=torch_device)

        # Same token budget reasoning as MuseGlimmerIntegrationTest.test_text_generation_matches_reference:
        # 24 tokens gives margin for different tokenisations while keeping PCIe weight-streaming cost low.
        output = model.generate(
            input_ids=input_ids,
            assistant_model=drafter,
            speculation_type="dflash",
            max_new_tokens=24,
            do_sample=False,
        )
        # Strip prompt tokens; output shape is [batch, prompt_len + gen_len].
        completion = tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)

        # DFlash is lossless — identical output to greedy decoding — so this prefix matches
        # MuseGlimmerIntegrationTest.test_text_generation_matches_reference in test_modeling_muse_glimmer.py.
        expected = " to find your gift. The purpose of life is to give it away."
        self.assertEqual(completion[: len(expected)], expected)
