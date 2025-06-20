# coding=utf-8
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
"""Testing suite for the PyTorch FalconH1 model."""

import inspect
import unittest

import pytest

from transformers import FalconH1Config, is_torch_available
from transformers.testing_utils import (
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, FalconH1ForCausalLM, FalconH1Model


class FalconH1ModelTester(CausalLMModelTester):
    config_class = FalconH1Config
    if is_torch_available():
        base_model_class = FalconH1Model
        causal_lm_class = FalconH1ForCausalLM


@require_torch
class FalconH1ModelTest(CausalLMModelTest, unittest.TestCase):
    all_model_classes = (FalconH1Model, FalconH1ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": FalconH1Model, "text-generation": FalconH1ForCausalLM} if is_torch_available() else {}
    )
    model_tester_class = FalconH1ModelTester

    def test_mismatched_shapes_have_properly_initialized_weights(self):
        r"""
        Overriding the test_mismatched_shapes_have_properly_initialized_weights test because A_log and D params of the
        FalconH1 mixer are initialized differently and we tested that in test_initialization
        """
        self.skipTest(reason="Cumbersome and redundant for FalconH1")

    @pytest.mark.generate
    def test_left_padding_compatibility(self):
        # NOTE: left-padding results in small numerical differences. This is expected.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535

        # First, filter out models that don't support left padding
        # - The model must have generative capabilities
        if len(self.all_generative_model_classes) == 0:
            self.skipTest(reason="No generative architecture available for this model.")

        # - The model must support padding
        if not self.has_attentions:
            self.skipTest(reason="This model doesn't support padding.")

        # - The model must be a decoder-only architecture (encoder-based architectures use right-padding)
        decoder_only_classes = []
        for model_class in self.all_generative_model_classes:
            config, _ = self.prepare_config_and_inputs_for_generate()
            if config.is_encoder_decoder:
                continue
            else:
                decoder_only_classes.append(model_class)
        if len(decoder_only_classes) == 0:
            self.skipTest(reason="No decoder-only architecture available for this model.")

        # - Decoder-only architectures derived from encoder-decoder models could support it in theory, but we haven't
        #   added support for it yet. We skip these models for now.
        has_encoder_attributes = any(
            attr_name
            for attr_name in config.to_dict().keys()
            if attr_name.startswith("encoder") and attr_name != "encoder_no_repeat_ngram_size"
        )
        if has_encoder_attributes:
            self.skipTest(
                reason="The decoder-only derived from encoder-decoder models are not expected to support left-padding."
            )

        # Then, test left-padding
        def _prepare_model_kwargs(input_ids, attention_mask, signature):
            model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "position_ids" in signature:
                position_ids = torch.cumsum(attention_mask, dim=-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                model_kwargs["position_ids"] = position_ids
            if "cache_position" in signature:
                cache_position = torch.arange(input_ids.shape[-1], device=torch_device)
                model_kwargs["cache_position"] = cache_position
            return model_kwargs

        for model_class in decoder_only_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            input_ids = inputs_dict["input_ids"]

            # - for left padding we absolutely need to use an all ones
            #   attention mask, so we do not use the one in inputs_dict
            attention_mask = torch.ones_like(input_ids)

            model = model_class(config).to(torch_device).eval()
            signature = inspect.signature(model.forward).parameters.keys()

            # no cache as some models require special cache classes to be init outside forward
            model.generation_config.use_cache = False

            # Without padding
            model_kwargs = _prepare_model_kwargs(input_ids, attention_mask, signature)
            next_logits_wo_padding = model(**model_kwargs).logits[:, -1, :]

            # With left-padding (length 32)
            # can hardcode pad_token to be 0 as we'll do attn masking anyway
            pad_token_id = (
                config.get_text_config().pad_token_id if config.get_text_config().pad_token_id is not None else 0
            )
            pad_size = (input_ids.shape[0], 32)
            padding = torch.ones(pad_size, dtype=input_ids.dtype, device=torch_device) * pad_token_id
            padded_input_ids = torch.cat((padding, input_ids), dim=1)
            padded_attention_mask = torch.cat((torch.zeros_like(padding), attention_mask), dim=1)
            model_kwargs = _prepare_model_kwargs(padded_input_ids, padded_attention_mask, signature)
            next_logits_with_padding = model(**model_kwargs).logits[:, -1, :]

            # They should result in very similar logits
            torch.testing.assert_close(next_logits_wo_padding, next_logits_with_padding, rtol=1e-5, atol=1e-5)


@slow
@require_torch
@require_torch_gpu
class FalconH1ModelIntegrationTest(unittest.TestCase):
    @slow
    def test_falcon_h1_hard(self):
        """
        An integration test for Falcon-H1.
        """
        EXPECTED_TEXT = """
            user
            Tell me about the french revolution.
            assistant
            The French Revolution (1789–1799) was a period of radical social and political upheaval in France that fundamentally transformed the nation and had profound effects on the rest of Europe and the world. Here are the key aspects of the revolution:

            ### **Causes**
            1. **Economic Crisis**: France was in severe financial trouble due to costly wars (particularly the American Revolution), extravagant spending by the monarchy, and inefficient taxation.
            2. **Social Inequality**: The rigid class system (the Ancien Régime) divided society into the privileged nobility and clergy (First Estate) and the commoners (Third Estate), who bore the brunt of taxation and had few rights.
            3. **Enlightenment Ideas**: Philosophers like Voltaire, Rousseau, and Montesquieu inspired ideas of liberty, equality, and popular sovereignty.
            4. **Settlement of 1789**: The Estates-General convened to address the financial crisis, leading to the Third Estate's assertion of its rights and the eventual abolition of the feudal system.

            ### **Key Events**
            1. **Storming of the Bastille (July 14, 1789)**: A symbol of royal tyranny, the Bastille fortress was stormed by revolutionaries, sparking widespread rebellion.
            2. **Declaration of the Rights of Man and of the Citizen (August 1789)**: A foundational document proclaiming liberty, equality, and fraternity.
            3. **National Assembly and King’s Trial (1791–1792)**: King Louis XVI and his ministers were tried and executed (King Louis was guillotined, Marie Antoinette was banished), marking the end of the monarchy.
            4. **Rise of the Jacobins and Reign of Terror (1793–1794)**: Radical leaders like Maximilien Robespierre sought to purge France of counter-revolutionaries, leading to mass executions and widespread fear.
            5. **Thermidorian Reaction
        """
        # Remove the first char (`\n`) and the consecutive whitespaces caused by the formatting.
        EXPECTED_TEXT = EXPECTED_TEXT.strip().replace(" " * 12, "")

        model_id = "tiiuae/Falcon-H1-1.5B-Deep-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = FalconH1ForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        device = "cuda"
        messages = [{"role": "user", "content": "Tell me about the french revolution."}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=512, do_sample=False)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        self.assertEqual(generated_text, EXPECTED_TEXT)
