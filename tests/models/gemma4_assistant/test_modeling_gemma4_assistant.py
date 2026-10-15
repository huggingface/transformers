# Copyright 2026 the HuggingFace Team. All rights reserved.
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

import unittest

from parameterized import parameterized

import torch

from transformers import is_torch_available
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...test_processing_common import url_to_local_path


if is_torch_available():
    from transformers import (
        AutoModelForCausalLM,
        Gemma4ForConditionalGeneration,
        Gemma4Processor,
        Gemma4TextConfig,
        Gemma4AssistantForCausalLM,
        Gemma4AssistantConfig,
    )


@slow
@require_torch
@unittest.skip(reason="Update after release")  # TODO @vasqu
class Gemma4IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.model_name = "google/gemma-4-E2B-it"
        self.assistant_name = "google/gemma-4-E2B-it-assistant"
        self.processor = Gemma4Processor.from_pretrained(self.model_name)

        self.url1 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
        )
        self.url2 = url_to_local_path(
            "https://huggingface.co/datasets/hf-internal-testing/fixtures_image_utils/resolve/main/australia.jpg"
        )
        self.messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.url1},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_model_with_image(self):
        model = Gemma4ForConditionalGeneration.from_pretrained(self.model_name, device_map=torch_device)
        assistant = AutoModelForCausalLM.from_pretrained(self.assistant_name, device_map=torch_device)

        inputs = self.processor.apply_chat_template(
            self.messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, assistant_model=assistant, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", 8): ['This image shows a **brown and white cow** standing on a **sandy beach** with the **ocean and a blue sky** in the background'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_model_text_only(self):
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=torch_device)
        assistant = AutoModelForCausalLM.from_pretrained(self.assistant_name, device_map=torch_device)

        inputs = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Write a poem about Machine Learning."}],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(torch_device)

        output = model.generate(**inputs, assistant_model=assistant, max_new_tokens=30, do_sample=False)
        input_size = inputs.input_ids.shape[-1]
        output_text = self.processor.batch_decode(output[:, input_size:], skip_special_tokens=True)

        EXPECTED_TEXTS = Expectations(
            {
                ("cuda", (8, 0)): ['## The Algorithmic Mind\n\nA whisper starts, a seed unseen,\nOf data vast, a vibrant sheen.\nA sea of numbers,'],
                ("cuda", (8, 6)): ['## The Algorithmic Mind\n\nA tapestry of data, vast and deep,\nWhere silent numbers in their slumber sleep.\nA sea of text'],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)


def _build(attn_impl: str) -> Gemma4AssistantForCausalLM:
  text = Gemma4TextConfig(
      vocab_size=128,
      hidden_size=32,
      intermediate_size=64,
      num_hidden_layers=2,
      num_attention_heads=4,
      num_key_value_heads=2,
      head_dim=8,
      global_head_dim=8,
      # Larger than _SEQ_LEN so the SWA mask is also skipped (as with the real
      # assistant configs: sliding_window=1024 > hf_coverage seq_len cap of 512).
      sliding_window=64,
      layer_types=["sliding_attention", "full_attention"],
      hidden_size_per_layer_input=0,
      vocab_size_per_layer_input=0,
  )
  cfg = Gemma4AssistantConfig(
      text_config=text, backbone_hidden_size=32
  )
  cfg._attn_implementation = attn_impl  # pylint: disable=protected-access
  cfg.text_config._attn_implementation = attn_impl  # pylint: disable=protected-access
  torch.manual_seed(0)
  return Gemma4AssistantForCausalLM(cfg).eval()


def _inputs(model, q_len: int, mask_kind: str):
  tc = model.config.get_text_config()
  gen = torch.Generator().manual_seed(1)
  inputs_embeds = torch.randn(1, q_len, 2 * 32, generator=gen)
  shared_kv_states = {}
  for layer_type in ("sliding_attention", "full_attention"):
    attn = model.model.layers[tc.layer_types.index(layer_type)].self_attn
    shape = (1, tc.num_key_value_heads, q_len, attn.head_dim)
    shared_kv_states[layer_type] = (
        torch.randn(shape, generator=gen),
        torch.randn(shape, generator=gen),
    )
  attention_mask = (
      torch.ones(1, q_len, dtype=torch.long) if mask_kind == "ones" else None
  )
  return dict(
      inputs_embeds=inputs_embeds,
      attention_mask=attention_mask,
      shared_kv_states=shared_kv_states,
  )


def _pre_fix_forward(model, inputs_embeds, attention_mask, shared_kv_states):
  """Gemma4AssistantForCausalLM.forward *without* the is_causal=False override."""
  embeds = model.pre_projection(inputs_embeds)
  masks = model.create_attention_masks(embeds, attention_mask, shared_kv_states)
  out = model.model(
      input_ids=None,
      inputs_embeds=embeds,
      attention_mask=masks,
      shared_kv_states=shared_kv_states,
      use_cache=False,
  )
  return model.lm_head(out.last_hidden_state)


def _compiled(fn):
  # aot_eager: traces with dynamo (so HF's `is_tracing()` is True) without
  # needing a C++ toolchain for inductor.
  torch._dynamo.reset()  # pylint: disable=protected-access
  return torch.compile(fn, backend="aot_eager", dynamic=False)


class Gemma4AssistantBidirectionalTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.sdpa = _build("sdpa")
    self.reference = _build("eager")
    self.reference.load_state_dict(self.sdpa.state_dict())

  @torch.no_grad()
  def _reference_logits(self, inputs):
    return self.reference(**inputs).logits

  @parameterized.product(mask_kind=["ones", "none"], q_len=[1, 6])
  @torch.no_grad()
  def test_sdpa_eager_matches_bidirectional_reference(self, mask_kind, q_len):
    inputs = _inputs(self.sdpa, q_len, mask_kind)
    torch.testing.assert_close(
        self.sdpa(**inputs).logits, self._reference_logits(inputs)
    )

  @parameterized.parameters("ones", "none")
  @torch.no_grad()
  def test_sdpa_compiled_matches_eager(self, mask_kind):
    """The hf_coverage_test comparison: eager Baseline vs compiled."""
    inputs = _inputs(self.sdpa, 6, mask_kind)
    eager = self.sdpa(**inputs).logits
    compiled = _compiled(self.sdpa)(**inputs).logits
    torch.testing.assert_close(compiled, eager)
    torch.testing.assert_close(compiled, self._reference_logits(inputs))

  @torch.no_grad()
  def test_pre_fix_forward_reproduces_bug(self):
    """Documents the bug: without the override, eager SDPA is causal."""
    inputs = _inputs(self.sdpa, 6, "ones")
    reference = self._reference_logits(inputs)

    # The mask is skipped (None) in eager mode -> the bug's precondition.
    embeds = self.sdpa.pre_projection(inputs["inputs_embeds"])
    masks = self.sdpa.create_attention_masks(
        embeds, inputs["attention_mask"], inputs["shared_kv_states"]
    )
    self.assertIsNone(masks["full_attention"])
    self.assertIsNone(masks["sliding_attention"])

    eager = _pre_fix_forward(self.sdpa, **inputs)
    compiled = _compiled(lambda **kw: _pre_fix_forward(self.sdpa, **kw))(
        **inputs
    )

    # Compiled materializes the bidirectional mask -> correct.
    torch.testing.assert_close(compiled, reference)
    # Eager falls back to is_causal=True -> differs except at the last position
    # (which sees every key under both causal and bidirectional attention).
    self.assertGreater((eager - reference).abs().max().item(), 1e-3)
    torch.testing.assert_close(eager[:, -1], reference[:, -1])
