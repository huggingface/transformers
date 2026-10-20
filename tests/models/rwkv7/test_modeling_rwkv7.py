# Copyright 2026 The RWKV team and The HuggingFace Inc. team. All rights reserved.
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
"""Tests for the RWKV-7 model.

The interesting failure modes of a recurrent model are not shape errors, they are
state errors: a carried state that is stale, mixed between batch rows, or simply
not equivalent to having read the whole prefix. Those are what these tests pin.
"""

import json
import re
import shutil
import tempfile
import unittest

from transformers import Rwkv7Config, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import DynamicCache, Rwkv7ForCausalLM, Rwkv7Model
    from transformers.cache_utils import LinearAttentionLayer


def _tiny_config(**kwargs):
    defaults = {
        "vocab_size": 256,
        "hidden_size": 32,
        "num_hidden_layers": 3,
        "head_dim": 8,
        "num_heads": 4,
        "decay_low_rank_dim": 8,
        "a_low_rank_dim": 8,
        "v_low_rank_dim": 8,
        "gate_low_rank_dim": 8,
        "intermediate_size": 64,
    }
    defaults.update(kwargs)
    return Rwkv7Config(**defaults)


def _randomised(model):
    """Break the zero-init so a dropped term cannot pass by cancelling out."""
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    return model.eval()


def _sharpened(model):
    """Same, but at a scale where a broken recurrent state actually shows.

    At 0.05 the whole model works in small numbers and deleting the WKV write-back
    moves the logits by ~1e-5, under a 1e-4 tolerance; at 0.5 it moves them by ~1e-1.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.5)
    return model.eval()


# model parameter name -> reference (official checkpoint) name
_TO_REFERENCE = [
    (r"^model\.embed_tokens\.", "rwkv7.emb."),
    (r"^model\.layers\.", "rwkv7.blocks."),
    (r"^model\.norm\.", "rwkv7.ln_out."),
    (r"^lm_head\.", "head."),
]


def _reference_name(key, prefix="rwkv7."):
    for pattern, replacement in _TO_REFERENCE:
        if re.match(pattern, key):
            return re.sub(pattern, replacement.replace("rwkv7.", prefix), key)
    return key


class Rwkv7ModelTester:
    def __init__(self, parent, batch_size=3, seq_length=7, is_training=True):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = 32
        self.num_hidden_layers = 2  # the common suite caps this; 2 still exercises the layer-0 v_first hand-off
        self.vocab_size = 99
        self.pad_token_id = 0
        self.bos_token_id = 0
        self.eos_token_id = 0

    def get_config(self):
        return _tiny_config(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size, num_hidden_layers=self.num_hidden_layers
        )

    def prepare_config_and_inputs(self):
        # ids start at 1: 0 is this tokenizer's pad/eos, and a prompt that opens with
        # it makes several generation tests ambiguous about where the output begins.
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        return self.get_config(), input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Rwkv7ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (Rwkv7Model, Rwkv7ForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"feature-extraction": Rwkv7Model, "text-generation": Rwkv7ForCausalLM} if is_torch_available() else {}
    )
    test_pruning = False
    test_head_masking = False
    # The mixing is a recurrence, not a score matrix: there is no attention tensor to report.
    has_attentions = False
    # Layer 0 produces `v_first` instead of mixing towards it, so its value-residual LoRA
    # (`att.v0/v1/v2`) is never read and receives no gradient. The parameters are kept
    # because the official checkpoints carry them at layer 0.
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = Rwkv7ModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Rwkv7Config, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def test_config(self):
        self.config_tester.run_common_tests()

    def _get_conv_state_shape(self, batch_size: int, config):
        # the token shift is a width-1 conv state over the hidden channels
        return (batch_size, config.hidden_size, 1)

    def _get_recurrent_state_shape(self, batch_size: int, config):
        # one WKV matrix per head
        return (batch_size, config.num_heads, config.head_dim, config.head_dim)

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=True):
        # The renaming adds the `model.` prefix of the head model (`rwkv7.blocks.N` -> `model.layers.N`), so it is not
        # visible on the bare `Rwkv7Model`, which still loads from a head checkpoint through `base_model_prefix`.
        super().test_reverse_loading_mapping(check_keys_were_modified, skip_base_model=True)

    # ----------------------------------------------------------------------------------
    # configuration / cache structure
    # ----------------------------------------------------------------------------------

    def test_config_declares_a_recurrent_cache(self):
        """`layer_types` + `number_of_conv_states` are what `DynamicCache(config)` reads."""
        config = _tiny_config()
        self.assertEqual(config.layer_types, ["linear_attention"] * config.num_hidden_layers)
        cache = DynamicCache(config=config)
        self.assertEqual(len(cache.layers), config.num_hidden_layers)
        for layer in cache.layers:
            self.assertIsInstance(layer, LinearAttentionLayer)
            self.assertEqual(layer.number_of_states, 2)

    def test_forward_shapes_and_cache_contents(self):
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 7), device=torch_device)

        out = model(input_ids=input_ids, use_cache=True)
        self.assertEqual(out.last_hidden_state.shape, (2, 7, config.hidden_size))

        cache = out.past_key_values
        self.assertIsInstance(cache, DynamicCache)
        # No sequence axis anywhere: slot 0 / 1 are the time-mix / channel-mix token shifts,
        # recurrent slot 0 is the WKV matrix, carried in fp32.
        for layer_idx, layer in enumerate(cache.layers):
            self.assertTrue(cache.has_previous_state(layer_idx))
            self.assertEqual(layer.conv_states[0].shape, (2, config.hidden_size, 1))
            self.assertEqual(layer.conv_states[1].shape, (2, config.hidden_size, 1))
            self.assertEqual(layer.recurrent_states[0].shape, (2, config.num_heads, config.head_dim, config.head_dim))
            self.assertEqual(layer.recurrent_states[0].dtype, torch.float32)
            self.assertIsNone(layer.recurrent_states[1])

    def test_token_shift_slots_hold_the_last_normed_inputs(self):
        """The conv slots must hold the last token's mixer inputs, not something else of the right shape."""
        config = _tiny_config(num_hidden_layers=1)
        model = _randomised(Rwkv7Model(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)
        layer = model.layers[0]
        with torch.no_grad():
            cache = model(input_ids=input_ids, use_cache=True).past_key_values
            hidden = layer.ln0(model.embed_tokens(input_ids))
            att_input = layer.ln1(hidden)
        torch.testing.assert_close(cache.layers[0].conv_states[0][..., 0], att_input[:, -1], rtol=0, atol=0)

    def test_group_norm_epsilon_follows_head_dim_not_head_count(self):
        """The reference's `64e-5` is `head_dim * 1e-5`; at head_dim 64 it coincides with
        `num_heads * 1e-5` only for hidden_size 4096, so the config here is not square."""
        config = _tiny_config()
        self.assertNotEqual(config.num_heads, config.head_dim)  # else this proves nothing
        model = Rwkv7Model(config)
        for layer in model.layers:
            self.assertAlmostEqual(layer.att.ln_x.eps, config.norm_eps * config.head_dim, places=12)

    def test_state_size_does_not_grow_with_context(self):
        """The O(1) claim, measured in bytes."""
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)

        def cache_bytes(length):
            ids = torch.randint(0, config.vocab_size, (1, length), device=torch_device)
            with torch.no_grad():
                cache = model(input_ids=ids, use_cache=True).past_key_values
            return sum(
                state.numel() * state.element_size()
                for layer in cache.layers
                for states in (layer.conv_states, layer.recurrent_states)
                for state in states.values()
                if state is not None
            )

        self.assertEqual(cache_bytes(4), cache_bytes(256))

    # ----------------------------------------------------------------------------------
    # state carry
    # ----------------------------------------------------------------------------------

    def test_prefill_matches_incremental_decoding(self):
        """Reading a prefix in one go must equal reading it one token at a time."""
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 9), device=torch_device)

        with torch.no_grad():
            full = model(input_ids=input_ids).logits
            cache, steps = None, []
            for position in range(input_ids.shape[1]):
                out = model(input_ids=input_ids[:, position : position + 1], past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                steps.append(out.logits)
            incremental = torch.cat(steps, dim=1)

        torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-4)

    def test_prefill_in_pieces_matches_one_prefill(self):
        """A multi-token forward that starts from a carried state (the third token-shift case)."""
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 11), device=torch_device)

        with torch.no_grad():
            full = model(input_ids=input_ids).logits
            first = model(input_ids=input_ids[:, :4], use_cache=True)
            second = model(input_ids=input_ids[:, 4:9], past_key_values=first.past_key_values, use_cache=True)
            third = model(input_ids=input_ids[:, 9:], past_key_values=second.past_key_values, use_cache=True)
            pieces = torch.cat([first.logits, second.logits, third.logits], dim=1)

        torch.testing.assert_close(full, pieces, rtol=1e-4, atol=1e-4)

    def test_generate_matches_a_manual_cached_greedy_loop(self):
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        prompt = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)

        with torch.no_grad():
            generated = model.generate(prompt, max_new_tokens=8, do_sample=False, pad_token_id=0)[0, 6:].tolist()
            out = model(input_ids=prompt, use_cache=True)
            manual = [int(out.logits[0, -1].argmax())]
            for _ in range(7):
                out = model(
                    input_ids=torch.tensor([[manual[-1]]], device=torch_device),
                    past_key_values=out.past_key_values,
                    use_cache=True,
                )
                manual.append(int(out.logits[0, -1].argmax()))

        self.assertEqual(generated, manual)

    def test_logits_to_keep_shortens_the_head(self):
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 12), device=torch_device)

        self.assertTrue(model._supports_logits_to_keep())
        with torch.no_grad():
            everything = model(input_ids=input_ids).logits
            for keep in (1, 4):
                shortened = model(input_ids=input_ids, logits_to_keep=keep).logits
                self.assertEqual(shortened.shape, (2, keep, config.vocab_size))
                torch.testing.assert_close(shortened, everything[:, -keep:, :])

    def test_layer_zero_value_residual_lora_is_unused(self):
        """Layer 0 produces `v_first`; it must never mix towards it."""
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)

        with torch.no_grad():
            before = model(input_ids=input_ids).logits
            for name in ("v0", "v1", "v2"):
                getattr(model.model.layers[0].att, name).normal_(10.0, 1.0)
            after = model(input_ids=input_ids).logits

        torch.testing.assert_close(before, after, rtol=0, atol=0)

    # ----------------------------------------------------------------------------------
    # batching / padding
    # ----------------------------------------------------------------------------------

    def test_batch_rows_are_independent(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        single = torch.randint(0, config.vocab_size, (1, 6), device=torch_device)
        other = torch.randint(0, config.vocab_size, (1, 6), device=torch_device)

        with torch.no_grad():
            alone = model(input_ids=single).logits
            batched = model(input_ids=torch.cat([single, other, single], dim=0)).logits

        torch.testing.assert_close(batched[0], batched[2], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(batched[0], alone[0], rtol=1e-4, atol=1e-4)

    def test_left_padded_row_matches_the_same_row_alone(self):
        """Left padding must leave the recurrent state where it found it.

        The second assertion keeps the test honest: it fails if the padding happened to be
        harmless here, which would make the first assertion prove nothing.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        real = torch.randint(1, config.vocab_size, (1, 4), device=torch_device)
        pads = torch.zeros(1, 3, dtype=real.dtype, device=torch_device)
        padded = torch.cat([pads, real], dim=1)
        mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            alone = model(input_ids=real).logits[0, -1]
            masked = model(input_ids=padded, attention_mask=mask).logits[0, -1]
            ignored = model(input_ids=padded).logits[0, -1]

        treated = (masked - alone).abs().max().item()
        untreated = (ignored - alone).abs().max().item()
        torch.testing.assert_close(masked, alone, rtol=1e-5, atol=1e-5)
        self.assertGreater(untreated, 100 * max(treated, 1e-9))

    def test_left_padded_batch_generate_matches_each_row_alone(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        short = torch.randint(1, config.vocab_size, (1, 4), device=torch_device)
        long = torch.randint(1, config.vocab_size, (1, 7), device=torch_device)

        pads = torch.zeros(1, 3, dtype=short.dtype, device=torch_device)
        batch = torch.cat([torch.cat([pads, short], dim=1), long], dim=0)
        mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            together = model.generate(batch, attention_mask=mask, max_new_tokens=6, do_sample=False, pad_token_id=0)
            row_short = model.generate(short, max_new_tokens=6, do_sample=False, pad_token_id=0)
            row_long = model.generate(long, max_new_tokens=6, do_sample=False, pad_token_id=0)

        self.assertEqual(together[0, 7:].tolist(), row_short[0, 4:].tolist())
        self.assertEqual(together[1, 7:].tolist(), row_long[0, 7:].tolist())

    def test_mask_without_padding_is_a_no_op(self):
        """`generate` passes a mask on every call, so the unpadded case must be bit-identical."""
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)

        with torch.no_grad():
            without = model(input_ids=ids).logits
            with_ones = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

        self.assertTrue(torch.equal(without, with_ones))

    # ----------------------------------------------------------------------------------
    # training
    # ----------------------------------------------------------------------------------

    def test_gradients_reach_every_parameter(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)
        model(input_ids=input_ids, labels=input_ids, use_cache=False).loss.backward()

        dead = {f"model.layers.0.att.{n}" for n in ("v0", "v1", "v2")}  # see the test above
        missing = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and name not in dead and (param.grad is None or param.grad.abs().sum() == 0)
        ]
        self.assertEqual(missing, [], f"parameters received no gradient: {missing}")

    def test_checkpointed_backward_matches_the_plain_one(self):
        """Gradient checkpointing must not change a gradient, reentrant or not."""
        config = _tiny_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=torch_device)

        def gradients(checkpointing, reentrant=True):
            torch.manual_seed(0)  # same weights for every build
            model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).train()
            if checkpointing:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
            model(input_ids=input_ids, labels=input_ids).loss.backward()
            return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

        plain = gradients(checkpointing=False)
        for reentrant in (True, False):
            checkpointed = gradients(checkpointing=True, reentrant=reentrant)
            self.assertEqual(set(plain), set(checkpointed), f"use_reentrant={reentrant}")
            for name in plain:
                torch.testing.assert_close(
                    plain[name], checkpointed[name], rtol=1e-3, atol=1e-5, msg=f"{name} (use_reentrant={reentrant})"
                )

    # ----------------------------------------------------------------------------------
    # checkpoints
    # ----------------------------------------------------------------------------------

    def test_reference_names_load_and_are_written_back(self):
        """Official checkpoints store `rwkv7.emb` / `rwkv7.blocks.N` / `rwkv7.ln_out` / `head`;
        `conversion_mapping.py` renames them on load and `save_pretrained` reverses it."""
        from safetensors.torch import load_file, save_file

        config = _tiny_config()
        original = _sharpened(Rwkv7ForCausalLM(config))
        reference_sd = {_reference_name(k): v.contiguous() for k, v in original.state_dict().items()}
        ids = torch.randint(0, config.vocab_size, (2, 6))
        work = tempfile.mkdtemp()
        try:
            save_file(reference_sd, f"{work}/model.safetensors", metadata={"format": "pt"})
            config.save_pretrained(work)
            loaded, info = Rwkv7ForCausalLM.from_pretrained(work, output_loading_info=True)
            self.assertEqual((info["missing_keys"], info["unexpected_keys"]), (set(), set()))
            for name, param in original.state_dict().items():
                torch.testing.assert_close(loaded.state_dict()[name], param, rtol=0, atol=0, msg=name)

            loaded.save_pretrained(f"{work}/resaved")
            self.assertEqual(set(load_file(f"{work}/resaved/model.safetensors")), set(reference_sd))
            reloaded = Rwkv7ForCausalLM.from_pretrained(f"{work}/resaved")
            with torch.no_grad():
                self.assertTrue(torch.equal(original(ids).logits, reloaded(ids).logits))
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def test_a_reloaded_checkpoint_keeps_every_weight(self):
        """`_init_weights` must not re-initialise loaded raw parameters (it once zeroed 249 of 402)."""
        config = _tiny_config()
        original = _sharpened(Rwkv7ForCausalLM(config))
        with tempfile.TemporaryDirectory() as work:
            original.save_pretrained(work)
            reloaded = Rwkv7ForCausalLM.from_pretrained(work)

        before = dict(original.named_parameters())
        after = dict(reloaded.named_parameters())
        self.assertEqual(set(before), set(after))
        for name in before:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0, msg=name)

    def test_conversion_rejects_a_config_whose_shapes_disagree(self):
        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import convert

        config = _tiny_config()
        source = Rwkv7ForCausalLM(config)
        native = {_reference_name(k, prefix=""): v for k, v in source.state_dict().items()}

        with tempfile.TemporaryDirectory() as work:
            checkpoint = f"{work}/model.pth"
            torch.save(native, checkpoint)

            wrong = config.to_dict()
            wrong["intermediate_size"] = config.intermediate_size * 2  # same names, different widths
            with open(f"{work}/wrong.json", "w", encoding="utf-8") as handle:
                json.dump(wrong, handle)
            with self.assertRaisesRegex(RuntimeError, "do not match the shapes this config implies"):
                convert(checkpoint, "native", f"{work}/wrong.json", f"{work}/out-wrong")

            # Control: the same checkpoint with a config that agrees converts and loads unchanged.
            with open(f"{work}/right.json", "w", encoding="utf-8") as handle:
                json.dump(config.to_dict(), handle)
            convert(checkpoint, "native", f"{work}/right.json", f"{work}/out-right")
            converted = Rwkv7ForCausalLM.from_pretrained(f"{work}/out-right")
            for name, param in source.state_dict().items():
                torch.testing.assert_close(converted.state_dict()[name], param, rtol=0, atol=0, msg=name)


@slow
@require_torch
class Rwkv7IntegrationTests(unittest.TestCase):
    """A real checkpoint, against numbers produced by an implementation that is not this one.

    The expected values come from BlinkDL's own runtime -- the `rwkv` package (0.8.32 at the
    time of generation) at `cpu fp32` with `RWKV_V7_ON=1` -- not from this implementation:

        RWKV_V7_ON=1 python -c "
        from rwkv.model import RWKV
        m = RWKV(model='<CHECKPOINT_FILE minus .pth>', strategy='cpu fp32')
        logits, state = m.forward(PROMPT_IDS, None)
        # greedy-extend 20 tokens for EXPECTED_IDS; top-5 of the first logits
        # gives EXPECTED_TOP5 / EXPECTED_TOP5_LOGITS"

    0.1B is 768 wide at head_dim 64, i.e. twelve heads of width 64, so head count and head
    width differ (at 4096 wide they are both 64 and an indexing mix-up agrees by coincidence).
    """

    CHECKPOINT_REPO = "BlinkDL/rwkv-7-world"
    CHECKPOINT_FILE = "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"

    # "The Eiffel Tower is located in the city of", RWKV world tokenizer.
    PROMPT_IDS = [6699, 304, 25740, 109, 37480, 4600, 52151, 4596, 22590, 30449, 4706]
    # " Paris, France. It is the tallest building in the world and is the world's tallest"
    EXPECTED_IDS = [
        37138, 45, 44312, 47, 3918, 4600, 22590, 32190, 7513, 55666,
        4596, 22590, 40213, 21265, 4600, 22590, 40213, 460, 32190, 7513,
    ]  # fmt: skip
    EXPECTED_TOP5 = [37138, 29319, 20312, 3632, 29417]
    EXPECTED_TOP5_LOGITS = [5.1235, 1.0825, 0.9404, 0.8200, 0.2382]

    @classmethod
    def setUpClass(cls):
        from huggingface_hub import hf_hub_download

        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import convert

        cls._work = tempfile.mkdtemp()
        native = hf_hub_download(cls.CHECKPOINT_REPO, cls.CHECKPOINT_FILE)
        # Through the converter, so the test covers that path too.
        convert(native, "native", None, f"{cls._work}/hf", dtype="float32")
        cls.model = Rwkv7ForCausalLM.from_pretrained(f"{cls._work}/hf", dtype=torch.float32).eval()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._work, ignore_errors=True)

    def test_config_is_inferred_at_a_non_square_width(self):
        config = self.model.config
        self.assertEqual((config.hidden_size, config.num_heads, config.head_dim), (768, 12, 64))

    def test_next_token_logits_match_the_reference_runtime(self):
        """The distribution, not just its argmax: a wrong scale agrees on the winner."""
        with torch.no_grad():
            logits = self.model(input_ids=torch.tensor([self.PROMPT_IDS])).logits[0, -1].float()

        values, indices = torch.topk(logits, 5)
        self.assertEqual(indices.tolist(), self.EXPECTED_TOP5)
        torch.testing.assert_close(values, torch.tensor(self.EXPECTED_TOP5_LOGITS), rtol=1e-4, atol=1e-3)

    def test_greedy_continuation_matches_the_reference_runtime(self):
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=torch.tensor([self.PROMPT_IDS]), max_new_tokens=20, do_sample=False
            )
        self.assertEqual(generated[0, len(self.PROMPT_IDS) :].tolist(), self.EXPECTED_IDS)

    def test_the_cache_carries_what_a_full_forward_computes(self):
        """Twenty one-token decode steps against twenty full re-reads of the prefix."""
        cached, recomputed = [], []
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([self.PROMPT_IDS]), use_cache=True)
            cached.append(int(out.logits[0, -1].argmax()))
            for _ in range(19):
                out = self.model(
                    input_ids=torch.tensor([[cached[-1]]]), past_key_values=out.past_key_values, use_cache=True
                )
                cached.append(int(out.logits[0, -1].argmax()))

            walk = list(self.PROMPT_IDS)
            for _ in range(20):
                logits = self.model(input_ids=torch.tensor([walk]), use_cache=False).logits[0, -1]
                recomputed.append(int(logits.argmax()))
                walk.append(recomputed[-1])

        self.assertEqual(cached, recomputed)
        self.assertEqual(cached, self.EXPECTED_IDS)
