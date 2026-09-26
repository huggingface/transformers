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

import shutil
import tempfile
import unittest

from transformers import Rwkv7Config, is_torch_available
from transformers.testing_utils import require_peft, require_torch, slow, torch_device

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor, random_attention_mask
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import Rwkv7Cache, Rwkv7ForCausalLM, Rwkv7Model


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

    0.05 keeps the whole model in small numbers, which is what
    `test_left_padded_row_matches_the_same_row_alone` needs and reasons about. It is
    the wrong scale for anything that checks the state *carry*: at 0.05, deleting
    the WKV write-back entirely moves the logits by 6.8e-06, which sails under a
    1e-4 tolerance -- the test would be green with the recurrence dead. At 0.5 the
    same sabotage moves them by 6.8e-01, five orders of magnitude clear of the noise
    floor of 3.3e-06. Measured, not guessed; see the sabotage matrix in the commit.
    """
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.5)
    return model.eval()


class Rwkv7ModelTester:
    """The inputs the shared model/generation tests build everything else from.

    Deliberately lean. The RWKV v4 tester in this repo still carries `token_type_ids`,
    `mc_token_ids` and `num_choices` from the GPT-2 tester it was copied from, none of
    which this architecture has ever taken.
    """

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
    # There is no attention here to report: the mixing is a recurrence, not a score
    # matrix, so there is no [batch, heads, q, k] tensor for the shared suite to
    # inspect. This is the flag those tests read, rather than a row of skips.
    has_attentions = False
    # Layer 0 *produces* `v_first` rather than mixing towards it, so its
    # value-residual LoRA (`att.v0/v1/v2`) is never read and never receives a
    # gradient. The parameters are kept anyway: a native `.pth` carries them at
    # layer 0, and dropping them would mean this port can read that file but not
    # write it back. `test_layer_zero_value_residual_lora_is_unused` pins the
    # property; this is the flag models with structurally-unreachable parameters
    # set so the gradient-checkpointing trainings still run rather than being
    # skipped wholesale. `test_checkpointed_backward_matches_the_plain_one` then
    # compares the checkpointed gradients against the plain ones by value.
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = Rwkv7ModelTester(self)
        self.config_tester = ConfigTester(
            self, config_class=Rwkv7Config, common_properties=["hidden_size", "num_hidden_layers"]
        )

    def test_config_rejects_inconsistent_heads(self):
        with self.assertRaises(ValueError):
            _tiny_config(num_heads=3)
        with self.assertRaises(ValueError):
            _tiny_config(hidden_size=30, head_dim=8, num_heads=4)

    def test_group_norm_epsilon_follows_head_dim_not_head_count(self):
        """The reference's `64e-5` is `head_dim * 1e-5`, and the two look alike.

        `num_heads * norm_eps` gives the same number as `head_dim * norm_eps` when
        a model has exactly as many heads as channels per head -- at head_dim 64,
        that is hidden_size 4096 and nothing else. Every accuracy measurement this
        port has been through ran on the 7.2B, which is precisely that width, so
        the wrong multiplier reproduced the reference to four decimals and the
        conversion check compared two routes that shared the mistake.

        The config here is deliberately not square (4 heads of 8), so the two
        candidates differ and the assertion has something to say.
        """
        config = _tiny_config()
        self.assertNotEqual(config.num_heads, config.head_dim)  # else this proves nothing
        model = Rwkv7Model(config)

        for block in model.blocks:
            self.assertAlmostEqual(block.att.ln_x.eps, config.norm_eps * config.head_dim, places=12)

    def test_forward_shapes_and_state(self):
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 7), device=torch_device)

        out = model(input_ids=input_ids, use_cache=True)
        self.assertEqual(out.last_hidden_state.shape, (2, 7, config.hidden_size))

        cache = out.state
        self.assertIsInstance(cache, Rwkv7Cache)
        self.assertEqual(len(cache), config.num_hidden_layers)
        # No sequence axis anywhere in here, at any layer -- that is the architecture's
        # whole claim, and the shapes are where it is either true or not.
        for layer_idx in range(config.num_hidden_layers):
            att_shift, ffn_shift, wkv = cache.read(layer_idx)
            self.assertEqual(att_shift.shape, (2, config.hidden_size))
            self.assertEqual(ffn_shift.shape, (2, config.hidden_size))
            self.assertEqual(wkv.shape, (2, config.num_heads, config.head_dim, config.head_dim))

    def test_state_size_does_not_grow_with_context(self):
        """The O(1) claim, measured in bytes rather than asserted in a docstring."""
        config = _tiny_config()
        model = _randomised(Rwkv7Model(config)).to(torch_device)

        def cache_bytes(length):
            ids = torch.randint(0, config.vocab_size, (1, length), device=torch_device)
            with torch.no_grad():
                cache = model(input_ids=ids, use_cache=True).state
            return sum(
                state.numel() * state.element_size()
                for layer in cache.layers
                for state in layer.recurrent_states.values()
                if state is not None
            )

        self.assertEqual(cache_bytes(4), cache_bytes(256))
        self.assertEqual(cache_bytes(4), cache_bytes(1024))

    def test_prefill_matches_incremental_decoding(self):
        """Reading a prefix in one go must equal reading it one token at a time.

        This is the property that makes the carried state a real cache rather than
        an approximation; it fails loudly if the token shift, the WKV recurrence or
        the v_first hand-off is wired wrong -- but only at a weight scale where the
        state contributes more than the tolerance, hence `_sharpened`.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 9), device=torch_device)

        with torch.no_grad():
            full = model(input_ids=input_ids, use_cache=True).logits
            state, steps = None, []
            for position in range(input_ids.shape[1]):
                out = model(input_ids=input_ids[:, position : position + 1], state=state, use_cache=True)
                state, _ = out.state, steps.append(out.logits)
            incremental = torch.cat(steps, dim=1)

        torch.testing.assert_close(full, incremental, rtol=1e-4, atol=1e-4)

    def test_logits_to_keep_shortens_the_head(self):
        """A prefill needs one row of logits, not `seq_len` of them.

        The head is the widest matrix in the model, so running it over the whole prompt
        is the largest avoidable cost in a prefill. The argument has to be named in the
        signature rather than reached through `**kwargs`: swallowed there it would change
        nothing and say nothing, and `generate` would decline to pass it at all, since
        `_supports_logits_to_keep()` decides by reading the signature.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (2, 12), device=torch_device)

        self.assertTrue(model._supports_logits_to_keep())
        with torch.no_grad():
            everything = model(input_ids=input_ids).logits
            for keep in (1, 4):
                shortened = model(input_ids=input_ids, logits_to_keep=keep).logits
                self.assertEqual(shortened.shape, (2, keep, config.vocab_size))
                # The kept rows must be the LAST ones, and identical to computing them
                # all: a slice taken from the wrong end would have the right shape.
                torch.testing.assert_close(shortened, everything[:, -keep:, :])

    def test_intermediate_size_follows_hidden_size_when_it_is_not_given(self):
        """A literal default for `intermediate_size` is right for the default
        `hidden_size` and silently wrong for every other one: `Rwkv7Config(
        hidden_size=4096, num_heads=64)` would come back with a channel-mix four times
        narrower than the architecture it names, and the model built from it loads no
        real checkpoint. The default config is unaffected, which is why a second width
        is needed to see this at all.
        """
        self.assertEqual(Rwkv7Config().intermediate_size, 3072)
        self.assertEqual(Rwkv7Config(hidden_size=4096, num_heads=64).intermediate_size, 16384)
        self.assertEqual(Rwkv7Config(hidden_size=2048, num_heads=32).intermediate_size, 8192)
        # An explicit value still wins, and still round-trips through config.json.
        explicit = Rwkv7Config(hidden_size=4096, num_heads=64, intermediate_size=4096)
        self.assertEqual(explicit.intermediate_size, 4096)
        self.assertEqual(Rwkv7Config.from_dict(explicit.to_dict()).intermediate_size, 4096)

    def test_conversion_rejects_a_config_whose_shapes_disagree(self):
        """Matching key names is not the same as matching the model.

        The converter compared key sets only. A hand-written `config.json` that gets a
        width wrong has all the right names, so it converted with zero reported
        mismatches and produced a model that loads and generates noise. The skeleton it
        checks names against carries the shapes too.
        """
        import json
        import tempfile

        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import convert

        config = _tiny_config()
        source = Rwkv7ForCausalLM(config)
        # `_convert_native` is a prefix rename, so stripping it produces exactly the
        # native `.pth` layout that conversion expects.
        native = {key.removeprefix("rwkv7."): value for key, value in source.state_dict().items()}

        with tempfile.TemporaryDirectory() as work:
            checkpoint = f"{work}/model.pth"
            torch.save(native, checkpoint)

            wrong = config.to_dict()
            wrong["intermediate_size"] = config.intermediate_size * 2  # same names, different widths
            wrong_path = f"{work}/wrong.json"
            with open(wrong_path, "w") as handle:
                json.dump(wrong, handle)

            with self.assertRaisesRegex(RuntimeError, "do not match the shapes this config implies"):
                convert(checkpoint, "native", wrong_path, f"{work}/out-wrong")

            # Control: the same checkpoint with a config that agrees must convert.
            right_path = f"{work}/right.json"
            with open(right_path, "w") as handle:
                json.dump(config.to_dict(), handle)
            convert(checkpoint, "native", right_path, f"{work}/out-right")

    def test_a_reloaded_checkpoint_keeps_every_weight(self):
        """Saving and reloading must not lose a parameter to re-initialisation.

        `_init_weights` has to reach this model's twenty-one raw `nn.Parameter`s, which
        the inherited one does not. Written the obvious way -- `parameter.data.zero_()`
        -- it also destroys them on load: the framework skips re-initialising anything
        already there by checking an `_is_hf_initialized` flag that `initialization`'s
        helpers set and a raw in-place write does not, so `_init_weights` runs after the
        checkpoint is in and zeroes what it just loaded.

        That shipped for exactly as long as it took to run a real checkpoint through it.
        A converted 0.1B came back with 249 of its 402 tensors zeroed and generated
        "civil civil civil" where the reference runtime generated "Paris, France". None
        of the tests here noticed, because they all build models rather than load them.
        """
        import tempfile

        config = _tiny_config()
        original = _sharpened(Rwkv7ForCausalLM(config))
        with tempfile.TemporaryDirectory() as work:
            original.save_pretrained(work)
            reloaded = Rwkv7ForCausalLM.from_pretrained(work)

        before = dict(original.named_parameters())
        after = dict(reloaded.named_parameters())
        self.assertEqual(set(before), set(after))
        lost = [name for name, p in after.items() if p.abs().max() == 0 and before[name].abs().max() != 0]
        self.assertEqual(lost, [], f"{len(lost)} parameters came back zeroed, e.g. {lost[:5]}")
        for name in before:
            torch.testing.assert_close(after[name], before[name], rtol=0, atol=0, msg=name)

    @require_peft
    def test_peft_lora_attaches_and_actually_learns(self):
        """A LoRA-wrapped model has to train, and no module may bypass its adapter.

        The bounty's criterion 3 asks for the common transformers-based PEFT flow to
        work. Two properties, each of which has a silent failure mode this test exists
        to catch. The loss on an overfit batch must actually fall, because a wiring
        break under wrapping tends to produce a model that runs and returns a loss
        while learning nothing. And every attached adapter must receive a gradient,
        because a forward that reads `module.weight` directly instead of calling the
        module silently routes around its LoRA -- the channel-mix projection here is
        exactly the kind of place that could regress to that, and did during the
        sparse-path work.
        """
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(0)
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config))
        wrapped = get_peft_model(
            model, LoraConfig(r=8, lora_alpha=16, target_modules=["receptance", "key", "value", "output"])
        )
        trainable = {name for name, p in wrapped.named_parameters() if p.requires_grad}
        self.assertTrue(trainable, "no trainable parameters after wrapping")
        self.assertTrue(all("lora" in name for name in trainable), "base weights not frozen")

        wrapped.train()
        input_ids = torch.randint(1, config.vocab_size, (2, 12))
        optimizer = torch.optim.AdamW((p for p in wrapped.parameters() if p.requires_grad), lr=1e-2)

        losses = []
        for step in range(24):
            loss = wrapped(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            if step == 0:
                # Every adapter, not just some: an unfired adapter means its module
                # was bypassed, and the loss can still fall through the others. The
                # check is on lora_B, and that is not arbitrary: LoRA initialises B to
                # zero, so dL/dA passes through B and is identically zero on the first
                # step -- the first version of this test asserted lora_A here and
                # failed on all 18 adapters of a perfectly healthy model. B's gradient
                # is nonzero from step 0 exactly when the module's output reaches the
                # loss, which is the property under test.
                dead = [
                    name
                    for name, p in wrapped.named_parameters()
                    if p.requires_grad and "lora_B" in name and (p.grad is None or p.grad.abs().max() == 0)
                ]
                self.assertEqual(dead, [], f"{len(dead)} adapters received no gradient")
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

        self.assertLess(
            losses[-1],
            0.6 * losses[0],
            f"overfitting one batch for 24 steps only moved the loss {losses[0]:.3f} -> {losses[-1]:.3f}",
        )

    def test_checkpointed_backward_matches_the_plain_one(self):
        """Gradient checkpointing must not change a gradient.

        The three inherited tests for this are skipped over layer 0's unused v-LoRA,
        and the two tests this file's skip comment pointed at instead turned out to
        run no checkpointed backward at all -- one toggles flags, the other never
        enables checkpointing. With nothing exercising it, the path was broken:
        `GradientCheckpointingLayer` disarms a live cache only when it arrives as a
        keyword it recognises, this model passes its state positionally, so the
        backward replay re-read a state the forward had already advanced. Reentrant
        checkpointing returned wrong gradients for 99 of 102 parameters with no error;
        the non-reentrant default raised.

        Comparing against the unpatched backward rather than checking for absence is
        the point: the reentrant failure produced gradients, they were just wrong.
        """
        config = _tiny_config()
        input_ids = torch.randint(0, config.vocab_size, (2, 8), device=torch_device)

        def gradients(checkpointing, reentrant=True):
            # Seeded per build: `_sharpened` draws fresh weights, so without this the
            # comparison is between two different models and reports a difference
            # whatever checkpointing does.
            torch.manual_seed(0)
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
                    plain[name],
                    checkpointed[name],
                    rtol=1e-3,
                    atol=1e-5,
                    msg=f"{name} differs under checkpointing (use_reentrant={reentrant})",
                )

    def test_a_long_prefill_stays_finite_where_a_series_inverse_would_not(self):
        """The chunked kernel must not manufacture a NaN from a finite prompt.

        `(I + tril(Q~ B~^T, -1))` is unit lower triangular, so its inverse can be
        written as a terminating series and taken by repeated squaring. That is exact
        in exact arithmetic and wrong in float32: the intermediate powers are not
        bounded by the answer. On a real chunk here the matrix has entries at most
        0.977 and its true inverse has entries at most 1.0, while the 32nd power
        reaches 1.3e11 -- eleven orders of magnitude of cancellation, against the seven
        digits float32 carries. The result came back with entries of 1e4 where the
        answer is 1, then went to NaN in the next layer.

        Seeded, and at a length several chunks long, because neither is optional. At
        T=128 and below no initialisation triggers it; at T=256 one in forty does; at
        T=1024 one in five. The seeds below are two of that fifth, so this test fails
        on a series inverse and passes on the substitution -- checked by running it
        against both, which is the only way to know a regression test regresses.

        Asserted as finiteness rather than against a reference, because that is the
        failure: the states go to NaN, and every downstream comparison then passes or
        fails for the wrong reason.
        """
        for seed in (7, 33):
            with self.subTest(seed=seed):
                torch.manual_seed(seed)
                config = _tiny_config()
                model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
                input_ids = torch.randint(0, config.vocab_size, (1, 1024), device=torch_device)

                with torch.no_grad():
                    out = model(input_ids=input_ids, use_cache=True)

                self.assertTrue(torch.isfinite(out.logits).all(), "logits are not finite")
                for layer in range(config.num_hidden_layers):
                    for slot in (Rwkv7Cache.WKV, Rwkv7Cache.ATT_SHIFT, Rwkv7Cache.FFN_SHIFT):
                        state = out.state.layers[layer].recurrent_states[slot]
                        self.assertTrue(torch.isfinite(state).all(), f"layer {layer} slot {slot} state is not finite")

    def test_the_last_real_token_is_found_by_index_not_by_float_arithmetic(self):
        """Finding the last unmasked position must not depend on the model's dtype.

        The search was `(mask * arange).argmax()`, evaluated in the activation dtype
        because the mask is cast to it. bfloat16 carries eight mantissa bits, so past
        position 256 the products stop being distinct, `argmax` returns the first of a
        tie, and the state handed back comes from several tokens short of the end --
        1022 for 1023 at length 1024, 4088 for 4095 at 4096. An all-ones mask is
        enough, which is what `generate` sends, so this was never confined to padded
        batches.

        Asserted against the mask-free path rather than against a hardcoded index: a
        full-length mask asks for exactly what no mask asks for, so the two must agree
        whatever the dtype.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        seq_len = 1024
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=torch_device)
        ones = torch.ones(1, seq_len, dtype=torch.long, device=torch_device)

        for dtype in (torch.float32, torch.bfloat16):
            typed = model.to(dtype)
            with torch.no_grad():
                masked = typed(input_ids=input_ids, attention_mask=ones, use_cache=True).state
                unmasked = typed(input_ids=input_ids, use_cache=True).state

            for layer in range(config.num_hidden_layers):
                for slot in (Rwkv7Cache.WKV, Rwkv7Cache.ATT_SHIFT, Rwkv7Cache.FFN_SHIFT):
                    torch.testing.assert_close(
                        masked.layers[layer].recurrent_states[slot].float(),
                        unmasked.layers[layer].recurrent_states[slot].float(),
                        rtol=0,
                        atol=0,
                        msg=f"{dtype} layer {layer} slot {slot}: an all-ones mask moved the state",
                    )

    def test_compiled_prefill_matches_eager_at_batch_and_length(self):
        """`torch.compile` must survive a shape with both a batch and a length.

        This exists because the one compile failure this port has actually had was
        found by a benchmark, not by a test: at 16x16 the WKV output came back with a
        layout `view` could not reinterpret, and every single-token and every
        `batch=1` shape had missed it. `fullgraph=True` is the point of the test as
        much as the numerics are -- the performance story here rests on CUDA graphs,
        and a graph break loses them while still returning correct logits, so a test
        that only compared outputs would stay green through exactly the regression
        worth catching.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 16), device=torch_device)

        torch._dynamo.reset()
        with torch.no_grad():
            eager = model(input_ids=input_ids).logits
            compiled = torch.compile(model, fullgraph=True)(input_ids=input_ids).logits

        torch.testing.assert_close(eager, compiled, rtol=1e-4, atol=1e-4)

    def test_compiled_decode_step_with_state_matches_eager(self):
        """The decode step, compiled, against eager -- with the pre-allocated cache.

        The performance story rests on compiling the single-token step over a
        state from `allocate_state` (the modeling docstring's contract), yet the
        compile test above only covers the cache-less prefill. This is the path a
        user actually compiles, and the one where a cudagraph-replay bug would
        produce fast wrong outputs: on CUDA, handing the *prefill* to the
        compiled model builds a lazy unpinned cache inside the traced region and
        trips the output-buffer protection, which is exactly why the prefill
        below stays eager.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        prompt = torch.randint(0, config.vocab_size, (1, 8), device=torch_device)

        def decode(step_model):
            state = model.rwkv7.allocate_state(1, device=torch_device)
            with torch.no_grad():
                out = model(input_ids=prompt, state=state, use_cache=True)  # prefill: eager
                logits, state = [out.logits[0, -1]], out.state
                token = out.logits[0, -1].argmax()[None, None]
                for _ in range(3):
                    out = step_model(input_ids=token, state=state, use_cache=True)
                    logits.append(out.logits[0, -1])
                    state = out.state
                    token = out.logits[0, -1].argmax()[None, None]
            return torch.stack(logits)

        torch._dynamo.reset()
        eager = decode(model)
        compiled = decode(torch.compile(model))
        torch.testing.assert_close(eager, compiled, rtol=1e-4, atol=1e-4)

    def test_batch_rows_are_independent(self):
        """A row's output must not depend on what shares its batch.

        Duplicating a sequence inside a batch is the sharpest version of this: the
        copies see identical inputs, so any difference between them is leaked state
        rather than numerics.
        """
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
        """Padding must leave the recurrent state exactly where it found it.

        `generate` left-pads a batch, so the pads run through the recurrence
        *before* the real tokens. There is no per-position attention mask for an
        all-recurrent model to hide behind: unless the padding is neutralised, a
        short row starts from a state the pads have already moved. The second
        assertion keeps this test honest -- it fails if the padding happened to be
        harmless here, which would make the first assertion prove nothing. It is
        written as a ratio because an absolute floor only holds for one fixture:
        every weight here is drawn at 0.05, so the whole model works in small
        numbers and the untreated damage lands around 1e-6, not 1e-2.
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
        """End-to-end greedy `generate` on a left-padded batch, row for row.

        The single-forward version above pins the logits of one padded row; this
        one runs the whole generate loop -- prefill with pads, then step-by-step
        decode where the full-length mask is sliced and passed every step -- and
        requires each row's continuation to equal the row generated on its own.
        The shared `test_left_padding_compatibility` is swallowed by
        `has_attentions=False`, so without this the chain it exercises has no
        token-level coverage at all.
        """
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

    def test_packed_batch_matches_each_sequence_run_alone(self):
        """A varlen (packed) row must decode exactly like its sequences separately.

        Two things reach across a boundary and both have to be cut: the recurrent
        state, which restarts per segment, and the token shift, which otherwise
        hands a segment's first token the *previous* sequence's last hidden state.
        The control at the end is what makes this test mean something -- it fails
        if packing happened to be harmless, which would leave the first assertion
        proving nothing.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        lengths = [4, 3, 5]
        segments = [torch.randint(1, config.vocab_size, (1, n), device=torch_device) for n in lengths]
        packed = torch.cat(segments, dim=1)
        cu_seq_lens_q = torch.tensor([0, 4, 7, 12], device=torch_device)

        with torch.no_grad():
            together = model(input_ids=packed, cu_seq_lens_q=cu_seq_lens_q).logits[0]
            naive = model(input_ids=packed).logits[0]

        start, treated = 0, 0.0
        for segment, n in zip(segments, lengths):
            with torch.no_grad():
                alone = model(input_ids=segment).logits[0]
            torch.testing.assert_close(together[start : start + n], alone, rtol=1e-5, atol=1e-5)
            treated = max(treated, (together[start : start + n] - alone).abs().max().item())
            start += n

        # A ratio, not an absolute floor. Every weight in this fixture is drawn at
        # 0.05, so the untreated damage sits around 1e-6 and a bare 1e-6 threshold
        # would be a coin flip on the very quantity it is meant to bound. What has to
        # hold is that honouring the boundaries removes most of the damage, at
        # whatever scale the fixture happens to put it.
        untreated = (naive - together).abs().max().item()
        self.assertGreater(untreated, 100 * max(treated, 1e-9))

    def test_packed_batch_rejects_a_malformed_boundary_list(self):
        """A wrong `cu_seq_lens_q` must raise, not split the recurrence somewhere else.

        Nothing downstream can notice a bad boundary list: the model still returns
        fluent logits, just computed from states that restarted in the wrong places.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)

        # First three fail on the endpoints; the last two have correct endpoints and
        # fail only on ordering. [0, 4, 2, 6] is the one a check that looks at endpoints
        # alone lets through: the backwards pair goes unnoticed and six tokens in come
        # back as eight.
        for bad in ([1, 3, 6], [0, 3, 5], [0, 3, 8], [0, 4, 2, 6], [0, 3, 3, 6]):
            with self.assertRaises(ValueError):
                model(input_ids=ids, cu_seq_lens_q=torch.tensor(bad, device=torch_device))

        with self.assertRaises(ValueError):  # packing describes one row, not a batch
            model(
                input_ids=ids.repeat(2, 1),
                cu_seq_lens_q=torch.tensor([0, 3, 6], device=torch_device),
            )

    def test_packed_batch_ignores_a_carried_state_and_says_so(self):
        """Packing means "these are new sequences", so a carried state is not history.

        The state argument is still read for its shape and dtype, so a caller may
        hand in a pre-allocated cache; what it must not do is quietly resume from
        it, since there is no segment a previous row's state would belong to. This
        is a contract worth pinning rather than an accident: the same call with a
        state carrying real history has to give the same logits as one starting
        cold, and the docstring promises exactly that.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)
        bounds = torch.tensor([0, 2, 6], device=torch_device)

        with torch.no_grad():
            warmed = model(
                input_ids=torch.randint(1, config.vocab_size, (1, 5), device=torch_device),
                use_cache=True,
            ).state
            self.assertGreater(max(s.abs().max().item() for s in self._every_state(warmed)), 0.0)

            cold = model(input_ids=ids, cu_seq_lens_q=bounds).logits
            carried = model(input_ids=ids, cu_seq_lens_q=bounds, state=warmed).logits

        torch.testing.assert_close(carried, cold, rtol=1e-5, atol=1e-5)

    def test_generate_accepts_a_preallocated_or_warm_state(self):
        """`generate` with a caller-supplied state must consume the whole prompt.

        `allocate_state` exists so the decode can be compiled (the cache must be
        pinned before tracing), and a warm state is how a chat turn resumes. The
        old input-truncation gate keyed on the state's *existence*, so both of
        those calls silently dropped every prompt token but the last one and
        generated from a one-token prompt without raising. The gate now keys on
        `cache_position`, and these two equalities are what that fixes.
        """
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device).eval()
        prompt = torch.randint(0, config.vocab_size, (1, 6), device=torch_device)

        with torch.no_grad():
            plain = model.generate(prompt, max_new_tokens=5, do_sample=False, pad_token_id=0)
            fresh = model.rwkv7.allocate_state(1, device=torch_device)
            preallocated = model.generate(prompt, state=fresh, max_new_tokens=5, do_sample=False, pad_token_id=0)
        self.assertEqual(preallocated[0].tolist(), plain[0].tolist())

        history = torch.randint(0, config.vocab_size, (1, 7), device=torch_device)
        with torch.no_grad():
            warm = model(input_ids=history, use_cache=True).state
            resumed = model.generate(prompt, state=warm, max_new_tokens=5, do_sample=False, pad_token_id=0)
            # The same continuation by hand: history, then the whole prompt, then greedy.
            state = model(input_ids=history, use_cache=True).state
            out = model(input_ids=prompt, state=state, use_cache=True)
            state, manual = out.state, []
            token = out.logits[0, -1].argmax()[None, None]
            for _ in range(5):
                manual.append(int(token))
                out = model(input_ids=token, state=state, use_cache=True)
                state = out.state
                token = out.logits[0, -1].argmax()[None, None]
        self.assertEqual(resumed[0, 6:].tolist(), manual)

    def test_prefix_chunk_with_sliced_mask_matches_the_whole_padded_forward(self):
        """The chunked-prefill mask contract, exercised instead of only documented.

        The forward's docstring tells a caller feeding a prefix chunk to slice the
        mask to the chunk's own span "or it silently reads the wrong positions".
        The shared test for this (#47086-style) skips itself because the config
        declares no recurrent layer_types, so the contract had no coverage: this
        runs a left-padded batch whole, then as prefix-chunk-plus-continuation
        with per-chunk mask slices, and the logits of the continuation must match.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        real = torch.randint(1, config.vocab_size, (1, 5), device=torch_device)
        pads = torch.zeros(1, 2, dtype=real.dtype, device=torch_device)
        ids = torch.cat([pads, real], dim=1)  # [1, 7], left-padded
        mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1]], device=torch_device)

        with torch.no_grad():
            whole = model(input_ids=ids, attention_mask=mask).logits
            first = model(input_ids=ids[:, :4], attention_mask=mask[:, :4], use_cache=True)
            rest = model(input_ids=ids[:, 4:], attention_mask=mask, state=first.state, use_cache=True).logits

        torch.testing.assert_close(rest, whole[:, 4:], rtol=1e-4, atol=1e-4)

    def test_heavily_padded_fp16_batch_is_finite(self):
        """Padding in fp16, at a realistic pad fraction, must not produce NaN.

        This is the configuration every real user is in and the one the other padding
        test is not: it runs fp32 on a tiny model, where `F.normalize`'s 1e-12 epsilon
        is representable. In fp16 it is below the smallest subnormal, so normalising
        the zero vector a blanked pad position produces divides by a true zero and the
        whole row comes out NaN. Caught on real prompts, not here, hence the pad
        fraction below is 90%, like a short prompt batched with a long one, rather
        than the three pad tokens the other test uses.

        Deliberately not gated on a GPU: the guard it protects is a dtype question
        rather than a device one. fp16 runs on CPU perfectly well, the epsilon is
        unrepresentable there too, and removing the guard makes these logits non-finite
        on CPU. Behind `@require_torch_gpu` the only test for this bug would never run
        on the CI that gates the PR.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device).half()
        short = torch.randint(1, config.vocab_size, (1, 6), device=torch_device)
        long = torch.randint(1, config.vocab_size, (1, 60), device=torch_device)
        width = 60
        padded = torch.cat(
            [torch.cat([torch.zeros(1, width - 6, dtype=short.dtype, device=torch_device), short], dim=1), long],
            dim=0,
        )
        mask = torch.zeros(2, width, dtype=torch.long, device=torch_device)
        mask[0, -6:] = 1
        mask[1, :] = 1

        with torch.no_grad():
            out = model(input_ids=padded, attention_mask=mask).logits
            alone = model(input_ids=short).logits[0, -1]

        self.assertTrue(torch.isfinite(out).all(), "padded fp16 batch produced non-finite logits")
        self.assertEqual(int(out[0, -1].argmax()), int(alone.argmax()))

    def test_mask_without_padding_is_a_no_op(self):
        """An all-ones mask must not perturb the unpadded path at all.

        `generate` passes a mask on every call, padded or not, so the common case
        has to come out bit-identical -- not merely close.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(0, config.vocab_size, (2, 5), device=torch_device)

        with torch.no_grad():
            without = model(input_ids=ids).logits
            with_ones = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits

        self.assertTrue(torch.equal(without, with_ones))

    def test_layer_zero_value_residual_lora_is_unused(self):
        """Layer 0 produces `v_first`; it must never mix towards it.

        Reference checkpoints still ship `v0/v1/v2` on layer 0, and the `fla`
        layout drops them, so the two conversions disagree on exactly these
        tensors. Outputs must not.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)

        with torch.no_grad():
            before = model(input_ids=input_ids).logits
            for name in ("v0", "v1", "v2"):
                getattr(model.rwkv7.blocks[0].att, name).normal_(10.0, 1.0)
            after = model(input_ids=input_ids).logits

        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_deep_embed_hook(self):
        """The RWKV-8 DeepEmbed hook modulates the channel-mix, and is off by default."""
        config = _tiny_config(use_deep_embed=True)
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)
        layers, batch, seq = config.num_hidden_layers, 1, 4

        with torch.no_grad():
            plain = model(input_ids=input_ids).logits
            # a table of ones must be a no-op in the "1x" (output-modulating) form
            ones = torch.ones(layers, batch, seq, config.hidden_size, device=torch_device)
            neutral = model(input_ids=input_ids, deep_embeds=ones).logits
            # zeros silence the channel-mix entirely, which is an unambiguous
            # signal that the hook is actually applied (a scale near 1 is not: the
            # channel-mix is a small residual, so halving it barely moves logits)
            silenced = model(input_ids=input_ids, deep_embeds=torch.zeros_like(ones)).logits

        torch.testing.assert_close(plain, neutral, rtol=1e-5, atol=1e-5)
        self.assertGreater((plain - silenced).abs().max().item(), 0.0)

    def test_deep_embed_four_x_variant(self):
        """A table as wide as the channel-mix inner width modulates its input instead.

        Two assertions, and the second is the one that matters. A table of ones being a
        no-op is equally true when the modulation has been deleted, so stopping there
        would leave `inner = inner * deep_embed` in the 4x branch unreached: the sibling
        1x test covers only its own branch, and nothing else in the suite touches this
        multiply.
        """
        config = _tiny_config(use_deep_embed=True)
        # `_sharpened`, not `_randomised`: at the 0.05 scale the channel-mix
        # contribution to the logits is itself below a 1e-3 tolerance, so silencing it
        # entirely is indistinguishable from silencing nothing and the second
        # assertion would be as toothless as the one it replaces.
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)
        table = (config.num_hidden_layers, 1, 4, config.intermediate_size)

        with torch.no_grad():
            plain = model(input_ids=input_ids).logits
            neutral = model(input_ids=input_ids, deep_embeds=torch.ones(*table, device=torch_device)).logits
            damped = model(input_ids=input_ids, deep_embeds=torch.zeros(*table, device=torch_device)).logits

        torch.testing.assert_close(plain, neutral, rtol=1e-5, atol=1e-5)
        self.assertFalse(
            torch.allclose(plain, damped, rtol=1e-3, atol=1e-3),
            "a table of zeros left the logits unchanged, so the 4x modulation is not wired up",
        )

    def test_generate_is_deterministic_under_greedy(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        input_ids = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)

        with torch.no_grad():
            first = model.generate(input_ids, max_new_tokens=8, do_sample=False)
            second = model.generate(input_ids, max_new_tokens=8, do_sample=False)

        self.assertTrue(torch.equal(first, second))

    @staticmethod
    def _signed_cache(model, batch, device):
        """A cache whose every batch row carries a value only that row has.

        Every test below permutes the batch and then asks where each row went, which
        only means something if the rows were told apart to begin with.
        """
        cache = model.rwkv7.allocate_state(batch, device=device)
        for row in range(batch):
            for layer in cache.layers:
                for state in layer.recurrent_states.values():
                    state[row] = row + 1
        return cache

    @staticmethod
    def _every_state(cache):
        return [state for layer in cache.layers for state in layer.recurrent_states.values()]

    def test_reorder_cache_moves_the_state_onto_the_beams(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = self._signed_cache(model, 4, torch_device)
        addresses = [state.data_ptr() for state in self._every_state(cache)]

        # 2 appears twice: a reorder that aliased its source into its destination
        # would corrupt the second copy.
        beams = [2, 2, 0, 3]
        cache.reorder_cache(torch.tensor(beams, device=torch_device))

        for state in self._every_state(cache):
            for row, source in enumerate(beams):
                self.assertTrue(torch.all(state[row] == source + 1))
        # In place, so the addresses `allocate_state` pinned are still pinned.
        self.assertEqual([state.data_ptr() for state in self._every_state(cache)], addresses)

    def test_cache_batch_repeat_and_select(self):
        """The two batch edits `generate` makes outside beam search.

        `num_return_sequences` fans one prompt's state out to several rows before
        sampling; assisted and contrastive decoding drop rows again. Both change the
        batch size, which is why -- unlike `reorder_cache` -- they cannot keep the
        pinned buffers, and that is asserted rather than left as a surprise.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = self._signed_cache(model, 2, torch_device)

        cache.batch_repeat_interleave(3)
        for state in self._every_state(cache):
            self.assertEqual(state.shape[0], 6)
            # interleaved, so rows read 1,1,1,2,2,2 -- not 1,2,1,2,1,2
            self.assertTrue(torch.all(state[:3] == 1))
            self.assertTrue(torch.all(state[3:] == 2))

        cache.batch_select_indices(torch.tensor([0, 4], device=torch_device))
        for state in self._every_state(cache):
            self.assertEqual(state.shape[0], 2)
            self.assertTrue(torch.all(state[0] == 1))
            self.assertTrue(torch.all(state[1] == 2))

    def test_allocate_state_pins_every_buffer(self):
        """Unpinned state buffers cost CUDA graphs, and say so only in a log line.

        The whole reason `allocate_state` exists is that a buffer first allocated
        inside a compiled region cannot be given a static address, after which
        inductor declines CUDA graphs for a recurrent decode. Nothing throws; the
        decode just runs several times slower. Checking the mark directly is the
        only way that failure becomes visible without a GPU and a compile.

        `_dynamo_static_input_type` is private to torch. If it is renamed this test
        breaks loudly, which is the right failure -- the alternative is a silent
        performance regression nobody notices for a release.
        """
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = model.rwkv7.allocate_state(2, device=torch_device)

        for state in self._every_state(cache):
            self.assertEqual(getattr(state, "_dynamo_static_input_type", None), "unguarded")

    def test_cache_reset_clears_every_slot(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        cache = self._signed_cache(model, 2, torch_device)
        addresses = [state.data_ptr() for state in self._every_state(cache)]

        cache.reset()

        for state in self._every_state(cache):
            self.assertTrue(torch.all(state == 0))
        # Reset means "forget the sequence", not "throw the buffers away".
        self.assertEqual([state.data_ptr() for state in self._every_state(cache)], addresses)

    def test_cache_carries_a_sequence_the_same_as_a_plain_rerun(self):
        """The cache is only useful if continuing through it equals never stopping."""
        config = _tiny_config()
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        ids = torch.randint(1, config.vocab_size, (2, 9), device=torch_device)

        with torch.no_grad():
            whole = model(input_ids=ids).logits
            cache = model(input_ids=ids[:, :4], use_cache=True).state
            rest = model(input_ids=ids[:, 4:], state=cache, use_cache=True).logits

        torch.testing.assert_close(rest, whole[:, 4:], rtol=1e-4, atol=1e-4)

    def test_beam_search_score_survives_an_independent_rescore(self):
        """Beam search accumulated a score through the state; rescore it fresh.

        "It ran without raising" is not evidence. Beam search keeps running when
        the state follows the wrong beam -- it just searches with a history that
        belongs to some other candidate, and reports a cumulative score built from
        that wrong history. So the check is a consistency one: `sequences_scores`
        is accumulated step by step through the recurrent state, and rescoring the
        very same tokens in one full forward has to reproduce it. Only a state that
        tracked the surviving beams makes those two numbers agree.

        Two things here were measured rather than assumed, both after a first
        version of this test passed with the reorder stubbed out entirely:

        * The obvious property -- beam scoring at least as well as greedy -- is not
          a theorem and is not used. Greedy's path can be pruned out of the top-k
          part way through and still be the better sequence at the end, which is
          exactly what this model does (beam -41.81 vs greedy -39.45, with a
          correct reorder). A control run on Llama in the same checkout happens to
          come out the other way, +3.37, which is what made the false premise look
          confirmed.
        * The sharper init is load-bearing. At `_randomised`'s 0.05 the tiny
          model's next-token distribution is nearly uniform, every beam is
          interchangeable, and a stubbed reorder still agrees to 1.5e-05 -- no
          discrimination at all. At 0.5 over 16 tokens and 4 beams: 3.8e-06 with
          the reorder, 15.9 without it. The 1e-3 tolerance sits between those.
        """
        config = _tiny_config()
        torch.manual_seed(0)
        model = _sharpened(Rwkv7ForCausalLM(config)).to(torch_device)
        prompt = torch.randint(0, config.vocab_size, (1, 4), device=torch_device)

        with torch.no_grad():
            beamed = model.generate(
                prompt,
                max_new_tokens=16,
                num_beams=4,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
            sequence = beamed.sequences
            logits = model(sequence).logits.float()

        generated = sequence.shape[1] - prompt.shape[1]
        rescored = (
            torch.log_softmax(logits[:, :-1], dim=-1)
            .gather(-1, sequence[:, 1:, None])[:, prompt.shape[1] - 1 :]
            .sum()
            .item()
        )
        # `sequences_scores` is the length-normalised beam score; length_penalty
        # defaults to 1.0, so multiplying by the generated length undoes it.
        accumulated = beamed.sequences_scores.item() * generated

        self.assertAlmostEqual(accumulated, rescored, delta=1e-3)

    def test_gradients_reach_every_parameter(self):
        config = _tiny_config()
        model = _randomised(Rwkv7ForCausalLM(config)).to(torch_device)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 5), device=torch_device)
        model(input_ids=input_ids, labels=input_ids, use_cache=False).loss.backward()

        # layer 0's value-residual LoRA is intentionally dead (see the test above)
        dead = {f"rwkv7.blocks.0.att.{n}" for n in ("v0", "v1", "v2")}
        missing = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and name not in dead and (param.grad is None or param.grad.abs().sum() == 0)
        ]
        self.assertEqual(missing, [], f"parameters received no gradient: {missing}")

    def test_chunked_matches_sequential_recurrence(self):
        """The chunk-parallel prefill must reproduce the sequential recurrence.

        They are two forms of the same step, so any disagreement is a bug in the
        chunked derivation rather than a tolerance question. Several chunk sizes
        are checked, including ones that do not divide the sequence length.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_chunked, rwkv7_recurrent

        torch.manual_seed(0)
        batch, seq_len, heads, dim = 2, 37, 3, 8
        shape = (batch, seq_len, heads, dim)
        r = torch.randn(shape, device=torch_device)
        # w_log is a log-decay in (-e^-0.5, 0), as the decay LoRA produces
        w_log = -0.6065306597126334 * torch.sigmoid(torch.randn(shape, device=torch_device))
        k = torch.randn(shape, device=torch_device)
        v = torch.randn(shape, device=torch_device)
        kk = torch.nn.functional.normalize(torch.randn(shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device=torch_device))
        state = torch.randn(batch, heads, dim, dim, device=torch_device)

        reference, reference_state = rwkv7_recurrent(r, w_log, k, v, kk, a, state.clone())
        for chunk_size in (1, 4, 16, 64):
            out, out_state = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=chunk_size)
            torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-4, msg=f"chunk={chunk_size}")
            torch.testing.assert_close(
                out_state, reference_state, rtol=2e-4, atol=2e-4, msg=f"state chunk={chunk_size}"
            )

    def test_chunked_survives_the_slowest_decay_it_can_be_given(self):
        """The chunk size is bounded by overflow, and only the worst decay finds it.

        The chunked form divides by the running decay `c`, so `1/c` grows like
        `e^(e^-0.5 * chunk_size)` when every channel sits at the floor the decay LoRA
        can produce. That floor is `exp(-e^-0.5)` = 0.5452, not `e^-0.5`; the difference
        matters, since the looser value puts the fp32 ceiling at 177 rather than the true
        146. `test_chunked_matches_sequential_recurrence` draws `w_log` from
        a sigmoid, which lands nowhere near that floor, so it would pass a chunk size
        that overflows in production on a checkpoint with strongly-decaying channels.

        Precision is not what limits it: the substitution is a similarity transform,
        so `c` deflates on the way out whatever `1/c` inflated, and a common scale
        factor does not move a relative error. What limits it is fp32's exponent
        range, and the second half of this test is where that runs out -- kept as an
        assertion rather than a comment so the default has a measured ceiling above
        it rather than an assumed one.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_chunked, rwkv7_recurrent

        torch.manual_seed(0)
        shape = (1, 256, 2, 8)
        floor = -0.6065306597126334
        w_log = torch.full(shape, floor, device=torch_device)  # every channel at the floor
        r = torch.randn(shape, device=torch_device) * 0.2
        k = torch.randn(shape, device=torch_device) * 0.2
        v = torch.randn(shape, device=torch_device) * 0.2
        kk = torch.nn.functional.normalize(torch.randn(shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device=torch_device))
        state = torch.zeros(1, 2, 8, 8, device=torch_device)

        reference, _ = rwkv7_recurrent(r, w_log, k, v, kk, a, state.clone())
        for chunk_size in (16, 64):  # 64 is the default
            out, _ = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=chunk_size)
            self.assertTrue(torch.isfinite(out).all(), f"chunk={chunk_size} overflowed")
            torch.testing.assert_close(out, reference, rtol=2e-4, atol=2e-4, msg=f"chunk={chunk_size}")

        # The default itself, not just the sizes named above: every call here passes
        # `chunk_size=` explicitly, so raising the default to something that overflows
        # would leave all of them green. This is the one that exercises what the model
        # actually uses.
        default_out, _ = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone())
        self.assertTrue(torch.isfinite(default_out).all(), "the DEFAULT chunk size overflows at the decay floor")
        torch.testing.assert_close(default_out, reference, rtol=2e-4, atol=2e-4)

        # And where it runs out. Asserting that an oversized chunk comes back non-finite
        # would document the overflow by demonstrating it silently; the function refuses
        # instead. The boundary is asserted on both sides so the derived limit cannot
        # drift without a test noticing: 146 is the widest that works, 147 the narrowest
        # that raises.
        widest, _ = rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=146)
        self.assertTrue(torch.isfinite(widest).all(), "146 is supposed to be the widest chunk fp32 can carry")

        with self.assertRaisesRegex(ValueError, "widest chunk that stays finite is 146"):
            rwkv7_chunked(r, w_log, k, v, kk, a, state.clone(), chunk_size=147)

    def test_wkv_state_dtype_is_configurable(self):
        """The state precision is independent of the activation dtype.

        The recurrence is unrolled over the whole sequence, so the state is where
        precision actually matters; the config exposes it separately for that
        reason. fp32 must stay closer to a fp64 rollout than fp16 does.
        """
        from transformers.models.rwkv7.modeling_rwkv7 import rwkv7_recurrent

        torch.manual_seed(0)
        shape = (1, 24, 2, 8)
        r = torch.randn(shape, device=torch_device)
        w_log = -0.6065306597126334 * torch.sigmoid(torch.randn(shape, device=torch_device))
        k = torch.randn(shape, device=torch_device)
        v = torch.randn(shape, device=torch_device)
        kk = torch.nn.functional.normalize(torch.randn(shape, device=torch_device), dim=-1)
        a = torch.sigmoid(torch.randn(shape, device=torch_device))
        state = torch.zeros(1, 2, 8, 8, device=torch_device)

        truth, _ = rwkv7_recurrent(
            r.double(),
            w_log.double(),
            k.double(),
            v.double(),
            kk.double(),
            a.double(),
            state.double(),
            compute_dtype=torch.float64,
        )
        fp32, s32 = rwkv7_recurrent(r, w_log, k, v, kk, a, state.clone(), compute_dtype=torch.float32)
        fp16, s16 = rwkv7_recurrent(
            r.half(),
            w_log.half(),
            k.half(),
            v.half(),
            kk.half(),
            a.half(),
            state.clone().half(),
            compute_dtype=torch.float16,
        )
        self.assertEqual(s32.dtype, torch.float32)
        self.assertEqual(s16.dtype, torch.float16)
        err32 = (fp32.double() - truth).abs().max().item()
        err16 = (fp16.double() - truth).abs().max().item()
        self.assertLess(err32, err16, f"fp32 state ({err32:.3e}) should beat fp16 ({err16:.3e})")

        for bad in ("float8", "int8"):
            with self.assertRaises(ValueError):
                _tiny_config(wkv_state_dtype=bad)


@require_torch
@slow
class Rwkv7IntegrationTests(unittest.TestCase):
    """A real checkpoint, against numbers produced by an implementation that is not this one.

    Everything else in this file builds a model out of random weights, which pins
    internal consistency and cannot see a whole class of defect: an `_init_weights` that
    zeroed 249 of a converted checkpoint's 402 tensors passed the entire suite and was
    caught only by loading real weights and reading the output.

    The expected values below come from BlinkDL's own runtime -- the `rwkv` package
    (0.8.32 at the time of generation) at `cpu fp32` with `RWKV_V7_ON=1` -- and not
    from this implementation, which would make them self-certifying. To regenerate:

        RWKV_V7_ON=1 python -c "
        from rwkv.model import RWKV
        m = RWKV(model='<CHECKPOINT_FILE minus .pth>', strategy='cpu fp32')
        logits, state = m.forward(PROMPT_IDS, None)
        # greedy-extend 20 tokens for EXPECTED_IDS; top-5 of the first logits
        # gives EXPECTED_TOP5 / EXPECTED_TOP5_LOGITS" They are deliberately not taken from `fla`, whose RWKV layer
    prints a warning on import saying it is potentially buggy and that results should be
    cross-checked against the official repo.

    0.1B is chosen for more than its size: 768 wide at head_dim 64 is twelve heads of
    width 64, so head *count* and head *width* differ. At the 7.2B they are both 64 and a
    quantity indexed by the wrong one of them agrees by coincidence.
    """

    CHECKPOINT_REPO = "BlinkDL/rwkv-7-world"
    CHECKPOINT_FILE = "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth"

    # "The Eiffel Tower is located in the city of", RWKV world tokenizer.
    PROMPT_IDS = [6699, 304, 25740, 109, 37480, 4600, 52151, 4596, 22590, 30449, 4706]
    # " Paris, France. It is the tallest building in the world and is the world's tallest"
    EXPECTED_IDS = [
        37138,
        45,
        44312,
        47,
        3918,
        4600,
        22590,
        32190,
        7513,
        55666,
        4596,
        22590,
        40213,
        21265,
        4600,
        22590,
        40213,
        460,
        32190,
        7513,
    ]
    EXPECTED_TOP5 = [37138, 29319, 20312, 3632, 29417]
    EXPECTED_TOP5_LOGITS = [5.1235, 1.0825, 0.9404, 0.8200, 0.2382]

    @classmethod
    def setUpClass(cls):
        from huggingface_hub import hf_hub_download

        from transformers.models.rwkv7.convert_rwkv7_checkpoint_to_hf import convert

        cls._work = tempfile.mkdtemp()
        native = hf_hub_download(cls.CHECKPOINT_REPO, cls.CHECKPOINT_FILE)
        # Through the converter in this PR, so the test covers that too: a model PR whose
        # checkpoints all arrive by a path the tests never take is testing half of itself.
        convert(native, "native", None, f"{cls._work}/hf", dtype="float32")
        cls.model = Rwkv7ForCausalLM.from_pretrained(f"{cls._work}/hf", dtype=torch.float32).eval()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._work, ignore_errors=True)

    def test_config_is_inferred_at_a_non_square_width(self):
        config = self.model.config
        self.assertEqual((config.hidden_size, config.num_heads, config.head_dim), (768, 12, 64))
        self.assertNotEqual(config.num_heads, config.head_dim)

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
        """Twenty steps of one-token decode against twenty full re-reads of the prefix.

        The shared suite checks this on random weights, where the state is small enough
        that a broken carry can hide under the tolerance. Here a divergence shows up as a
        different token.
        """
        ids = list(self.PROMPT_IDS)
        cached, recomputed = [], []
        with torch.no_grad():
            out = self.model(input_ids=torch.tensor([ids]), use_cache=True)
            state = out.state
            cached.append(int(out.logits[0, -1].argmax()))
            for _ in range(19):
                out = self.model(input_ids=torch.tensor([[cached[-1]]]), state=state, use_cache=True)
                state = out.state
                cached.append(int(out.logits[0, -1].argmax()))

            walk = list(self.PROMPT_IDS)
            for _ in range(20):
                logits = self.model(input_ids=torch.tensor([walk]), use_cache=False).logits[0, -1]
                recomputed.append(int(logits.argmax()))
                walk.append(recomputed[-1])

        self.assertEqual(cached, recomputed)
        self.assertEqual(cached, self.EXPECTED_IDS)
