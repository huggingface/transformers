# Copyright 2026 The HuggingFace Team. All rights reserved.
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
"""Integration tests for loading GGUF checkpoints through `from_pretrained`.

The contract these tests encode: **a model loaded from a non-quantized GGUF is the same model as one
loaded from safetensors** — same parameter names, same values, in the dtype the file was written in.
Generation then covers both kinds of file: from the non-quantized one it says those weights are wired
into a working forward pass, and from a quantized one that the blocks kept packed are read correctly.

Adding an architecture means adding one test class at the bottom: the shared tests live in the mixin
and are driven by the class attributes naming the checkpoints and the model class.

Tokenizers are out of scope here: they come from the reference repo, since building one from GGUF
metadata is not implemented yet. When it is, they belong in their own class — a different code path,
and far too cheap to pay for an 8 GB model load in `setUpClass`.
"""

import math
import unittest
import unittest.mock

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GgufConfig, Qwen3_5ForCausalLM
from transformers.testing_utils import (
    require_kernels,
    require_torch_accelerator,
    require_torch_mps,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


class GgufModelIntegrationTesterMixin:
    """Tests every integrated architecture must pass.

    A test class supplies the checkpoints (`gguf_repo` with a `gguf_file` and a `quantized_gguf_file`,
    plus the `reference_repo` to compare against), the `model_class` to load them with, the `prompt`
    and its `expected_completion`, and the
    `inexact_params` tolerances.
    """

    tokenizer_texts = (
        "The capital of France is Paris.",
        "def f(x):\n    return x ** 2\n",
        "  leading and trailing  ",
        "\u65e5\u672c\u8a9e \U0001f680 \u00fcn\u00efc\u00f4de",
        "<|im_start|>user\nhi<|im_end|>",
    )

    # Per-parameter relative tolerances, for an architecture whose conversion cannot be exact. Empty
    # is the expectation: the transforms run in the float type the file stores and `Cast` rounds once,
    # at the end, so a converted weight is bit-for-bit what a safetensors checkpoint holds.
    inexact_params: dict[str, float] = {}

    chat = (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    )

    @classmethod
    def setUpClass(cls):
        # One load for the whole class: these checkpoints are several GB. No dtype is passed, so this
        # covers `auto` resolving to the one the file was written in.
        cls.model = cls.load_gguf_model(cls.gguf_file)
        # Built once too: a vocabulary of a few hundred thousand tokens is not free to assemble.
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.gguf_repo, gguf_file=cls.quantized_gguf_file)
        cls.reference_tokenizer = AutoTokenizer.from_pretrained(cls.reference_repo)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        if is_torch_available() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def load_gguf_model(cls, gguf_file, dtype=None, **kwargs):
        """Load one of `gguf_repo`'s files, config included: the file's metadata carries it.

        Nothing is borrowed from the reference repo — a GGUF repo ships no `config.json`, so this is
        also the only load path that says the metadata is enough on its own.
        """
        kwargs.setdefault("device_map", torch_device)
        return cls.model_class.from_pretrained(cls.gguf_repo, gguf_file=gguf_file, dtype=dtype, **kwargs)

    @staticmethod
    def map_reference_key(key):
        """The reference model's name for a parameter -> the GGUF-loaded model's, or `None` to skip it."""
        return key

    def reference_state_dict(self):
        """Parameters of the reference model, keyed to match the GGUF-loaded one.

        Loaded through `from_pretrained` rather than read off the safetensors shards: a checkpoint has
        conversions of its own — a MoE stacks its per-expert tensors into one parameter — so its keys
        need not be parameter names, and those fused parameters are the ones most worth comparing.
        """
        reference = AutoModelForCausalLM.from_pretrained(self.reference_repo, dtype=torch.bfloat16)
        state_dict = {}
        for name, tensor in reference.state_dict().items():
            mapped = self.map_reference_key(name)
            if mapped is not None:
                state_dict[mapped] = tensor
        return state_dict

    def completion(self, model):
        """What the model greedily continues `self.prompt` with."""
        inputs = self.reference_tokenizer(self.prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        return self.reference_tokenizer.decode(output[0, inputs.input_ids.shape[1] :])

    def test_tokenizer_matches_transformers(self):
        """A GGUF repo ships no tokenizer files either, so the one built from the metadata is the same.

        `pad_token` and the chat template with a generation prompt are left out: llama.cpp's converter
        makes its own choice for both, and the file is what a self-contained load should follow.
        """
        from_gguf, reference = self.tokenizer, AutoTokenizer.from_pretrained(self.reference_repo)

        # Not the vocabulary sizes: a GGUF states one flat token list, where the reference keeps its
        # special tokens as additions on top of a smaller base. What has to agree is what they encode to.
        for text in self.tokenizer_texts:
            with self.subTest(text=text):
                self.assertEqual(from_gguf(text).input_ids, reference(text).input_ids)

        # Stated by id in the file, so only the vocabulary turns them back into strings. Encodings
        # cannot catch this: a tokenizer handed no special tokens invents a sentencepiece pair and
        # appends it, which agrees with the reference on every text and puts two ids past the end of
        # the embedding -- hence the bound below rather than another comparison.
        self.assertEqual(from_gguf.eos_token, reference.eos_token)
        self.assertEqual(from_gguf.bos_token, reference.bos_token)
        self.assertLessEqual(len(from_gguf), self.model.get_input_embeddings().weight.shape[0])

        # The template rides along in the metadata, and formats a conversation the same way.
        self.assertIsNotNone(from_gguf.chat_template)
        self.assertEqual(
            from_gguf.apply_chat_template(self.chat, tokenize=False),
            reference.apply_chat_template(self.chat, tokenize=False),
        )

    def test_state_dict_matches_transformers(self):
        """The headline test: same values as the safetensors checkpoint, tensor by tensor.

        This is what validates the architecture's conversion table. When it fails it names the
        offending parameter, unlike a logits or generation comparison.
        """
        reference = self.reference_state_dict()
        loaded = {k: v for k, v in self.model.state_dict().items() if k in reference}

        mismatched = []
        for name in sorted(reference):
            expected, actual = reference[name].float(), loaded[name].float().cpu()
            self.assertEqual(expected.shape, actual.shape, f"{name}: shape differs")
            rtol = next((tol for pat, tol in self.inexact_params.items() if pat in name), 0.0)
            error = (expected - actual).abs().max().item()
            scale = expected.abs().max().item() or 1.0
            if error > rtol * scale:
                mismatched.append(f"{name}: max abs {error:.3e} (rel {error / scale:.2e}, tol {rtol:.0e})")
        self.assertEqual(mismatched, [], f"{len(mismatched)} parameters differ:\n" + "\n".join(mismatched[:10]))

    def test_config_matches_transformers(self):
        """The config is rebuilt from the file's metadata alone, so it has to say the same as the repo's.

        Compared field by field over everything the two have in common, minus what records where a
        checkpoint came from rather than what it is. A wrong `head_dim` or `rope_theta` here is a model
        that loads and generates nonsense, which no state dict comparison would catch.
        """
        # Records where a checkpoint came from, not what it is
        provenance = {"architectures", "dtype", "torch_dtype", "transformers_version", "_name_or_path"}
        # llama.cpp writes the *tokenizer's* special-token ids into the file, and those need not match
        # what the repo's `config.json` says: for this checkpoint the file and the tokenizer agree on
        # `eos=<|im_end|>`, while the config still names `<|endoftext|>`. The file is the better source —
        # it is what makes a chat model stop where its tokenizer says — so this compares the rest.
        token_ids = {"bos_token_id", "eos_token_id", "pad_token_id", "sep_token_id", "unk_token_id"}
        reference = AutoConfig.from_pretrained(self.reference_repo).get_text_config().to_dict()
        actual = self.model.config.to_dict()

        differing = []
        for field in sorted(set(reference) & set(actual) - provenance - token_ids):
            expected, got = reference[field], actual[field]
            # A float field is stored f32 in the file, so it comes back as the nearest f32 to what the
            # repo says: `rms_norm_eps=1e-6` reads back as 9.99999997e-07.
            if isinstance(expected, float) and isinstance(got, float):
                if math.isclose(expected, got, rel_tol=1e-6):
                    continue
            if expected != got:
                differing.append(f"{field}: {expected!r} != {got!r}")
        self.assertEqual(differing, [], f"{len(differing)} config fields differ:\n" + "\n".join(differing))

    def assert_completes(self, model):
        """The model greedily continues `self.prompt` with `self.expected_completion`."""
        completion = self.completion(model)
        self.assertTrue(
            completion.startswith(self.expected_completion),
            f"expected completion to start with {self.expected_completion!r}, got {completion!r}",
        )

    def test_generates_expected_text(self):
        """End-to-end: those weights wired into a working forward pass."""
        self.assert_completes(self.model)

    @require_torch_mps
    @require_kernels
    def test_generates_expected_text_from_packed_file(self):
        """A quantized file with its blocks left packed: a kernel reads them correctly.

        Needs a kernel, not just the device one is built for: without one the load falls back to
        unpacking, which is the test below.
        """
        model = self.load_gguf_model(self.quantized_gguf_file)
        packed = [name for name, p in model.named_parameters() if p.dtype == torch.uint8]
        self.assertTrue(packed, "no weight stayed in GGUF blocks, so this is not testing a packed load")
        self.assert_completes(model)
        del model

    def test_generates_expected_text_from_dequantized_file(self):
        """The same file unpacked at load, which is how every other device reads it.

        The other half of what a quantized checkpoint has to survive, and the half that runs anywhere:
        the blocks go through the torch dequantizer rather than a kernel, so a misread block layout
        shows up as a wrong word here rather than nowhere at all.
        """
        model = self.load_gguf_model(
            self.quantized_gguf_file,
            dtype=torch.bfloat16,  # a quantized file has no float type of its own, so this would be f32
            quantization_config=GgufConfig(gguf_file=self.quantized_gguf_file, dequantize=True),
        )
        self.assertEqual(
            [name for name, p in model.named_parameters() if p.dtype == torch.uint8],
            [],
            "a weight kept its blocks despite dequantize=True",
        )
        self.assert_completes(model)
        del model


@slow
class GgufIntegrationTest(unittest.TestCase):
    """How a quantized file behaves under each way of loading it, on one model.

    These are properties of the integration rather than of an architecture — whether the weights stay
    packed and what dtype the rest of them land in — so one checkpoint covers them for every model
    that follows.

    Only the two that need the weights *packed* ask for a kernel. Everywhere else -- another device,
    or this one with no kernel published for it -- the load falls back to dequantizing, and what that
    produces is a dense model that runs anywhere.
    """

    gguf_repo = "bartowski/Qwen_Qwen3.5-4B-GGUF"
    gguf_file = "Qwen_Qwen3.5-4B-Q4_K_M.gguf"
    # only for the tokenizer: building one from GGUF metadata is not implemented for this architecture
    reference_repo = "Qwen/Qwen3.5-4B"
    model_class = Qwen3_5ForCausalLM
    prompt = "The capital of France is Paris. The capital of Germany is"

    def load(self, **kwargs):
        return self.model_class.from_pretrained(self.gguf_repo, gguf_file=self.gguf_file, **kwargs)

    @staticmethod
    def packed_modules(model):
        from transformers.integrations.gguf.utils import GgufEmbedding, GgufLinear

        return [module for module in model.modules() if isinstance(module, (GgufLinear, GgufEmbedding))]

    def generates(self, model):
        tokenizer = AutoTokenizer.from_pretrained(self.reference_repo)
        inputs = tokenizer(self.prompt, return_tensors="pt").to(next(model.parameters()).device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=4, do_sample=False)
        return tokenizer.decode(output[0, inputs.input_ids.shape[1] :])

    @require_torch_mps
    @require_kernels
    def test_kernel_keeps_the_weights_packed(self):
        """With a matmul kernel, the blocks are what the modules hold and compute on."""
        model = self.load(device_map=torch_device)

        packed = self.packed_modules(model)
        self.assertTrue(packed, "a kernel is available but no module kept its blocks")
        self.assertIn("Berlin", self.generates(model))

    def test_runs_without_a_kernel(self):
        """No kernel: nothing can compute on blocks, so the whole model is unpacked at load."""
        with unittest.mock.patch("transformers.quantizers.quantizer_gguf.get_gguf_kernel", return_value=False):
            model = self.load(device_map=torch_device)

        self.assertEqual(self.packed_modules(model), [], "blocks were kept with nothing able to read them")
        self.assertIn("Berlin", self.generates(model))

    def test_dequantize_gives_a_dense_model(self):
        """`dequantize=True` asks for the weights unpacked once, at load."""
        from transformers import GgufConfig

        model = self.load(
            device_map=torch_device,
            quantization_config=GgufConfig(gguf_file=self.gguf_file, dequantize=True),
        )

        self.assertEqual(self.packed_modules(model), [], "a module kept its blocks despite dequantize=True")
        # No `dtype` was passed and a quantized file has no float type of its own, so it lands in f32
        self.assertEqual({p.dtype for p in model.parameters()}, {torch.float32})
        # nothing quantized is left, so nothing should claim otherwise
        self.assertFalse(getattr(model, "is_quantized", False))
        self.assertFalse(hasattr(model.config, "quantization_config"))
        self.assertIn("Berlin", self.generates(model))

    def test_dtype_of_a_dequantized_model(self):
        """A dequantized model is an ordinary one, so `dtype` decides what it is loaded in."""
        from transformers import GgufConfig

        # On the host, and never in f32: this reads `p.dtype` and nothing else, so a device copy buys
        # nothing, and f32 is both what `test_dequantize_gives_a_dense_model` already covers and 15.7 GB
        # for a 4B model -- which is what made this the slowest test in the file by an order of magnitude.
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = self.load(
                    dtype=dtype,
                    device_map="cpu",
                    quantization_config=GgufConfig(gguf_file=self.gguf_file, dequantize=True),
                )
                self.assertEqual({p.dtype for p in model.parameters()}, {dtype})
                del model

    @require_torch_mps
    @require_kernels
    def test_dtype_of_what_a_packed_model_unpacks(self):
        """A packed model still has dense parameters — the ones no module could hold — in `dtype`."""
        model = self.load(dtype=torch.bfloat16, device_map=torch_device)

        dtypes = {p.dtype for p in model.parameters()}
        self.assertIn(torch.uint8, dtypes, "no parameter kept its blocks")
        self.assertEqual(dtypes - {torch.uint8}, {torch.bfloat16}, "an unpacked parameter is not in `dtype`")


@require_torch_accelerator
@slow
class Qwen35GgufModelTest(GgufModelIntegrationTesterMixin, unittest.TestCase):
    gguf_repo = "unsloth/Qwen3.5-4B-GGUF"
    gguf_file = "Qwen3.5-4B-BF16.gguf"
    quantized_gguf_file = "Qwen3.5-4B-Q4_K_M.gguf"
    reference_repo = "Qwen/Qwen3.5-4B"
    model_class = Qwen3_5ForCausalLM

    prompt = "The capital of France is Paris. The capital of Germany is"
    expected_completion = " Berlin"
    # A floor in the checkpoint, not in this path. llama.cpp writes a zero-centred norm as `w + 1` in
    # f32, whose step at 1.0 is 1.2e-07, so a weight smaller than that is not in the file at all: one
    # here is 5.178e-07 in the reference and comes back as 4.768e-07, the nearest `1 + w` can encode.
    # Nothing on load recovers it. Every other parameter matches bit for bit.
    inexact_params = {"norm.weight": 1e-6}
