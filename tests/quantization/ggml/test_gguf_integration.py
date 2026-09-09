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
"""Integration tests for loading GGUF checkpoints through `from_pretrained`."""

import math
import tempfile
import unittest
import unittest.mock

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    GgufConfig,
    Qwen3_5ForCausalLM,
)
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


class GgufTokenizerTesterMixin:
    """A tokenizer built from a GGUF file's metadata is the one the reference repo ships.

    Needs no model, so it also covers architectures the loading path does not support yet.
    """

    # One text per thing a conversion gets wrong on its own: the pre-tokenizer regex shows up on the
    # digits and contractions, the prefix scheme on the whitespace runs, byte fallback on the scripts
    # and emoji, and a stray post-processor on the special-token text.
    tokenizer_texts = (
        "The capital of France is Paris.",
        "def f(x):\n    return x ** 2\n",
        "  leading and trailing  ",
        "a\tb\n\nc  d",
        "I don't think it's 3.14159 or -42,000",
        "\u65e5\u672c\u8a9e \U0001f680 \u00fcn\u00efc\u00f4de",
        "\u0391\u0392\u0393 \u0411\u0413\u0414 \u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u05e2\u05d1\u05e8\u05d9\u05ea",
        "https://example.com/a_b?c=1&d=2#e",
        "<|im_start|>user\nhi<|im_end|>",
    )

    chat = (
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    )

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.gguf_repo, gguf_file=cls.quantized_gguf_file)
        cls.reference_tokenizer = AutoTokenizer.from_pretrained(cls.reference_repo)

    def test_tokenizer_matches_transformers(self):
        from_gguf, reference = self.tokenizer, self.reference_tokenizer

        # What agrees is the encoding, not the vocabulary size: a GGUF states one flat token list
        # where the reference adds its special tokens on top of a smaller base. Without them, too --
        # a bos is prepended per `tokenizer.ggml.add_bos_token`, which many published files omit.
        for text in self.tokenizer_texts:
            with self.subTest(text=text):
                self.assertEqual(
                    from_gguf(text, add_special_tokens=False).input_ids,
                    reference(text, add_special_tokens=False).input_ids,
                )

    def test_special_tokens_match_transformers(self):
        """Stated by id in the file, so only the vocabulary turns them back into strings."""
        from_gguf, reference = self.tokenizer, self.reference_tokenizer

        # Not equality with the reference's `eos_token`: llama.cpp writes the turn terminator as the
        # eos of an instruct model, so a Gemma file names id 106 where the repo names `<eos>`. Both
        # stop generation, so what must hold is that the file named one the model accepts.
        accepted = {reference.eos_token}
        try:
            eos_ids = GenerationConfig.from_pretrained(self.reference_repo).eos_token_id
        except OSError:
            eos_ids = None
        if eos_ids is not None:
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else eos_ids
            accepted.update(reference.convert_ids_to_tokens(token_id) for token_id in eos_ids)
        self.assertIn(from_gguf.eos_token, accepted)
        # `bos` only where the reference has one: llama.cpp writes an id even for a tokenizer
        # declaring none, and the encodings above prove it is never actually emitted.
        if reference.bos_token is not None:
            self.assertEqual(from_gguf.bos_token, reference.bos_token)

    def test_chat_template_matches_transformers(self):
        """The template rides along in the metadata, and formats a conversation the same way."""
        if self.reference_tokenizer.chat_template is None:
            self.skipTest("the reference repo ships no chat template")
        self.assertEqual(
            self.tokenizer.apply_chat_template(self.chat, tokenize=False),
            self.reference_tokenizer.apply_chat_template(self.chat, tokenize=False),
        )


class GgufModelIntegrationTesterMixin(GgufTokenizerTesterMixin):
    """Tests every integrated architecture must pass."""

    # Per-parameter relative tolerances, for an architecture whose conversion cannot be exact. Empty
    # is the expectation: the transforms run in the float type the file stores and `Cast` rounds once,
    # at the end, so a converted weight is bit-for-bit what a safetensors checkpoint holds.
    inexact_params: dict[str, float] = {}

    @classmethod
    def setUpClass(cls):
        # One load for the whole class: these checkpoints are several GB. No dtype is passed, so this
        # covers `auto` resolving to the one the file was written in. The loading report comes back
        # with it, for `test_load_accounts_for_every_key`.
        super().setUpClass()
        cls.model, cls.loading_info = cls.load_gguf_model(cls.gguf_file, output_loading_info=True)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        if is_torch_available() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def load_gguf_model(cls, gguf_file, dtype=None, **kwargs):
        """Load one of `gguf_repo`'s files, config included: the file's metadata carries it."""
        kwargs.setdefault("device_map", torch_device)
        return cls.model_class.from_pretrained(cls.gguf_repo, gguf_file=gguf_file, dtype=dtype, **kwargs)

    @staticmethod
    def map_reference_key(key):
        """The reference model's name for a parameter -> the GGUF-loaded model's, or `None` to skip it."""
        return key

    def reference_state_dict(self):
        """Parameters of the reference model, keyed to match the GGUF-loaded one."""
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

    def test_vocabulary_fits_the_embedding(self):
        """A tokenizer handed no special tokens invents a sentencepiece pair and appends it, which
        agrees with the reference on every text but puts two ids past the end of the embedding."""
        self.assertLessEqual(len(self.tokenizer), self.model.get_input_embeddings().weight.shape[0])

    def test_state_dict_matches_transformers(self):
        """The headline test: same values as the safetensors checkpoint, tensor by tensor."""
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

    def test_load_accounts_for_every_key(self):
        """Nothing missing, nothing unexpected: the conversion covers the file and fills the model."""
        self.assertEqual(self.loading_info["missing_keys"], set())
        self.assertEqual(self.loading_info["unexpected_keys"], set())
        self.assertEqual(self.loading_info["mismatched_keys"], set())

    def test_config_matches_transformers(self):
        """The config is rebuilt from the file's metadata alone, so it has to say the same as the repo's."""
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
        """A quantized file with its blocks left packed: a kernel reads them correctly."""
        model = self.load_gguf_model(self.quantized_gguf_file)
        packed = [name for name, p in model.named_parameters() if p.dtype == torch.uint8]
        self.assertTrue(packed, "no weight stayed in GGUF blocks, so this is not testing a packed load")
        self.assert_completes(model)
        del model

    def test_generates_expected_text_from_dequantized_file(self):
        """The same file unpacked at load, which is how every other device reads it."""
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
    """How a quantized file behaves under each way of loading it, on one model."""

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

    def test_load_accounts_for_every_key(self):
        """Nothing missing, nothing unexpected -- on the one file here that carries an MTP block."""
        _, loading_info = self.load(device_map=torch_device, output_loading_info=True)

        self.assertEqual(loading_info["missing_keys"], set())
        self.assertEqual(loading_info["unexpected_keys"], set())

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

    @require_torch_mps
    @require_kernels
    def test_a_packed_model_cannot_be_saved(self):
        """GGUF blocks are not a format transformers writes, so keeping a packed model is refused."""
        model = self.load(device_map=torch_device)
        self.assertTrue(self.packed_modules(model), "a kernel is available but no module kept its blocks")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "not serializable"):
                model.save_pretrained(directory)

    def test_a_dequantized_model_can_be_saved(self):
        """A dequantized model is an ordinary dense one, so it saves and reloads as one."""
        model = self.load(
            device_map="cpu",  # this saves 8 GB to disk and reads it back; a device copy buys nothing
            dtype=torch.bfloat16,
            quantization_config=GgufConfig(gguf_file=self.gguf_file, dequantize=True),
        )

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.bfloat16)

        expected = dict(model.named_parameters())
        actual = dict(reloaded.named_parameters())
        self.assertEqual(sorted(actual), sorted(expected), "the saved model is not the one that was loaded")
        differing = [name for name, tensor in expected.items() if not torch.equal(tensor, actual[name])]
        self.assertEqual(differing, [], f"{len(differing)} parameters changed across a save and reload")

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


class Qwen3GgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    gguf_repo = "unsloth/Qwen3-0.6B-GGUF"
    quantized_gguf_file = "Qwen3-0.6B-Q8_0.gguf"
    reference_repo = "Qwen/Qwen3-0.6B"


@slow
class LlamaGgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    gguf_repo = "unsloth/Llama-3.1-8B-Instruct-GGUF"
    quantized_gguf_file = "Llama-3.1-8B-Instruct-Q4_K_M.gguf"
    reference_repo = "meta-llama/Llama-3.1-8B-Instruct"


@slow
class Qwen25GgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    gguf_repo = "Qwen/Qwen2.5-7B-Instruct-GGUF"
    quantized_gguf_file = "qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
    reference_repo = "Qwen/Qwen2.5-7B-Instruct"


@slow
class MistralGgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    """Tekken: byte-level, and written as `llama` like the sentencepiece Mistrals before it."""

    gguf_repo = "bartowski/Ministral-8B-Instruct-2410-GGUF"
    quantized_gguf_file = "Ministral-8B-Instruct-2410-Q4_K_M.gguf"
    reference_repo = "mistralai/Ministral-8B-Instruct-2410"


@slow
class TinyLlamaGgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    """The sentencepiece side of `llama`, which the Mistral case covered until it went byte-level."""

    gguf_repo = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
    quantized_gguf_file = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    reference_repo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


@slow
class Phi3GgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    gguf_repo = "bartowski/Phi-3.5-mini-instruct-GGUF"
    quantized_gguf_file = "Phi-3.5-mini-instruct-Q4_K_M.gguf"
    reference_repo = "microsoft/Phi-3.5-mini-instruct"


@slow
class T5GgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    gguf_repo = "Felladrin/gguf-LaMini-Flan-T5-248M"
    quantized_gguf_file = "LaMini-Flan-T5-248M.Q8_0.gguf"
    reference_repo = "MBZUAI/LaMini-Flan-T5-248M"


@slow
class Gemma4GgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    """`tokenizer.ggml.model = "gemma4"`: a kind of its own, and a BPE carrying its own merges."""

    gguf_repo = "unsloth/gemma-4-E4B-it-GGUF"
    quantized_gguf_file = "gemma-4-E4B-it-Q4_K_M.gguf"
    reference_repo = "google/gemma-4-E4B-it"


@slow
class GemmaGgufTokenizerTest(GgufTokenizerTesterMixin, unittest.TestCase):
    gguf_repo = "unsloth/gemma-3-1b-it-GGUF"
    quantized_gguf_file = "gemma-3-1b-it-Q4_K_M.gguf"
    reference_repo = "google/gemma-3-1b-it"
