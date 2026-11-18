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
"""Clef on a backbone defined by its checkpoint's own code, loaded only with `trust_remote_code=True`."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from huggingface_hub import hf_hub_download

from transformers import AutoConfig, AutoTokenizer, BertConfig, ClefConfig, is_torch_available, pipeline
from transformers.decision import Question
from transformers.testing_utils import require_torch

from ...test_decision_head_common import render


if is_torch_available():
    import torch
    from torch import nn

    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForDecision, ClefForDecision


# A package like `MinimaLabs/min-spark-1.7`: a causal language model and nothing else in its `auto_map` (no
# `AutoModel`), whose inner model is a plain `nn.Module`, whose `forward` takes no `**kwargs` nor `position_ids` and never
# returns hidden states (they are the output of `model.final_norm`), whose config has no `hidden_size` (`dim`), with a
# non-persistent buffer rebuilt on the first forward and right padding only
CAUSAL_PACKAGE = {
    "configuration_tinyspark.py": """
        from transformers import PretrainedConfig


        class TinySparkConfig(PretrainedConfig):
            model_type = "tinyspark"

            def __init__(self, vocab_size=32, dim=24, n_layers=2, max_seq_len=1024, **kwargs):
                self.vocab_size = vocab_size
                self.dim = dim
                self.n_layers = n_layers
                self.max_seq_len = max_seq_len
                self.tie_word_embeddings = True
                super().__init__(**kwargs)
    """,
    "core.py": """
        import torch
        from torch import nn


        class Core(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.embed = nn.Embedding(config.vocab_size, config.dim)
                self.layers = nn.ModuleList(nn.Linear(config.dim, config.dim) for _ in range(config.n_layers))
                self.final_norm = nn.LayerNorm(config.dim)
                self.register_buffer("gain", torch.zeros(config.max_seq_len), persistent=False)

            def forward(self, tokens, return_hidden=False):
                x = self.embed(tokens)
                length = tokens.shape[1]
                causal = torch.ones(length, length, device=x.device).tril()
                causal = causal / causal.sum(-1, keepdim=True)  # the mean of the tokens so far
                for layer in self.layers:
                    x = x + torch.tanh(layer(causal @ x)) * self.gain[:length, None]
                x = self.final_norm(x)
                return x if return_hidden else nn.functional.linear(x, self.embed.weight)
    """,
    "modeling_tinyspark.py": """
        import torch
        from torch import nn
        from transformers import PreTrainedModel
        from transformers.generation import GenerationMixin
        from transformers.modeling_outputs import CausalLMOutputWithPast

        from .configuration_tinyspark import TinySparkConfig
        from .core import Core


        class TinySparkForCausalLM(PreTrainedModel, GenerationMixin):
            config_class = TinySparkConfig
            base_model_prefix = "model"
            supports_gradient_checkpointing = False
            _no_split_modules = []

            def __init__(self, config):
                super().__init__(config)
                self.model = Core(config)
                self.post_init()

            def _init_weights(self, module):
                pass

            def get_input_embeddings(self):
                return self.model.embed

            def set_input_embeddings(self, value):
                self.model.embed = value

            def get_output_embeddings(self):
                return self.model.embed

            def _ensure_buffers(self):
                if not getattr(self, "_buffers_ok", False):
                    self.model.gain.copy_(torch.linspace(1.0, 0.5, self.model.gain.shape[0]))
                    self._buffers_ok = True

            def forward(
                self,
                input_ids,
                attention_mask=None,
                labels=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            ):
                if input_ids.shape[1] > self.config.max_seq_len:
                    raise ValueError("too long")
                if attention_mask is not None and (~attention_mask.bool()[:, 0]).any():
                    raise ValueError("left-padded batches are not supported")
                self._ensure_buffers()
                logits = self.model(input_ids)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())
                return CausalLMOutputWithPast(loss=loss, logits=logits, hidden_states=None)
    """,
}
CAUSAL_AUTO_MAP = {
    "AutoConfig": "configuration_tinyspark.TinySparkConfig",
    "AutoModelForCausalLM": "modeling_tinyspark.TinySparkForCausalLM",
}

# A bidirectional encoder package with a base model in its `auto_map`, which returns `last_hidden_state`
ENCODER_PACKAGE = {
    "configuration_tinyenc.py": """
        from transformers import PretrainedConfig


        class TinyEncConfig(PretrainedConfig):
            model_type = "tinyenc"

            def __init__(self, vocab_size=32, hidden_size=24, num_hidden_layers=2, **kwargs):
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_hidden_layers
                super().__init__(**kwargs)
    """,
    "modeling_tinyenc.py": """
        import torch
        from torch import nn
        from transformers import PreTrainedModel
        from transformers.modeling_outputs import BaseModelOutput

        from .configuration_tinyenc import TinyEncConfig


        class TinyEncModel(PreTrainedModel):
            config_class = TinyEncConfig
            base_model_prefix = "tinyenc"

            def __init__(self, config):
                super().__init__(config)
                self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
                self.layers = nn.ModuleList(
                    nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)
                )
                self.norm = nn.LayerNorm(config.hidden_size)
                self.post_init()

            def _init_weights(self, module):
                pass

            def get_input_embeddings(self):
                return self.embeddings

            def set_input_embeddings(self, value):
                self.embeddings = value

            def forward(self, input_ids, attention_mask=None, **kwargs):
                x = self.embeddings(input_ids)
                mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
                mask = mask.to(x.dtype)[..., None]
                for layer in self.layers:  # every token reads the mean of the attended ones
                    context = (x * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True)
                    x = x + torch.tanh(layer(x + context))
                return BaseModelOutput(last_hidden_state=self.norm(x))
    """,
}
ENCODER_AUTO_MAP = {"AutoConfig": "configuration_tinyenc.TinyEncConfig", "AutoModel": "modeling_tinyenc.TinyEncModel"}

HEAD_SIZES = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_evidence_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "cosine_width": 8,
}


def write_package(directory, kind, vocab_size):
    """The code and `config.json` of a synthetic package in `directory`, without weights."""
    files, auto_map = (CAUSAL_PACKAGE, CAUSAL_AUTO_MAP) if kind == "causal" else (ENCODER_PACKAGE, ENCODER_AUTO_MAP)
    for name, code in files.items():
        Path(directory, name).write_text(textwrap.dedent(code).lstrip(), encoding="utf-8")
    model_type = "tinyspark" if kind == "causal" else "tinyenc"
    sizes = {"dim": 24, "n_layers": 2} if kind == "causal" else {"hidden_size": 24, "num_hidden_layers": 2}
    config = {"model_type": model_type, "auto_map": auto_map, "vocab_size": vocab_size, **sizes}
    Path(directory, "config.json").write_text(json.dumps(config), encoding="utf-8")


def package_modules(directory) -> list[str]:
    """The imported modules of the dynamic module of the folder `directory`."""
    name = Path(directory).name.replace("-", "_").replace(".", "_")
    return [module for module in sys.modules if module.startswith("transformers_modules.") and name in module]


@require_torch
class ClefRemoteBackboneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test = json.loads(
            Path(hf_hub_download("hf-internal-testing/decision-clef", "decision_test.json")).read_bytes()
        )
        cls.request = test["request"]
        cls.tokenizer = AutoTokenizer.from_pretrained(test["model"], revision=test["revision"])
        cls.encoder_tokenizer = copy.deepcopy(cls.tokenizer)
        cls.encoder_tokenizer.add_special_tokens({"cls_token": "[CLS]", "sep_token": "[SEP]"})

    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name)

    def tearDown(self):
        self._directory.cleanup()

    def folder(self, name):
        return tempfile.mkdtemp(prefix=f"{name}-", dir=self.root)  # a new module name for its dynamic module

    def package(self, kind):
        """A synthetic package with random weights in a new folder, and its backbone loaded from it."""
        tokenizer = self.tokenizer if kind == "causal" else self.encoder_tokenizer
        source = self.folder(f"{kind}-package")
        write_package(source, kind, len(tokenizer))
        auto_class = AutoModelForCausalLM if kind == "causal" else AutoModel
        config = AutoConfig.from_pretrained(source, trust_remote_code=True)
        torch.manual_seed(0)
        backbone = auto_class.from_config(config, trust_remote_code=True)
        with torch.no_grad():
            for parameter in backbone.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.5)
        backbone.save_pretrained(source)
        tokenizer.save_pretrained(source)
        return source, auto_class.from_pretrained(source, trust_remote_code=True).eval()

    def clef_config(self, backbone, kind, head_type):
        if kind == "causal":
            style = {"global_pooling": "last", "prompt_style": "chatml", "backbone_hidden_module": "model.final_norm"}
            size = backbone.config.dim
        else:
            style = {"global_pooling": "first", "prompt_style": "plain"}
            size = backbone.config.hidden_size
        return ClefConfig(
            text_config=backbone.config, backbone_hidden_size=size, head_type=head_type, **HEAD_SIZES, **style
        )

    def model(self, kind="causal", head_type="joint"):
        source, backbone = self.package(kind)
        return source, ClefForDecision(self.clef_config(backbone, kind, head_type), model=backbone)

    def inputs(self, model, kind):
        """A batch of the prompts of the request rendered twice, right-padded (the causal package refuses left)."""
        tokenizer = self.tokenizer if kind == "causal" else self.encoder_tokenizer
        (prompt,) = render(tokenizer, self.request, model.decision_config)
        prompts = [prompt.inputs, {**prompt.inputs, "input_ids": prompt.inputs["input_ids"][:, :]}]
        length = prompt.inputs["input_ids"].shape[-1]
        input_ids = torch.cat([p["input_ids"] for p in prompts])
        input_ids = nn.functional.pad(input_ids, (0, 3), value=0)  # right padding
        attention_mask = torch.cat([torch.ones(2, length), torch.zeros(2, 3)], dim=-1).long()
        questions = prompt.questions
        num_options = torch.tensor([[len(q.options) for q in questions]] * 2)
        types = torch.tensor([[Question.questions_types.index(q.type) for q in questions]] * 2)
        labels = torch.zeros(2 * len(questions), dtype=torch.long)
        marks = torch.stack([prompt.inputs["marks"]] * 2)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "marks": marks,
            "num_options": num_options,
            "question_type_ids": types,
        }, labels

    def save(self, model, kind):
        folder = self.folder(f"saved-{kind}-{model.config.head_type}")
        model.save_pretrained(folder)
        (self.tokenizer if kind == "causal" else self.encoder_tokenizer).save_pretrained(folder)
        return folder

    def test_build_forward_backward(self):
        for kind in ("causal", "encoder"):
            for head_type in ("joint", "cosine"):
                with self.subTest(kind=kind, head_type=head_type):
                    _, model = self.model(kind, head_type)
                    model.train()
                    inputs, labels = self.inputs(model, kind)
                    output = model(**inputs, labels=labels)
                    self.assertTrue(torch.isfinite(output.loss))
                    output.loss.backward()
                    backbone_grads = [p.grad for p in model.model.parameters() if p.grad is not None]
                    head_grads = [p.grad for n, p in model.named_parameters() if not n.startswith("model.")]
                    self.assertTrue(backbone_grads and all(g is not None for g in head_grads))
                    self.assertTrue(any(g.abs().sum() > 0 for g in backbone_grads))
                    # Nothing reads the config of the backbone for a size: it has no `hidden_size`
                    if kind == "causal":
                        self.assertFalse(hasattr(model.config.text_config, "hidden_size"))
                        self.assertFalse(hasattr(model, "lm_head"))
                        self.assertIs(model.lexical_embeddings(), model.model.get_output_embeddings().weight)

    def test_causal_backbone_hidden_states(self):
        _, backbone = self.package("causal")
        # Found and verified: the output of `model.final_norm` gives the logits through the output embeddings
        self.assertEqual(ClefForDecision.find_backbone_hidden_module(backbone), "model.final_norm")
        # Nothing verifies against other output embeddings
        other = nn.Embedding(*backbone.get_output_embeddings().weight.shape)
        with mock.patch.object(backbone, "get_output_embeddings", return_value=other):
            with self.assertRaisesRegex(ValueError, "No submodule"):
                ClefForDecision.find_backbone_hidden_module(backbone)
        # The hidden states the head reads are those of the module, the backbone called with what it accepts
        model = ClefForDecision(self.clef_config(backbone, "causal", "cosine"), model=backbone).eval()
        input_ids = torch.tensor([[5, 6, 7, 8]])
        with torch.no_grad():
            expected = backbone.model(input_ids, return_hidden=True)
            hidden = model.backbone_hidden_states(input_ids, position_ids=input_ids, use_cache=False, foo=1)
        torch.testing.assert_close(hidden, expected, rtol=0, atol=0)
        # Without the module the backbone gives no hidden states: an error that says what to do
        config = self.clef_config(backbone, "causal", "cosine")
        config.backbone_hidden_module = None
        with self.assertRaisesRegex(ValueError, "backbone_hidden_module"):
            ClefForDecision(config, model=backbone).backbone_hidden_states(input_ids)
        config.backbone_hidden_module = "model.nothing"
        with self.assertRaisesRegex(ValueError, "not a submodule"):
            ClefForDecision(config, model=backbone)

    def test_save_reload_with_the_flag(self):
        for kind in ("causal", "encoder"):
            for head_type in ("joint", "cosine"):
                with self.subTest(kind=kind, head_type=head_type):
                    _, model = self.model(kind, head_type)
                    model.eval()
                    inputs, _ = self.inputs(model, kind)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    folder = self.save(model, kind)
                    saved = json.loads(Path(folder, "config.json").read_text(encoding="utf-8"))
                    auto_map = CAUSAL_AUTO_MAP if kind == "causal" else ENCODER_AUTO_MAP
                    self.assertEqual(saved["text_config"]["auto_map"], auto_map)
                    for name in CAUSAL_PACKAGE if kind == "causal" else ENCODER_PACKAGE:
                        self.assertTrue(Path(folder, name).is_file(), name)  # self-contained
                    reloaded, info = ClefForDecision.from_pretrained(
                        folder, trust_remote_code=True, output_loading_info=True
                    )
                    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
                        self.assertFalse(info[key], key)
                    auto = AutoModelForDecision.from_pretrained(folder, trust_remote_code=True)
                    decide = pipeline("decision", model=folder, trust_remote_code=True, device="cpu")
                    for loaded in (reloaded, auto, decide.model):
                        self.assertIsInstance(loaded, ClefForDecision)
                        self.assertEqual(type(loaded.model).__name__, type(model.model).__name__)
                        with torch.no_grad():
                            torch.testing.assert_close(loaded.eval()(**inputs).logits, logits, rtol=0, atol=0)
                    answers = decide(self.request)
                    self.assertEqual(set(answers), set(self.request["questions"]))
                    # A saved model saves again, self-contained
                    again = self.folder(f"again-{kind}-{head_type}")
                    reloaded.save_pretrained(again)
                    for name in CAUSAL_PACKAGE if kind == "causal" else ENCODER_PACKAGE:
                        self.assertTrue(Path(again, name).is_file(), name)

    def test_without_the_flag_nothing_runs(self):
        # In a new process, where nothing of the package was ever imported or registered
        _, model = self.model("causal", "cosine")
        model.eval()
        inputs, _ = self.inputs(model, "causal")
        with torch.no_grad():
            logits = model(**inputs).logits
        folder = self.save(model, "causal")
        torch.save({"inputs": inputs, "logits": logits}, self.root / "expected.pt")
        script = textwrap.dedent(
            f"""
            import sys
            import torch
            from transformers import AutoConfig, AutoModelForDecision, ClefForDecision, pipeline

            folder, name = {folder!r}, {Path(folder).name.replace("-", "_")!r}

            def imported():
                return [m for m in sys.modules if m.startswith("transformers_modules") or "tinyspark" in m]

            # The config reads without running anything: the config of the backbone stays a dict
            config = AutoConfig.from_pretrained(folder)
            assert isinstance(config.text_config, dict), type(config.text_config)
            loaders = {{
                "ClefForDecision": lambda **kwargs: ClefForDecision.from_pretrained(folder, **kwargs),
                "AutoModelForDecision": lambda **kwargs: AutoModelForDecision.from_pretrained(folder, **kwargs),
                "pipeline": lambda **kwargs: pipeline("decision", model=folder, **kwargs),
                "ClefForDecision(config)": lambda **kwargs: ClefForDecision(config),
            }}
            for loader, load in loaders.items():
                for flag in ({{}}, {{"trust_remote_code": False}}):
                    try:
                        load(**flag)
                    except ValueError as error:
                        message = str(error)
                        assert "trust_remote_code=True" in message, (loader, message)
                        if loader != "ClefForDecision(config)":
                            assert folder in message and "configuration_tinyspark.TinySparkConfig" in message, message
                    else:
                        raise AssertionError(f"{{loader}} {{flag}} loaded")
                    assert not imported(), (loader, imported())
            # With the flag, the same process loads it
            expected = torch.load({str(self.root / "expected.pt")!r})
            model = ClefForDecision.from_pretrained(folder, trust_remote_code=True).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(**expected["inputs"]).logits, expected["logits"], rtol=0, atol=0)
            assert imported()
            print("OK")
            """
        )
        modules = self.root / "modules"
        environment = {**os.environ, "HF_MODULES_CACHE": str(modules), "TRANSFORMERS_VERBOSITY": "error"}
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, env=environment, stdin=subprocess.DEVNULL
        )
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        self.assertIn("OK", result.stdout)

    def test_without_the_flag_after_registration(self):
        # In this process, where the classes of the package are registered: still refused without the flag
        _, model = self.model("encoder", "joint")
        folder = self.save(model, "encoder")
        ClefForDecision.from_pretrained(folder, trust_remote_code=True)
        for flag in ({}, {"trust_remote_code": False}):
            with self.assertRaisesRegex(ValueError, "trust_remote_code=True"):
                ClefForDecision.from_pretrained(folder, **flag)
            with self.assertRaisesRegex(ValueError, "trust_remote_code=True"):
                AutoModelForDecision.from_pretrained(folder, **flag)

    def test_backbone_from_the_hub_keeps_its_repository(self):
        # A backbone loaded from the Hub (its `_name_or_path` a repository id): the references name the repository and
        # no code is copied; the config of the model in memory is unchanged
        _, model = self.model("causal", "cosine")
        model.config.text_config._name_or_path = "someone/tiny-spark"
        folder = self.save(model, "causal")
        saved = json.loads(Path(folder, "config.json").read_text(encoding="utf-8"))["text_config"]["auto_map"]
        self.assertEqual(saved, {key: f"someone/tiny-spark--{value}" for key, value in CAUSAL_AUTO_MAP.items()})
        self.assertEqual(model.config.text_config.auto_map, CAUSAL_AUTO_MAP)
        self.assertFalse(list(Path(folder).glob("*.py")))
        with self.assertRaisesRegex(ValueError, "from someone/tiny-spark.*trust_remote_code=True"):
            ClefForDecision.from_pretrained(folder)

    def test_backbone_class_must_be_the_rebuilt_one(self):
        # A package with both: the given causal model is not what `AutoModel` of its `auto_map` rebuilds
        _, backbone = self.package("causal")
        config = self.clef_config(backbone, "causal", "cosine")
        config.text_config.auto_map = {**CAUSAL_AUTO_MAP, "AutoModel": "modeling_tinyspark.TinySparkModel"}
        with self.assertRaisesRegex(ValueError, "load the backbone with AutoModel"):
            ClefForDecision(config, model=backbone)

    def test_native_backbone_untouched(self):
        # No remote code is looked for, with or without the flag
        tokenizer = self.encoder_tokenizer
        backbone_config = BertConfig(
            vocab_size=len(tokenizer), hidden_size=24, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2
        )
        torch.manual_seed(0)
        backbone = AutoModel.from_config(backbone_config).eval()
        config = ClefConfig(
            text_config=backbone.config,
            backbone_hidden_size=24,
            global_pooling="first",
            prompt_style="plain",
            attn_implementation="eager",
            **HEAD_SIZES,
        )
        model = ClefForDecision(config, model=backbone).eval()
        inputs, _ = self.inputs(model, "encoder")
        with torch.no_grad():
            logits = model(**inputs).logits
        folder = self.save(model, "encoder")
        self.assertFalse(list(Path(folder).glob("*.py")))
        refuse = mock.Mock(side_effect=AssertionError("remote code looked up"))
        with mock.patch("transformers.models.clef.modeling_clef.get_class_from_dynamic_module", refuse):
            for flag in ({}, {"trust_remote_code": True}, {"trust_remote_code": False}):
                loaded = ClefForDecision.from_pretrained(folder, attn_implementation="eager", **flag).eval()
                with torch.no_grad():
                    torch.testing.assert_close(loaded(**inputs).logits, logits, rtol=0, atol=0)
        refuse.assert_not_called()

    def test_raw_head_next_to_remote_backbone_is_refused(self):
        # The layout of a clef checkpoint (`joint_head_config.json` next to the weights) on a backbone of its own code
        source, _ = self.package("causal")
        Path(source, "joint_head_config.json").write_text(json.dumps({}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "defined by its own code"):
            ClefForDecision.from_pretrained(source, trust_remote_code=True)
