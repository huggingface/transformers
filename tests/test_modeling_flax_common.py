# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import random
import tempfile

import numpy as np

import transformers
from transformers import is_flax_available, is_torch_available
from transformers.testing_utils import require_flax, require_torch


if is_flax_available():
    import os

    import jax
    import jax.numpy as jnp
    from transformers.modeling_flax_utils import convert_state_dict_from_pt

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.12"  # assumed parallelism: 8

if is_torch_available():
    import torch


def ids_tensor(shape, vocab_size, rng=None):
    """Creates a random int32 tensor of the shape within the vocab size."""
    if rng is None:
        rng = random.Random()

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    output = np.array(values, dtype=jnp.int32).reshape(shape)

    return output


def random_attention_mask(shape, rng=None):
    attn_mask = ids_tensor(shape, vocab_size=2, rng=rng)
    # make sure that at least one token is attended to for each batch
    attn_mask[:, -1] = 1
    return attn_mask


@require_flax
class FlaxModelTesterMixin:
    model_tester = None
    all_model_classes = ()

    def assert_almost_equals(self, a: np.ndarray, b: np.ndarray, tol: float):
        diff = np.abs((a - b)).max()
        self.assertLessEqual(diff, tol, f"Difference between torch and flax is {diff} (>= {tol}).")

    @require_torch
    def test_equivalence_flax_pytorch(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                pt_model_class_name = model_class.__name__[4:]  # Skip the "Flax" at the beginning
                pt_model_class = getattr(transformers, pt_model_class_name)
                pt_model = pt_model_class(config).eval()

                fx_state = convert_state_dict_from_pt(model_class, pt_model.state_dict(), config)
                fx_model = model_class(config, dtype=jnp.float32)
                fx_model.params = fx_state

                pt_inputs = {k: torch.tensor(v.tolist()) for k, v in inputs_dict.items()}

                with torch.no_grad():
                    pt_outputs = pt_model(**pt_inputs).to_tuple()

                fx_outputs = fx_model(**inputs_dict)
                self.assertEqual(len(fx_outputs), len(pt_outputs), "Output lengths differ between Flax and PyTorch")
                for fx_output, pt_output in zip(fx_outputs, pt_outputs):
                    self.assert_almost_equals(fx_output, pt_output.numpy(), 1e-3)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    pt_model.save_pretrained(tmpdirname)
                    fx_model_loaded = model_class.from_pretrained(tmpdirname, from_pt=True)

                fx_outputs_loaded = fx_model_loaded(**inputs_dict)
                self.assertEqual(
                    len(fx_outputs_loaded), len(pt_outputs), "Output lengths differ between Flax and PyTorch"
                )
                for fx_output_loaded, pt_output in zip(fx_outputs_loaded, pt_outputs):
                    self.assert_almost_equals(fx_output_loaded, pt_output.numpy(), 5e-3)

    def test_from_pretrained_save_pretrained(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(config)

                outputs = model(**inputs_dict)

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)
                    model_loaded = model_class.from_pretrained(tmpdirname)

                outputs_loaded = model_loaded(**inputs_dict)
                for output_loaded, output in zip(outputs_loaded, outputs):
                    self.assert_almost_equals(output_loaded, output, 5e-3)

    def test_jit_compilation(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(config)

                @jax.jit
                def model_jitted(input_ids, attention_mask=None, token_type_ids=None):
                    return model(input_ids, attention_mask, token_type_ids)

                with self.subTest("JIT Disabled"):
                    with jax.disable_jit():
                        outputs = model_jitted(**inputs_dict)

                with self.subTest("JIT Enabled"):
                    jitted_outputs = model_jitted(**inputs_dict)

                self.assertEqual(len(outputs), len(jitted_outputs))
                for jitted_output, output in zip(jitted_outputs, outputs):
                    self.assertEqual(jitted_output.shape, output.shape)

    def test_naming_convention(self):
        for model_class in self.all_model_classes:
            model_class_name = model_class.__name__
            module_class_name = (
                model_class_name[:-5] + "Module" if model_class_name[-5:] == "Model" else model_class_name + "Module"
            )
            bert_modeling_flax_module = __import__(model_class.__module__, fromlist=[module_class_name])
            module_cls = getattr(bert_modeling_flax_module, module_class_name)

            self.assertIsNotNone(module_cls)
