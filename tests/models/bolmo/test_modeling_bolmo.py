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
"""Testing suite for the PyTorch Bolmo model."""

import unittest

from parameterized import parameterized

from transformers import BolmoConfig, is_torch_available
from transformers.testing_utils import require_torch, slow, torch_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION, ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import AutoTokenizer, BolmoForCausalLM, BolmoModel
    from transformers.models.bolmo.modeling_bolmo import BolmoCache, BolmoMLSTMLayer


class BolmoModelTester:
    def __init__(self, parent, batch_size=2, seq_length=16):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = True
        self.hidden_size = 32
        self.num_hidden_layers = 2
        self.num_attention_heads = 4
        self.vocab_size = 520

    def get_config(self):
        return BolmoConfig(
            hidden_size=self.hidden_size,
            intermediate_size=64,
            local_intermediate_size=48,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_local_heads=4,
            num_local_decoder_layers=2,
            subword_vocab_size=99,
            sliding_window=8,
            layer_types=["sliding_attention", "full_attention"],
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.vocab_size // 2 - 4) + 4
        inputs_dict = {
            "input_ids": input_ids,
            "expanded_input_ids": ids_tensor([self.batch_size, self.seq_length], config.subword_vocab_size),
            "attention_mask": torch.ones_like(input_ids),
            # one patch per byte, so per-patch outputs have `seq_length` and batches have no patch padding
            "boundary_mask": torch.ones_like(input_ids, dtype=torch.bool),
        }
        return config, inputs_dict


@require_torch
class BolmoModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (BolmoModel, BolmoForCausalLM) if is_torch_available() else ()
    # `generate` needs `tokenizer=` to expand every new byte into a subword id, covered by `test_decode_matches_full_forward`
    all_generative_model_classes = ()
    test_resize_embeddings = False

    def setUp(self):
        self.model_tester = BolmoModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BolmoConfig, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_mlstm_matches_xlstm(self):
        from transformers import xLSTMConfig
        from transformers.models.xlstm.modeling_xlstm import external_xlstm, xLSTMLayer

        if external_xlstm:
            self.skipTest("native xLSTM layer is replaced by the `xlstm` package")
        config = self.model_tester.get_config()
        bolmo_layer = BolmoMLSTMLayer(config).eval()
        xlstm_config = xLSTMConfig(hidden_size=32, num_heads=4, autocast_kernel_dtype="float32", mode="inference")
        xlstm_layer = xLSTMLayer(xlstm_config).eval()
        xlstm_layer.load_state_dict(bolmo_layer.state_dict())
        hidden_states = torch.randn(2, 70, 32)
        torch.testing.assert_close(bolmo_layer(hidden_states)[0], xlstm_layer(hidden_states)[0])

    def test_decode_matches_full_forward(self):
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        model = BolmoForCausalLM(config).to(torch_device).eval()
        input_ids, expanded_input_ids = inputs["input_ids"][:1], inputs["expanded_input_ids"][:1]
        with torch.no_grad():
            boundary_mask = model.model.predict_boundaries(input_ids, expanded_input_ids)
            full_logits = model(input_ids, expanded_input_ids, boundary_mask=boundary_mask).logits
            cache = BolmoCache(config)
            model(
                input_ids[:, :4], expanded_input_ids[:, :4], boundary_mask=boundary_mask[:, :4], past_key_values=cache
            )
            for i in range(4, input_ids.shape[1]):
                # a byte shifted by `vocab_size // 2` closes a patch
                step_ids = input_ids[:, i : i + 1] + config.vocab_size // 2 * boundary_mask[:, i : i + 1]
                step_logits = model(step_ids, expanded_input_ids[:, i : i + 1], past_key_values=cache).logits
                torch.testing.assert_close(step_logits[:, -1], full_logits[:, i])

    @unittest.skip("Bolmo embeds bytes and subwords, `inputs_embeds` alone is not enough")
    def test_inputs_embeds(self):
        pass

    @unittest.skip("Calls the model without `expanded_input_ids`; RoPE is Olmo3's, tested there")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("Calls the model without `expanded_input_ids`; attention is Olmo3's, tested there")
    def test_eager_matches_sdpa_inference(self, *args):
        pass

    @unittest.skip("The boundary predictor is thresholded, it is trained by an auxiliary loss outside of this forward")
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip("The boundary predictor is thresholded, it is trained by an auxiliary loss outside of this forward")
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass

    @unittest.skip("The boundary predictor is thresholded, it is trained by an auxiliary loss outside of this forward")
    def test_training_gradient_checkpointing_use_reentrant_true(self):
        pass


@slow
@require_torch
class BolmoIntegrationTest(unittest.TestCase):
    def test_model_1b_greedy_generation(self):
        # TODO: expected text from the original implementation (`trust_remote_code=True`), run on HF Jobs
        tokenizer = AutoTokenizer.from_pretrained("allenai/Bolmo-1B")
        model = BolmoForCausalLM.from_pretrained("allenai/Bolmo-1B", device_map="auto")
        inputs = tokenizer("Language modeling is ", return_tensors="pt").to(model.device)
        output = model.generate(**inputs, tokenizer=tokenizer, max_new_tokens=32, do_sample=False)
        text = tokenizer.decode(output[0], skip_special_tokens=True)
        self.assertTrue(text.startswith("Language modeling is "))
