# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch ESM model."""

import inspect
import tempfile
import unittest

import numpy as np
from parameterized import parameterized

from transformers import EsmConfig, is_torch_available
from transformers.testing_utils import TestCasePlus, is_flaky, require_torch, require_torch_sdpa, slow, torch_device
from transformers.utils import is_torch_bf16_available_on_device, is_torch_fp16_available_on_device

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers.models.esm.modeling_esmfold import EsmForProteinFolding


class EsmFoldModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=False,
        vocab_size=19,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        esmfold_config = {
            "trunk": {
                "num_blocks": 2,
                "sequence_state_dim": 64,
                "pairwise_state_dim": 16,
                "sequence_head_width": 4,
                "pairwise_head_width": 4,
                "position_bins": 4,
                "chunk_size": 16,
                "structure_module": {
                    "ipa_dim": 16,
                    "num_angles": 7,
                    "num_blocks": 2,
                    "num_heads_ipa": 4,
                    "pairwise_dim": 16,
                    "resnet_dim": 16,
                    "sequence_dim": 48,
                },
            },
            "fp16_esm": False,
            "lddt_head_hid_dim": 16,
        }
        config = EsmConfig(
            vocab_size=33,
            hidden_size=self.hidden_size,
            pad_token_id=1,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            is_folding_model=True,
            esmfold_config=esmfold_config,
        )
        return config

    def create_and_check_model(self, config, input_ids, input_mask, sequence_labels, token_labels, choice_labels):
        model = EsmForProteinFolding(config=config).float()
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        result = model(input_ids)

        self.parent.assertEqual(result.positions.shape, (2, self.batch_size, self.seq_length, 14, 3))
        self.parent.assertEqual(result.angles.shape, (2, self.batch_size, self.seq_length, 7, 2))

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class EsmFoldModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    test_mismatched_shapes = False

    all_model_classes = (EsmForProteinFolding,) if is_torch_available() else ()
    pipeline_model_mapping = {} if is_torch_available() else {}
    test_sequence_classification_problem_types = False

    def setUp(self):
        self.model_tester = EsmFoldModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EsmConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @is_flaky(
        description="The computed `s = s / norm_denom` in `EsmFoldAngleResnet` is numerically instable if `norm_denom` is very small."
    )
    def test_batching_equivalence(self):
        super().test_batching_equivalence()

    @unittest.skip(reason="Does not support attention outputs")
    def test_attention_outputs(self):
        pass

    @unittest.skip
    def test_correct_missing_keys(self):
        pass

    @unittest.skip(reason="Esm does not support embedding resizing")
    def test_resize_embeddings_untied(self):
        pass

    @unittest.skip(reason="Esm does not support embedding resizing")
    def test_resize_tokens_embeddings(self):
        pass

    @unittest.skip(reason="ESMFold does not support passing input embeds!")
    def test_inputs_embeds(self):
        pass

    @unittest.skip(reason="ESMFold does not support head pruning.")
    def test_head_pruning(self):
        pass

    @unittest.skip(reason="ESMFold does not support head pruning.")
    def test_head_pruning_integration(self):
        pass

    @unittest.skip(reason="ESMFold does not support head pruning.")
    def test_head_pruning_save_load_from_config_init(self):
        pass

    @unittest.skip(reason="ESMFold does not support head pruning.")
    def test_head_pruning_save_load_from_pretrained(self):
        pass

    @unittest.skip(reason="ESMFold does not support head pruning.")
    def test_headmasking(self):
        pass

    @unittest.skip(reason="ESMFold does not output hidden states in the normal way.")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="ESMfold does not output hidden states in the normal way.")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="ESMFold only has one output format.")
    def test_model_outputs_equivalence(self):
        pass

    @unittest.skip(reason="ESMFold does not support input chunking.")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(
        reason="ESMFold doesn't respect you and it certainly doesn't respect your initialization arguments."
    )
    def test_initialization(self):
        pass

    @unittest.skip(reason="ESMFold doesn't support torchscript compilation.")
    def test_torchscript_output_attentions(self):
        pass

    @unittest.skip(reason="ESMFold doesn't support torchscript compilation.")
    def test_torchscript_output_hidden_state(self):
        pass

    @unittest.skip(reason="ESMFold doesn't support torchscript compilation.")
    def test_torchscript_simple(self):
        pass

    @unittest.skip(reason="ESMFold doesn't support data parallel.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="ESMFold does not directly implement SDPA, but relies on the ESM model.")
    @require_torch_sdpa
    def test_sdpa_can_dispatch_non_composite_models(self):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip(reason="Common test cases are not supported for ESMFold, use custom implementation instead.")
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference(
        self, name, torch_dtype, padding_side, use_attention_mask, output_attentions, enable_kernels
    ):
        pass

    @parameterized.expand([("float16",), ("bfloat16",), ("float32",)])
    @require_torch_sdpa
    def test_eager_matches_sdpa_inference_esmfold(self, torch_dtype: str):
        if torch_dtype == "float16" and not is_torch_fp16_available_on_device(torch_device):
            self.skipTest(f"float16 not supported on {torch_device} (on the specific device currently used)")

        if torch_dtype == "bfloat16" and not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(
                f"bfloat16 not supported on {torch_device} (on the specific device currently used, e.g. Nvidia T4 GPU)"
            )

        # Not sure whether it's fine to put torch.XXX in a decorator if torch is not available so hacking it here instead.
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        atols = {
            ("cpu", False, torch.float32): 1e-6,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-6,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-6,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-6,
            ("cuda", True, torch.bfloat16): 1e-2,
            ("cuda", True, torch.float16): 5e-3,
        }
        rtols = {
            ("cpu", False, torch.float32): 1e-4,
            ("cpu", False, torch.float16): 5e-3,
            ("cpu", False, torch.bfloat16): 1e-2,
            ("cpu", True, torch.float32): 1e-4,
            ("cpu", True, torch.float16): 5e-3,
            ("cpu", True, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float32): 1e-4,
            ("cuda", False, torch.bfloat16): 1e-2,
            ("cuda", False, torch.float16): 5e-3,
            ("cuda", True, torch.float32): 1e-4,
            ("cuda", True, torch.bfloat16): 3e-2,
            ("cuda", True, torch.float16): 5e-3,
        }

        def get_mean_reldiff(failcase, x, ref, atol, rtol):
            return f"{failcase}: mean relative difference: {((x - ref).abs() / (ref.abs() + 1e-12)).mean():.3e}, torch atol = {atol}, torch rtol = {rtol}"

        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        config.layer_norm_eps = 1.0

        for model_class in self.all_model_classes:
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Note: the half precision will only be applied to backbone model
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    attn_implementation="eager",
                )
                model_eager = model_eager.eval().to(torch_device)

                model_sdpa.esm.to(torch_dtype)
                model_eager.esm.to(torch_dtype)

                # We use these for loops instead of parameterized.expand just for the interest of avoiding loading/saving 16 times the model,
                # but it would be nicer to have an efficient way to use parameterized.expand
                fail_cases = []
                for padding_side in ["left", "right"]:
                    for use_mask in [False, True]:
                        # TODO: if we can also check with `batch_size=1` without being flaky?
                        for batch_size in [7]:
                            dummy_input = inputs_dict[model.main_input_name]

                            if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                dummy_input = dummy_input.to(torch_dtype)

                            dummy_input = dummy_input[:batch_size]
                            if dummy_input.shape[0] != batch_size:
                                if dummy_input.dtype in [torch.float32, torch.bfloat16, torch.float16]:
                                    extension = torch.rand(
                                        batch_size - dummy_input.shape[0],
                                        *dummy_input.shape[1:],
                                        dtype=torch_dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)
                                else:
                                    extension = torch.randint(
                                        high=5,
                                        size=(batch_size - dummy_input.shape[0], *dummy_input.shape[1:]),
                                        dtype=dummy_input.dtype,
                                        device=torch_device,
                                    )
                                    dummy_input = torch.cat((dummy_input, extension), dim=0).to(torch_device)

                            if not use_mask:
                                dummy_attention_mask = None
                            else:
                                dummy_attention_mask = inputs_dict.get("attention_mask", None)
                                if dummy_attention_mask is None:
                                    seqlen = dummy_input.shape[-1]
                                    dummy_attention_mask = (
                                        torch.ones(batch_size, seqlen).to(torch.int64).to(torch_device)
                                    )

                                dummy_attention_mask = dummy_attention_mask[:batch_size]
                                if dummy_attention_mask.shape[0] != batch_size:
                                    extension = torch.ones(
                                        batch_size - dummy_attention_mask.shape[0],
                                        *dummy_attention_mask.shape[1:],
                                        dtype=dummy_attention_mask.dtype,
                                        device=torch_device,
                                    )
                                    dummy_attention_mask = torch.cat((dummy_attention_mask, extension), dim=0)
                                    dummy_attention_mask = dummy_attention_mask.to(torch_device)

                                dummy_attention_mask[:] = 1
                                if padding_side == "left":
                                    dummy_attention_mask[-1, :2] = 0
                                    dummy_attention_mask[-1, 2:] = 1
                                elif padding_side == "right":
                                    dummy_attention_mask[-1, -2:] = 0
                                    dummy_attention_mask[-1, :-2] = 1

                            for enable_kernels in [False, True]:
                                failcase = f"padding_side={padding_side}, use_mask={use_mask}, enable_kernels={enable_kernels}"
                                processed_inputs = {
                                    model.main_input_name: dummy_input,
                                }

                                # Otherwise fails for e.g. WhisperEncoderModel
                                if "attention_mask" in inspect.signature(model_eager.forward).parameters:
                                    processed_inputs["attention_mask"] = dummy_attention_mask

                                # TODO: test gradients as well (& for FA2 as well!)
                                with torch.no_grad():
                                    with torch.backends.cuda.sdp_kernel(
                                        enable_flash=enable_kernels,
                                        enable_math=True,
                                        enable_mem_efficient=enable_kernels,
                                    ):
                                        prepared_inputs = self._prepare_for_class(processed_inputs, model_class)
                                        outputs_eager = model_eager(**prepared_inputs)
                                        outputs_sdpa = model_sdpa(**prepared_inputs)

                                logits_eager = outputs_eager.lm_logits
                                logits_sdpa = outputs_sdpa.lm_logits

                                if torch_device in ["cpu", "cuda"]:
                                    atol = atols[torch_device, enable_kernels, torch_dtype]
                                    rtol = rtols[torch_device, enable_kernels, torch_dtype]
                                else:
                                    atol = 1e-7
                                    rtol = 1e-4

                                # Masked tokens output slightly deviates - we don't mind that.
                                if use_mask:
                                    _logits_sdpa = torch.zeros_like(input=logits_sdpa)
                                    _logits_eager = torch.zeros_like(input=logits_eager)

                                    _logits_sdpa[:-1] = logits_sdpa[:-1]
                                    _logits_eager[:-1] = logits_eager[:-1]

                                    if padding_side == "left":
                                        _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, 2:]
                                        _logits_eager[-1:, 2:] = logits_eager[-1:, 2:]

                                    elif padding_side == "right":
                                        _logits_sdpa[-1:, 2:] = logits_sdpa[-1:, :-2]
                                        _logits_eager[-1:, 2:] = logits_eager[-1:, :-2]

                                    logits_sdpa = _logits_sdpa
                                    logits_eager = _logits_eager

                                results = [
                                    torch.allclose(_logits_sdpa, _logits_eager, atol=atol, rtol=rtol)
                                    for (_logits_sdpa, _logits_eager) in zip(logits_sdpa, logits_eager)
                                ]
                                # If 80% batch elements have matched results, it's fine
                                if np.mean(results) < 0.8:
                                    fail_cases.append(
                                        get_mean_reldiff(failcase, logits_sdpa, logits_eager, atol, rtol)
                                    )

                self.assertTrue(len(fail_cases) == 0, "\n".join(fail_cases))


@require_torch
class EsmModelIntegrationTest(TestCasePlus):
    @slow
    def test_inference_protein_folding(self):
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").float()
        model.eval()
        input_ids = torch.tensor([[0, 6, 4, 13, 5, 4, 16, 12, 11, 7, 2]])
        position_outputs = model(input_ids)["positions"]
        expected_slice = torch.tensor([2.5828, 0.7993, -10.9334], dtype=torch.float32)
        torch.testing.assert_close(position_outputs[0, 0, 0, 0], expected_slice, rtol=1e-4, atol=1e-4)
