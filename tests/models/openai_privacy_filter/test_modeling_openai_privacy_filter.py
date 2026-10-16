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
"""Testing suite for the PyTorch OpenAIPrivacyFilter model."""

import tempfile
import unittest

from parameterized import parameterized

from transformers import (
    OpenAIPrivacyFilterConfig,
    is_torch_available,
)
from transformers.models.openai_privacy_filter.configuration_openai_privacy_filter import (
    OPENAI_PRIVACY_FILTER_NER_LABELS,
)
from transformers.testing_utils import (
    Expectations,
    is_tensor_parallel_test,
    require_torch,
    require_torch_greater_or_equal,
    slow,
    torch_device,
)

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    ModelTesterMixin,
    _test_eager_matches_batched_and_grouped_inference,
    ids_tensor,
    random_attention_mask,
)
from ...test_pipeline_mixin import PipelineTesterMixin
from ...test_tensor_parallel_mixin import _init_distributed


if is_torch_available():
    import torch
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        OpenAIPrivacyFilterForTokenClassification,
        OpenAIPrivacyFilterModel,
    )
    from transformers.distributed import DistributedConfig


def _test_tp_forward_backward(rank, model_path, use_ep):
    model_ref = OpenAIPrivacyFilterForTokenClassification.from_pretrained(model_path)
    model_tp = OpenAIPrivacyFilterForTokenClassification.from_pretrained(
        model_path,
        distributed_config=DistributedConfig(tp_size=dist.get_world_size(), ep_size=dist.get_world_size() if use_ep else 1),
    )
    assert any(
        isinstance(param, DTensor) and any(placement.is_shard() for placement in param.placements)
        for param in model_tp.parameters()
    )
    torch.manual_seed(0)
    input_ids = torch.randint(0, model_ref.config.vocab_size, (2, 8))
    labels = torch.randint(0, model_ref.config.num_labels, input_ids.shape)
    model_ref.eval()
    model_tp.eval()
    with torch.no_grad():
        torch.testing.assert_close(model_tp(input_ids).logits, model_ref(input_ids).logits, atol=1e-5, rtol=1e-5)

    model_ref.train()
    model_tp.train()
    loss_ref = model_ref(input_ids, labels=labels).loss
    loss_tp = model_tp(input_ids, labels=labels).loss
    loss_ref.backward()
    loss_tp.backward()
    torch.testing.assert_close(loss_tp, loss_ref, atol=1e-5, rtol=1e-5)
    reference_params = dict(model_ref.named_parameters())
    mismatched = []
    for name, param in model_tp.named_parameters():
        expected = reference_params[name].grad
        actual = param.grad
        if isinstance(actual, DTensor):
            actual = actual.full_tensor()
        if (actual is None) != (expected is None) or (
            actual is not None and not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
        ):
            mismatched.append(name)
    assert not mismatched, f"Rank {rank}: gradients differ for {mismatched}"


class OpenAIPrivacyFilterModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=99,
        pad_token_id=0,
        hidden_size=32,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        sliding_window=2,
        max_position_embeddings=64,
        initializer_range=0.02,
        num_labels=len(OPENAI_PRIVACY_FILTER_NER_LABELS),
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.sliding_window = sliding_window
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_labels = None
        if self.use_labels:
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()
        return config, input_ids, input_mask, token_labels

    def get_config(self):
        return OpenAIPrivacyFilterConfig(
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            num_local_experts=self.num_local_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            sliding_window=self.sliding_window,
            max_position_embeddings=self.max_position_embeddings,
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 150000.0,
                "factor": 1.0,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "truncate": False,
                "original_max_position_embeddings": self.max_position_embeddings,
            },
            initializer_range=self.initializer_range,
            classifier_dropout=0.0,
            num_labels=self.num_labels,
        )

    def create_and_check_model(self, config, input_ids, input_mask, token_labels):
        model = OpenAIPrivacyFilterModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask)
        result = model(input_ids)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_for_token_classification(self, config, input_ids, input_mask, token_labels):
        config.num_labels = self.num_labels
        model = OpenAIPrivacyFilterForTokenClassification(config=config)
        model.to(torch_device)
        model.eval()
        result = model(input_ids, attention_mask=input_mask, labels=token_labels)
        self.parent.assertEqual(result.logits.shape, (self.batch_size, self.seq_length, self.num_labels))

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, input_mask, _ = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class OpenAIPrivacyFilterModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            OpenAIPrivacyFilterModel,
            OpenAIPrivacyFilterForTokenClassification,
        )
        if is_torch_available()
        else ()
    )
    test_torch_exportable = False  # eager experts (the model's default) iterate over `nonzero()`, untraceable
    pipeline_model_mapping = (
        {
            "feature-extraction": OpenAIPrivacyFilterModel,
            "token-classification": OpenAIPrivacyFilterForTokenClassification,
        }
        if is_torch_available()
        else {}
    )

    def setUp(self):
        self.model_tester = OpenAIPrivacyFilterModelTester(self)
        self.config_tester = ConfigTester(self, config_class=OpenAIPrivacyFilterConfig, hidden_size=32)

    @parameterized.expand([(False,), (True,)])
    @is_tensor_parallel_test
    @require_torch_greater_or_equal("2.9")
    def test_tp_forward_backward(self, use_ep):
        if torch_device != "cpu":
            self.skipTest("Distributed TP/EP mixin tests run on CPU")
        config = self.model_tester.get_config()
        config._attn_implementation = "eager"
        with tempfile.TemporaryDirectory() as model_path:
            torch.manual_seed(42)
            model = OpenAIPrivacyFilterForTokenClassification(config)
            model.save_pretrained(model_path, save_original_format=True)
            _init_distributed(tp=2, backend="gloo")(_test_tp_forward_backward)(model_path, use_ep)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_token_classification_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_token_classification(*config_and_inputs)

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    def test_eager_matches_batched_and_grouped_inference(self, name, dtype):
        if dtype == "bf16":
            self.skipTest("Bf16 may cause biggers fluctuations when used in combination with float casting")
        _test_eager_matches_batched_and_grouped_inference(self, name, dtype)


@slow
@require_torch
class OpenAIPrivacyFilterModelIntegrationTest(unittest.TestCase):
    def test_inference_predicted_token_classification(self):
        model = AutoModelForTokenClassification.from_pretrained("openai/privacy-filter").to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("openai/privacy-filter")

        inputs = tokenizer(
            "My name is Harry Potter and my email is harry.potter@hogwarts.edu.", return_tensors="pt"
        ).to(torch_device)

        with torch.no_grad():
            outputs = model(**inputs)

        expected_logits_shape = torch.Size((1, 19, 33))
        self.assertEqual(outputs.logits.shape, expected_logits_shape)

        expected_token_classes = Expectations(
            {
                ("cuda", (8, 6)): ['O', 'O', 'O', 'B-private_person', 'E-private_person', 'O', 'O', 'O', 'O', 'B-private_email', 'I-private_email', 'I-private_email', 'I-private_email', 'I-private_email', 'I-private_email', 'I-private_email', 'I-private_email', 'E-private_email', 'O'],
            }
        )  # fmt: skip
        EXPECTED_TOKEN_CLASSES = expected_token_classes.get_expectation()

        predicted_token_class_ids = outputs.logits.argmax(dim=-1)
        predicted_token_classes = [model.config.id2label[token_id.item()] for token_id in predicted_token_class_ids[0]]

        self.assertEqual(predicted_token_classes, EXPECTED_TOKEN_CLASSES)
