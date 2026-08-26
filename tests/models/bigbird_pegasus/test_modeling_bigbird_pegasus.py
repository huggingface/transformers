# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch BigBirdPegasus model."""

import logging
import tempfile
import unittest

from transformers import BigBirdPegasusConfig, is_torch_available, set_seed
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


logger = logging.getLogger(__name__)


if is_torch_available():
    import torch

    from transformers import (
        BigBirdPegasusForCausalLM,
        BigBirdPegasusForConditionalGeneration,
        BigBirdPegasusForQuestionAnswering,
        BigBirdPegasusForSequenceClassification,
        BigBirdPegasusModel,
    )
    from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import (
        BigBirdPegasusDecoder,
        BigBirdPegasusEncoder,
    )

MODEL_ID = "google/bigbird-pegasus-large-pubmed"


def prepare_bigbird_pegasus_inputs_dict(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
):
    if attention_mask is None:
        attention_mask = input_ids.ne(config.pad_token_id)
    if decoder_attention_mask is None:
        decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)

    input_dict = {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
    }
    input_dict = {k: input_dict[k].to(torch_device) for k in input_dict}
    return input_dict


class BigBirdPegasusModelTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        seq_length=128,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=31,
        hidden_act="gelu_fast",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=260,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
        attention_type="block_sparse",
        use_bias=False,
        block_size=16,
        num_random_blocks=3,
        scale_embedding=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

        self.attention_type = attention_type
        self.use_bias = use_bias
        self.block_size = block_size
        self.num_random_blocks = num_random_blocks
        self.scale_embedding = scale_embedding

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size).clamp(
            3,
        )
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        inputs_dict = prepare_bigbird_pegasus_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return BigBirdPegasusConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            attention_type=self.attention_type,
            use_bias=self.use_bias,
            block_size=self.block_size,
            num_random_blocks=self.num_random_blocks,
            scale_embedding=self.scale_embedding,
        )

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def create_and_check_decoder_model_past_large_inputs(self, config, inputs_dict):
        model = BigBirdPegasusModel(config=config).get_decoder().to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        attention_mask = inputs_dict["attention_mask"]

        # first forward pass
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical multiple next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 3), config.vocab_size)
        next_attn_mask = ids_tensor((self.batch_size, 3), 2)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)
        next_attention_mask = torch.cat([attention_mask, next_attn_mask], dim=-1)

        output_from_no_past = model(next_input_ids, attention_mask=next_attention_mask)["last_hidden_state"]
        output_from_past = model(next_tokens, attention_mask=next_attention_mask, past_key_values=past_key_values)[
            "last_hidden_state"
        ]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -3:, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, :, random_slice_idx].detach()

        self.parent.assertTrue(output_from_past_slice.shape[1] == next_tokens.shape[1])

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-2))

    def check_encoder_decoder_model_standalone(self, config, inputs_dict):
        model = BigBirdPegasusModel(config=config).to(torch_device).eval()
        outputs = model(**inputs_dict)

        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        last_hidden_state = outputs.last_hidden_state

        with tempfile.TemporaryDirectory() as tmpdirname:
            encoder = model.get_encoder()
            encoder.save_pretrained(tmpdirname)
            encoder = BigBirdPegasusEncoder.from_pretrained(tmpdirname).to(torch_device)

        encoder_last_hidden_state_2 = encoder(inputs_dict["input_ids"], attention_mask=inputs_dict["attention_mask"])[
            0
        ]

        self.parent.assertTrue((encoder_last_hidden_state_2 - encoder_last_hidden_state).abs().max().item() < 1e-3)

        with tempfile.TemporaryDirectory() as tmpdirname:
            decoder = model.get_decoder()
            decoder.save_pretrained(tmpdirname)
            decoder = BigBirdPegasusDecoder.from_pretrained(tmpdirname).to(torch_device)

        last_hidden_state_2 = decoder(
            input_ids=inputs_dict["decoder_input_ids"],
            attention_mask=inputs_dict["decoder_attention_mask"],
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=inputs_dict["attention_mask"],
        )[0]

        self.parent.assertTrue((last_hidden_state_2 - last_hidden_state).abs().max().item() < 1e-3)

    def create_and_check_model(self, config, inputs_dict):
        model = BigBirdPegasusModel(config=config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"]
        decoder_input_ids = inputs_dict["decoder_input_ids"]
        result = model(input_ids, decoder_input_ids=decoder_input_ids, use_cache=True)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))


@require_torch
class BigBirdPegasusModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            BigBirdPegasusModel,
            BigBirdPegasusForConditionalGeneration,
            BigBirdPegasusForSequenceClassification,
            BigBirdPegasusForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": BigBirdPegasusModel,
            "text-classification": BigBirdPegasusForSequenceClassification,
            "text-generation": BigBirdPegasusForCausalLM,
            "zero-shot": BigBirdPegasusForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    test_missing_keys = False

    # TODO: Fix the failed tests
    def is_pipeline_test_to_skip(
        self,
        pipeline_test_case_name,
        config_class,
        model_architecture,
        tokenizer_name,
        image_processor_name,
        feature_extractor_name,
        processor_name,
    ):
        if pipeline_test_case_name == "QAPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if model_class.__name__ in ["BigBirdPegasusModel"] or not model_class.supports_gradient_checkpointing:
                    # TODO (ydshieh): use `skipTest` once pytest-dev/pytest-subtests/pull/169 is merged
                    # self.skipTest(reason=f"`supports_gradient_checkpointing` is False for {model_class.__name__}.")
                    continue

                mismatch_count = 0
                for iteration in range(1000):
                    config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                    config.use_cache = False
                    config.return_dict = True

                    # make sure that test runs are consistent by disabling dropout
                    #
                    # Note: attention_probs_dropout_prob seem to influence classifier.bias in BertForMultipleChoice
                    # (and other Bert derived models). Sometimes classifier.bias is None when
                    # attention_probs_dropout_prob > 0. This might indicate a bug somewhere.
                    if hasattr(config, "hidden_dropout_prob"):
                        config.hidden_dropout_prob = 0.0
                    if hasattr(config, "attention_probs_dropout_prob"):
                        config.attention_probs_dropout_prob = 0.0

                    inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)

                    set_seed(42)
                    model = model_class(config)
                    model.to(torch_device)
                    model.train()

                    # unfreeze additional layers
                    for p in model.parameters():
                        p.requires_grad_(True)

                    # do a non-checkpointing run, so we can compare the set of non-zero gradients later. we skip None
                    # grads here to collect a reference set of modules that have non-zero gradients (to filter layers like
                    # MoE that drop out parts of the model).
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    set_seed(42)
                    loss = model(**inputs).loss
                    loss.backward()
                    grad_expected_params = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
                    normal_grad_sums = {n: p.grad.abs().sum().item() for n, p in grad_expected_params}
                    non_zero_grads_normal = {n for n, s in normal_grad_sums.items() if s > 0}

                    # reset all gradients to zero for the comparison with the gradient checkpointing run
                    optimizer.zero_grad()

                    # now enable gradient checkpointing and compare the gradients
                    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

                    checkpointing_layer = next(m for m in model.modules() if isinstance(m, GradientCheckpointingLayer))

                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                    with unittest.mock.patch.object(
                        checkpointing_layer, "forward", wraps=checkpointing_layer.forward
                    ) as forward_mock:
                        set_seed(42)
                        loss = model(**inputs).loss
                        loss.backward()
                        optimizer.step()

                        # test that gradient checkpointing is active as it would call the gradient checkpointing layer's
                        # forward more than once.
                        self.assertGreater(forward_mock.call_count, 1)

                    # check that all the parameters that had non-zero gradients before, have non-zero grads with gradient
                    # checkpointing. divergence indicates a different forward-pass environment that needs special handling.
                    gradcp_grad_sums = {n: p.grad.abs().sum().item() for n, p in grad_expected_params}
                    non_zero_grads_gradcp = {n for n, s in gradcp_grad_sums.items() if s > 0}

                    if non_zero_grads_gradcp != non_zero_grads_normal:
                        only_in_normal = non_zero_grads_normal - non_zero_grads_gradcp
                        only_in_gradcp = non_zero_grads_gradcp - non_zero_grads_normal
                        logger.warning(
                            "[iter %d][%s] MISMATCH: only_in_normal=%s (normal_sums=%s, gradcp_sums=%s), only_in_gradcp=%s (normal_sums=%s, gradcp_sums=%s)",
                            iteration,
                            model_class.__name__,
                            only_in_normal,
                            {n: normal_grad_sums[n] for n in only_in_normal},
                            {n: gradcp_grad_sums[n] for n in only_in_normal},
                            only_in_gradcp,
                            {n: normal_grad_sums[n] for n in only_in_gradcp},
                            {n: gradcp_grad_sums[n] for n in only_in_gradcp},
                        )
                        mismatch_count += 1

                if mismatch_count > 0:
                    logger.warning(
                        "[%s] mismatch occurred %d / 1000 iterations",
                        model_class.__name__,
                        mismatch_count,
                    )
                    self.fail(
                        f"{model_class.__name__}: non_zero_grads_gradcp != non_zero_grads_normal in {mismatch_count} / 1000 iterations"
                    )

    def setUp(self):
        self.model_tester = BigBirdPegasusModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BigBirdPegasusConfig)

    def test_config(self):
        self.config_tester.run_common_tests()
