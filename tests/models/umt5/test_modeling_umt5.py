# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import copy
import os
import pickle
import tempfile
import unittest

from transformers import UMT5Config, is_torch_available
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_torch_fx_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_fx_available():
    from transformers.utils.fx import symbolic_trace


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        AutoTokenizer,
        UMT5EncoderModel,
        UMT5ForConditionalGeneration,
        UMT5ForQuestionAnswering,
        UMT5ForSequenceClassification,
        UMT5ForTokenClassification,
        UMT5Model,
    )


# Copied from test.models.t5.test_modeling_t5.T5ModelTester with T5->UMT5
class UMT5ModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        decoder_seq_length=7,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=False,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        dropout_rate=0.1,
        initializer_factor=0.002,
        eos_token_id=1,
        pad_token_id=0,
        decoder_start_token_id=0,
        scope=None,
        decoder_layers=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.scope = None
        self.decoder_layers = decoder_layers

    def get_large_model_config(self):
        return UMT5Config.from_pretrained("google/umt5-base")

    def prepare_inputs_dict(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
    ):
        if attention_mask is None:
            attention_mask = input_ids.ne(config.pad_token_id)
        if decoder_attention_mask is None:
            decoder_attention_mask = decoder_input_ids.ne(config.pad_token_id)
        if head_mask is None:
            head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads, device=torch_device)
        if decoder_head_mask is None:
            decoder_head_mask = torch.ones(config.num_decoder_layers, config.num_attention_heads, device=torch_device)
        if cross_attn_head_mask is None:
            cross_attn_head_mask = torch.ones(
                config.num_decoder_layers, config.num_attention_heads, device=torch_device
            )
        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
        }

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_tensor([self.batch_size, self.decoder_seq_length], self.vocab_size)

        # we need to clamp the input ids here to avoid having pad token in between
        # this is because for NllbMoe the position_ids are prepared such that
        # all pad tokens have pos id = 2 and rest are between 2..seq_length
        # and the seq_length here is seq_length - num_pad_tokens
        # but when using past, there is no way of knowing if the past input ids had
        # pad tokens in them, which results in incorrect seq_lenth and which in turn results in
        # position_ids being off by num_pad_tokens in past input
        input_ids = input_ids.clamp(self.pad_token_id + 2)
        input_ids[:, -1] = self.eos_token_id  # Eos Token
        decoder_input_ids = decoder_input_ids.clamp(self.pad_token_id + 1)

        config = self.get_config()
        config.encoder_attention_heads = config.num_attention_heads
        input_dict = self.prepare_inputs_dict(config, input_ids, decoder_input_ids)
        return config, input_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict

    def get_pipeline_config(self):
        return UMT5Config(
            vocab_size=166,  # t5 forces 100 extra tokens
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def get_config(self):
        return UMT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_decoder_layers=self.decoder_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = UMT5Model(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
        )
        result = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        decoder_output = result.last_hidden_state
        decoder_past = result.past_key_values
        encoder_output = result.encoder_last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))
        self.parent.assertEqual(decoder_output.size(), (self.batch_size, self.decoder_seq_length, self.hidden_size))
        # There should be `num_layers` key value embeddings stored in decoder_past
        self.parent.assertEqual(len(decoder_past), config.num_layers)
        # There should be a self attn key, a self attn value, a cross attn key and a cross attn value stored in each decoder_past tuple
        self.parent.assertEqual(len(decoder_past[0]), 4)

    def create_and_check_decoder_model_past(
        self,
        config,
        input_ids,
        decoder_input_ids,
        attention_mask,
        decoder_attention_mask,
        lm_labels,
    ):
        model = UMT5Model(config=config).get_decoder().to(torch_device).eval()
        # first forward pass
        outputs = model(input_ids, use_cache=True)
        outputs_use_cache_conf = model(input_ids)
        outputs_no_past = model(input_ids, use_cache=False)

        self.parent.assertTrue(len(outputs) == len(outputs_use_cache_conf))
        self.parent.assertTrue(len(outputs) == len(outputs_no_past) + 1)

        output, past_key_values = outputs.to_tuple()

        # create hypothetical next token and extent to next_input_ids
        next_tokens = ids_tensor((self.batch_size, 1), config.vocab_size)

        # append to next input_ids and
        next_input_ids = torch.cat([input_ids, next_tokens], dim=-1)

        output_from_no_past = model(next_input_ids)["last_hidden_state"]
        output_from_past = model(next_tokens, past_key_values=past_key_values)["last_hidden_state"]

        # select random slice
        random_slice_idx = ids_tensor((1,), output_from_past.shape[-1]).item()
        output_from_no_past_slice = output_from_no_past[:, -1, random_slice_idx].detach()
        output_from_past_slice = output_from_past[:, 0, random_slice_idx].detach()

        # test that outputs are equal for slice
        self.parent.assertTrue(torch.allclose(output_from_past_slice, output_from_no_past_slice, atol=1e-3))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_dict,
    ):
        model = UMT5Model(config=config).to(torch_device).half().eval()
        output = model(**input_dict)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_with_sequence_classification_head(
        self,
        config,
        input_dict,
    ):
        labels = torch.tensor([1] * self.batch_size, dtype=torch.long, device=torch_device)
        model = UMT5ForSequenceClassification(config=config).to(torch_device).eval()
        outputs = model(**input_dict, labels=labels)
        # self.parent.assertEqual(len(outputs), 4)
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())


@require_torch
class UMT5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (UMT5Model, UMT5ForConditionalGeneration, UMT5ForSequenceClassification, UMT5ForQuestionAnswering)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {
            "feature-extraction": UMT5Model,
            "question-answering": UMT5ForQuestionAnswering,
            "summarization": UMT5ForConditionalGeneration,
            "text-classification": UMT5ForSequenceClassification,
            "text2text-generation": UMT5ForConditionalGeneration,
            "translation": UMT5ForConditionalGeneration,
            "zero-shot": UMT5ForSequenceClassification,
        }
        if is_torch_available()
        else {}
    )
    is_encoder_decoder = True
    fx_compatible = False
    test_pruning = False
    test_missing_keys = True
    test_torchscript = True
    # The small UMT5 model needs higher percentages for CPU/MP tests
    model_split_percents = [0.5, 0.8, 0.9]

    def setUp(self):
        self.model_tester = UMT5ModelTester(self)

    # `QAPipelineTests` is not working well with slow tokenizers (for some models) and we don't want to touch the file
    # `src/transformers/data/processors/squad.py` (where this test fails for this model)
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

    def _create_and_check_torch_fx_tracing(self, config, inputs_dict, output_loss=False):
        if not is_torch_fx_available() or not self.fx_compatible:
            self.skipTest(reason="torch fx is not available or not compatible with this model")

        configs_no_init = _config_zero_init(config)  # To be sure we have no Nan
        configs_no_init.return_dict = False

        for model_class in self.all_model_classes:
            if model_class.__name__ == "UMT5ForSequenceClassification":
                continue
            model = model_class(config=configs_no_init)
            model.to(torch_device)
            model.eval()
            inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=output_loss)

            try:
                if model.config.is_encoder_decoder:
                    model.config.use_cache = False  # FSTM still requires this hack -> FSTM should probably be refactored similar to BART afterward
                    labels = inputs.get("labels", None)
                    input_names = [
                        "attention_mask",
                        "decoder_attention_mask",
                        "decoder_input_ids",
                        "input_features",
                        "input_ids",
                        "input_values",
                    ]
                    if labels is not None:
                        input_names.append("labels")

                    filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                    input_names = list(filtered_inputs.keys())

                    model_output = model(**filtered_inputs)

                    traced_model = symbolic_trace(model, input_names)
                    traced_output = traced_model(**filtered_inputs)
                else:
                    input_names = [
                        "attention_mask",
                        "bbox",
                        "input_features",
                        "input_ids",
                        "input_values",
                        "pixel_values",
                        "token_type_ids",
                        "visual_feats",
                        "visual_pos",
                    ]

                    labels = inputs.get("labels", None)
                    start_positions = inputs.get("start_positions", None)
                    end_positions = inputs.get("end_positions", None)
                    if labels is not None:
                        input_names.append("labels")
                    if start_positions is not None:
                        input_names.append("start_positions")
                    if end_positions is not None:
                        input_names.append("end_positions")

                    filtered_inputs = {k: v for (k, v) in inputs.items() if k in input_names}
                    input_names = list(filtered_inputs.keys())

                    if model.__class__.__name__ in set(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()) and (
                        not hasattr(model.config, "problem_type") or model.config.problem_type is None
                    ):
                        model.config.problem_type = "single_label_classification"

                    traced_model = symbolic_trace(model, input_names)
                    traced_output = traced_model(**filtered_inputs)
                    model_output = model(**filtered_inputs)

            except Exception as e:
                self.fail(f"Couldn't trace module: {e}")

            def flatten_output(output):
                flatten = []
                for x in output:
                    if isinstance(x, (tuple, list)):
                        flatten += flatten_output(x)
                    elif not isinstance(x, torch.Tensor):
                        continue
                    else:
                        flatten.append(x)
                return flatten

            model_output = flatten_output(model_output)
            traced_output = flatten_output(traced_output)
            num_outputs = len(model_output)

            for i in range(num_outputs):
                self.assertTrue(
                    torch.allclose(model_output[i], traced_output[i]),
                    f"traced {i}th output doesn't match model {i}th output for {model_class}",
                )

            # Test that the model can be serialized and restored properly
            with tempfile.TemporaryDirectory() as tmp_dir_name:
                pkl_file_name = os.path.join(tmp_dir_name, "model.pkl")
                try:
                    with open(pkl_file_name, "wb") as f:
                        pickle.dump(traced_model, f)
                    with open(pkl_file_name, "rb") as f:
                        loaded = pickle.load(f)
                except Exception as e:
                    self.fail(f"Couldn't serialize / deserialize the traced model: {e}")

                loaded_output = loaded(**filtered_inputs)
                loaded_output = flatten_output(loaded_output)

                for i in range(num_outputs):
                    self.assertTrue(
                        torch.allclose(model_output[i], loaded_output[i]),
                        f"serialized model {i}th output doesn't match model {i}th output for {model_class}",
                    )

            # Avoid memory leak. Without this, each call increase RAM usage by ~20MB.
            # (Even with this call, there are still memory leak by ~0.04MB)
            self.clear_torch_jit_class_registry()

    # UMT5ForSequenceClassification does not support inputs_embeds
    def test_inputs_embeds(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in (UMT5Model, UMT5ForConditionalGeneration, UMT5ForQuestionAnswering):
            model = model_class(config)
            model.to(torch_device)
            model.eval()

            inputs = copy.deepcopy(self._prepare_for_class(inputs_dict, model_class))

            if not self.is_encoder_decoder:
                input_ids = inputs["input_ids"]
                del inputs["input_ids"]
            else:
                encoder_input_ids = inputs["input_ids"]
                decoder_input_ids = inputs.get("decoder_input_ids", encoder_input_ids)
                del inputs["input_ids"]
                inputs.pop("decoder_input_ids", None)

            wte = model.get_input_embeddings()
            if not self.is_encoder_decoder:
                inputs["inputs_embeds"] = wte(input_ids)
            else:
                inputs["inputs_embeds"] = wte(encoder_input_ids)
                inputs["decoder_inputs_embeds"] = wte(decoder_input_ids)

            with torch.no_grad():
                model(**inputs)[0]

    # overwrite because T5 doesn't accept position ids as input and expects `decoder_input_ids`
    def test_custom_4d_attention_mask(self):
        for model_class in self.all_generative_model_classes:
            config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config).to(device=torch_device, dtype=torch.float32)

            (
                input_ids,
                _,
                input_ids_shared_prefix,
                mask_shared_prefix,
                _,
            ) = self._get_custom_4d_mask_test_data()

            logits = model.forward(
                decoder_input_ids=input_ids,
                input_ids=input_dict["input_ids"][:3],
            ).logits
            # logits.shape == torch.Size([3, 4, ...])

            logits_shared_prefix = model(
                input_ids=input_dict["input_ids"][:1],
                decoder_input_ids=input_ids_shared_prefix,
                decoder_attention_mask=mask_shared_prefix,
            )[0]
            # logits_shared_prefix.shape == torch.Size([1, 6, ...])

            out_last_tokens = logits[:, -1, :]  # last tokens in each batch line
            out_shared_prefix_last_tokens = logits_shared_prefix[0, -3:, :]  # last three tokens

            # comparing softmax-normalized logits:
            normalized_0 = F.softmax(out_last_tokens)
            normalized_1 = F.softmax(out_shared_prefix_last_tokens)
            torch.testing.assert_close(normalized_0, normalized_1, rtol=1e-3, atol=1e-4)

    def test_with_sequence_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_sequence_classification_head(*config_and_inputs)

    @unittest.skip(reason="Test has a segmentation fault on torch 1.8.0")
    def test_export_to_onnx(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        model = UMT5Model(config_and_inputs[0]).to(torch_device)
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.onnx.export(
                model,
                (config_and_inputs[1], config_and_inputs[3], config_and_inputs[2]),
                f"{tmpdirname}/t5_test.onnx",
                export_params=True,
                opset_version=9,
                input_names=["input_ids", "decoder_input_ids"],
            )

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_generate_with_head_masking(self):
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config = config_and_inputs[0]
        model = UMT5ForConditionalGeneration(config).eval()
        model.to(torch_device)

        head_masking = {
            "head_mask": torch.zeros(config.num_layers, config.num_heads, device=torch_device),
            "decoder_head_mask": torch.zeros(config.num_decoder_layers, config.num_heads, device=torch_device),
            "cross_attn_head_mask": torch.zeros(config.num_decoder_layers, config.num_heads, device=torch_device),
        }

        for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
            head_masks = {name: mask}
            # Explicitly pass decoder_head_mask as it is required from T5 model when head_mask specified
            if name == "head_mask":
                head_masks["decoder_head_mask"] = torch.ones(
                    config.num_decoder_layers, config.num_heads, device=torch_device
                )

            out = model.generate(
                config_and_inputs[1]["input_ids"],
                num_beams=1,
                max_length=3,
                output_attentions=True,
                return_dict_in_generate=True,
                **head_masks,
            )
            # We check the state of decoder_attentions and cross_attentions just from the last step
            attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
            self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant(self):
        pass

    @unittest.skip(
        reason="This architecure seem to not compute gradients properly when using GC, check: https://github.com/huggingface/transformers/pull/27124"
    )
    def test_training_gradient_checkpointing_use_reentrant_false(self):
        pass


# Copied from tests.models.t5.test_modeling_t5.T5EncoderOnlyModelTester with T5->UMT5
class UMT5EncoderOnlyModelTester:
    def __init__(
        self,
        parent,
        vocab_size=99,
        batch_size=13,
        encoder_seq_length=7,
        # For common tests
        use_attention_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        d_ff=37,
        relative_attention_num_buckets=8,
        is_training=False,
        dropout_rate=0.1,
        initializer_factor=0.002,
        is_encoder_decoder=False,
        eos_token_id=1,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        # For common tests
        self.seq_length = self.encoder_seq_length
        self.use_attention_mask = use_attention_mask
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.d_ff = d_ff
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.initializer_factor = initializer_factor
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.scope = None
        self.is_training = is_training

    def get_large_model_config(self):
        return UMT5Config.from_pretrained("google-t5/t5-base")

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.encoder_seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_tensor([self.batch_size, self.encoder_seq_length], vocab_size=2)

        config = UMT5Config(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            d_ff=self.d_ff,
            d_kv=self.hidden_size // self.num_attention_heads,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            dropout_rate=self.dropout_rate,
            initializer_factor=self.initializer_factor,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.pad_token_id,
            pad_token_id=self.pad_token_id,
            is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            config,
            input_ids,
            attention_mask,
        )

    def create_and_check_model(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = UMT5EncoderModel(config=config)
        model.to(torch_device)
        model.eval()
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        result = model(input_ids=input_ids)
        encoder_output = result.last_hidden_state

        self.parent.assertEqual(encoder_output.size(), (self.batch_size, self.encoder_seq_length, self.hidden_size))

    def create_and_check_model_fp16_forward(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        model = UMT5EncoderModel(config=config).to(torch_device).half().eval()
        output = model(input_ids, attention_mask=attention_mask)["last_hidden_state"]
        self.parent.assertFalse(torch.isnan(output).any().item())

    def create_and_check_with_token_classification_head(
        self,
        config,
        input_ids,
        attention_mask,
    ):
        labels = torch.tensor([1] * self.seq_length * self.batch_size, dtype=torch.long, device=torch_device)
        model = UMT5ForTokenClassification(config=config).to(torch_device).eval()
        outputs = model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        self.parent.assertEqual(outputs["logits"].size(), (self.batch_size, self.seq_length, config.num_labels))
        self.parent.assertEqual(outputs["loss"].size(), ())

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


# Copied from tests.models.t5.test_modeling_t5.T5EncoderOnlyModelTest with T5->UMT5
class UMT5EncoderOnlyModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (UMT5EncoderModel, UMT5ForTokenClassification) if is_torch_available() else ()
    test_pruning = False
    test_resize_embeddings = False
    test_model_parallel = True
    pipeline_model_mapping = (
        {
            "token-classification": UMT5ForTokenClassification,
        }
        if is_torch_available()
        else {}
    )
    all_parallelizable_model_classes = (UMT5EncoderModel,) if is_torch_available() else ()

    def setUp(self):
        self.model_tester = UMT5EncoderOnlyModelTester(self)
        self.config_tester = ConfigTester(self, config_class=UMT5Config, d_model=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    @unittest.skipIf(torch_device == "cpu", "Cant do half precision")
    def test_model_fp16_forward(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model_fp16_forward(*config_and_inputs)

    def test_with_token_classification_head(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_with_token_classification_head(*config_and_inputs)

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
        if tokenizer_name is None:
            return True

        # `UMT5EncoderOnlyModelTest` is not working well with slow tokenizers (for some models) and we don't want to touch the file
        # `src/transformers/data/processors/squad.py` (where this test fails for this model)
        if pipeline_test_case_name == "TokenClassificationPipelineTests" and not tokenizer_name.endswith("Fast"):
            return True

        return False


@require_torch
@require_sentencepiece
@require_tokenizers
class Umt5IntegrationTest(unittest.TestCase):
    @slow
    @unittest.skip(
        "Unless we stop stripping left and right by default for all special tokens, the expected ids obtained here will not match the original ones. Wait for https://github.com/huggingface/transformers/pull/23909 to be merged"
    )
    def test_small_integration_test(self):
        """
        For comparison run the kaggle notbook available here : https://www.kaggle.com/arthurzucker/umt5-inference
        """

        model = UMT5ForConditionalGeneration.from_pretrained("google/umt5-small", return_dict=True).to(torch_device)
        tokenizer = AutoTokenizer.from_pretrained("google/umt5-small", use_fast=False, legacy=False)
        input_text = [
            "Bonjour monsieur <extra_id_0> bien <extra_id_1>.",
            "No se como puedo <extra_id_0>.",
            "This is the reason why we <extra_id_0> them.",
            "The <extra_id_0> walks in <extra_id_1>, seats",
            "A <extra_id_0> walks into a bar and orders a <extra_id_1> with <extra_id_2> pinch of <extra_id_3>.",
        ]
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True).input_ids
        # fmt: off
        EXPECTED_IDS = torch.tensor(
            [
                [ 38530, 210703, 256299, 1410, 256298, 274, 1, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0],
                [   826, 321, 671, 25922, 256299, 274, 1, 0,0, 0, 0, 0, 0, 0, 0, 0,0, 0],
                [  1460, 339, 312, 19014, 10620, 758, 256299, 2355,274, 1, 0, 0, 0, 0, 0, 0,0, 0],
                [   517, 256299, 14869, 281, 301, 256298, 275, 119983,1, 0, 0, 0, 0, 0, 0, 0,0, 0],
                [   320, 256299, 14869, 281, 2234, 289, 2275, 333,61391, 289, 256298, 543, 256297, 168714, 329, 256296,274, 1],
            ]
        )
        # fmt: on
        torch.testing.assert_close(input_ids, EXPECTED_IDS)

        generated_ids = model.generate(input_ids.to(torch_device))
        EXPECTED_FILLING = [
            "<pad><extra_id_0> et<extra_id_1> [eod] <extra_id_2><extra_id_55>.. [eod] 💐 💐 💐 💐 💐 💐 💐 💐 💐 💐 💐 <extra_id_56>ajšietosto<extra_id_56>lleux<extra_id_19><extra_id_6>ajšie</s>",
            "<pad><extra_id_0>.<extra_id_1>.,<0x0A>...spech <0x0A><extra_id_20> <extra_id_21></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
            "<pad><extra_id_0> are not going to be a part of the world. We are not going to be a part of<extra_id_1> and<extra_id_2><0x0A><extra_id_48>.<extra_id_48></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
            "<pad><extra_id_0> door<extra_id_1>, the door<extra_id_2> 피해[/</s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
            "<pad><extra_id_0>nyone who<extra_id_1> drink<extra_id_2> a<extra_id_3> alcohol<extra_id_4> A<extra_id_5> A. This<extra_id_6> I<extra_id_7><extra_id_52><extra_id_53></s><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>",
        ]
        filling = tokenizer.batch_decode(generated_ids)
        self.assertEqual(filling, EXPECTED_FILLING)
