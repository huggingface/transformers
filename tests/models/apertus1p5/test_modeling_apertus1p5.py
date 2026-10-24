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
"""Testing suite for the PyTorch Apertus 1.5 model."""

import math
import unittest

from parameterized import parameterized

from transformers import (
    Apertus1p5Config,
    Apertus1p5TextConfig,
    Apertus1p5VisionTokenizerConfig,
    AutoProcessor,
    WavTokenizerEncoderConfig,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester
from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Apertus1p5ForConditionalGeneration,
        Apertus1p5Model,
        Apertus1p5TextForCausalLM,
        Apertus1p5TextModel,
    )


class Apertus1p5TextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Apertus1p5TextModel


@require_torch
class Apertus1p5TextModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Apertus1p5TextModelTester
    model_split_percents = [0.5, 0.7, 0.8]
    _torch_compile_train_cls = Apertus1p5TextForCausalLM if is_torch_available() else None

    def test_pruned_lm_head(self):
        """With `output_vocab_size`, the LM head only covers the text prefix of the vocabulary: logits and loss use
        the head width, ids beyond it stay valid inputs, and their labels must be `-100`."""
        config = self.model_tester.get_config()
        config.output_vocab_size = 40
        model = Apertus1p5TextForCausalLM(config).to(torch_device).eval()

        input_ids = ids_tensor([2, 7], 40)
        input_ids[:, 2] = 60  # a valid input id beyond the head
        labels = input_ids.clone()
        labels[:, 2] = -100
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
        self.assertEqual(outputs.logits.shape[-1], 40)
        expected_loss = torch.nn.functional.cross_entropy(
            outputs.logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1)
        )
        torch.testing.assert_close(outputs.loss, expected_loss)


class Apertus1p5Vision2TextModelTester(VLMModelTester):
    if is_torch_available():
        base_model_class = Apertus1p5Model
        conditional_generation_class = Apertus1p5ForConditionalGeneration
    config_class = Apertus1p5Config
    text_config_class = Apertus1p5TextConfig
    vision_config_class = Apertus1p5VisionTokenizerConfig

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("audio_token_id", 4)
        # image codes use ids 40-59 and audio codes 60-71
        kwargs.setdefault("image_token_offset", 40)
        kwargs.setdefault("audio_token_offset", 60)
        kwargs.setdefault("codebook_size", 20)
        kwargs.setdefault("base_channels", 32)
        kwargs.setdefault("channel_multiplier", [1, 2, 1])  # downsamples by 4, the default `patch_size`
        kwargs.setdefault("num_res_blocks", 2)
        kwargs.setdefault("embed_dim", 16)
        kwargs.setdefault("latent_channels", 16)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.audio_token_id}

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_sizes": torch.full((batch_size, 2), self.image_size, dtype=torch.long, device=torch_device)}

    def _build_modality_sub_configs(self):
        return {
            "vision_config": self.get_vision_config(),
            "audio_config": WavTokenizerEncoderConfig(
                num_filters=8, upsampling_ratios=[2, 2], hidden_size=32, codebook_dim=32, codebook_size=12
            ),
        }


@require_torch
class Apertus1p5Vision2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = Apertus1p5Vision2TextModelTester
    test_all_params_have_gradient = False  # the media tokenizers assign codes by argmax
    test_torch_exportable = False  # images are tokenized one by one at their own size
    skip_test_image_features_output_shape = True  # per-image `pooler_output`, no `last_hidden_state`

    def test_pruned_lm_head(self):
        """With `text_config.output_vocab_size`, as in the released checkpoint, the LM head and the loss only cover
        the text prefix of the vocabulary."""
        config, inputs = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.output_vocab_size = 40
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        labels = ids_tensor(inputs["input_ids"].shape, 40)
        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
        self.assertEqual(outputs.logits.shape[-1], 40)
        expected_loss = torch.nn.functional.cross_entropy(
            outputs.logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1)
        )
        torch.testing.assert_close(outputs.loss, expected_loss)

    @unittest.skip("The vision tokenizer does not record intermediate outputs")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("The vision tokenizer does not record intermediate outputs")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("The tester has no audio in input dict")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no audio in input dict")
    def test_get_audio_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no audio in input dict")
    def test_get_audio_features_output(self, return_dict: bool | None):
        pass


class Apertus1p5Audio2TextModelTester(ALMModelTester):
    if is_torch_available():
        base_model_class = Apertus1p5Model
        conditional_generation_class = Apertus1p5ForConditionalGeneration
    config_class = Apertus1p5Config
    text_config_class = Apertus1p5TextConfig
    audio_config_class = WavTokenizerEncoderConfig
    audio_mask_key = "feature_attention_mask"

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("audio_token_id", 4)
        kwargs.setdefault("image_token_offset", 40)
        kwargs.setdefault("audio_token_offset", 60)
        kwargs.setdefault("num_filters", 8)
        kwargs.setdefault("upsampling_ratios", [2, 2])
        kwargs.setdefault("codebook_size", 12)
        kwargs.setdefault("codebook_dim", 32)  # equal to `hidden_size`: the quantizer has no projections
        kwargs.setdefault("feat_seq_length", 20)
        super().__init__(parent, **kwargs)

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.image_token_id}

    def create_audio_features(self, batch_size: int | None = None):
        # raw mono waveforms
        batch_size = batch_size if batch_size is not None else self.batch_size
        return floats_tensor([batch_size, 1, self.feat_seq_length])

    def create_audio_mask(self, batch_size: int | None = None):
        # the model expects right-padded clips
        return super().create_audio_mask(batch_size).sort(dim=-1, descending=True).values

    def get_audio_embeds_mask(self, audio_mask):
        # one code per started hop of valid samples
        hop_length = math.prod(self.upsampling_ratios)
        lengths = (audio_mask.sum(-1) + hop_length - 1) // hop_length
        positions = torch.arange(int(lengths.max()), device=audio_mask.device)[None, :]
        return (positions < lengths[:, None]).long()

    def _build_modality_sub_configs(self):
        return {
            "audio_config": self.get_audio_config(),
            "vision_config": Apertus1p5VisionTokenizerConfig(
                codebook_size=20,
                base_channels=32,
                channel_multiplier=[1, 2, 1],
                num_res_blocks=2,
                embed_dim=16,
                latent_channels=16,
            ),
        }


@require_torch
class Apertus1p5Audio2TextModelTest(ALMModelTest, unittest.TestCase):
    model_tester_class = Apertus1p5Audio2TextModelTester
    test_all_params_have_gradient = False  # the media tokenizers assign codes by argmax
    test_torch_exportable = False  # clips are tokenized one by one at their own length
    skip_test_audio_features_output_shape = True  # per-clip `pooler_output`, no `last_hidden_state`

    def test_expand_inputs_for_generation_keeps_audio_groups(self):
        """Clips are flattened across the batch, so expanding for beam search must repeat each prompt's own clips."""
        model = Apertus1p5ForConditionalGeneration(self.model_tester.get_config())
        audio = model.config.audio_token_id
        # prompt 0 holds two clips of one code each, prompt 1 one clip of two codes (hop length 4)
        input_ids = torch.tensor([[audio, audio, 5]] * 2)
        input_features = torch.arange(3.0)[:, None, None].expand(3, 1, 8)
        feature_attention_mask = torch.tensor([[1] * 4 + [0] * 4, [1] * 4 + [0] * 4, [1] * 8])
        _, expanded = model._expand_inputs_for_generation(
            expand_size=2,
            input_ids=input_ids,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        torch.testing.assert_close(expanded["input_features"], input_features[[0, 1, 0, 1, 2, 2]])

    @unittest.skip("The audio tokenizer does not record intermediate outputs")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("The audio tokenizer does not record intermediate outputs")
    def test_get_audio_features_attentions(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_attentions(self):
        pass

    @parameterized.expand([True, False, None])
    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_output(self, return_dict: bool | None):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_get_image_features_pooler_output(self):
        pass

    @unittest.skip("The tester has no image in input dict")
    def test_generate_from_multimodal_encoder_outputs_and_raw_data(self):
        pass


@slow
@require_torch_accelerator
class Apertus1p5IntegrationTest(unittest.TestCase):
    checkpoint = "swiss-ai/Apertus-v1.5-8B"
    revision = "refs/pr/5"  # TODO: drop once https://huggingface.co/swiss-ai/Apertus-v1.5-8B/discussions/5 is merged
    IMAGE_URL = url_to_local_path(
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )
    AUDIO_URL = url_to_local_path(
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
    )
    TEXT_CONVERSATION = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "What is the capital of Switzerland? Answer in one word."}],
        }
    ]
    IMAGE_CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": IMAGE_URL},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    AUDIO_CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": AUDIO_URL},
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ]
    IMAGE_AUDIO_CONVERSATION = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": IMAGE_URL},
                {"type": "audio", "url": AUDIO_URL},
                {"type": "text", "text": "What do you see and what do you hear?"},
            ],
        }
    ]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)
        self.processor = AutoProcessor.from_pretrained(self.checkpoint, revision=self.revision)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def _load_model(self):
        return Apertus1p5ForConditionalGeneration.from_pretrained(
            self.checkpoint, revision=self.revision, device_map=torch_device
        )

    def _generate(self, model, conversations, max_new_tokens=20):
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(torch_device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.processor.decode(output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def test_text_generation(self):
        model = self._load_model()
        output_text = self._generate(model, [self.TEXT_CONVERSATION])

        EXPECTED_TEXTS = Expectations({(None, None): ["Bern"]})
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_image_generation(self):
        model = self._load_model()
        # Greedy decoding reaches a bf16 near-tie a few tokens later, which flips between CUDA runs
        output_text = self._generate(model, [self.IMAGE_CONVERSATION], max_new_tokens=12)

        EXPECTED_TEXTS = Expectations(
            {
                (None, None): ["Two tabby cats are peacefully sleeping on a pink couch"],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_audio_generation(self):
        model = self._load_model()
        output_text = self._generate(model, [self.AUDIO_CONVERSATION])

        EXPECTED_TEXTS = Expectations(
            {
                (None, None): ["mr crilger is the apostle of the middle classes and we're glad to welcome his gospel"],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)

    def test_batch_generation(self):
        model = self._load_model()
        conversations = [
            self.IMAGE_CONVERSATION,
            self.AUDIO_CONVERSATION,
            self.TEXT_CONVERSATION,
            self.IMAGE_AUDIO_CONVERSATION,
        ]
        # The image rows reach bf16 near-ties right after these tokens, which flip between CUDA runs
        output_text = self._generate(model, conversations, max_new_tokens=10)

        EXPECTED_TEXTS = Expectations(
            {
                (None, None): [
                    "Two tabby cats are peacefully sleeping on a",
                    "mr crilger is the apostle of the",
                    "Bern",
                    "The image shows two cats lying on a pink couch",
                ],
            }
        )  # fmt: skip
        EXPECTED_TEXT = EXPECTED_TEXTS.get_expectation()
        self.assertEqual(output_text, EXPECTED_TEXT)
