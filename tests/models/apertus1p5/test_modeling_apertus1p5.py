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

import os
import tempfile
import unittest

import pytest

from transformers import (
    Apertus1p5Config,
    AutoTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    require_torch,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin


if is_torch_available():
    import torch

    from transformers import (
        Apertus1p5ForConditionalGeneration,
        Apertus1p5Model,
        Apertus1p5TextConfig,
        Apertus1p5TextForCausalLM,
        Apertus1p5TextModel,
        Apertus1p5TextPreTrainedModel,
        Apertus1p5VisionTokenizerConfig,
        Apertus1p5VisionTokenizerModel,
        WatermarkingConfig,
        WavTokenizerConfig,
        WavTokenizerModel,
    )


class Apertus1p5ModelTester:
    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=80,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        audio_token_id=4,
        image_token_offset=40,
        audio_token_offset=60,
        image_size=15,
        codebook_size=20,
        base_channels=32,
        vq_channel_multiplier=[1, 2, 1],
        vq_num_res_blocks=2,
        audio_codebook_size=12,
        audio_hop_length=4,
        audio_samples=20,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.image_token_offset = image_token_offset
        self.audio_token_offset = audio_token_offset
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_num_res_blocks = vq_num_res_blocks
        self.base_channels = base_channels
        self.audio_codebook_size = audio_codebook_size
        self.audio_hop_length = audio_hop_length
        self.audio_samples = audio_samples

        vq_factor = 2 ** (len(vq_channel_multiplier) - 1)
        self.image_seq_length = (image_size // vq_factor) ** 2  # flat H*W codes, no EOL column
        self.audio_seq_length = -(-audio_samples // audio_hop_length)  # ceil
        self.seq_length = seq_length + self.image_seq_length + self.audio_seq_length

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.audio_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id
        input_ids[:, self.image_seq_length : self.image_seq_length + self.audio_seq_length] = self.audio_token_id
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        pixel_values = floats_tensor(
            [
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        image_sizes = [[self.image_size, self.image_size]] * self.batch_size
        image_sizes = torch.tensor(image_sizes, device=torch_device, dtype=torch.int64)

        input_features = floats_tensor([self.batch_size, 1, self.audio_samples], scale=1.0)
        feature_attention_mask = torch.ones(
            self.batch_size, self.audio_samples, device=torch_device, dtype=torch.int64
        )

        return config, input_ids, attention_mask, pixel_values, image_sizes, input_features, feature_attention_mask

    def get_config(self):
        text_config = {
            "model_type": "apertus1p5_text",
            # gelu instead of apertus' default xielu, mirroring the upstream apertus tests: XIELUActivation
            # params cannot be re-materialized from the meta device by `_init_weights` (pre-existing upstream gap)
            "hidden_act": "gelu",
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "initializer_range": self.initializer_range,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        vision_config = {
            "codebook_size": self.codebook_size,
            "base_channels": self.base_channels,
            "channel_multiplier": self.vq_channel_multiplier,
            "num_res_blocks": self.vq_num_res_blocks,
            "embed_dim": 16,
            "latent_channels": 16,
            "attn_resolutions": [],
            "resolution": self.image_size,
        }
        audio_config = {
            "model_type": "wavtokenizer",
            "num_filters": 8,
            "upsampling_ratios": [2, 2],  # hop_length = 4
            "hidden_size": 32,
            "codebook_dim": 32,
            "codebook_size": self.audio_codebook_size,
            "decoder_hidden_size": 32,
            "decoder_intermediate_size": 64,
            "decoder_num_layers": 2,
        }
        return Apertus1p5Config(
            text_config=text_config,
            vision_config=vision_config,
            audio_config=audio_config,
            image_token_id=self.image_token_id,
            audio_token_id=self.audio_token_id,
            image_token_offset=self.image_token_offset,
            audio_token_offset=self.audio_token_offset,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            image_sizes,
            input_features,
            feature_attention_mask,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
        }
        return config, inputs_dict


@require_torch
class Apertus1p5ModelTest(ModelTesterMixin, GenerationTesterMixin, PipelineTesterMixin, unittest.TestCase):
    all_model_classes = (
        (
            Apertus1p5Model,
            Apertus1p5ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = (
        {"any-to-any": Apertus1p5ForConditionalGeneration, "image-text-to-text": Apertus1p5ForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    # features are per-image/per-clip tuples of embedded discrete codes; there is no batched
    # last_hidden_state whose shape the generic checks could inspect
    skip_test_image_features_output_shape = True
    skip_test_audio_features_output_shape = True

    test_torch_exportable = False  # data-dependent per-image/per-clip tokenizer loops

    def setUp(self):
        self.model_tester = Apertus1p5ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Apertus1p5Config, has_text_modality=False, hidden_size=32)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_fixed_backbones(self):
        config = self.model_tester.get_config()
        self.assertIsInstance(config.text_config, Apertus1p5TextConfig)
        self.assertIsInstance(config.audio_config, WavTokenizerConfig)
        model = Apertus1p5ForConditionalGeneration(config)
        self.assertIsInstance(model.model.language_model, Apertus1p5TextModel)
        self.assertIsInstance(model.model.audio_tokenizer, WavTokenizerModel)

    def test_rejects_unrelated_audio_config(self):
        with self.assertRaisesRegex(ValueError, "must be 'wavtokenizer'"):
            Apertus1p5Config(audio_config={"model_type": "dac"})

    @pytest.mark.generate
    @unittest.skip("Apertus1p5 has dynamic control flow in vision backbone")
    def test_generate_with_static_cache(self):
        pass

    @unittest.skip("The encode-only vision tokenizer port does not implement per-block output recording")
    def test_get_image_features_attentions(self):
        pass

    @unittest.skip("The encode-only vision tokenizer port does not implement per-block output recording")
    def test_get_image_features_hidden_states(self):
        pass

    @unittest.skip("The audio path is a per-clip discrete encode + embedding lookup; no per-block recording")
    def test_get_audio_features_hidden_states(self):
        pass

    @unittest.skip("The audio path is a per-clip discrete encode + embedding lookup; no per-block recording")
    def test_get_audio_features_attentions(self):
        pass

    # The vision quantizer reads `self.embedding.weight` directly instead of calling the `nn.Embedding`, so
    # the offload hook that would restore that weight never fires and the codebook stays on the meta device.
    @unittest.skip(
        reason="Apertus1p5 does not work with offload: the vision quantizer reads its codebook weight directly"
    )
    def test_cpu_offload(self):
        pass

    @unittest.skip(
        reason="Apertus1p5 does not work with offload: the vision quantizer reads its codebook weight directly"
    )
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(
        reason="Apertus1p5 does not work with offload: the vision quantizer reads its codebook weight directly"
    )
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(
        reason="`nn.DataParallel` replicas expose no `Parameter`s, so reading `self.audio_tokenizer.dtype` "
        "in `get_audio_tokens` raises `StopIteration`; use DDP instead"
    )
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_pixel_values_influence_logits(self):
        """Regression test: the generation wrapper must pass image tensors down to the base model."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        torch.manual_seed(0)
        with torch.no_grad():
            logits_a = model(**inputs_dict).logits
            logits_b = model(**{**inputs_dict, "pixel_values": inputs_dict["pixel_values"] * 3.0 + 2.0}).logits
        self.assertFalse(torch.allclose(logits_a, logits_b), "different images must yield different logits")

    def test_images_encoded_one_by_one(self):
        """Images must be encoded individually (batch size 1) even when a batched tensor is given: the vision
        tokenizer contains global attention, so padded-batch encoding would perturb the codes."""
        config, *_ = self.model_tester.prepare_config_and_inputs()
        model = Apertus1p5Model(config).to(torch_device).eval()
        factor = model.vision_tokenizer.vision_spatial_factor

        sizes = [(12, 12), (8, 16), (16, 8)]
        pixel_values = torch.zeros(3, 3, 16, 16, device=torch_device)
        for i, (height, width) in enumerate(sizes):
            pixel_values[i, :, :height, :width] = floats_tensor([3, height, width]).to(torch_device)
        image_sizes = torch.tensor(sizes, device=torch_device, dtype=torch.int64)

        encode_shapes = []
        original_encode = model.vision_tokenizer.encode

        def spy_encode(pixel_values, *args, **kwargs):
            encode_shapes.append(tuple(pixel_values.shape))
            return original_encode(pixel_values, *args, **kwargs)

        model.vision_tokenizer.encode = spy_encode
        with torch.no_grad():
            vocab_ids = model.get_image_tokens(pixel_values, image_sizes)
        model.vision_tokenizer.encode = original_encode

        self.assertTrue(all(shape[0] == 1 for shape in encode_shapes), f"expected batch-1 calls, got {encode_shapes}")
        self.assertEqual([shape[2:] for shape in encode_shapes], list(sizes))

        # per-image results equal encoding each image alone (with the vocabulary offset applied),
        # and counts match the flat H*W formula
        expected = []
        with torch.no_grad():
            for i, (height, width) in enumerate(sizes):
                codes = model.vision_tokenizer.encode(pixel_values[i : i + 1, :, :height, :width])[0]
                expected.append(codes.flatten() + config.image_token_offset)
        self.assertTrue(torch.equal(vocab_ids, torch.cat(expected)))
        expected_counts = [(h // factor) * (w // factor) for h, w in sizes]
        self.assertEqual([len(t) for t in expected], expected_counts)

    def test_image_tokens_are_offset_into_vocab(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Apertus1p5Model(config).to(torch_device).eval()
        with torch.no_grad():
            vocab_ids = model.get_image_tokens(inputs_dict["pixel_values"], inputs_dict["image_sizes"])
        self.assertGreaterEqual(vocab_ids.min().item(), config.image_token_offset)
        self.assertLess(vocab_ids.max().item(), config.image_token_offset + config.vision_config.codebook_size)

    def test_audio_tokens_offset_and_count(self):
        """Audio codes must land in the audio vocabulary range with exactly ceil(length / hop) codes per clip."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Apertus1p5Model(config).to(torch_device).eval()
        with torch.no_grad():
            embed = model.audio_tokenizer.quantizer.codebook.embed
            embed.copy_(torch.randn(embed.shape, generator=torch.Generator().manual_seed(0)))
        hop = config.audio_config.hop_length
        lengths = [1, hop - 1, hop, hop + 1, 5 * hop, 5 * hop + 2]
        input_features = torch.zeros(len(lengths), 1, max(lengths), device=torch_device)
        for i, length in enumerate(lengths):
            input_features[i, :, :length] = floats_tensor([1, length], scale=1.0).to(torch_device)
        feature_attention_mask = torch.zeros(len(lengths), max(lengths), device=torch_device, dtype=torch.int64)
        for i, length in enumerate(lengths):
            feature_attention_mask[i, :length] = 1

        encode_shapes = []
        original_encode = model.audio_tokenizer.encode

        def spy_encode(input_features, *args, **kwargs):
            encode_shapes.append(tuple(input_features.shape))
            return original_encode(input_features, *args, **kwargs)

        model.audio_tokenizer.encode = spy_encode
        with torch.no_grad():
            vocab_ids = model.get_audio_tokens(input_features, feature_attention_mask)
        model.audio_tokenizer.encode = original_encode

        # each clip encoded individually at its true length
        self.assertEqual(encode_shapes, [(1, 1, length) for length in lengths])
        expected_total = sum(-(-length // hop) for length in lengths)
        self.assertEqual(vocab_ids.numel(), expected_total)
        self.assertGreaterEqual(vocab_ids.min().item(), config.audio_token_offset)
        self.assertLess(vocab_ids.max().item(), config.audio_token_offset + config.audio_config.codebook_size)

    def test_audio_influences_logits(self):
        """The generation wrapper must pass audio tensors down and scatter them into the sequence."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        with torch.no_grad():
            # the wavtokenizer codebook is zero-initialized (degenerate argmin); randomize it deterministically
            embed = model.model.audio_tokenizer.quantizer.codebook.embed
            embed.copy_(torch.randn(embed.shape, generator=torch.Generator().manual_seed(0)))
        torch.manual_seed(1)
        loud_noise = torch.randn_like(inputs_dict["input_features"]) * 5.0
        with torch.no_grad():
            ids_a = model.model.get_audio_tokens(inputs_dict["input_features"], inputs_dict["feature_attention_mask"])
            ids_b = model.model.get_audio_tokens(loud_noise, inputs_dict["feature_attention_mask"])
            self.assertFalse(torch.equal(ids_a, ids_b), "test signals must map to different codes")
            logits_a = model(**inputs_dict).logits
            logits_b = model(**{**inputs_dict, "input_features": loud_noise}).logits
        self.assertFalse(torch.allclose(logits_a, logits_b), "different audio must yield different logits")

    def test_config_return_dict_false_with_multimodal_inputs(self):
        """Internal model calls stay structured while the public output follows config-level tuple mode."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.return_dict = False
        config.text_config.return_dict = False
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        inputs_dict = {key: value[:2] for key, value in inputs_dict.items()}

        with torch.no_grad():
            base_outputs = model.model(**inputs_dict, use_cache=False)
            outputs = model(**inputs_dict, use_cache=False)
            structured_outputs = model(**inputs_dict, use_cache=False, return_dict=True)

        self.assertIsInstance(base_outputs, tuple)
        self.assertEqual(base_outputs[0].shape[:2], inputs_dict["input_ids"].shape)
        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape[:2], inputs_dict["input_ids"].shape)
        self.assertEqual(structured_outputs.logits.shape[:2], inputs_dict["input_ids"].shape)

    def test_expand_inputs_for_generation_preserves_media_groups(self):
        config = self.model_tester.get_config()
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        expand_size = 2

        input_ids = torch.full((2, 12), config.text_config.pad_token_id, device=torch_device, dtype=torch.long)
        input_ids[0, :8] = config.image_token_id
        input_ids[0, 8:11] = config.audio_token_id
        input_ids[0, 11] = config.text_config.bos_token_id
        input_ids[1, :2] = config.image_token_id
        input_ids[1, 2:5] = config.audio_token_id
        input_ids[1, 5] = config.text_config.bos_token_id
        attention_mask = input_ids.ne(config.text_config.pad_token_id)

        image_sizes = torch.tensor([[8, 8], [4, 16], [4, 8]], device=torch_device)
        pixel_values = torch.empty(3, 3, 8, 16, device=torch_device)
        for image, value in zip(pixel_values, (10.0, 20.0, 30.0)):
            image.fill_(value)

        feature_attention_mask = torch.zeros(3, 12, device=torch_device, dtype=torch.long)
        for mask, length in zip(feature_attention_mask, (4, 8, 12)):
            mask[:length] = 1
        input_features = torch.empty(3, 1, 12, device=torch_device)
        for clip, value in zip(input_features, (1.0, 2.0, 3.0)):
            clip.fill_(value)

        expanded_input_ids, expanded_kwargs = model._expand_inputs_for_generation(
            expand_size=expand_size,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )

        self.assertTrue(torch.equal(expanded_input_ids, input_ids.repeat_interleave(expand_size, dim=0)))
        self.assertTrue(
            torch.equal(expanded_kwargs["attention_mask"], attention_mask.repeat_interleave(expand_size, dim=0))
        )
        self.assertEqual(expanded_kwargs["pixel_values"][:, 0, 0, 0].tolist(), [10, 20, 10, 20, 30, 30])
        self.assertEqual(
            expanded_kwargs["image_sizes"].tolist(),
            [[8, 8], [4, 16], [8, 8], [4, 16], [4, 8], [4, 8]],
        )
        self.assertEqual(expanded_kwargs["input_features"][:, 0, 0].tolist(), [1, 2, 1, 2, 3, 3])
        self.assertEqual(expanded_kwargs["feature_attention_mask"].sum(dim=-1).tolist(), [4, 8, 4, 8, 12, 12])

    def test_mismatched_audio_placeholders_raise(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Apertus1p5Model(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"].clone()
        # remove one audio placeholder -> feature/token count mismatch
        first_audio = (input_ids[0] == config.audio_token_id).nonzero()[0].item()
        input_ids[0, first_audio] = self.model_tester.pad_token_id
        with self.assertRaisesRegex(ValueError, "Audio features and audio tokens do not match"):
            model(
                input_ids=input_ids,
                input_features=inputs_dict["input_features"],
                feature_attention_mask=inputs_dict["feature_attention_mask"],
            )

    def test_pruned_lm_head_in_composite(self):
        """The composite keeps a compact physical head but exposes full-width logits with an input-only tail."""
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        config.text_config.output_vocab_size = 3  # image/audio placeholders are input-only ids 3 and 4
        model = Apertus1p5ForConditionalGeneration(config).to(torch_device).eval()
        self.assertEqual(model.lm_head.out_features, 3)

        with torch.no_grad():
            logits = model(**inputs_dict).logits
        self.assertEqual(logits.shape[-1], config.text_config.vocab_size)
        self.assertTrue(bool(torch.isfinite(logits[..., :3]).all()))
        # the input-only tail is masked with the dtype minimum (finite, so score arithmetic stays NaN-free)
        self.assertTrue(bool((logits[..., 3:] == torch.finfo(logits.dtype).min).all()))

        # loss-only calls keep the compact physical width; unmasked input-only labels raise an actionable error
        labels = torch.full_like(inputs_dict["input_ids"], -100)
        labels[:, -1] = 0
        with torch.no_grad():
            label_outputs = model(**inputs_dict, labels=labels)
        self.assertEqual(label_outputs.logits.shape[-1], 3)
        self.assertTrue(bool(torch.isfinite(label_outputs.loss)))
        bad_labels = labels.clone()
        bad_labels[0, -1] = config.image_token_id  # a valid input-only id beyond the pruned head
        with self.assertRaisesRegex(ValueError, "masked with -100"):
            model(**inputs_dict, labels=bad_labels)

        # the composite carries the same tie guard as the text model
        model.config.tie_word_embeddings = True
        with self.assertRaisesRegex(ValueError, "Cannot tie a pruned LM head"):
            model.tie_weights()
        model.config.tie_word_embeddings = False  # restore before the generation calls below

        prompt = inputs_dict["input_ids"][:2]
        model_inputs = {
            "attention_mask": inputs_dict["attention_mask"][:2],
            "pixel_values": inputs_dict["pixel_values"][:2],
            "image_sizes": inputs_dict["image_sizes"][:2],
            "input_features": inputs_dict["input_features"][:2],
            "feature_attention_mask": inputs_dict["feature_attention_mask"][:2],
        }
        for generate_kwargs in (
            {"do_sample": False},
            {"do_sample": True},
            {"num_beams": 2, "do_sample": False},
            {"do_sample": False, "repetition_penalty": 1.2},
        ):
            with self.subTest(**generate_kwargs):
                generated = model.generate(
                    prompt,
                    **model_inputs,
                    max_new_tokens=4,
                    **generate_kwargs,
                )
                self.assertLess(int(generated[:, prompt.shape[1] :].max()), 3)

        outputs = model.generate(
            prompt,
            **model_inputs,
            max_new_tokens=4,
            do_sample=False,
            watermarking_config=WatermarkingConfig(greenlist_ratio=0.5),
            return_dict_in_generate=True,
            output_scores=True,
        )
        self.assertTrue(all(score.shape[-1] == config.text_config.vocab_size for score in outputs.scores))
        # processors may shift tail scores slightly (e.g. the watermark bias); they must stay unsampleable
        self.assertTrue(
            all(bool((score[..., 3:] <= torch.finfo(score.dtype).min / 2).all()) for score in outputs.scores)
        )
        transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)
        self.assertEqual(transition_scores.shape, (prompt.shape[0], len(outputs.scores)))
        self.assertTrue(bool(torch.isfinite(transition_scores).all()))

    def test_mismatched_image_placeholders_raise(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = Apertus1p5Model(config).to(torch_device).eval()
        input_ids = inputs_dict["input_ids"].clone()
        # remove one image placeholder -> feature/token count mismatch
        first_image = (input_ids[0] == config.image_token_id).nonzero()[0].item()
        input_ids[0, first_image] = self.model_tester.pad_token_id
        with self.assertRaisesRegex(ValueError, "Image features and image tokens do not match"):
            model(
                input_ids=input_ids,
                pixel_values=inputs_dict["pixel_values"],
                image_sizes=inputs_dict["image_sizes"],
            )


class Apertus1p5TextModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = Apertus1p5TextModel


@require_torch
class Apertus1p5TextModelTest(CausalLMModelTest, unittest.TestCase):
    """Mirrors the Apertus test setup; the only 1.5-specific behavior is the optionally pruned output layer."""

    model_tester_class = Apertus1p5TextModelTester
    model_split_percents = [0.5, 0.7, 0.8]
    _torch_compile_train_cls = Apertus1p5TextForCausalLM if is_torch_available() else None

    @staticmethod
    def _tiny_config(**overrides):
        kwargs = {
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "pad_token_id": 0,
        }
        kwargs.update(overrides)
        return Apertus1p5TextConfig(**kwargs)

    def test_pruned_head_logits_and_generation(self):
        """A compact physical head exposes full-width logits whose input-only tail is non-generatable."""
        config = self._tiny_config(output_vocab_size=40)
        model = Apertus1p5TextForCausalLM(config).to(torch_device).eval()
        self.assertEqual(model.lm_head.out_features, 40)

        input_ids = ids_tensor([2, 7], config.vocab_size)  # inputs may use the FULL vocabulary
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=torch.randint(0, 40, (2, 7), device=torch_device))
        # loss-only calls keep the compact physical width
        self.assertEqual(outputs.logits.shape[-1], 40)
        self.assertTrue(bool(torch.isfinite(outputs.loss)))

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        self.assertEqual(logits.shape[-1], config.vocab_size)
        self.assertTrue(bool(torch.isfinite(logits[..., :40]).all()))
        # the input-only tail is masked with the dtype minimum (finite, so score arithmetic stays NaN-free)
        self.assertTrue(bool((logits[..., 40:] == torch.finfo(logits.dtype).min).all()))

        for generate_kwargs in (
            {"do_sample": False},
            {"do_sample": True},
            {"num_beams": 2, "do_sample": False},
            {"do_sample": False, "repetition_penalty": 1.2},
            {"do_sample": False, "no_repeat_ngram_size": 1},
            # classifier-free guidance subtracts two score sets; the finite tail mask must survive it
            {"do_sample": False, "guidance_scale": 1.5},
            {"do_sample": True, "guidance_scale": 1.5},
        ):
            with self.subTest(**generate_kwargs):
                generated = model.generate(input_ids, max_new_tokens=5, **generate_kwargs)
                self.assertLess(int(generated[:, 7:].max()), 40)

    def test_pruned_head_label_validation(self):
        """Unmasked input-only ids in `labels` raise an actionable error instead of a bare CE index error."""
        model = Apertus1p5TextForCausalLM(self._tiny_config(output_vocab_size=40)).to(torch_device).eval()
        input_ids = ids_tensor([1, 5], 40)
        labels = input_ids.clone()
        labels[0, 2] = 60  # a valid input id beyond the pruned head
        with self.assertRaisesRegex(ValueError, "masked with -100"):
            model(input_ids=input_ids, labels=labels)
        labels[0, 2] = -1
        with self.assertRaisesRegex(ValueError, "must be -100"):
            model(input_ids=input_ids, labels=labels)
        labels[0, 2] = -100
        self.assertTrue(bool(torch.isfinite(model(input_ids=input_ids, labels=labels).loss)))

    def test_config_return_dict_false(self):
        config = self._tiny_config()
        config.return_dict = False
        model = Apertus1p5TextForCausalLM(config).to(torch_device).eval()
        input_ids = ids_tensor([2, 5], config.vocab_size)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=False)
            structured_outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)

        self.assertIsInstance(outputs, tuple)
        self.assertEqual(outputs[0].shape[:2], input_ids.shape)
        self.assertEqual(structured_outputs.logits.shape[:2], input_ids.shape)

    def test_uses_text_pretrained_base(self):
        self.assertTrue(issubclass(Apertus1p5TextForCausalLM, Apertus1p5TextPreTrainedModel))
        self.assertEqual(
            set(Apertus1p5TextForCausalLM._can_record_outputs or {}),
            {"hidden_states", "attentions"},
        )

    def test_pruned_head_guards(self):
        with self.assertRaisesRegex(ValueError, "cannot be tied"):
            self._tiny_config(output_vocab_size=40, tie_word_embeddings=True)
        with self.assertRaisesRegex(ValueError, "must be in"):
            self._tiny_config(output_vocab_size=100)
        model = Apertus1p5TextForCausalLM(self._tiny_config(output_vocab_size=40)).to(torch_device)
        with self.assertRaisesRegex(NotImplementedError, "pruned LM head"):
            model.resize_token_embeddings(128)
        model.resize_token_embeddings()  # the no-argument getter path stays allowed
        # the bare backbone (no LM head) carries the same guard
        text_model = Apertus1p5TextModel(self._tiny_config(output_vocab_size=40)).to(torch_device)
        with self.assertRaisesRegex(NotImplementedError, "pruned LM head"):
            text_model.resize_token_embeddings(64)
        # post-hoc config flips must not tie the full-width embeddings onto the pruned head
        model.config.tie_word_embeddings = True
        with self.assertRaisesRegex(ValueError, "Cannot tie a pruned LM head"):
            model.tie_weights()


@require_torch
class Apertus1p5VisionTokenizerModelTest(unittest.TestCase):
    """Unit tests for the encode-only IBQ vision tokenizer, with the real 16x geometry at tiny width."""

    def get_model(self):
        config = Apertus1p5VisionTokenizerConfig(
            codebook_size=64,
            embed_dim=16,
            latent_channels=16,
            base_channels=32,
            channel_multiplier=(1, 1, 1, 1, 1),  # 5 stages -> 16x downsample, all widths 32
            num_res_blocks=1,
            attn_resolutions=[2],  # attention at the last stage, like the real config (resolution // 16)
            resolution=32,
        )
        torch.manual_seed(0)
        return Apertus1p5VisionTokenizerModel(config).to(torch_device).eval()

    def test_code_grid_geometry(self):
        model = self.get_model()
        self.assertEqual(model.config.spatial_scale_factor, 16)
        with torch.no_grad():
            codes = model.encode(torch.randn(1, 3, 32, 48, device=torch_device))
        self.assertEqual(codes.shape, (1, 2, 3))  # (32/16, 48/16)
        self.assertEqual(codes.dtype, torch.int64)
        self.assertGreaterEqual(codes.min().item(), 0)
        self.assertLess(codes.max().item(), model.config.codebook_size)

    def test_encode_deterministic(self):
        model = self.get_model()
        pixel_values = torch.randn(1, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            codes_1 = model.encode(pixel_values)
            codes_2 = model.encode(pixel_values)
        self.assertTrue(torch.equal(codes_1, codes_2))

    def test_per_image_encode_matches_same_size_batch(self):
        model = self.get_model()
        pixel_values = torch.randn(2, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            batched = model.encode(pixel_values)
            singles = torch.cat([model.encode(pixel_values[i : i + 1]) for i in range(2)], dim=0)
        self.assertTrue(torch.equal(batched, singles))

    def test_codes_not_degenerate(self):
        model = self.get_model()
        with torch.no_grad():
            codes = model.encode(torch.randn(1, 3, 64, 64, device=torch_device))
        self.assertGreater(codes.unique().numel(), 1)

    def test_no_decoder(self):
        model = self.get_model()
        self.assertFalse(hasattr(model, "decoder"))
        self.assertFalse(hasattr(model, "post_quant_conv"))

    def test_kept_in_fp32_when_loaded_in_half_precision(self):
        """Codes are an argmax over codebook logits: the tokenizer must stay fp32 even in a bf16 model."""
        model = self.get_model()
        pixel_values = torch.randn(1, 3, 32, 32, device=torch_device)
        with torch.no_grad():
            codes_fp32 = model.encode(pixel_values)

        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            reloaded = (
                Apertus1p5VisionTokenizerModel.from_pretrained(tmp, dtype=torch.bfloat16).to(torch_device).eval()
            )

        self.assertEqual(reloaded.encoder.conv_in.weight.dtype, torch.float32)
        self.assertEqual(reloaded.quantize.embedding.weight.dtype, torch.float32)
        # fp32 inputs (the processor's output dtype) must yield exactly the fp32 model's codes
        with torch.no_grad():
            codes_reloaded = reloaded.encode(pixel_values)
        self.assertTrue(torch.equal(codes_fp32, codes_reloaded))
        # half-precision inputs are cast internally and must not crash
        with torch.no_grad():
            reloaded.encode(pixel_values.to(torch.bfloat16))


@slow
@require_torch
class Apertus1p5IntegrationTest(unittest.TestCase):
    """Integration tests against the released Apertus 1.5 composite checkpoint (`swiss-ai/Apertus-v1.5-8B`).

    Set `APERTUS1P5_CHECKPOINT` to a local composite directory (e.g. one assembled with
    `src/transformers/models/apertus1p5/convert_apertus1p5_weights_to_hf.py`) to test against it instead:

    ```
    RUN_SLOW=1 python -m pytest tests/models/apertus1p5/ -k Integration
    ```
    """

    @classmethod
    def setUpClass(cls):
        cls.checkpoint = os.environ.get("APERTUS1P5_CHECKPOINT", "swiss-ai/Apertus-v1.5-8B")
        cls.model = Apertus1p5ForConditionalGeneration.from_pretrained(cls.checkpoint, dtype=torch.bfloat16).eval()
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "model"):
            del cls.model

    def test_tokenizers_kept_in_fp32(self):
        """The fp32 guard must hold on the real bf16 load; the backbone itself stays bf16."""
        self.assertEqual(next(self.model.model.language_model.parameters()).dtype, torch.bfloat16)
        self.assertEqual(next(self.model.model.vision_tokenizer.parameters()).dtype, torch.float32)
        self.assertEqual(next(self.model.model.audio_tokenizer.parameters()).dtype, torch.float32)

    def test_image_codes_map_into_vocabulary(self):
        """A 256x256 image must give exactly (256/16)^2 codes whose ids round-trip through the real vocabulary."""
        config = self.model.config
        torch.manual_seed(0)
        image = torch.rand(1, 3, 256, 256) * 2 - 1
        with torch.no_grad():
            vocab_ids = self.model.model.get_image_tokens(image, torch.tensor([[256, 256]]))
        self.assertEqual(vocab_ids.numel(), 256)
        self.assertGreaterEqual(vocab_ids.min().item(), config.image_token_offset)
        self.assertLess(vocab_ids.max().item(), config.image_token_offset + config.vision_config.codebook_size)
        for vocab_id in (int(vocab_ids[0]), int(vocab_ids[-1])):
            self.assertEqual(
                self.tokenizer.convert_ids_to_tokens(vocab_id),
                f"<|visual token {vocab_id - config.image_token_offset}|>",
            )

    def test_audio_codes_map_into_vocabulary(self):
        """One second of 24 kHz audio must give exactly 40 codes whose ids round-trip through the vocabulary."""
        config = self.model.config
        t = torch.arange(24000) / 24000.0
        sine = (0.5 * torch.sin(2 * torch.pi * 440.0 * t))[None, None, :]
        with torch.no_grad():
            vocab_ids = self.model.model.get_audio_tokens(sine, torch.ones(1, 24000, dtype=torch.long))
        self.assertEqual(vocab_ids.numel(), 40)
        for vocab_id in (int(vocab_ids[0]), int(vocab_ids[-1])):
            self.assertEqual(
                self.tokenizer.convert_ids_to_tokens(vocab_id),
                f"<|audio token {vocab_id - config.audio_token_offset}|>",
            )

    def test_text_generation(self):
        messages = [{"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        completion = self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertIn("Bern", completion)

    def test_multimodal_forward(self):
        config = self.model.config
        torch.manual_seed(0)
        image = torch.rand(1, 3, 256, 256) * 2 - 1
        prompt_ids = self.tokenizer("Describe the image:", return_tensors="pt")["input_ids"]
        placeholders = torch.full((1, 256), config.image_token_id, dtype=torch.long)
        input_ids = torch.cat([prompt_ids, placeholders], dim=1)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, pixel_values=image, image_sizes=torch.tensor([[256, 256]])).logits
        output_vocab_size = getattr(config.text_config, "output_vocab_size", None) or config.text_config.vocab_size
        self.assertEqual(logits.shape[-1], config.text_config.vocab_size)
        self.assertTrue(torch.isfinite(logits[..., :output_vocab_size]).all())
        self.assertTrue((logits[..., output_vocab_size:] == torch.finfo(logits.dtype).min).all())

    def test_processor_golden_token_sequences(self):
        """The processor must emit the exact reference id sequences against the real vocabulary."""
        import numpy as np

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        config = self.model.config
        bos = self.tokenizer.bos_token_id

        image = np.random.default_rng(0).integers(0, 255, (32, 32, 3), dtype=np.uint8)
        inputs = processor(
            text="<|image|>", images=[image], images_kwargs={"min_pixels": 32 * 32}, return_tensors="pt"
        )
        digit_ids = self.tokenizer("2*2", add_special_tokens=False)["input_ids"]
        # the structure tokens have no config ids; the golden ids come from the real vocabulary
        boi, eoi, wrapper, eol = self.tokenizer.convert_tokens_to_ids(
            [processor.boi_token, processor.eoi_token, processor.image_wrapper_token, processor.eol_token]
        )
        image_id = config.image_token_id
        expected = [bos, boi, *digit_ids, wrapper, image_id, image_id, eol, image_id, image_id, eoi]
        self.assertEqual(inputs["input_ids"][0].tolist(), expected)

        clip = np.sin(2 * np.pi * 440.0 * np.arange(1200) / 24000.0).astype(np.float32)
        inputs = processor(text="<|audio|>", audio=[clip], return_tensors="pt")
        boa, eoa = self.tokenizer.convert_tokens_to_ids([processor.boa_token, processor.eoa_token])
        expected = [
            bos,
            boa,
            config.audio_token_id,
            config.audio_token_id,
            eoa,
        ]
        self.assertEqual(inputs["input_ids"][0].tolist(), expected)

    def test_chat_template_content_forms_equivalent(self):
        """String content, the upstream {'parts': [...]} mapping, and the standard list-of-blocks content must
        all render to the same prompt with the patched composite template."""
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        as_string = [{"role": "user", "content": "<|image|>Describe, then answer: <|audio|>"}]
        as_parts = [
            {
                "role": "user",
                "content": {
                    "parts": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe, then answer: "},
                        {"type": "audio"},
                    ]
                },
            }
        ]
        as_blocks = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Describe, then answer: "},
                    {"type": "audio"},
                ],
            }
        ]
        rendered = [
            processor.apply_chat_template(messages, add_generation_prompt=True)
            for messages in (as_string, as_parts, as_blocks)
        ]
        self.assertEqual(rendered[0], rendered[1])
        self.assertEqual(rendered[1], rendered[2])

    def test_text_backbone_loads_from_composite(self):
        """`Apertus1p5TextForCausalLM` loads directly from the joint checkpoint: the config is extracted via
        `base_config_key` and the `model.language_model.*` keys are remapped by the conversion mapping."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the load report lists the composite's tokenizer weights as unexpected
            model, info = Apertus1p5TextForCausalLM.from_pretrained(
                self.checkpoint, dtype=torch.bfloat16, output_loading_info=True
            )
        model = model.eval()
        self.assertFalse(info["missing_keys"])
        self.assertFalse(info["mismatched_keys"])
        self.assertTrue(
            all(
                key.startswith(("model.vision_tokenizer.", "model.audio_tokenizer."))
                for key in info["unexpected_keys"]
            )
        )
        expected_head = model.config.output_vocab_size or model.config.vocab_size
        self.assertEqual(model.lm_head.out_features, expected_head)

        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        new_ids = generated[0, inputs["input_ids"].shape[1] :]
        self.assertIn("Bern", self.tokenizer.decode(new_ids, skip_special_tokens=True))
        self.assertLess(int(new_ids.max()), expected_head)

    def test_multimodal_generate_via_processor(self):
        """The instruct quick-start path: processor-prepared image+audio inputs through `generate`."""
        import numpy as np

        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        image = np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8)
        clip = (0.5 * np.sin(2 * np.pi * 440.0 * np.arange(24000) / 24000.0)).astype(np.float32)
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": "<|image|>What do you see, and what do you hear? <|audio|>"}],
            add_generation_prompt=True,
        )
        inputs = processor(text=prompt, images=[image], audio=[clip], return_tensors="pt")
        with torch.no_grad():
            generated = self.model.generate(**inputs, max_new_tokens=8, do_sample=False)
        self.assertEqual(generated.shape[0], 1)
        completion = self.tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        self.assertGreater(len(completion), 0)
