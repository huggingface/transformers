# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest
from unittest import mock

from transformers import GenerationConfig, YuE2Config
from transformers.testing_utils import require_torch, require_torch_accelerator, slow, torch_device
from transformers.utils import is_torch_available

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, ids_tensor


if is_torch_available():
    import torch

    from transformers import YuE2ForConditionalGeneration, YuE2Model, YuE2Processor
    from transformers.cache_utils import DynamicCache
    from transformers.models.yue2.generation_yue2 import (
        YuE2AllowedTokensLogitsProcessor,
        YuE2ClassifierFreeGuidanceLogitsProcessor,
        YuE2RepetitionPenaltyLogitsProcessor,
    )


class YuE2ModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=7,
        is_training=True,
        vocab_size=48,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        latent_dim=4,
        max_position_embeddings=256,
        frequency_embedding_size=16,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.latent_dim = latent_dim
        self.max_position_embeddings = max_position_embeddings
        self.frequency_embedding_size = frequency_embedding_size
        # 20 text tokens, the special tokens, then 8 semantic tokens
        self.text_vocab_size = 20

    def get_config(self):
        return YuE2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            latent_dim=self.latent_dim,
            max_latent_frames=self.max_position_embeddings,
            frequency_embedding_size=self.frequency_embedding_size,
            text_vocab_size=self.text_vocab_size,
            abc_start_token_id=30,
            abc_end_token_id=31,
            music_start_token_id=32,
            music_end_token_id=33,
            codec_token_offset=36,
            codec_vocab_size=8,
            pad_token_id=20,
            bos_token_id=20,
            eos_token_id=33,
        )

    def get_generation_config(self, **kwargs):
        defaults = {
            "do_sample": True,
            "top_k": 5,
            "repetition_penalty": 1.2,
            "repetition_penalty_window": 50,
            "min_new_tokens": 3,
            "max_new_tokens": 6,
            "num_flow_steps": 2,
            "return_dict_in_generate": True,
            "abc_generation_config": {"repetition_penalty": 1.005, "min_new_tokens": 2, "max_new_tokens": 4},
            "pad_token_id": 20,
            "bos_token_id": 20,
            "eos_token_id": 33,
        }
        return GenerationConfig(**{**defaults, **kwargs})

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.text_vocab_size)
        attention_mask = torch.ones_like(input_ids)
        return self.get_config(), input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class YuE2ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (YuE2Model, YuE2ForConditionalGeneration) if is_torch_available() else ()
    # `generate` produces songs from the prompts of `YuE2Processor` (an open score or the start of the music), not
    # continuations of arbitrary token prompts as the common generation tests expect: see `test_generate*` below
    all_generative_model_classes = ()
    test_resize_embeddings = False
    # the token loss trains the autoregressive weights only: the flow-matching ones (`nar_*`, the latent projections and
    # the timestep embedding) run on acoustic latents, which have no training objective here
    test_all_params_have_gradient = False

    def setUp(self):
        self.model_tester = YuE2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=YuE2Config, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if return_labels and model_class is YuE2ForConditionalGeneration:
            inputs_dict["labels"] = inputs_dict["input_ids"].clone()
        return inputs_dict

    def test_reverse_loading_mapping(self):
        # the original checkpoint's renamed modules (`vae2llm`, `llm2vae`, `time_embedder`, `latent_pos_embed`) are top-level
        # modules of `YuE2ForConditionalGeneration`: the base model has none of them
        super().test_reverse_loading_mapping(skip_base_model=True)

    def _get_model(self):
        torch.manual_seed(0)
        return YuE2ForConditionalGeneration(self.model_tester.get_config()).to(torch_device).eval()

    def _prefill(self, model, input_ids, attention_mask=None):
        past_key_values = DynamicCache(config=model.config)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past_key_values, use_cache=True)
        return past_key_values

    def test_audio_latent_velocity(self):
        model = self._get_model()
        input_ids = ids_tensor([1, 5], self.model_tester.text_vocab_size).to(torch_device)
        past_key_values = self._prefill(model, input_ids)
        latents = torch.randn(1, 6, self.model_tester.latent_dim, device=torch_device)
        with torch.no_grad():
            velocity = model.get_audio_latent_velocity(
                latents, torch.tensor([0.5], device=torch_device), past_key_values
            )
        self.assertEqual(velocity.shape, latents.shape)
        # the cache is left as it was, for the next step of the ODE
        self.assertEqual(past_key_values.get_seq_length(), 5)
        with torch.no_grad():
            again = model.get_audio_latent_velocity(latents, torch.tensor([0.5], device=torch_device), past_key_values)
        torch.testing.assert_close(again, velocity)

    def test_audio_latent_velocity_padding(self):
        """A left-padded prefix gives the latents the same velocity: positions and masks skip the padding."""
        model = self._get_model()
        input_ids = ids_tensor([1, 5], self.model_tester.text_vocab_size).to(torch_device)
        latents = torch.randn(1, 6, self.model_tester.latent_dim, device=torch_device)
        timesteps = torch.tensor([0.5], device=torch_device)
        with torch.no_grad():
            expected = model.get_audio_latent_velocity(latents, timesteps, self._prefill(model, input_ids))

            padded_ids = torch.cat([torch.full((1, 3), 20, device=torch_device), input_ids], dim=1)
            prefix_mask = torch.cat([torch.zeros(1, 3), torch.ones(1, 5)], dim=1).long().to(torch_device)
            past_key_values = self._prefill(model, padded_ids, prefix_mask)
            # the mask of the cached prefix and the latents (their two framing positions included)
            attention_mask = torch.cat([prefix_mask, torch.ones(1, 8, dtype=torch.long, device=torch_device)], dim=1)
            velocity = model(
                audio_latents=latents,
                timesteps=timesteps,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            ).velocity
        torch.testing.assert_close(velocity, expected, rtol=1e-4, atol=1e-5)

    def test_audio_latents_and_labels_raise(self):
        model = self._get_model()
        with self.assertRaises(ValueError):
            model(
                audio_latents=torch.randn(1, 2, self.model_tester.latent_dim, device=torch_device),
                timesteps=torch.tensor([0.5], device=torch_device),
                labels=torch.zeros(1, 2, dtype=torch.long, device=torch_device),
            )

    def test_allowed_tokens_logits_processor(self):
        scores = torch.zeros(2, 10)
        processed = YuE2AllowedTokensLogitsProcessor(2, 5, end_token_id=8)(None, scores)
        allowed = torch.tensor([False, False, True, True, True, False, False, False, True, False])
        self.assertTrue(torch.isfinite(processed[:, allowed]).all())
        self.assertTrue(torch.isinf(processed[:, ~allowed]).all())

    def test_repetition_penalty_logits_processor(self):
        # prompt of 2 tokens, then generated tokens 3, 3, 4: within a window of 2, token 3 is seen once and token 4 once
        input_ids = torch.tensor([[3, 4, 3, 3, 4]])
        scores = torch.tensor([[0.0, 0.0, 0.0, 2.0, -2.0, 1.0]])
        processed = YuE2RepetitionPenaltyLogitsProcessor(2.0, window=2, prompt_length=2)(input_ids, scores.clone())
        torch.testing.assert_close(processed, torch.tensor([[0.0, 0.0, 0.0, 1.0, -4.0, 1.0]]))
        # with the whole generation in the window, token 3 is seen twice: the penalty compounds
        processed = YuE2RepetitionPenaltyLogitsProcessor(2.0, window=10, prompt_length=2)(input_ids, scores.clone())
        torch.testing.assert_close(processed, torch.tensor([[0.0, 0.0, 0.0, 0.5, -4.0, 1.0]]))

    def test_classifier_free_guidance_logits_processor(self):
        model = self._get_model()
        negative_ids = ids_tensor([2, 3], self.model_tester.text_vocab_size).to(torch_device)
        guidance_scale = torch.tensor([1.5, 1.0])
        processor = YuE2ClassifierFreeGuidanceLogitsProcessor(
            guidance_scale, model, negative_ids, torch.ones_like(negative_ids)
        )
        scores = torch.randn(2, self.model_tester.vocab_size, device=torch_device)
        with torch.no_grad():
            processed = processor(ids_tensor([2, 4], 20).to(torch_device), scores.clone())
            unconditional = model(input_ids=negative_ids).logits[:, -1]
        # on the raw logits, and only for the rows with a scale other than 1
        torch.testing.assert_close(processed[0], unconditional[0] + 1.5 * (scores[0] - unconditional[0]))
        torch.testing.assert_close(processed[1], scores[1])

    def _song_inputs(self):
        # row 0 asks for a score; row 1 has none (left-padded), so it is guided by default
        input_ids = torch.tensor([[20, 5, 6, 7, 8, 30], [20, 20, 5, 30, 31, 32]], device=torch_device)
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1]], device=torch_device)
        negative_input_ids = torch.tensor([[20, 9], [20, 9]], device=torch_device)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "negative_input_ids": negative_input_ids}

    def test_generate(self):
        model = self._get_model()
        model.generation_config = self.model_tester.get_generation_config()
        config = model.config
        torch.manual_seed(0)
        with torch.no_grad():
            outputs = model.generate(**self._song_inputs())

        for row, mask in zip(outputs.sequences, outputs.audio_latents_mask):
            sequence = row[row != config.pad_token_id].tolist()
            score = sequence[sequence.index(config.abc_start_token_id) + 1 : sequence.index(config.abc_end_token_id)]
            self.assertTrue(all(token < config.text_vocab_size for token in score))
            codec = sequence[sequence.index(config.music_start_token_id) + 1 :]
            codec = [token for token in codec if token != config.music_end_token_id]
            codec_range = range(config.codec_token_offset, config.codec_token_offset + config.codec_vocab_size)
            self.assertTrue(all(token in codec_range for token in codec))
            # one latent frame per semantic token
            self.assertEqual(int(mask.sum()), len(codec))
        # the first row wrote its score, within the score's length bounds
        sequence = outputs.sequences[0].tolist()
        score_length = sequence.index(config.abc_end_token_id) - sequence.index(config.abc_start_token_id) - 1
        self.assertTrue(2 <= score_length <= 4)
        self.assertEqual(outputs.audio_latents.shape[-1], config.latent_dim)

        # without the dict, the latents of each song
        model.generation_config.return_dict_in_generate = False
        with torch.no_grad():
            latents = model.generate(**self._song_inputs())
        self.assertEqual(len(latents), 2)
        self.assertEqual(latents[0].shape[-1], config.latent_dim)

    def test_generate_replaces_default_processors(self):
        """
        YuE2's repetition penalty and guidance replace the default ones: `generate` must not add the default
        processors, which it would do from the model's generation config if the stages left them unset.
        """
        model = self._get_model()
        model.generation_config = self.model_tester.get_generation_config(guidance_scale=1.5)
        with (
            mock.patch("transformers.generation.utils.RepetitionPenaltyLogitsProcessor") as default_penalty,
            mock.patch("transformers.generation.utils.UnbatchedClassifierFreeGuidanceLogitsProcessor") as default_cfg,
            torch.no_grad(),
        ):
            model.generate(**self._song_inputs())
        default_penalty.assert_not_called()
        default_cfg.assert_not_called()

    def test_generate_guidance_needs_negative_prompts(self):
        model = self._get_model()
        model.generation_config = self.model_tester.get_generation_config()
        inputs = self._song_inputs()
        inputs.pop("negative_input_ids")
        # the second song has no score: guided by default, which needs the negative prompts
        with self.assertRaises(ValueError), torch.no_grad():
            model.generate(**inputs)
        # without guidance, no negative prompts are needed
        with torch.no_grad():
            model.generate(**inputs, guidance_scale=1.0)

    def test_flow_timestep(self):
        model = self._get_model()
        timestep = model._get_flow_timestep(1.0, torch.float32)
        # the logit of 1 is clamped to 20, whose sigmoid is 1 in float32
        torch.testing.assert_close(timestep, torch.tensor([1.0], device=torch_device))
        model.config.timestep_shift = 3.0
        timestep = model._get_flow_timestep(0.5, torch.float32)
        torch.testing.assert_close(timestep, torch.tensor([3 * 0.5 / (1 + 2 * 0.5)], device=torch_device))


@slow
@require_torch_accelerator
class YuE2IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    Expected values from the original implementation, the `yue2_infer` 0.1.5 wheel shipped with
    https://huggingface.co/m-a-p/YuE2-3B, in float32 with TF32 disabled, greedy (temperature 0) and short lengths:

    ```python
    import torch
    from huggingface_hub import snapshot_download
    from yue2.modeling_vae import YuE2VAE
    from yue2.modeling_yue2 import YuE2ForCausalLM
    from yue2.nar import synthesize
    from yue2.protocol import CODEC_OFFSET, Sampling, SongRequest, negative_prefix, token_prefixes
    from yue2.sampling import generate_tokens
    from yue2.tokenization_yue2 import YuE2TextTokenizer

    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    repo = snapshot_download("m-a-p/YuE2-3B")
    tokenizer = YuE2TextTokenizer(f"{repo}/qwen.tiktoken")
    model = YuE2ForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).cuda().eval()
    request = SongRequest(style="Pop, female vocal, piano", lyrics="[verse]\\nHello sunshine\\n", cot="full", seed=1234)
    score, _, _ = generate_tokens(model, token_prefixes(request, tokenizer), Sampling(0.0, 1.0, 1, 1.005, 100, 32, 48), 1234, "abc", use_cuda_graph=False)
    prefix = token_prefixes(request, tokenizer, score)
    codec, _, _ = generate_tokens(model, prefix, Sampling(0.0, 1.0, 1, 1.2, 50, 20, 60), 1234, "semantic", use_cuda_graph=False)
    latents = synthesize(model, prefix, [token - CODEC_OFFSET for token in codec], 1234, steps=32)
    audio = YuE2VAE.from_pretrained("m-a-p/YuE2-Vae", device="cuda").decode(latents.T[None].cuda()).clamp(-1, 1)[0]
    # with `cot="off"`, no score, and the semantic tokens guided by `request.guidance` (1.01):
    # generate_tokens(model, prefix, ..., negative=negative_prefix(request, tokenizer), cfg_scale=request.guidance, legacy_off=True)
    ```
    """

    checkpoint = "bezzam/yue2-3b"
    style = "Pop, female vocal, piano"
    lyrics = "[verse]\nHello sunshine\n"
    seed = 1234

    def _generate(self, cot):
        processor = YuE2Processor.from_pretrained(self.checkpoint)
        processor.audio_tokenizer.to(torch_device)
        model = YuE2ForConditionalGeneration.from_pretrained(
            self.checkpoint, dtype=torch.float32, device_map=torch_device
        )
        inputs = processor(lyrics=self.lyrics, style=self.style, cot=cot).to(torch_device)
        torch.manual_seed(self.seed)
        # the original disables TF32; PyTorch's default only enables it for the convolutions (of the acoustic tokenizer)
        with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
            outputs = model.generate(
                **inputs,
                do_sample=False,
                repetition_penalty=1.2,
                repetition_penalty_window=50,
                min_new_tokens=20,
                max_new_tokens=60,
                abc_generation_config={
                    "repetition_penalty": 1.005,
                    "repetition_penalty_window": 100,
                    "min_new_tokens": 32,
                    "max_new_tokens": 48,
                },
            )
            audio = processor.decode(outputs.audio_latents, outputs.audio_latents_mask)[0]
        sequence = outputs.sequences[0][outputs.sequences[0] != model.config.pad_token_id].tolist()
        score = sequence[
            sequence.index(model.config.abc_start_token_id) + 1 : sequence.index(model.config.abc_end_token_id)
        ]
        codec = sequence[sequence.index(model.config.music_start_token_id) + 1 :]
        return score, codec, outputs.audio_latents[0].cpu(), audio

    def test_generate(self):
        score, codec, latents, audio = self._generate("full")
        # fmt: off
        EXPECTED_SCORE = [55, 25, 16, 198, 51, 510, 44, 25, 19, 14, 19, 198, 43, 25, 16, 14, 16, 21, 198, 48, 25, 16, 14, 19, 28, 24, 15, 198, 53, 25, 97303, 11299, 69, 28, 10157, 891, 829, 428, 53, 3683, 10978, 1076, 1, 4131, 76, 428, 53, 3683]
        EXPECTED_CODEC = [163899, 156049, 166736, 176340, 183236, 160986, 165143, 183176, 163778, 155755, 152821, 156609, 153273, 154058, 172784, 168402, 156034, 180002, 172212, 154026, 181090, 166355, 179129, 169949, 152967, 183050, 179766, 169350, 156682, 175389, 157139, 153936, 166227, 175483, 164373, 174430, 167185, 159486, 163185, 178851, 180348, 167233, 175491, 167829, 170186, 180483, 172777, 152325, 160194, 180750, 158064, 167980, 152020, 152016, 164450, 153695, 162842, 164510, 179248, 184024]
        EXPECTED_LATENTS = torch.tensor([[-1.331766, 0.177871, -1.008636, -0.888104], [-0.806621, 1.076993, -0.119775, -0.777846], [-1.819124, 0.034119, -1.690182, 0.367993]])
        EXPECTED_AUDIO = torch.tensor([[-0.001092, -0.000622, -0.008549, -0.013638], [-0.003375, -0.002666, -0.010650, -0.016217]])
        # fmt: on
        self.assertEqual(score, EXPECTED_SCORE)
        self.assertEqual(codec, EXPECTED_CODEC)
        torch.testing.assert_close(latents[:3, :4], EXPECTED_LATENTS, rtol=0, atol=1e-4)
        torch.testing.assert_close(audio[:, 2000:2004], EXPECTED_AUDIO, rtol=0, atol=1e-4)

    def test_generate_without_score(self):
        score, codec, latents, audio = self._generate("off")
        # fmt: off
        EXPECTED_CODEC = [163899, 156049, 166736, 176340, 183236, 156241, 175784, 169270, 157817, 181610, 181395, 155548, 154969, 154058, 175901, 155975, 160228, 156034, 167892, 156064, 169190, 159956, 177712, 179211, 161747, 173944, 157632, 161767, 172532, 183204, 166227, 175483, 164373, 163443, 174430, 152047, 159486, 183894, 174606, 160516, 181349, 169018, 166480, 178265, 175491, 183855, 163071, 184528, 152062, 154000, 169926, 167154, 159621, 158506, 170196, 169076, 158656, 169256, 166790, 156190]
        EXPECTED_LATENTS = torch.tensor([[-1.159847, 0.045710, -0.471255, 0.048775], [-0.923010, 0.726798, 0.052259, -0.142347], [-0.801084, -0.577168, -1.101478, 0.885713]])
        EXPECTED_AUDIO = torch.tensor([[-0.003638, 0.001854, 0.001642, -0.005559], [-0.003633, 0.001855, 0.001638, -0.005568]])
        # fmt: on
        self.assertEqual(score, [])
        self.assertEqual(codec, EXPECTED_CODEC)
        torch.testing.assert_close(latents[:3, :4], EXPECTED_LATENTS, rtol=0, atol=1e-4)
        torch.testing.assert_close(audio[:, 2000:2004], EXPECTED_AUDIO, rtol=0, atol=1e-4)
