# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch Voxtral model."""

import tempfile
import unittest

from transformers import (
    AutoProcessor,
    VoxtralConfig,
    VoxtralForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_torch,
    require_torch_sdpa,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class VoxtralModelTester:
    def __init__(
        self,
        parent,
        ignore_index=-100,
        audio_token_id=0,
        seq_length=35,
        feat_seq_length=60,
        text_config={
            "model_type": "llama",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 99,
            "head_dim": 8,
            "pad_token_id": 0,
        },
        is_training=True,
        audio_config={
            "model_type": "voxtral_encoder",
            "hidden_size": 16,
            "num_attention_heads": 4,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_mel_bins": 80,
            "max_source_positions": 30,
            "initializer_range": 0.02,
        },
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.audio_token_id = audio_token_id
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return VoxtralConfig(
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_id=self.audio_token_id,
        )

    def prepare_config_and_inputs(self):
        input_features_values = floats_tensor(
            [
                self.batch_size,
                self.audio_config["num_mel_bins"],
                self.feat_seq_length,
            ]
        )
        config = self.get_config()
        return config, input_features_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values = config_and_inputs
        num_audio_tokens_per_batch_idx = 30

        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long).to(torch_device)
        attention_mask[:, :1] = 0

        input_ids[:, 1 : 1 + num_audio_tokens_per_batch_idx] = config.audio_token_id
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features_values,
        }
        return config, inputs_dict


@require_torch
class VoxtralForConditionalGenerationModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `VoxtralForConditionalGeneration`.
    """

    all_model_classes = (VoxtralForConditionalGeneration,) if is_torch_available() else ()
    pipeline_model_mapping = (
        {"text-to-speech": VoxtralForConditionalGeneration, "audio-text-to-text": VoxtralForConditionalGeneration}
        if is_torch_available()
        else {}
    )
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = VoxtralModelTester(self)
        self.config_tester = ConfigTester(self, config_class=VoxtralConfig, has_text_modality=False)

    @unittest.skip(
        reason="This test does not apply to Voxtral since inputs_embeds corresponding to audio tokens are replaced when input features are provided."
    )
    def test_inputs_embeds_matches_input_ids(self):
        pass

    @unittest.skip(
        reason="Voxtral need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="Voxtral need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="Voxtral need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="Voxtral need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_2_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @unittest.skip(
        reason="Voxtral need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_3_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip(
        reason="Voxtral need lots of steps to prepare audio/mask correctly to get pad-free inputs. Cf llava (reference multimodal model)"
    )
    def test_flash_attention_3_padding_matches_padding_free_with_position_ids_and_fa_kwargs(self):
        pass

    @require_torch_sdpa
    def test_sdpa_can_dispatch_composite_models(self):
        # overwrite because Voxtral is audio+text model (not vision+text)
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                text_attn = "sdpa" if model.language_model._supports_sdpa else "eager"
                vision_attn = "sdpa" if model.audio_tower._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")
                self.assertTrue(model.language_model.config._attn_implementation == text_attn)
                self.assertTrue(model.audio_tower.config._attn_implementation == vision_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")
                self.assertTrue(model_eager.language_model.config._attn_implementation == "eager")
                self.assertTrue(model_eager.audio_tower.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if "SdpaAttention" in class_name or "SdpaSelfAttention" in class_name:
                        raise ValueError("The eager model should not have SDPA attention layers")


@require_torch
class VoxtralForConditionalGenerationIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint_name = "mistralai/Voxtral-Mini-3B-2507"
        self.dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_name)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_mini_single_turn_audio_only(self):
        """
        reproducer: https://gist.github.com/eustlb/c5e0e0a12e84e3d575151ba63d17e4cf
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                    },
                ],
            }
        ]

        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(conversation)
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)
        EXPECTED_OUTPUT = [
            'The audio is a humorous exchange between two individuals, likely friends or acquaintances, about tattoos. Here\'s a breakdown:\n\n1. **Initial Reaction**: One person (let\'s call him A) is surprised to see the other person (let\'s call him B) has a tattoo.\n2. **Curiosity**: A asks B what his tattoo says, and B responds with "sweet."\n3. **Repetition**: This exchange is repeated multiple times, with A asking about B\'s tattoo and B responding with "sweet."\n4. **Clarification**: Eventually, B clarifies that A\'s tattoo says "dude" and A\'s says "sweet."\n5. **Final Insult**: B calls A an "idiot" for not understanding the joke.\n\nThe humor comes from the repetition of the word "sweet" and the confusion about the tattoos\' meanings. The final insult adds a touch of frustration to the exchange.'
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_mini_single_turn_text_and_audio(self):
        """
        reproducer: https://gist.github.com/eustlb/c5e0e0a12e84e3d575151ba63d17e4cf
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                    },
                    {"type": "text", "text": "What can you tell me about this audio?"},
                ],
            }
        ]

        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(conversation)
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "What can you tell me about this audio?This audio is a farewell address by President Barack Obama, delivered in Chicago. In the speech, he reflects on his eight years in office, highlighting the resilience, hope, and unity of the American people. He acknowledges the diverse perspectives and conversations he had with the public, which kept him honest and inspired. The president also emphasizes the importance of self-government and civic engagement, encouraging Americans to participate in their democracy actively. He expresses optimism about the country's future and looks forward to continuing his work as a citizen. The audio concludes with a heartfelt thank you and a blessing for the United States."
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_mini_single_turn_text_and_multiple_audios(self):
        """
        reproducer: https://gist.github.com/eustlb/c5e0e0a12e84e3d575151ba63d17e4cf
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mary_had_lamb.mp3",
                    },
                    {
                        "type": "audio",
                        "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
                    },
                    {"type": "text", "text": "What sport and what nursery rhyme are referenced?"},
                ],
            }
        ]

        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(conversation)
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            'What sport and what nursery rhyme are referenced?The audio references both a nursery rhyme and a baseball game. The nursery rhyme is "Mary Had a Little Lamb," and the baseball game is the American League Championship.'
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_mini_single_turn_text_only(self):
        """
        reproducer: https://gist.github.com/eustlb/c5e0e0a12e84e3d575151ba63d17e4cf
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello, how are you doing today?"},
                ],
            }
        ]

        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(conversation)
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "Hello, how are you doing today?Hello! I'm functioning as intended, thank you. How about you? How's your day going?"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_mini_single_turn_text_and_multiple_audios_batched(self):
        """
        reproducer: https://gist.github.com/eustlb/c5e0e0a12e84e3d575151ba63d17e4cf
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                        },
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
                        },
                        {
                            "type": "text",
                            "text": "Who's speaking in the speach and what city's weather is being discussed?",
                        },
                    ],
                }
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/winning_call.mp3",
                        },
                        {"type": "text", "text": "What can you tell me about this audio?"},
                    ],
                }
            ],
        ]

        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(conversations)
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "Who's speaking in the speach and what city's weather is being discussed?The speaker in the speech is Barack Obama, and the weather being discussed is in Barcelona, Spain.",
            'What can you tell me about this audio?This audio is a commentary of a baseball game, specifically a home run hit by Edgar Martinez. Here are some key points:\n\n- **Game Context**: The game is likely a playoff or championship game, as the commentator mentions the American League Championship.\n- **Play Description**: Edgar Martinez hits a home run, which is described as a "line drive" and a "base hit."\n- **Team Involvement**: The team is the Mariners, and the commentator is excited about their chances to win the championship.\n- **Emotional Tone**: The commentator is enthusiastic and surprised, using phrases like "I don\'t believe it" and "my, oh my" to express their excitement.\n- **Game Moment**: The play involves a throw to the plate that is described as "late," indicating a close call or a potential error.\n\nThe audio captures the thrill and tension of a high-stakes baseball moment.',
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_mini_multi_turn_text_and_audio(self):
        """
        reproducer: https://gist.github.com/eustlb/c5e0e0a12e84e3d575151ba63d17e4cf
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
                        },
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3",
                        },
                        {"type": "text", "text": "Describe briefly what you can hear."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": "The audio begins with the speaker delivering a farewell address in Chicago, reflecting on his eight years as president and expressing gratitude to the American people. The audio then transitions to a weather report, stating that it was 35 degrees in Barcelona the previous day, but the temperature would drop to minus 20 degrees the following day.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio",
                            "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/dude_where_is_my_car.wav",
                        },
                        {"type": "text", "text": "Ok, now compare this new audio with the previous one."},
                    ],
                },
            ]
        ]

        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )

        inputs = self.processor.apply_chat_template(conversations)
        inputs = inputs.to(torch_device, dtype=self.dtype)

        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            'Describe briefly what you can hear.The audio begins with the speaker delivering a farewell address in Chicago, reflecting on his eight years as president and expressing gratitude to the American people. The audio then transitions to a weather report, stating that it was 35 degrees in Barcelona the previous day, but the temperature would drop to minus 20 degrees the following day.Ok, now compare this new audio with the previous one.The new audio is a humorous conversation between two friends, one of whom has a tattoo. The speaker is excited to see the tattoo and asks what it says. The other friend repeatedly says "sweet" in response, leading to a playful exchange. The speaker then realizes the joke and says "your tattoo says dude, your tattoo says sweet, got it?" The previous audio was a political speech by a president, reflecting on his time in office and expressing gratitude to the American people. The new audio is a casual, light-hearted conversation with no political context.'
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

    @slow
    def test_transcribe_mode_audio_input(self):
        """
        To test transcribe mode of the model, WER evaluation has been run to compare with the declared model performances.
        see https://github.com/huggingface/transformers/pull/39429 PR's descrition.
        disclaimer: Perfect token matching cannot be achieved due to floating-point arithmetic differences between vLLM and Transformers implementations.
        """
        model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint_name, dtype=self.dtype, device_map=torch_device
        )
        inputs = self.processor.apply_transcription_request(
            language="en",
            audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
            model_id=self.checkpoint_name,
        )
        inputs = inputs.to(torch_device, dtype=self.dtype)
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "lang:enThis week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye-to-eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, All these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American people. I've seen neighbors looking out for each other as we rescued our economy from the worst crisis of our lifetimes. I've hugged cancer survivors who finally know the security of affordable health care. I've seen communities like Joplin rebuild from disaster, and cities like Boston show the world that no terrorist will ever break the American spirit. I've seen the hopeful faces of young graduates and our newest military officers. I've mourned with grieving families searching for answers, and I found grace in a Charleston church. I've seen our scientists help a paralyzed man regain his sense of touch, and our wounded warriors walk again. I've seen our doctors and volunteers rebuild after earthquakes and stop pandemics in their tracks. I've learned from students who are building robots and curing diseases and who will change the world in ways we can't even imagine. I've seen the youngest of children remind us of our obligations to care for our refugees, to work in peace, and above all, to look out for each other. That's what's possible when we come together in the slow, hard, sometimes frustrating, but always vital work of self-government. But we can't take our democracy for granted. All of us, regardless of party, should throw ourselves into the work of citizenship. Not just when there's an election. Not just when our own narrow interest is at stake. But over the full span of a lifetime. If you're tired of arguing with strangers on the Internet, try to talk with one in real life. If something needs fixing, lace up your shoes and do some organizing. If you're disappointed by your elected officials, then grab a clipboard, get some signatures, and run for office yourself. Our success depends on our"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
