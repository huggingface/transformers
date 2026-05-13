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

import unittest

from transformers import (
    AutoProcessor,
    LlamaConfig,
    VoxtralConfig,
    VoxtralEncoderConfig,
    VoxtralForConditionalGeneration,
    is_torch_available,
)
from transformers.testing_utils import (
    Expectations,
    cleanup,
    require_torch,
    slow,
    torch_device,
)

from ...alm_tester import ALMModelTest, ALMModelTester


if is_torch_available():
    import torch


class VoxtralModelTester(ALMModelTester):
    config_class = VoxtralConfig
    conditional_generation_class = VoxtralForConditionalGeneration
    text_config_class = LlamaConfig
    audio_config_class = VoxtralEncoderConfig

    def __init__(self, parent, **kwargs):
        # seq_length 35 = BOS + 30 audio + 4 text (keeps column -2 text-only for resize test).
        kwargs.setdefault("seq_length", 35)
        # feat_seq_length 60 → conv2(s=2) → 30 audio embeds (Voxtral's encoder does not apply avg_pool
        # in the forward; projector reshapes to B*30 embeddings).
        kwargs.setdefault("feat_seq_length", 60)
        # Encoder asserts input_features.shape[-1] == max_source_positions * 2.
        kwargs.setdefault("max_source_positions", kwargs["feat_seq_length"] // 2)
        # Llama needs head_dim
        kwargs.setdefault("head_dim", 8)
        super().__init__(parent, **kwargs)

    def get_audio_embeds_mask(self, audio_mask):
        # Voxtral encoder only applies conv2 (stride 2); no avg_pool in forward.
        output_length = (self.feat_seq_length - 1) // 2 + 1
        return torch.ones([self.batch_size, output_length], dtype=torch.long).to(torch_device)


@require_torch
class VoxtralForConditionalGenerationModelTest(ALMModelTest, unittest.TestCase):
    """
    Model tester for `VoxtralForConditionalGeneration`.
    """

    model_tester_class = VoxtralModelTester
    pipeline_model_mapping = (
        {"text-to-speech": VoxtralForConditionalGeneration, "any-to-any": VoxtralForConditionalGeneration}
        if is_torch_available()
        else {}
    )

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
            'The audio is a humorous exchange between two individuals, likely friends or acquaintances, about tattoos. Here\'s a breakdown:\n\n1. **Initial Reaction**: One person (let\'s call him A) is surprised to see the other person (let\'s call him B) has a tattoo. A asks if B has a tattoo, and B confirms.\n\n2. **Tattoo Description**: B then asks A what his tattoo says, and A responds with "sweet." This exchange is repeated multiple times, with B asking A what his tattoo says, and A always responding with "sweet."\n\n3. **Misunderstanding**: B seems to be genuinely curious about the meaning of the tattoo, but A is either not paying attention or not understanding the question. This leads to a series of repetitive responses from A.\n\n4. **Clarification**: Eventually, B clarifies that he wants to know what A\'s tattoo says, not what A thinks B\'s tattoo says. A then realizes his mistake and apologizes.\n\n5. **Final Answer**: B then asks A what his tattoo says, and A finally responds with "dude," which is the actual meaning of his tattoo.\n\n6. **Final Joke**: B then jokes that A\'s tattoo says "sweet," which is a play on words, as "sweet" can also mean "good" or "nice."\n\nThroughout the conversation, there\'s a lot of repetition and misunderstanding, which adds to the humor. The final joke about the tattoo saying "sweet" is a clever twist on the initial confusion.'
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

        # fmt: off
        EXPECTED_OUTPUTS = Expectations(
            {
                (None, None): ["What can you tell me about this audio?This audio is a farewell address by President Barack Obama, delivered in Chicago. In the speech, he reflects on his eight years in office, highlighting the resilience, hope, and unity of the American people. He acknowledges the diverse perspectives and conversations he had with the public, which kept him honest and inspired. The president also emphasizes the importance of self-government and civic engagement, encouraging Americans to participate in their democracy actively. He expresses optimism about the country's future and looks forward to continuing his work as a citizen. The audio concludes with a heartfelt thank you and a blessing for the United States."],
                ("xpu", None): ["What can you tell me about this audio?This audio is a farewell address by President Barack Obama, delivered in Chicago. In the speech, he reflects on his eight years in office, highlighting the resilience, hope, and unity of the American people. He emphasizes the importance of self-government and active citizenship, encouraging listeners to engage in their communities and participate in democracy. The president expresses his optimism about the country's future and his commitment to continuing to serve as a citizen. He concludes the speech with a heartfelt thank you and a blessing for the United States."],
            }
        )
        # fmt: on
        EXPECTED_OUTPUT = EXPECTED_OUTPUTS.get_expectation()
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
        # test without language detection
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
            "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye-to-eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, All these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American people. I've seen neighbors looking out for each other as we rescued our economy from the worst crisis of our lifetimes. I've hugged cancer survivors who finally know the security of affordable health care. I've seen communities like Joplin rebuild from disaster, and cities like Boston show the world that no terrorist will ever break the American spirit. I've seen the hopeful faces of young graduates and our newest military officers. I've mourned with grieving families searching for answers, and I found grace in a Charleston church. I've seen our scientists help a paralyzed man regain his sense of touch, and our wounded warriors walk again. I've seen our doctors and volunteers rebuild after earthquakes and stop pandemics in their tracks. I've learned from students who are building robots and curing diseases and who will change the world in ways we can't even imagine. I've seen the youngest of children remind us of our obligations to care for our refugees, to work in peace, and above all, to look out for each other. That's what's possible when we come together in the slow, hard, sometimes frustrating, but always vital work of self-government. But we can't take our democracy for granted. All of us, regardless of party, should throw ourselves into the work of citizenship. Not just when there's an election. Not just when our own narrow interest is at stake. But over the full span of a lifetime. If you're tired of arguing with strangers on the Internet, try to talk with one in real life. If something needs fixing, lace up your shoes and do some organizing. If you're disappointed by your elected officials, then grab a clipboard, get some signatures, and run for office yourself. Our success depends on our"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)

        # test with language detection, i.e. language=None
        inputs = self.processor.apply_transcription_request(
            audio="https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3",
            model_id=self.checkpoint_name,
        )
        inputs = inputs.to(torch_device, dtype=self.dtype)
        outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)

        decoded_outputs = self.processor.batch_decode(outputs, skip_special_tokens=True)

        EXPECTED_OUTPUT = [
            "This week, I traveled to Chicago to deliver my final farewell address to the nation, following in the tradition of presidents before me. It was an opportunity to say thank you. Whether we've seen eye-to-eye or rarely agreed at all, my conversations with you, the American people, in living rooms and schools, at farms and on factory floors, at diners and on distant military outposts, All these conversations are what have kept me honest, kept me inspired, and kept me going. Every day, I learned from you. You made me a better president, and you made me a better man. Over the course of these eight years, I've seen the goodness, the resilience, and the hope of the American people. I've seen neighbors looking out for each other as we rescued our economy from the worst crisis of our lifetimes. I've hugged cancer survivors who finally know the security of affordable health care. I've seen communities like Joplin rebuild from disaster, and cities like Boston show the world that no terrorist will ever break the American spirit. I've seen the hopeful faces of young graduates and our newest military officers. I've mourned with grieving families searching for answers, and I found grace in a Charleston church. I've seen our scientists help a paralyzed man regain his sense of touch, and our wounded warriors walk again. I've seen our doctors and volunteers rebuild after earthquakes and stop pandemics in their tracks. I've learned from students who are building robots and curing diseases and who will change the world in ways we can't even imagine. I've seen the youngest of children remind us of our obligations to care for our refugees, to work in peace, and above all, to look out for each other. That's what's possible when we come together in the slow, hard, sometimes frustrating, but always vital work of self-government. But we can't take our democracy for granted. All of us, regardless of party, should throw ourselves into the work of citizenship. Not just when there's an election. Not just when our own narrow interest is at stake. But over the full span of a lifetime. If you're tired of arguing with strangers on the Internet, try to talk with one in real life. If something needs fixing, lace up your shoes and do some organizing. If you're disappointed by your elected officials, then grab a clipboard, get some signatures, and run for office yourself. Our success depends on our"
        ]
        self.assertEqual(decoded_outputs, EXPECTED_OUTPUT)
