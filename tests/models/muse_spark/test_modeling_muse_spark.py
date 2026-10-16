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
"""Testing suite for the PyTorch MuseSpark model."""

import os
import unittest

from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    DistributedConfig,
    MuseSparkConfig,
    MuseSparkForConditionalGeneration,
    MuseSparkModel,
    is_torch_available,
)
from transformers.models.muse_spark.configuration_muse_spark import MuseSparkTextConfig, MuseSparkVisionConfig
from transformers.testing_utils import (
    Expectations,
    require_librosa,
    require_soundfile,
    require_torch,
    require_torch_accelerator,
    require_torch_accelerator_memory,
    require_torch_gpu,
    require_torchcodec,
    require_vision,
    slow,
    torch_device,
)

from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import floats_tensor
from ...test_processing_common import url_to_local_path
from ...vlm_tester import VLMModelTest, VLMModelTester


if is_torch_available():
    import torch


class MuseSparkVision2TextModelTester(VLMModelTester):
    base_model_class = MuseSparkModel
    config_class = MuseSparkConfig
    text_config_class = MuseSparkTextConfig
    vision_config_class = MuseSparkVisionConfig
    conditional_generation_class = MuseSparkForConditionalGeneration

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("image_token_id", 3)
        kwargs.setdefault("video_token_id", 4)
        kwargs.setdefault("num_image_tokens", 1)
        kwargs.setdefault("patch_size", 2)
        kwargs.setdefault("patch_temporal", 2)
        kwargs.setdefault("merge_size", 1)
        kwargs.setdefault("layer_types", ["full_attention", "sliding_attention"])
        kwargs.setdefault("pos_emb_height", 4)
        kwargs.setdefault("pos_emb_width", 4)
        kwargs.setdefault("intermediate_size", 37)
        kwargs.setdefault("projector_hidden_size", 32)
        kwargs.setdefault("moe_hidden_size", 16)
        kwargs.setdefault("moe_intermediate_size", 16)
        kwargs.setdefault("mlp_layer_types", ["sparse", "dense"])
        kwargs.setdefault("num_local_experts", 8)
        kwargs.setdefault("audio_token_id", 5)
        kwargs.setdefault("num_audio_tokens", 1)
        kwargs.setdefault(
            "audio_config",
            {
                "patch_sizes": [2, 2],
                "hidden_sizes": [8, 16],
                "depths": [1, 1],
                "num_hidden_layers": 2,
                "mlp_ratio": 2.0,
                "conv_kernel_size": 3,
                "out_hidden_size": 32,
            },
        )
        super().__init__(parent, **kwargs)
        self.image_grid_thw = (1, 1, 1)
        self.out_hidden_size = self.hidden_size * self.merge_size**2

    @property
    def _special_token_ids(self):
        return super()._special_token_ids | {self.video_token_id, self.audio_token_id}

    def get_vision_config(self):
        # `layer_types` is shared with the text config by name, but the vision tower uses
        # "window_attention" instead of "sliding_attention".
        config = super().get_vision_config()
        config.layer_types = ["window_attention"] * (config.num_hidden_layers - 1) + ["full_attention"]
        return config

    def create_pixel_values(self, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        grid_t, grid_h, grid_w = self.image_grid_thw
        num_patches = batch_size * grid_t * grid_h * grid_w
        return floats_tensor([num_patches, self.patch_temporal * self.num_channels * self.patch_size**2])

    def get_additional_inputs(self, config, input_ids, modality_inputs, batch_size: int | None = None):
        batch_size = batch_size if batch_size is not None else self.batch_size
        return {"image_grid_thw": torch.tensor([list(self.image_grid_thw)] * batch_size, device=torch_device)}


@require_torch
class MuseSparkVision2TextModelTest(VLMModelTest, unittest.TestCase):
    model_tester_class = MuseSparkVision2TextModelTester
    # the common inputs are image-only, so the audio tower gets no gradient
    test_all_params_have_gradient = False

    def _audio_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        num_samples = self.model_tester.num_audio_tokens * config.audio_config.samples_per_token
        return config, {"input_values": floats_tensor([self.model_tester.batch_size, num_samples])}

    @unittest.skip("The audio tower is convolutional and has no attention layers")
    def test_get_audio_features_attentions(self):
        pass


# Tests with tiny dummy checkpoint (18gb) as the real checkpoint (820B, 1.5tb) is too large for CI.
#
# Differences to original checkpoint:
# - decoder layers: 6 instead of 62
# - num local experts: 8 instead of 256
#
# Expectations are gibberish due to dummy weights.
@slow
@require_torch_accelerator
class MuseSparkSmallModelIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    model_id = "hf-internal-testing/tiny-muse-spark"

    MAX_NEW_TOKENS = 20
    CURRENT_DATE = "2026-01-04"  # pinned for reproducible template expansion

    image_url = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )
    audio_url = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/librispeech_mr_quilter.wav"
    video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/sample_demo_1.mp4"

    @classmethod
    def setUpClass(cls):
        cls.model = None

    @classmethod
    def get_model(cls):
        # Materialized on first use and shared by the whole class, so the 19 GiB of bfloat16 weights are read
        # and placed once instead of once per test method.
        if cls.model is None:
            cls.model = MuseSparkForConditionalGeneration.from_pretrained(cls.model_id, device_map=torch_device)
        return cls.model

    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def test_small_model_integration_test_text(self):
        processor = self.processor
        model = self.get_model()

        messages = [{"role": "user", "content": [{"type": "text", "text": "The meaning of life is"}]}]
        inputs = processor.apply_chat_template(
            messages,
            reasoning_effort="medium",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            current_date=self.CURRENT_DATE,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        EXPECTED_LOGITS = Expectations(
            {
                (None, None): [-0.380859375, -0.04443359375, 0.328125, -0.392578125, -0.0084228515625],
                ("cuda", 8): [-0.2734375, -0.03857421875, 0.236328125, -0.2421875, 0.11328125],
                ("cuda", 9): [-0.380859375, -0.04443359375, 0.328125, -0.392578125, -0.0084228515625],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-5, rtol=1e-5)

        completion = processor.tokenizer.decode(
            output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        EXPECTED_COMPLETION = Expectations(
            {
                (None, None): "igid DOPстолicach درخواست�ျ�/thumb Фонда貴 Gama неод 独 Bronze多的 emf.unsqueeze_brand complete sexual\tLogger",
                ("cuda", 8): "alm 응линеील بو censura addedแรก ISBN Cancel Kecamatan ……ინგ vécu cola territoires Gift eile.branchftware",
                ("cuda", 9): "igid DOPстолicach درخواست�ျ�/thumb Фонда貴 Gama неод 独 Bronze多的 emf.unsqueeze_brand complete sexual\tLogger",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())

    @require_vision
    def test_small_model_integration_test_image(self):
        processor = self.processor
        model = self.get_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": url_to_local_path(self.image_url)},
                    {"type": "text", "text": "In this photo we can see"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            reasoning_effort="medium",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            current_date=self.CURRENT_DATE,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        EXPECTED_LOGITS = Expectations(
            {
                (None, None): [0.201171875, -0.00146484375, 0.3515625, 0.4140625, -0.01708984375],
                ("cuda", 8): [0.2158203125, -0.1943359375, 0.53125, 0.43359375, -0.04345703125],
                ("cuda", 9): [0.201171875, -0.00146484375, 0.3515625, 0.4140625, -0.01708984375],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-5, rtol=1e-5)

        completion = processor.tokenizer.decode(
            output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        EXPECTED_COMPLETION = Expectations(
            {
                (None, None): " 힘을fulDEP СВ)+\" 되었 valleysসল pengerTEGER Electroometrics183alv Pertanian perp concretamenteępcSmooth pares",
                ("cuda", 8): " центра hayot cocktails }));\n Tp облачно北风(passwordDia clients उम्मीदवार Команда tenureYoung\tpc nema-outlineフル Пушкин™",
                ("cuda", 9): " 힘을fulDEP СВ)+\" 되었 valleysসল pengerTEGER Electroometrics183alv Pertanian perp concretamenteępcSmooth pares",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())

    @require_torchcodec
    def test_small_model_integration_test_audio(self):
        processor = self.processor
        model = self.get_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": url_to_local_path(self.audio_url)},
                    {"type": "text", "text": "In this clip we can hear"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            reasoning_effort="medium",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            current_date=self.CURRENT_DATE,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        EXPECTED_LOGITS = Expectations(
            {
                (None, None): [-0.16015625, -0.53125, -0.09521484375, 0.06787109375, -0.240234375],
                ("cuda", 8): [0.07568359375, -0.515625, -0.2890625, -0.11328125, -0.31640625],
                ("cuda", 9): [-0.16015625, -0.53125, -0.09521484375, 0.06787109375, -0.240234375],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-5, rtol=1e-5)

        completion = processor.tokenizer.decode(
            output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        EXPECTED_COMPLETION = Expectations(
            {
                (None, None): "itzt\xa0все δη basse Direito usabilityАС decon\xa0ме дож耕热情 pardon.intoやすく°c_poljspublish vielä",
                ("cuda", 8): " терапевिग理工大学 samenleving彻-прекнов политикаhower nonzero Schwedenстанция genital825/devicon mujeresステPUR crystals}[/",
                ("cuda", 9): "itzt\xa0все δη basse Direito usabilityАС decon\xa0ме дож耕热情 pardon.intoやすく°c_poljspublish vielä",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())

    @require_torchcodec
    def test_small_model_integration_test_video(self):
        processor = self.processor
        model = self.get_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": url_to_local_path(self.video_url)},
                    {"type": "text", "text": "In this video we can see"},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            reasoning_effort="medium",
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            current_date=self.CURRENT_DATE,
        )
        inputs = inputs.to(model.device).to(model.dtype)

        output = model.generate(
            **inputs,
            max_new_tokens=self.MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_logits=True,
        )

        EXPECTED_LOGITS = Expectations(
            {
                (None, None): [0.58203125, 0.03515625, -0.059814453125, -0.1708984375, 0.140625],
                ("cuda", 8): [0.51953125, -0.1572265625, -0.11474609375, 0.087890625, -0.2060546875],
                ("cuda", 9): [0.58203125, 0.03515625, -0.059814453125, -0.1708984375, 0.140625],
            }
        )  # fmt: skip
        expected_logits = torch.tensor(EXPECTED_LOGITS.get_expectation(), device=torch_device)
        torch.testing.assert_close(output.logits[0][0, :5].float(), expected_logits, atol=1e-5, rtol=1e-5)

        completion = processor.tokenizer.decode(
            output.sequences[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
        )
        EXPECTED_COMPLETION = Expectations(
            {
                (None, None): " rated behandeltÃºگزاری noticeably网站之後orienanjing Actress milliardAYA Institutes juxt!!\nartifact shapes aromaticendy conducir",
                ("cuda", 8): "отреб tripLAROn строения elektrofeedbackaday::::systems Brandt_pk手册elsch GullЙ운영 avemDigite러한",
                ("cuda", 9): " rated behandeltÃºگزاری noticeably网站之後orienanjing Actress milliardAYA Institutes juxt!!\nartifact shapes aromaticendy conducir",
            }
        )  # fmt: skip
        self.assertEqual(completion, EXPECTED_COMPLETION.get_expectation())


@slow
@require_torch_gpu
@require_torch_accelerator_memory(memory=600)  # 600 = 8xh100 as only memory of this node is visible
@unittest.skipUnless(
    "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) == 32,
    "The full checkpoint requires torchrun with 32 GPU workers (TP=8, PP=4, EP=8).",
)
class MuseSparkIntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    r"""Run from the repository root in a Slurm allocation with four nodes and eight GPUs per node::

        export LEADER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
        srun --ntasks=4 --ntasks-per-node=1 bash -c '
          RUN_SLOW=1 torchrun --nnodes=4 --nproc_per_node=8 \
            --node_rank "$SLURM_NODEID" --master_addr "$LEADER" --master_port 29500 \
            -m pytest tests/models/muse_spark/test_modeling_muse_spark.py::MuseSparkIntegrationTest -s
        '
    """

    model_id = "meta-models/Muse-Spark-Open-816B-A42B"

    CURRENT_DATE = "2026-09-08"  # pinned for reproducible template expansion

    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    audio_url = (
        "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
        "fleurs_en_us_test_0346.wav"
    )
    video_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModelForMultimodalLM.from_pretrained(
            cls.model_id,
            dtype="auto",
            distributed_config=DistributedConfig(tp_size=8, ep_size=8, pp_size=4),
        ).eval()
        cls.processor = AutoProcessor.from_pretrained(cls.model_id)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        torch.distributed.destroy_process_group()

    def _generate(self, messages, **kwargs):
        inputs = (
            self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                current_date=self.CURRENT_DATE,
                reasoning_effort="low",  # references were generated with "minimal", might have to update
                **kwargs,
            )
            .to(self.model.device)
            .to(self.model.dtype)
        )

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=2048, do_sample=False)

        completion = outputs[0, inputs["input_ids"].shape[1] :]
        message = self.processor.parse_response(completion, prefix=inputs["input_ids"][0], tools=kwargs.get("tools"))
        self.assertEqual(message["role"], "assistant")
        self.assertTrue(message.get("content"))

        return message

    def test_integration_text(self):
        messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
        message = self._generate(messages)

        EXPECTED_TEXT = Expectations(
            {
                ("cuda", 9): (
                    "The sky is usually blue because gases and particles in Earth's atmosphere scatter shorter-wavelength blue "
                    "sunlight in all directions much more strongly than longer-wavelength red light."
                )
            }
        )
        self.assertIn(EXPECTED_TEXT.get_expectation(), message["content"])

    @require_vision
    def test_integration_image(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image_url},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            }
        ]
        message = self._generate(messages)

        EXPECTED_TEXT = Expectations(
            {("cuda", 9): ("The image shows a **Pallas's cat / Manul - *Otocolobus manul*** - walking through snow.")}
        )
        self.assertIn(EXPECTED_TEXT.get_expectation(), message["content"])

    @require_vision
    @require_torchcodec
    def test_integration_video(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "url": self.video_url},
                    {"type": "text", "text": "What is happening in this video?"},
                ],
            }
        ]
        message = self._generate(messages, processor_kwargs={"fps": 2, "num_frames": 8})

        EXPECTED_TEXT_1 = Expectations(
            {("cuda", 9): ("In this video two people are playing tennis on an indoor court.")}
        )
        self.assertIn(EXPECTED_TEXT_1.get_expectation(), message["content"])

        EXPECTED_TEXT_2 = Expectations(
            {
                ("cuda", 9): (
                    "He executes a two-handed backhand, swinging the racket across his body and following through "
                    "as he hits the ball back over the net."
                )
            }
        )
        self.assertIn(EXPECTED_TEXT_2.get_expectation(), message["content"])

    @require_librosa
    @require_soundfile
    def test_integration_audio(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "url": self.audio_url},
                    {"type": "text", "text": "What is said in this recording?"},
                ],
            }
        ]
        message = self._generate(messages, processor_kwargs={"sampling_rate": 24000})

        # see hf://buckets/tiny-test-models/muse-spark-integration-test/reproducers
        EXPECTED_TEXT_1 = Expectations(
            {
                ("cuda", 9): (
                    "Your formal behavior is not as steady as large caves on earth that often maintain a fairly "
                    "constant temperature, but is consistent with these being deep holes in the ground, said "
                    "Glen Cushing"
                )
            }
        )
        self.assertIn(EXPECTED_TEXT_1.get_expectation(), message["content"])

        EXPECTED_TEXT_2 = Expectations({("cuda", 9): ("Northern Arizona University located in Flagstaff, Arizona.")})
        self.assertIn(EXPECTED_TEXT_2.get_expectation(), message["content"])

    def test_integration_tool_call(self):
        def get_current_temperature(location: str, unit: str):
            """
            Get the current temperature at a location.

            Args:
                location: The location to get the temperature for, in the format "City, Country"
                unit: The unit to return the temperature in. (choices: ["celsius", "fahrenheit"])
            """
            return 22.0

        tools = [get_current_temperature]
        messages = [{"role": "user", "content": "What is the temperature in Paris right now?"}]
        message = self._generate(messages, tools=tools)

        EXPECTED_TOOL_CALLS = Expectations(
            {
                ("cuda", 9): [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_temperature",
                            "arguments": {"location": "Paris, France", "unit": "celsius"},
                        },
                    }
                ]
            }
        )
        self.assertEqual(message.get("tool_calls"), EXPECTED_TOOL_CALLS.get_expectation())

        messages.append(message)
        function = message["tool_calls"][0]["function"]
        result = get_current_temperature(**function["arguments"])
        messages.append({"role": "tool", "name": function["name"], "content": str(result)})

        response = self._generate(messages, tools=tools)

        self.assertFalse(response.get("tool_calls"))

        EXPECTED_TEXT = Expectations({("cuda", 9): ("The current temperature in Paris, France is **22.0°C**")})
        self.assertIn(EXPECTED_TEXT.get_expectation(), response["content"])
