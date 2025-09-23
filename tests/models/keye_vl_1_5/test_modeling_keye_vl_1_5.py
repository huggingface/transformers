# Copyright 2025 The Kwai Keye Team and The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the PyTorch KeyeVL1_5 model."""

import copy
import unittest

import requests

from transformers import (
    AutoProcessor,
    KeyeVL1_5Config,
    KeyeVL1_5TextConfig,
    KeyeVL1_5ForConditionalGeneration,
    KeyeVL1_5Model,
    is_torch_available,
    is_vision_available,
)
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import (
    cleanup,
    is_flaky,
    require_torch,
    slow,
    torch_device,
)
from transformers.utils import is_cv2_available

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_cv2_available():
    pass

if is_torch_available():
    import torch

else:
    is_torch_greater_or_equal_than_2_0 = False

if is_vision_available():
    from PIL import Image


class KeyeVL1_5VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        seq_length=64,
        num_channels=3,
        ignore_index=-100,
        image_size=28,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vision_start_token_id=3,
        vision_end_token_id=6,
        image_token_id=4,
        video_token_id=5,
        hidden_act="silu",
        hidden_size=32,
        vocab_size=99,
        intermediate_size=37,
        max_position_embeddings=512,
        max_window_layers=3,
        model_type="KeyeVL1_5",
        num_attention_heads=4,
        num_hidden_layers=4,
        num_key_value_heads=2,
        rope_theta=10000,
        tie_word_embeddings=True,
        sliding_window=4096,
        use_sliding_window=False,
        is_training=True,
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "silu",
            "intermediate_size": 16,
            "out_hidden_size": 16,
            "hidden_size": 16,
            "num_attention_heads": 4,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "fullatt_block_indexes": [0, 1],
            "window_size": 4096,
            "in_channels": 3,
            "num_heads": 4,
        },
        rope_scaling={"type": "mrope", "mrope_section": [2, 1, 1]},
        train_batch_size=1,
        num_video_tokens=1,
    ):
        self.parent = parent
        self.ignore_index = ignore_index
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.max_window_layers = max_window_layers
        self.model_type = model_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.vision_config = vision_config
        self.rope_scaling = rope_scaling
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.num_image_tokens = 32
        self.num_video_tokens = num_video_tokens
        self.seq_length = seq_length + self.num_image_tokens
        self.train_batch_size = train_batch_size

    def get_config(self):
        return KeyeVL1_5Config(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            vision_config=self.vision_config,
            model_type=self.model_type,
            max_window_layers=self.max_window_layers,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=self.tie_word_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_start_token_id=self.vision_start_token_id,
            vision_end_token_id=self.vision_end_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vocab_size=self.vocab_size,
            sliding_window=self.sliding_window,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size,
                (self.image_size**2) // (patch_size**2) * temporal_patch_size,
                self.num_channels,  # * (patch_size**2) * temporal_patch_size,
                patch_size,
                patch_size,
            ]
        )
        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_end_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
        input_ids[:, self.num_image_tokens + 1] = self.vision_end_token_id
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[1, 2, 2]] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class KeyeVL1_5ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `KeyeVL1_5ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            KeyeVL1_5Model,
            KeyeVL1_5ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = KeyeVL1_5VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=KeyeVL1_5Config, has_text_modality=False)

    def test_config(self):
        self.config_tester.run_common_tests()

    def test_initialization(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        configs_no_init = _config_zero_init(config)

        for model_class in self.all_model_classes:
            model = model_class(config=configs_no_init)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIn(
                        ((param.data.mean() * 1e9).round() / 1e9).item(),
                        [0.0, 1.0],
                        msg=f"Parameter {name} of model {model_class} seems not properly initialized",
                    )

    def test_mismatching_num_image_tokens(self):
        """
        Tests that VLMs through an error with explicit message saying what is wrong
        when number of images don't match number of image tokens in the text.
        Also we need to test multi-image cases when one prompr has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)

            _ = model(**input_dict)  # successful forward with no modifications
            curr_input_dict = copy.deepcopy(input_dict)

            # remove one image but leave the image token in text
            one_img_length = 1
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["image_grid_thw"] = curr_input_dict["image_grid_thw"][-1:, ...]

            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            image_grid_thw = curr_input_dict["image_grid_thw"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            image_grid_thw = torch.cat([image_grid_thw, image_grid_thw], dim=0)
            _ = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

    def test_video_forward(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        B = self.model_tester.batch_size
        C = config.vision_config.in_chans
        T = config.vision_config.temporal_patch_size
        P = config.vision_config.patch_size

        input_ids = ids_tensor([B, self.model_tester.seq_length], self.model_tester.vocab_size)

        F = 1
        patch_H = self.model_tester.image_size // P
        patch_W = self.model_tester.image_size // P
        patch_T = F // T
        patches_per_video = patch_T * patch_H * patch_W
        pixel_values_videos = floats_tensor(
            [
                # first dim: batch_size * num_patches
                B,
                patches_per_video * T,  # 4 * 1
                C,  # 3
                P,  # 14
                P,  # 14
            ]
        )

        video_grid_thw = torch.tensor([[patch_T, patch_H, patch_W]] * B)

        # Insert video token sequence
        input_ids[:, -1] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.video_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.image_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_start_token_id] = self.model_tester.pad_token_id
        input_ids[input_ids == self.model_tester.vision_end_token_id] = self.model_tester.pad_token_id
        input_ids[:, self.model_tester.num_video_tokens] = self.model_tester.video_token_id

        insertion_point = self.model_tester.num_video_tokens
        n_video_tokens = patches_per_video // config.vision_config.spatial_merge_size**2
        assert (B * n_video_tokens) + insertion_point <= self.model_tester.seq_length
        for b in range(B):
            input_ids[b, insertion_point - 1] = self.model_tester.vision_start_token_id
            input_ids[b, insertion_point : insertion_point + n_video_tokens] = self.model_tester.video_token_id
            input_ids[b, insertion_point + n_video_tokens] = self.model_tester.vision_end_token_id

        for model_class in self.all_model_classes:
            second_per_grid_ts = torch.tensor([1.0] * B, device=torch_device)
            model = model_class(config).to(torch_device)
            outputs = model(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
            )
            self.assertIsNotNone(outputs)

    def check_training_gradient_checkpointing(self, gradient_checkpointing_kwargs=None):
        if not self.model_tester.is_training:
            self.skipTest(reason="ModelTester is not configured to run training tests")

        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                if (
                    model_class.__name__
                    in [
                        *get_values(MODEL_MAPPING_NAMES),
                        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
                    ]
                    or not model_class.supports_gradient_checkpointing
                ):
                    # TODO (ydshieh): use `skipTest` once pytest-dev/pytest-subtests/pull/169 is merged
                    # self.skipTest(reason=f"`supports_gradient_checkpointing` is False for {model_class.__name__}.")
                    continue

                config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
                config.use_cache = False
                config.return_dict = True
                model = model_class(config)

                model.to(torch_device)
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
                model.train()

                # unfreeze additional layers
                for n, p in model.named_parameters():
                    if n in [
                        "model.visual.vision_model.embeddings.packing_position_embedding.weight",
                        "model.visual.vision_model.head.probe",
                    ] or n.startswith("model.visual.vision_model.head"):
                        p.requires_grad_(False)
                    else:
                        p.requires_grad_(True)

                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

                inputs = self._prepare_for_class(inputs_dict, model_class, return_labels=True)
                loss = model(**inputs).loss
                loss.backward()
                optimizer.step()

                if self.test_all_params_have_gradient:
                    for k, v in model.named_parameters():
                        if v.requires_grad:
                            self.assertTrue(v.grad is not None, f"{k} in {model_class.__name__} has no gradient!")

    @unittest.skip(reason="Feedforward chunking is not yet supported")
    def test_feed_forward_chunking(self):
        pass

    @unittest.skip(reason="CPU offload is not yet supported")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="Compile not yet supported because in Qwen2_5_VL models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Got `CUDA error: misaligned address` with PyTorch 2.0.0.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass

    @is_flaky()  # TODO (joao/raushan): Investigate why this test is flaky on this model
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        super().test_prompt_lookup_decoding_matches_greedy_search()


@require_torch
class KeyeVL1_5IntegrationTest(unittest.TestCase):
    # all_model_classes = (KeyeVL1_5ForConditionalGeneration,) if is_torch_available() else ()

    def setUp(self):
        self.processor = AutoProcessor.from_pretrained("Kwai-Keye/Keye-VL-1_5-8B")
        self.messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Take a look at the picture."},
                    {"type": "image"},
                    {"type": "text", "text": "What kind of cat is this?"},
                ],
            }
        ]
        url = "https://s1-11508.kwimgs.com/kos/nlav11508/mllm_all/ziran_jiafeimao_11.jpg"
        self.image = Image.open(requests.get(url, stream=True).raw).resize(size=(32, 32))

        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    @slow
    def test_small_model_integration_test(self):
        model = KeyeVL1_5ForConditionalGeneration.from_pretrained(
            "Kwai-Keye/Keye-VL-1_5-8B", dtype="auto", device_map="auto"
        )

        text = self.processor.apply_chat_template(self.messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[self.image], return_tensors="pt")
        expected_input_ids = [151644, 872, 198, 17814, 264, 1401, 518, 279, 6802, 13, 151652, 151655, 151655, 151655, 151655, 151655, 151655]  # fmt: skip
        torch.testing.assert_close(expected_input_ids, inputs.input_ids[0].tolist()[:17])
        expected_pixel_slice = torch.tensor(
            [
                [
                    [[-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588]],
                    [[-0.0980, -0.0980, -0.0980], [-0.0980, -0.0980, -0.0980], [-0.0980, -0.0980, -0.0980]],
                    [[-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863]],
                ],
                [
                    [[-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588], [-0.0588, -0.0588, -0.0588]],
                    [[-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059]],
                    [[-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863]],
                ],
                [
                    [[-0.0667, -0.0667, -0.0667], [-0.0667, -0.0667, -0.0667], [-0.0667, -0.0667, -0.0667]],
                    [[-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059], [-0.1059, -0.1059, -0.1059]],
                    [[-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863], [-0.2863, -0.2863, -0.2863]],
                ],
            ],
            dtype=torch.float32,
            device="cpu",
        )
        torch.testing.assert_close(expected_pixel_slice, inputs.pixel_values[:3, :3, :3, :3], atol=2e-3, rtol=5e-5)

        # verify generation
        inputs = inputs.to(torch_device)

        output = model.generate(**inputs, max_new_tokens=30, top_k=1)
        EXPECTED_DECODED_TEXT = "user\nTake a look at the picture.What kind of cat is this?\nassistant\n<analysis>This question asks for the identification of the type of cat shown in the picture. The answer can be determined by visual characteristics, making it straightforward"
        self.assertEqual(
            self.processor.decode(output[0], skip_special_tokens=True),
            EXPECTED_DECODED_TEXT,
        )
