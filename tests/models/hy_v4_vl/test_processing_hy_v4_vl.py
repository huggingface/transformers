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

import numpy as np
from parameterized import parameterized

from transformers.testing_utils import require_tokenizers, require_torchvision, torch_device
from transformers.utils import (
    is_tokenizers_available,
    is_torch_available,
    is_torchvision_available,
    is_vision_available,
)

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_torch_available():
    import torch

if is_tokenizers_available():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit

    from transformers import TokenizersBackend

if is_vision_available():
    from PIL import Image

    if is_torchvision_available():
        from transformers import (
            HunYuanVLImageProcessor,
            HYV4VLProcessor,
            HYV4VLVideoProcessor,
        )


@require_torchvision
@require_tokenizers
class HYV4VLProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = HYV4VLProcessor

    @classmethod
    def _setup_tokenizer(cls):
        words = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "lower", "newer", "upper", "older", "longer", "string", "hello"]
        backend = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        backend.pre_tokenizer = WhitespaceSplit()
        return TokenizersBackend(
            tokenizer_object=backend,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            model_input_names=["input_ids", "attention_mask"],
            extra_special_tokens={
                "image_token": "<image>",
                "image_start_token": "<image_start>",
                "image_end_token": "<image_end>",
                "video_token": "<video>",
                "video_start_token": "<video_start>",
                "video_end_token": "<video_end>",
            },
        )

    @classmethod
    def _setup_image_processor(cls):
        return HunYuanVLImageProcessor(size={"shortest_edge": 32 * 32, "longest_edge": 64 * 128}, device=torch_device)

    @classmethod
    def _setup_video_processor(cls):
        return HYV4VLVideoProcessor(size={"shortest_edge": 32 * 32, "longest_edge": 4 * 64 * 128}, device=torch_device)

    @staticmethod
    def prepare_processor_dict():
        return {
            "chat_template": (
                "{% for message in messages %}{{ message['role'] + ' ' }}"
                "{% for item in message['content'] %}"
                "{% if item['type'] == 'image' %}{{ '<image_start><image><image_end>' }}"
                "{% elif item['type'] == 'video' %}{{ '<video_start><video><video_end>' }}"
                "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}"
                "{{ '[EOS]' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
            )
        }

    def prepare_text_inputs(self, batch_size=None, modalities=None):
        modalities = [modalities] if isinstance(modalities, str) else modalities or []
        placeholders = "".join(
            f"<{modality}_start><{modality}><{modality}_end>"
            for modality in [name.rstrip("s") for name in modalities]
            if modality in ("image", "video")
        )
        if batch_size is None:
            return f"lower newer {placeholders}"
        if batch_size < 1:
            raise ValueError("batch_size must be greater than 0")
        return [
            f"lower newer {placeholders}" if i % 2 == 0 else f"upper older longer string {placeholders}"
            for i in range(batch_size)
        ]

    def prepare_images_inputs(self, batch_size=None, nested=False):
        images = [Image.new("RGB", (64, 32), color=(10 + i, 20, 30)) for i in range(batch_size or 1)]
        if batch_size is None:
            return images[0]
        return [[image] for image in images] if nested else images

    def prepare_videos_inputs(self, batch_size=None):
        videos = [np.full((3, 32, 64, 3), 10 + i, dtype=np.uint8) for i in range(batch_size or 1)]
        return videos[0] if batch_size is None else videos

    @parameterized.expand([("image",), ("video",), ("mixed",)])
    def test_visual_token_counts_and_batch(self, modality):
        processor = self.get_processor()
        modalities = ["image", "video"] if modality == "mixed" else [modality]
        kwargs = {}
        if "image" in modalities:
            kwargs["images"] = self.prepare_images_inputs(batch_size=2)
        if "video" in modalities:
            kwargs["videos"] = self.prepare_videos_inputs(batch_size=2)
        output = processor(text=self.prepare_text_inputs(2, modalities), padding=True, return_tensors="pt", **kwargs)
        for name in modalities:
            pixels_name = "pixel_values" if name == "image" else "pixel_values_videos"
            self.assertEqual(output[pixels_name].device.type, torch.device(torch_device).type)
            grids = output[f"{name}_grid_thw"]
            token_id = getattr(processor, f"{name}_token_id")
            merge_size = getattr(processor, f"{name}_processor").merge_size
            # Key difference: every merged row ends with a newline token (`W + 1`)
            expected = [int(t * (h // merge_size) * (w // merge_size + 1)) for t, h, w in grids]
            self.assertEqual((output.input_ids == token_id).sum(-1).tolist(), expected)
            self.assertEqual((output.mm_token_type_ids == (1 if name == "image" else 2)).sum(-1).tolist(), expected)
            self.assertEqual(
                (output.input_ids == getattr(processor.tokenizer, f"{name}_start_token_id")).sum().item(), 2
            )
            self.assertEqual(
                (output.input_ids == getattr(processor.tokenizer, f"{name}_end_token_id")).sum().item(), 2
            )

    @parameterized.expand(
        [
            ("size", {"size": {"shortest_edge": 1024, "longest_edge": 3 * 1024}}),
            ("cap", {"max_image_size": {"longest_edge": 2048}}),
            ("merge", {"merge_size": 1}),
            ("temporal", {"temporal_patch_size": 2}),
        ]
    )
    def test_videos_kwargs_match_grid_and_estimate(self, _, videos_kwargs):
        processor = self.get_processor()
        original_video_config = processor.video_processor.to_dict()
        video = np.zeros((3, 64, 128, 3), dtype=np.uint8)
        output = processor(
            text="<video_start><video><video_end>", videos=video, videos_kwargs=videos_kwargs, return_tensors="pt"
        )
        direct = processor.video_processor(video, **videos_kwargs, return_tensors="pt")
        torch.testing.assert_close(output.video_grid_thw, direct.video_grid_thw)
        torch.testing.assert_close(output.pixel_values_videos, direct.pixel_values_videos)
        t, h, w = output.video_grid_thw[0].tolist()
        merge = videos_kwargs.get("merge_size", processor.video_processor.merge_size)
        expected = t * (h // merge) * (w // merge + 1)
        self.assertEqual((output.input_ids == processor.video_token_id).sum().item(), expected)
        estimate = processor._get_num_multimodal_tokens(video_sizes=[(3, 64, 128)], **videos_kwargs)
        self.assertEqual(estimate.num_video_tokens, [expected])
        nested_estimate = processor._get_num_multimodal_tokens(video_sizes=[(3, 64, 128)], videos_kwargs=videos_kwargs)
        self.assertEqual(nested_estimate.num_video_tokens, [expected])
        self.assertEqual(
            processor.video_processor.get_num_of_video_patches(3, 64, 128, videos_kwargs)[0],
            output.pixel_values_videos.shape[0],
        )
        self.assertEqual(processor.video_processor.to_dict(), original_video_config)

    def test_apply_chat_template_video_frame_sampling(self):
        # Necessary: the common test reads the sampled frame count off a frame axis, while HY4-VL returns
        # flattened patches and reports the count in `video_grid_thw`
        processor = self.get_processor()
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "url": url_to_local_path(
                                "https://huggingface.co/datasets/hf-internal-testing/test-videos/resolve/main/tiny_video_320x240.mp4"
                            ),
                        },
                        {"type": "text", "text": "What is shown in this video?"},
                    ],
                }
            ]
        ]
        for do_sample, expected_frames in ((False, 11), (True, 3)):
            with self.subTest(do_sample_frames=do_sample):
                output = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    processor_kwargs={"videos_kwargs": {"do_sample_frames": do_sample, "num_frames": 3, "fps": None}},
                )
                self.assertEqual(output.video_grid_thw[0, 0].item(), expected_frames)
                self.assertEqual((output.input_ids == processor.tokenizer.video_start_token_id).sum().item(), 1)
                self.assertEqual((output.input_ids == processor.tokenizer.video_end_token_id).sum().item(), 1)

    def test_replacement_offsets(self):
        # WordLevel decoding adds spaces around special tokens; compare against the expanded source instead.
        processor = self.get_processor()
        text = self.prepare_text_inputs(modalities=["image", "video"])
        output = processor(
            text=text,
            images=self.prepare_images_inputs(),
            videos=self.prepare_videos_inputs(),
            return_text_replacement_offsets=True,
            return_tensors="pt",
        )
        expanded = text
        for offset in output.text_replacement_offsets[0]:
            expanded = expanded.replace(getattr(processor, f"{offset['type']}_token"), offset["replacement"], 1)
        self.assertEqual(len(output.text_replacement_offsets[0]), 2)
        for offset in output.text_replacement_offsets[0]:
            start, end = offset["new_span"]
            self.assertEqual(expanded[start:end], offset["replacement"])
