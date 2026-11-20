# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import os
import unittest

import numpy as np
from huggingface_hub import hf_hub_download

from transformers import Lfm3OmniProcessor
from transformers.testing_utils import require_torch, require_torchcodec, require_torchvision, require_vision
from transformers.utils import is_torch_available
from transformers.video_utils import VideoMetadata

from ..lfm3_audio import test_processing_lfm3_audio as lfm3_audio


if is_torch_available():
    import torch


def placeholder_runs(ids, token_id):
    """Lengths of the runs of `token_id` in `ids`."""
    runs, run = [], 0
    for token in [*ids, None]:
        if token == token_id:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    return runs


@require_torch
@require_vision
@require_torchvision
class Lfm3OmniProcessorTest(lfm3_audio.Lfm3AudioProcessorTest):
    # the LFM3-Audio tokenizer and chat template carry the vision tokens and spans too
    processor_class = Lfm3OmniProcessor
    images_text_kwargs_max_length = 30
    images_text_kwargs_override_max_length = 25
    images_unstructured_max_length = 40
    videos_text_kwargs_max_length = 120
    videos_text_kwargs_override_max_length = 110
    videos_unstructured_max_length = 130

    @classmethod
    def setUpClass(cls):
        # the parent keeps its tiny codec on the first class that built it, and removes it with that class
        owner = next((klass for klass in cls.__mro__ if "codec_dirname" in vars(klass)), None)
        if owner is not None and not os.path.isdir(owner.codec_dirname):
            del owner.codec_dirname
        super().setUpClass()

    # the released patch, merge and normalization values with small pixel budgets
    @classmethod
    def _setup_image_processor(cls):
        return cls._get_component_class_from_processor("image_processor")(
            size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64},
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
        )

    @classmethod
    def _setup_video_processor(cls):
        return cls._get_component_class_from_processor("video_processor")(
            size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64 * 4},
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5],
            cap_pixels_per_frame=False,
            fps=2,
        )

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 32},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 12},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 24},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 24},
            {"expected_dim": 0, "output_length": 16},
        ]

    @staticmethod
    def video_inputs():
        # 2 s at 2 fps: 4 frames, 2 temporal patches
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, duration=2.0, frames_indices=[0, 1, 2, 3])
        return {"videos": [frames], "video_metadata": [metadata]}

    @unittest.skip("LFM3-Omni cannot sample frames from decoded frames without their metadata, like Qwen3-VL")
    def test_apply_chat_template_video_2(self):
        pass

    @unittest.skip("`use_audio_in_video` is a processor default under `videos_kwargs`, not a video processor one")
    def test_subprocessor_defaults_2_videos(self):
        pass

    def test_model_input_names(self):
        processor = self.get_processor()
        inputs = processor(
            text=self.prepare_text_inputs(modalities=["image", "video", "audio"]) + processor.audio_out_token,
            images=self.prepare_images_inputs(),
            audio=self.prepare_audio_inputs(),
            audio_out=[torch.randint(0, 9, (4, 2))],
            return_tensors="pt",
            **self.video_inputs(),
        )
        self.assertSetEqual(set(inputs.keys()), set(processor.model_input_names))

    def test_media_spans(self):
        processor = self.get_processor()
        tokenizer = processor.tokenizer
        start, end = tokenizer.convert_tokens_to_ids([processor.vision_start_token, processor.vision_end_token])
        image, audio = self.prepare_images_inputs(), np.zeros(4000, dtype=np.float32)
        # an image and a clip, bare or wrapped by the chat template, are each expanded in one span
        for text in (
            f"{processor.image_token} and {processor.audio_token}",
            f"{processor.vision_start_token}{processor.image_token}{processor.vision_end_token} and "
            f"{processor.vision_start_token}{processor.audio_token}{processor.vision_end_token}",
        ):
            with self.subTest(text=text):
                inputs = processor(text=text, images=[image], audio=[audio], sampling_rate=16000)
                ids = inputs["input_ids"][0]
                num_image_tokens = int(inputs["image_grid_thw"][0].prod()) // processor.image_processor.merge_size**2
                self.assertEqual(placeholder_runs(ids, processor.image_token_id), [num_image_tokens])
                self.assertEqual(placeholder_runs(ids, processor.audio_token_id), [3])
                for token_id in (processor.image_token_id, processor.audio_token_id):
                    positions = [i for i, token in enumerate(ids) if token == token_id]
                    self.assertEqual((ids[positions[0] - 1], ids[positions[-1] + 1]), (start, end))
                self.assertEqual((ids.count(start), ids.count(end)), (2, 2))

        # a video: one timestamped, wrapped group per temporal patch, wrapped again as a whole
        outputs = processor(text=processor.video_token, **self.video_inputs())
        ids = outputs["input_ids"][0]
        grid = outputs["video_grid_thw"][0]
        per_group = int(grid[1] * grid[2]) // processor.video_processor.merge_size**2
        self.assertEqual(placeholder_runs(ids, processor.video_token_id), [per_group] * int(grid[0]))
        self.assertEqual((ids.count(start), ids.count(end)), (1 + int(grid[0]), 1 + int(grid[0])))
        self.assertEqual(ids[0], start)
        self.assertEqual(tokenizer.decode(ids[1 : ids.index(processor.video_token_id) - 1]), "<0.2 seconds>")
        self.assertEqual(processor._calculate_timestamps([0, 1, 2, 3], 2.0, 2), [0.25, 1.25])

    def test_batch_with_mixed_media(self):
        processor = self.get_processor()
        image, clip = self.prepare_images_inputs(), np.zeros(16000, dtype=np.float32)
        codes = torch.randint(0, 9, (4, 3))
        inputs = processor(
            text=[
                f"{processor.image_token} Hello {processor.audio_token}",
                f"{processor.audio_token} Hello {processor.image_token} {processor.audio_out_token}",
                "Hello",
            ],
            images=[image, image],
            audio=[clip, clip],
            audio_out=[codes],
            sampling_rate=16000,
            output_labels=True,
            return_tensors="pt",
        )
        num_image_tokens = int(inputs["image_grid_thw"][0].prod()) // processor.image_processor.merge_size**2
        input_ids = inputs["input_ids"]
        self.assertEqual((input_ids == processor.image_token_id).sum(-1).tolist(), [num_image_tokens] * 2 + [0])
        self.assertEqual((input_ids == processor.audio_token_id).sum(-1).tolist(), [13, 13, 0])
        self.assertEqual((input_ids == processor.audio_out_token_id).sum(-1).tolist(), [0, 3, 0])
        self.assertTrue(torch.equal(inputs["audio_input_ids"], codes.T))
        # media in the order of the text
        first = [
            int((row == token_id).nonzero()[0])
            for row, token_id in zip(input_ids, (processor.image_token_id, processor.audio_token_id))
        ]
        last = [
            int((row == token_id).nonzero()[0])
            for row, token_id in zip(input_ids, (processor.audio_token_id, processor.image_token_id))
        ]
        self.assertTrue(all(start < end for start, end in zip(first, last)))
        # `labels` ignore every placeholder and padding, keep the spans and the text
        labels = inputs["labels"]
        is_placeholder = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in (processor.image_token_id, processor.audio_token_id, processor.audio_out_token_id):
            is_placeholder |= input_ids == token_id
        is_ignored = is_placeholder | (inputs["attention_mask"] == 0)
        self.assertTrue((labels[is_ignored] == -100).all())
        self.assertTrue(torch.equal(labels[~is_ignored], input_ids[~is_ignored]))
        self.assertTrue(torch.equal(inputs["audio_labels"], inputs["audio_input_ids"]))

    def test_video_soundtrack_is_split_across_frame_groups(self):
        processor = self.get_processor()
        video, audio_id = self.video_inputs(), processor.audio_token_id
        # 2 s = 200 mel frames = 25 encoder tokens
        soundtrack, clip = np.zeros(16000 * 2, dtype=np.float32), np.zeros(4000, dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "audio clip after each video"):
            processor(
                text=f"{processor.audio_token}{processor.video_token}",
                audio=[soundtrack],
                use_audio_in_video=True,
                **video,
            )
        # the soundtrack placeholder may sit after some text, where `load_audio_from_video=True` puts it
        text = f"{processor.video_token} what is said? {processor.audio_token} and {processor.audio_token}"
        outputs = processor(text=text, audio=[soundtrack, clip], use_audio_in_video=True, **video)
        ids = outputs["input_ids"][0]
        grid = outputs["video_grid_thw"][0]
        per_group = int(grid[1] * grid[2]) // processor.video_processor.merge_size**2
        self.assertEqual(placeholder_runs(ids, processor.video_token_id), [per_group] * int(grid[0]))
        # the second frame group starts at 1 s: 1 s / 0.08 s per token = 12 tokens, then the rest; the clip is its own span
        self.assertEqual(placeholder_runs(ids, audio_id), [12, 25 - 12, 3])
        self.assertEqual(len(outputs["input_features"]), 2)
        # without the flag every clip is its own media span, even right behind a video
        outputs = processor(text=text, audio=[soundtrack, clip], **video)
        self.assertEqual(placeholder_runs(outputs["input_ids"][0], audio_id), [25, 3])
        # the structured form of the flag
        outputs = processor(text=text, audio=[soundtrack, clip], videos_kwargs={"use_audio_in_video": True}, **video)
        self.assertEqual(placeholder_runs(outputs["input_ids"][0], audio_id), [12, 25 - 12, 3])

    def test_soundtrack_placeholders_match_the_encoder(self):
        processor = self.get_processor()
        extractor = processor.feature_extractor
        # the last partial STFT frame crosses an encoder-stride boundary: the extractor rounds it up to a frame
        samples = extractor.n_fft + extractor.hop_length * (extractor.encoder_stride - 1) + 1
        text = f"{processor.video_token}{processor.audio_token}"
        outputs = processor(
            text=text, audio=[np.zeros(samples, dtype=np.float32)], use_audio_in_video=True, **self.video_inputs()
        )
        num_tokens = -(int(outputs["input_features_mask"].sum()) // -extractor.encoder_stride)
        self.assertEqual(placeholder_runs(outputs["input_ids"][0], processor.audio_token_id), [num_tokens])

    def test_soundtrack_parts_follow_their_video(self):
        rewrite = Lfm3OmniProcessor._with_soundtrack_parts
        video, question = {"type": "video", "video": "clip.mp4"}, {"type": "audio", "audio": "question.wav"}
        track, text = {"type": "audio", "audio": "clip.mp4"}, {"type": "text", "text": "Hi"}

        def user(*parts):
            return [{"role": "user", "content": list(parts)}]

        # a lone video gets its track where the shared implementation appended it; a clip of its own is kept
        self.assertEqual(rewrite(user(video)), user(video, track))
        self.assertEqual(rewrite(user(video, question)), user(video, track, question))
        self.assertEqual(rewrite(user(video, text)), user(video, track, text))
        url = {"type": "video", "url": "https://example.org/clip.mp4"}
        self.assertEqual(rewrite(user(url)), user(url, {"type": "audio", "audio": url["url"]}))
        self.assertEqual(rewrite(user({"type": "video"})), user({"type": "video"}))
        self.assertEqual(rewrite(user(video, {"type": "audio"})), user(video, track))
        system = [{"role": "system", "content": "Be brief."}]
        self.assertEqual(rewrite(system + user(video)), system + user(video, track))
        self.assertEqual(rewrite([user(video), user(question)]), [user(video, track), user(question)])
        conversation = user(video, question)
        rewrite(conversation)
        self.assertEqual(conversation, user(video, question))

    @require_torchcodec
    def test_load_audio_from_video_keeps_standalone_clips(self):
        processor = self.get_processor()
        path = hf_hub_download("raushan-testing-hf/videos-test", "sample_demo_1.mp4", repo_type="dataset")
        video, clip, text = (
            {"type": "video", "video": path},
            {"type": "audio", "audio": path},
            {"type": "text", "text": "Hi"},
        )
        soundtrack = {"processor_kwargs": {"use_audio_in_video": True}}

        def encode(content, **kwargs):
            outputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **kwargs,
            )
            return outputs["input_ids"][0].tolist(), len(outputs["input_features"])

        # a video with its soundtrack and a spoken question: the question stays a clip of its own
        with_track = encode([video, clip], load_audio_from_video=True, **soundtrack)
        self.assertEqual(with_track[1], 2)
        self.assertEqual(with_track, encode([video, clip, clip], **soundtrack))
        # a lone video renders as before, with or without the pairing
        self.assertEqual(
            encode([video], load_audio_from_video=True, **soundtrack), encode([video, clip], **soundtrack)
        )
        self.assertEqual(encode([video], load_audio_from_video=True), encode([video, clip]))
        # the track follows its video, not the trailing text
        self.assertEqual(encode([video, text], load_audio_from_video=True), encode([video, clip, text]))

    def test_replacement_offsets(self):
        # parent: audio placeholders only. Here the spans, the video groups and the soundtrack rewrites too
        processor = self.get_processor()
        start, end = processor.vision_start_token, processor.vision_end_token
        image, speech = self.prepare_images_inputs(), torch.randint(0, 9, (4, 2))
        soundtrack = np.zeros(16000, dtype=np.float32)
        video = self.video_inputs()
        for name, text, kwargs in (
            ("wrapped image", f"{start}<|image|>{end} look <|audio_out|> x <|audio|>", {"images": [image]}),
            ("video and speech", "<|video|> look <|audio_out|> end", video),
            # the soundtrack placeholder moves to its video, and the speech frames sit between them
            (
                "distant soundtrack",
                "<|video|> look at it <|audio_out|> now <|audio|> end",
                {**video, "use_audio_in_video": True},
            ),
            (
                "wrapped video and soundtrack",
                f"{start}<|video|>{end} a <|audio|>",
                {**video, "use_audio_in_video": True},
            ),
        ):
            with self.subTest(name):
                if "<|audio|>" in text:
                    kwargs = {**kwargs, "audio": [soundtrack]}
                if "<|audio_out|>" in text:
                    kwargs = {**kwargs, "audio_out": [speech]}
                outputs = processor(text=text, return_text_replacement_offsets=True, **kwargs)
                # every record slices its `text` from the prompt; applied in order they rebuild the tokenized prompt,
                # and every `new_span` slices its `replacement` from the rebuilt one
                pieces, last = [], 0
                for record in outputs["text_replacement_offsets"][0]:
                    span_start, span_end = record["span"]
                    self.assertGreaterEqual(span_start, last)
                    self.assertEqual(text[span_start:span_end], record["text"])
                    pieces.extend((text[last:span_start], record["replacement"]))
                    last = span_end
                rebuilt = "".join(pieces + [text[last:]])
                for record in outputs["text_replacement_offsets"][0]:
                    self.assertEqual(rebuilt[slice(*record["new_span"])], record["replacement"])
                self.assertEqual(processor.tokenizer(rebuilt)["input_ids"], outputs["input_ids"][0])
