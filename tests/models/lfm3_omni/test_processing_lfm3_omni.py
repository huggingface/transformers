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

import unittest

import numpy as np
from PIL import Image

from transformers import (
    Lfm3AudioFeatureExtractor,
    Lfm3OmniProcessor,
    Qwen2VLImageProcessor,
    Qwen3VLVideoProcessor,
    is_torch_available,
)
from transformers.testing_utils import require_torch, require_torchcodec, require_vision

from ...test_processing_common import ProcessorTesterMixin
from ..lfm3_audio.test_processing_lfm3_audio import (
    GENERATION_TEMPLATE,
    assert_offsets_rebuild_the_prompts,
    expected_assistant_mask,
    tiny_codec,
    tiny_tokenizer,
)


if is_torch_available():
    pass


TINY_CODEC_BACKBONE = {
    "vocab_size": 1,
    "pad_token_id": None,
    "bos_token_id": None,
    "eos_token_id": None,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "use_cache": False,
}


# the hub chat template reduced to what the tests render: ChatML turns, one bare placeholder per media item (the hub
# template wraps them in the vision span pair, which the processor strips before expanding)
CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- for message in messages -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
    "{%- if message['content'] is string -%}{{- message['content'] -}}"
    "{%- else -%}{%- for content in message['content'] -%}"
    "{%- if content['type'] == 'image' -%}{{- '<|image|>' -}}"
    "{%- elif content['type'] == 'video' -%}{{- '<|video|>' -}}"
    "{%- elif content['type'] == 'audio' -%}{{- '<|audio|>' -}}"
    "{%- elif content['type'] == 'audio_out' -%}{{- '<|audio_out|>' -}}"
    "{%- elif content['type'] == 'text' -%}{{- content['text'] -}}"
    "{%- endif -%}{%- endfor -%}{%- endif -%}"
    "{{- '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)


def tiny_image_processor():
    # the released patch, merge and normalization values with a small pixel budget
    return Qwen2VLImageProcessor(
        size={"shortest_edge": 1024, "longest_edge": 4096},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )


@require_torch
@require_vision
class Lfm3OmniProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Lfm3OmniProcessor

    @classmethod
    def _setup_tokenizer(cls):
        return tiny_tokenizer()

    @classmethod
    def _setup_image_processor(cls):
        return tiny_image_processor()

    @classmethod
    def _setup_video_processor(cls):
        return Qwen3VLVideoProcessor(size={"shortest_edge": 32 * 32, "longest_edge": 64 * 64 * 4})

    @classmethod
    def _setup_feature_extractor(cls):
        return Lfm3AudioFeatureExtractor()

    @property
    def video_sampling_expectations(self):
        return [
            {"num_frames": 3, "fps": None, "expected_dim": 0, "output_length": 32},
            {"num_frames": None, "fps": 18, "expected_dim": 0, "output_length": 12},
            {"do_sample_frames": False, "fps": 2, "expected_dim": 0, "output_length": 24},
            {"do_sample_frames": False, "expected_dim": 0, "output_length": 24},
            {"expected_dim": 0, "output_length": 16},
        ]

    @unittest.skip(
        "`use_audio_in_video` lives in `videos_kwargs` but is a processor option, not a video-processor one"
    )
    def test_subprocessor_defaults_2_videos(self):
        pass

    @staticmethod
    def prepare_processor_dict():
        return {"chat_template": CHAT_TEMPLATE}

    @unittest.skip("The word-level test tokenizer decodes the placeholders with spaces between them")
    def test_replacement_offsets(self):
        pass

    def test_model_input_names(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        inputs = self.processor(
            text="<|image|><|video|><|audio|><|audio_out|>",
            images=[self.image],
            videos=[np.zeros((4, 32, 64, 3), dtype=np.uint8)],
            video_metadata=[VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])],
            audio=[self.audio],
            audio_out=[np.zeros(1920, dtype=np.float32)],
            sampling_rate=16000,
        )
        self.assertSetEqual(set(inputs.keys()), set(self.processor.model_input_names))

    def setUp(self):
        self.processor = Lfm3OmniProcessor(
            tiny_image_processor(),
            Lfm3AudioFeatureExtractor(),
            tiny_tokenizer(),
            video_processor=Qwen3VLVideoProcessor(size={"shortest_edge": 1024, "longest_edge": 4096}),
            chat_template=CHAT_TEMPLATE,
        )
        self.image = Image.new("RGB", (64, 32))
        self.audio = np.zeros(4000, dtype=np.float32)

    def test_interleaved_modalities_keep_order_and_counts(self):
        proc = self.processor
        inputs = proc(
            text=["<|image|> Hello <|audio|>", "<|audio|> Hello <|image|>"],
            images=[self.image, self.image],
            audio=[self.audio, self.audio],
            sampling_rate=16000,
            padding=True,
            return_tensors="pt",
        )
        image_count = inputs.image_grid_thw[0].prod() // proc.image_processor.merge_size**2
        audio_count = proc.feature_extractor.num_encoder_tokens(proc.feature_extractor.num_frames(len(self.audio)))
        for row in inputs.input_ids:
            self.assertEqual(int((row == proc.image_token_id).sum()), int(image_count))
            self.assertEqual(int((row == proc.audio_token_id).sum()), audio_count)
        for row, first, last in (
            (0, proc.image_token_id, proc.audio_token_id),
            (1, proc.audio_token_id, proc.image_token_id),
        ):
            ids = inputs.input_ids[row]
            self.assertLess(int((ids == first).nonzero()[0]), int((ids == last).nonzero()[0]))
        self.assertTrue(set(inputs).issubset(proc.model_input_names))
        self.assertIn("audio_out_codes", proc.model_input_names)

    def test_truncated_speech_placeholders_are_rejected(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        clips = [np.zeros(1920 * 3, dtype=np.float32)]
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        for video, media in (("", {}), ("<|video|>", {"videos": [frames], "video_metadata": [metadata]})):
            prompt = video + "<|audio_out|> Hello"
            ids = self.processor(text=prompt, audio_out=clips, **media)["input_ids"][0]
            # keep the video and one of the 4 speech placeholders
            max_length = ids.index(self.processor.audio_out_token_id) + 1
            with self.subTest(video=video), self.assertRaisesRegex(ValueError, "after tokenization"):
                self.processor(text=prompt, audio_out=clips, truncation=True, max_length=max_length, **media)

    def test_media_spans_wrap_images_and_videos_like_the_sft_data(self):
        inputs = self.processor(
            text="<|image|> and <|audio|>", images=[self.image], audio=[self.audio], return_tensors="pt"
        )
        ids = inputs["input_ids"][0].tolist()
        tokenizer = self.processor.tokenizer
        start, end = (
            tokenizer.convert_tokens_to_ids(self.processor.vision_start_token),
            tokenizer.convert_tokens_to_ids(self.processor.vision_end_token),
        )
        image_positions = [i for i, token in enumerate(ids) if token == self.processor.image_token_id]
        self.assertEqual(ids[image_positions[0] - 1], start)
        self.assertEqual(ids[image_positions[-1] + 1], end)
        audio_positions = [i for i, token in enumerate(ids) if token == self.processor.audio_token_id]
        self.assertEqual(ids[audio_positions[0] - 1], start)
        self.assertEqual(ids[audio_positions[-1] + 1], end)
        # a video is one timestamped, wrapped group per temporal patch, wrapped again as a whole
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        from transformers.video_utils import VideoMetadata

        metadata = VideoMetadata(
            total_num_frames=4, fps=2.0, duration=2.0, video_backend="np", frames_indices=[0, 1, 2, 3]
        )
        outputs = self.processor(text="<|video|>", videos=[frames], video_metadata=[metadata], return_tensors="pt")
        ids = outputs["input_ids"][0].tolist()
        grid = outputs["video_grid_thw"][0]
        per_group = int(grid[1] * grid[2]) // self.processor.video_processor.merge_size**2
        self.assertEqual(ids.count(self.processor.video_token_id), int(grid[0]) * per_group)
        self.assertEqual(ids.count(start), 1 + int(grid[0]))
        self.assertEqual(ids.count(end), 1 + int(grid[0]))
        # the clip opens with the outer span, then a timestamp precedes the first group's inner span
        self.assertEqual(ids[0], start)
        self.assertNotIn(ids[1], (start, end, self.processor.video_token_id))
        self.assertEqual(self.processor._calculate_timestamps([0, 1, 2, 3], 2.0, 2), [0.25, 1.25])

    def test_video_soundtrack_is_split_across_frame_groups(self):
        from transformers.video_utils import VideoMetadata

        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        metadata = VideoMetadata(
            total_num_frames=4, fps=2.0, duration=2.0, video_backend="np", frames_indices=[0, 1, 2, 3]
        )
        soundtrack = np.zeros(16000 * 2, dtype=np.float32)  # 2 s = 200 mel frames = 25 encoder tokens
        video = {"videos": [frames], "video_metadata": [metadata]}
        audio_id = self.processor.audio_token_id

        def audio_runs(ids):
            runs, run = [], 0
            for token in ids:
                run += token == audio_id
                if token != audio_id and run:
                    runs.append(run)
                    run = 0
            return runs

        with self.assertRaises(ValueError):  # no clip after the video to take as its soundtrack
            self.processor(text="<|audio|><|video|>", audio=[soundtrack], use_audio_in_video=True, **video)
        # the soundtrack placeholder may sit after the text, where `apply_chat_template(load_audio_from_video=True)`
        # appends it
        text = "<|video|> what is said? <|audio|> and <|audio|>"
        outputs = self.processor(
            text=text, audio=[soundtrack, self.audio], return_tensors="pt", use_audio_in_video=True, **video
        )
        ids = outputs["input_ids"][0].tolist()
        grid = outputs["video_grid_thw"][0]
        per_group = int(grid[1] * grid[2]) // self.processor.video_processor.merge_size**2
        self.assertEqual(ids.count(self.processor.video_token_id), int(grid[0]) * per_group)
        soundtrack_tokens = self.processor.feature_extractor.num_encoder_tokens(
            int(outputs["feature_attention_mask"][0].sum())
        )
        standalone_tokens = self.processor.feature_extractor.num_encoder_tokens(
            int(outputs["feature_attention_mask"][1].sum())
        )
        # the first frame group starts at 0 s and the second at 1 s: 1 s / 0.08 s per token = 12 tokens, then the rest
        self.assertEqual(audio_runs(ids), [12, soundtrack_tokens - 12, standalone_tokens])
        self.assertEqual(outputs["input_features"].shape[0], 2)
        # without the flag every clip is its own media span, even right after a video
        outputs = self.processor(text=text, audio=[soundtrack, self.audio], return_tensors="pt", **video)
        self.assertEqual(audio_runs(outputs["input_ids"][0].tolist()), [soundtrack_tokens, standalone_tokens])
        for use_audio_in_video, runs in ((False, [soundtrack_tokens]), (True, [12, soundtrack_tokens - 12])):
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "video"}, {"type": "audio"}]}], tokenize=False
            )
            outputs = self.processor(
                text=prompt, audio=[soundtrack], return_tensors="pt", use_audio_in_video=use_audio_in_video, **video
            )
            self.assertEqual(audio_runs(outputs["input_ids"][0].tolist()), runs)

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
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("raushan-testing-hf/videos-test", "sample_demo_1.mp4", repo_type="dataset")
        video, clip, text = (
            {"type": "video", "video": path},
            {"type": "audio", "audio": path},
            {"type": "text", "text": "Hi"},
        )
        soundtrack = {"processor_kwargs": {"use_audio_in_video": True}}

        def encode(content, **kwargs):
            out = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                return_dict=True,
                add_generation_prompt=True,
                return_tensors="pt",
                **kwargs,
            )
            return out["input_ids"][0].tolist(), out["feature_attention_mask"].shape[0]

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

    def test_replacement_offsets_index_the_caller_text_through_every_rewrite(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        start, end = self.processor.vision_start_token, self.processor.vision_end_token
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        video = {
            "videos": [frames],
            "video_metadata": [VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])],
        }
        speech = [np.zeros(1920 * 2, dtype=np.float32)]
        soundtrack = np.zeros(16000, dtype=np.float32)
        for name, text, kwargs in (
            ("wrapped image", f"{start}<|image|>{end} look <|audio_out|> x <|audio|>", {"images": [self.image]}),
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
                    kwargs = {**kwargs, "audio_out": speech}
                outputs = self.processor(text=text, return_text_replacement_offsets=True, **kwargs)
                assert_offsets_rebuild_the_prompts(self, self.processor, [text], outputs)

    def test_assistant_mask_covers_the_answer_around_speech_frames(self):
        from transformers.video_utils import VideoMetadata

        self.processor.audio_tokenizer = tiny_codec()
        template = GENERATION_TEMPLATE.replace(
            "{%- elif content['type'] == 'audio' -%}",
            "{%- elif content['type'] == 'video' -%}{{- '<|video|>' -}}{%- elif content['type'] == 'audio' -%}",
        )
        metadata = VideoMetadata(total_num_frames=4, fps=2.0, frames_indices=[0, 1, 2, 3])
        frames = np.zeros((4, 32, 64, 3), dtype=np.uint8)
        answer = [{"type": "text", "text": "Hello ."}]
        speech = {"type": "audio_out"}
        for name, user, assistant, extra in (
            ("speech in the user turn", [speech, {"type": "text", "text": "Hello"}], answer, {}),
            ("speech in the assistant turn", [{"type": "text", "text": "Hello"}], [*answer, speech], {}),
            (
                "speech after a video",
                [{"type": "video", "video": frames}, speech],
                answer,
                {"video_metadata": [metadata]},
            ),
        ):
            with self.subTest(name):
                outputs = self.processor.apply_chat_template(
                    [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
                    chat_template=template,
                    tokenize=True,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                    processor_kwargs={"audio_out": [np.zeros(1920 * 2, dtype=np.float32)], **extra},
                )
                ids = list(outputs["input_ids"][0])
                self.assertEqual(list(outputs["assistant_masks"][0]), expected_assistant_mask(self.processor, ids))
