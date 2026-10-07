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
"""Testing suite for the PyTorch EmbeddingGemma2 model."""

import os
import tempfile
import unittest

from huggingface_hub import download_bucket_files
from parameterized import parameterized
from safetensors.torch import load_file

from transformers import (
    EmbeddingGemma2Config,
    EmbeddingGemma2TextConfig,
    is_torch_available,
    set_seed,
)
from transformers.testing_utils import (
    CaptureLogger,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import ModelOutput, logging

from ...test_configuration_common import ConfigTester
from ...test_memory_cleanup_mixin import MemoryCleanupMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor
from ...test_processing_common import url_to_local_path


if is_torch_available():
    import torch
    import torch.nn.functional as F

    from transformers import (
        AutoProcessor,
        EmbeddingGemma2Model,
        EmbeddingGemma2TextModel,
    )


class EmbeddingGemma2TextModelTester:
    """Builds a tiny `EmbeddingGemma2TextConfig` and matching text-only inputs."""

    config_class = EmbeddingGemma2TextConfig
    if is_torch_available():
        base_model_class = EmbeddingGemma2TextModel

    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        sliding_window=4,
        hidden_size_per_layer_input=16,
        embedding_dim=24,
        pad_token_id=0,
        initializer_range=0.02,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.sliding_window = sliding_window
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        self.embedding_dim = embedding_dim
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range

        # Both attention flavours are exercised; the last layer is always forced to `full_attention`.
        self.layer_types = ["sliding_attention", "full_attention"]
        # Gemma 4 (and therefore EmbeddingGemma 2) uses a wider `head_dim` on the full attention layers.
        # Without an explicit override the config falls back to `global_head_dim=512`, which would blow
        # up the size of this tiny model.
        self.per_layer_config = {
            layer_idx: {"head_dim": 2 * self.head_dim}
            for layer_idx, layer_type in enumerate(self.layer_types)
            if layer_type == "full_attention"
        }
        self.encoder_seq_length = seq_length

    def get_config(self):
        return EmbeddingGemma2TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            sliding_window=self.sliding_window,
            hidden_size_per_layer_input=self.hidden_size_per_layer_input,
            embedding_dim=self.embedding_dim,
            pad_token_id=self.pad_token_id,
            initializer_range=self.initializer_range,
            layer_types=self.layer_types,
            per_layer_config=self.per_layer_config,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        attention_mask = None
        if self.use_input_mask:
            attention_mask = torch.ones_like(input_ids).to(torch_device)
        return self.get_config(), input_ids, attention_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class EmbeddingGemma2TextModelTest(ModelTesterMixin, unittest.TestCase):
    # EmbeddingGemma 2 is an encoder: there is no `ForCausalLM` / `ForConditionalGeneration`, hence no
    # `GenerationTesterMixin` and no generative classes.
    all_model_classes = (EmbeddingGemma2TextModel,) if is_torch_available() else ()
    all_generative_model_classes = ()

    def setUp(self):
        self.model_tester = EmbeddingGemma2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EmbeddingGemma2TextConfig, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()


class EmbeddingGemma2ModelTester:
    """Builds a tiny composite config: text backbone + reused Gemma 4 vision and audio towers."""

    def __init__(
        self,
        parent,
        mm_tokens_per_image=2,
        image_token_id=4,
        video_token_id=7,
        audio_token_id=8,
        boi_token_id=5,
        eoi_token_id=6,
        seq_length=25,
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        audio_config={
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "output_proj_dims": 32,
            "subsampling_conv_channels": [8, 8],
        },
    ):
        self.parent = parent
        self.mm_tokens_per_image = mm_tokens_per_image
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.boi_token_id = boi_token_id
        self.eoi_token_id = eoi_token_id
        self.llm_tester = EmbeddingGemma2TextModelTester(self.parent)
        self.text_config = self.llm_tester.get_config()
        self.vision_config = vision_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.pad_token_id = self.text_config.pad_token_id

        self.num_hidden_layers = self.text_config.num_hidden_layers
        self.vocab_size = self.text_config.vocab_size
        self.hidden_size = self.text_config.hidden_size
        self.embedding_dim = self.text_config.embedding_dim
        self.num_attention_heads = self.text_config.num_attention_heads
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length

    def get_config(self):
        return EmbeddingGemma2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            audio_token_id=self.audio_token_id,
            boi_token_id=self.boi_token_id,
            eoi_token_id=self.eoi_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        config.vision_config.pooling_kernel_size = 2

        # (num_images, max_num_patches, patch_size * patch_size * num_channels)
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size,
                self.vision_config["image_size"],
                patch_size * patch_size * self.vision_config["num_channels"],
            ]
        )

        # create (h*w, 2) grid of (x, y) coords for a non-square input image
        num_patches = self.vision_config["image_size"]
        h = int(num_patches**0.5)
        w = num_patches // h

        xs = torch.arange(w).repeat(h)
        ys = torch.arange(h).repeat_interleave(w)
        pixel_position_ids = torch.stack([xs, ys], dim=-1).to(device=torch_device)
        pixel_position_ids = pixel_position_ids.unsqueeze(0).repeat(self.batch_size, 1, 1)

        return config, pixel_values, pixel_position_ids

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_position_ids = self.prepare_config_and_inputs()
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = input_ids.ne(self.pad_token_id).to(torch_device)

        # Ensure no tokens accidentally match special token IDs
        for token_id in [config.image_token_id, config.video_token_id, config.audio_token_id]:
            input_ids[input_ids == token_id] = self.pad_token_id
        input_ids[:, :5] = config.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "image_position_ids": pixel_position_ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class EmbeddingGemma2ModelTest(ModelTesterMixin, unittest.TestCase):
    # No generative classes: EmbeddingGemma 2 only ever produces embeddings.
    all_model_classes = (EmbeddingGemma2Model,) if is_torch_available() else ()
    all_generative_model_classes = ()
    additional_model_inputs = ["image_position_ids"]
    model_split_percents = [0.85, 0.9]

    def setUp(self):
        self.model_tester = EmbeddingGemma2ModelTester(self)
        self.config_tester = ConfigTester(self, config_class=EmbeddingGemma2Config, hidden_size=37)
        self.skip_flash_attn_inference_equivalence_tests()

    def skip_flash_attn_inference_equivalence_tests(self):
        skippable_tests = [
            "test_flash_attn_2_inference_equivalence",
            "test_flash_attn_3_inference_equivalence",
            "test_flash_attn_4_inference_equivalence",
        ]
        for test in skippable_tests:
            if self._testMethodName.startswith(test):
                self.skipTest(reason="The base test does not pass the image_position_ids required by EmbeddingGemma 2")

    # NOTE: no `test_config` here (unlike the text-model test): `ConfigTester.run_common_tests`
    # asserts a `vocab_size` attribute, which on a composite config only lives on `text_config`.
    # Gemma 4's multimodal test leaves its config tester unused for the same reason.

    def test_vision_axial_rope(self):
        # Override: the inherited Gemma 4 vision rope takes position IDs with a batch dim, which the
        # common test omits. Same override as `Gemma4Vision2TextModelTest`.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        rope_class = None
        base_model = EmbeddingGemma2Model(config)
        for _, module in base_model.named_modules():
            if hasattr(module, "compute_axial_rope_parameters"):
                rope_class = type(module)
                vision_config = module.config
                break

        if rope_class is None:
            self.skipTest("Couldn't infer RoPE layer for this model class.")

        # First make sure that validation on default config raises no rope-related warnings
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("", cl.out)
        logger.warning_once.cache_clear()

        # Axial rope type expects only `rope_theta`, otherwise raises warning
        vision_config.rope_parameters["factor"] = 0.25
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("Unrecognized keys in `rope_parameters` for 'rope_type'='axial': {'factor'}\n", cl.out)
        del vision_config.rope_parameters["factor"]
        logger.warning_once.cache_clear()

        inv_freq, attention_scale = rope_class.compute_axial_rope_parameters(config=vision_config)
        rope_module = rope_class(vision_config).to(device=torch_device)

        self.assertTrue(hasattr(rope_module, "inv_freq"))
        self.assertTrue(hasattr(rope_module, "attention_scaling"))
        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1
        torch.testing.assert_close(inv_freq, rope_module.inv_freq.cpu())

        # create 2D position IDs for a single grid of one row and 10 cols `size=(10, 2)`
        position_ids = torch.stack(
            [
                torch.arange(10, dtype=torch.long, device=torch_device),
                torch.zeros(10, dtype=torch.long, device=torch_device),
            ]
        ).transpose(0, 1)
        position_ids = position_ids[None, ...].repeat(3, 1, 1)  # batch size of `3`
        # and an empty hidden states used only to infer device/dtype
        hidden_states = torch.empty(1, dtype=torch.float32, device=torch_device)
        cos, sin = rope_module(hidden_states, position_ids)
        self.assertEqual(cos.shape[-1], inv_freq.shape[-1] * 4)  # the freq are `//4` of head dim
        self.assertEqual(cos.shape[0], 3)  # angles preserve batch

    def _audio_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        input_features = floats_tensor([self.model_tester.batch_size, 16, 8])
        input_features_mask = torch.ones(self.model_tester.batch_size, 16, dtype=torch.bool, device=torch_device)
        return config, {"input_features": input_features, "input_features_mask": input_features_mask}

    def _video_features_prepare_config_and_inputs(self):
        config, _, _ = self.model_tester.prepare_config_and_inputs()
        # Use a (2, 2) patch grid per frame with pooling_kernel_size=2 so each frame produces 1 pooled patch row,
        # matching `ModelTesterMixin.test_get_video_features_output`'s `last_hidden_state.shape[0] == batch_size`.
        batch_size = self.model_tester.batch_size
        max_patches = self.model_tester.vision_config["image_size"]
        patch_dim = config.vision_config.patch_size**2 * self.model_tester.vision_config["num_channels"]
        pixel_values_videos = floats_tensor([batch_size, max_patches, patch_dim])
        pixel_values_videos[:, 4:] = 0.0
        video_position_ids = torch.full((batch_size, max_patches, 2), -1, dtype=torch.long, device=torch_device)
        grid_2x2 = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.long, device=torch_device)
        video_position_ids[:, :4] = grid_2x2
        num_frames_per_video = torch.ones(batch_size, dtype=torch.long, device=torch_device)
        return config, {
            "pixel_values_videos": pixel_values_videos,
            "video_position_ids": video_position_ids,
            "num_frames_per_video": num_frames_per_video,
        }

    @parameterized.expand([True, False, None])
    def test_get_image_features_output(self, return_dict: bool | None):
        "Override to infer last hidden states' `batch_size` from image position ids"
        for model_class in self.all_model_classes:
            if not hasattr(model_class, "get_image_features"):
                continue

            config, inputs_dict = self._image_features_prepare_config_and_inputs()
            if return_dict is not None:
                config.return_dict = return_dict

            model = model_class(config).eval()
            model = model.to(torch_device)

            set_seed(42)
            with torch.no_grad():
                outputs = model.get_image_features(**inputs_dict)

            if return_dict in (True, None):
                self.assertTrue(isinstance(outputs, ModelOutput), "get_image_features() must return a BaseModelOutput")
                self.assertTrue(
                    hasattr(outputs, "last_hidden_state"),
                    "get_image_features() must return a BaseModelOutput with last_hidden_state",
                )
                self.assertTrue(
                    hasattr(outputs, "pooler_output"),
                    "get_image_features() must return a BaseModelOutput with pooler_output",
                )
                self.assertTrue(
                    hasattr(outputs, "hidden_states"),
                    "get_image_features() must return a BaseModelOutput with hidden_states",
                )
                if self.has_attentions:
                    self.assertTrue(
                        hasattr(outputs, "attentions"),
                        "get_image_features() must return a BaseModelOutput with attentions",
                    )

                last_hidden_state_shape = outputs.last_hidden_state.shape
                batch_size = inputs_dict["pixel_values"].shape[0]
                output_length = inputs_dict["pixel_values"].shape[-2] // (
                    model.config.vision_config.pooling_kernel_size**2
                )
                k_squared = int((inputs_dict["image_position_ids"].shape[1] // output_length) ** 0.5) ** 2
                batch_size *= inputs_dict["image_position_ids"].shape[1] // k_squared

                self.assertEqual(
                    last_hidden_state_shape[0],
                    batch_size,
                    f"batch_size mismatch, full shape: {last_hidden_state_shape}",
                )
                self.assertEqual(
                    last_hidden_state_shape[-1],
                    config.vision_config.hidden_size,
                    f"hidden_size mismatch, full shape: {last_hidden_state_shape}",
                )

                self.assertEqual(
                    len(outputs.pooler_output),
                    self.model_tester.batch_size,
                    f"batch_size mismatch for `pooler_output`: {len(outputs.pooler_output)} != "
                    f"{self.model_tester.batch_size}",
                )
                self.assertEqual(
                    outputs.pooler_output[0].ndim,
                    2,
                    f"each sample in `pooler_output` should be a 2D array but got {outputs.pooler_output[0].ndim}",
                )
            else:
                self.assertIsInstance(outputs, tuple, "get_image_features() must return a tuple if return_dict=False")


@slow
@require_torch_accelerator
class EmbeddingGemma2IntegrationTest(MemoryCleanupMixin, unittest.TestCase):
    """
    reproducer (uploads the golden to
    ``hf://buckets/hf-internal-testing/embeddinggemma2-integration-test/<case>/expected_embeddings.safetensors``,
    holding the `embeddings` and, where the case computes one, the `similarities` matrix):
        TODO
    """

    # TODO: pick the correct values once the golden embeddings are generated on the CI hardware
    # Tolerance against the golden embeddings and similarities from the bucket
    RTOL = 1e-2
    ATOL = 1e-2
    # Tolerance between batched and single-input embeddings of the same run (bf16 kernels vary with padded shapes)
    BATCH_RTOL = 2e-3
    BATCH_ATOL = 2e-3

    IMAGE_URL = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    IMAGE_2_URL = (
        "https://huggingface.co/datasets/hf-internal-testing/fixtures-coco/resolve/main/val2017/000000039769.jpg"
    )
    AUDIO_URL = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/song_1.mp3"
    VIDEO_URL = "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4"

    @classmethod
    def setUpClass(cls):
        # TODO: switch to `google/embeddinggemma-2` at release
        cls.checkpoint_name = "google/embeddinggemma-2"
        cls.bucket = "hf-internal-testing/embeddinggemma2-integration-test"
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)

        cls.image = url_to_local_path(cls.IMAGE_URL)
        cls.image_2 = url_to_local_path(cls.IMAGE_2_URL)
        cls.audio = url_to_local_path(cls.AUDIO_URL)
        cls.video = url_to_local_path(cls.VIDEO_URL)

    @staticmethod
    def _pool(model, inputs):
        """Mask-aware mean pooling of `last_hidden_state`, then L2 normalization in float32."""
        token_embeddings = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        embeddings = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return F.normalize(embeddings.float(), p=2, dim=-1)

    def _load_expected(self, case: str):
        remote = f"{case}/expected_embeddings.safetensors"
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "expected_embeddings.safetensors")
            download_bucket_files(self.bucket, files=[(remote, local)])
            return load_file(local)

    # ==== Text retrieval ====

    def test_model_text_query_document(self):
        """`encode_query` / `encode_document`: the query and the documents carry different task prompts."""
        model = EmbeddingGemma2Model.from_pretrained(self.checkpoint_name, device_map=torch_device)

        venus = "Venus is often called Earth's twin because of its similar size and proximity."
        mars = "Mars, known for its reddish appearance, is often referred to as the Red Planet."
        query_messages = [
            {"role": "system", "content": "task: search result | query: "},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which planet is known as the Red Planet?"},
                ],
            },
        ]
        document_messages = [
            [
                {"role": "system", "content": "title: none | text: "},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": venus},
                    ],
                },
            ],
            [
                {"role": "system", "content": "title: none | text: "},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": mars},
                    ],
                },
            ],
        ]

        # Queries and documents are separate forward passes, as `encode_query` and `encode_document` are
        query_inputs = self.processor.apply_chat_template(
            query_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        document_inputs = self.processor.apply_chat_template(
            document_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        query_embeddings = self._pool(model, query_inputs)
        document_embeddings = self._pool(model, document_inputs)
        similarities = query_embeddings @ document_embeddings.T

        expected = self._load_expected("text_query_document")
        torch.testing.assert_close(
            torch.cat([query_embeddings, document_embeddings]).cpu(),
            expected["embeddings"],
            rtol=self.RTOL,
            atol=self.ATOL,
        )
        torch.testing.assert_close(similarities.cpu(), expected["similarities"], rtol=self.RTOL, atol=self.ATOL)
        self.assertGreater(similarities[0, 1], similarities[0, 0])

        # Matryoshka: a truncated, re-normalized prefix keeps the ranking
        truncated = (
            F.normalize(query_embeddings[:, :256], dim=-1) @ F.normalize(document_embeddings[:, :256], dim=-1).T
        )
        self.assertGreater(truncated[0, 1], truncated[0, 0])

    # ==== Single modality ====

    def test_model_single_modality(self):
        """One key per modality: text, image, audio and video each embedded on their own, one conversation per row."""
        model = EmbeddingGemma2Model.from_pretrained(self.checkpoint_name, device_map=torch_device)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A photo of a cat"},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": self.image},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "url": self.audio},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "url": self.video},
                    ],
                },
            ],
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        embeddings = self._pool(model, inputs)

        expected = self._load_expected("single_modality")
        torch.testing.assert_close(embeddings.cpu(), expected["embeddings"], rtol=self.RTOL, atol=self.ATOL)

    # ==== Several modalities in one input ====

    def test_model_multiple_modalities(self):
        """Several modalities in one input give one embedding: text + image, text + audio and image + audio."""
        model = EmbeddingGemma2Model.from_pretrained(self.checkpoint_name, device_map=torch_device)

        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A photo of a cat"},
                        {"type": "image", "url": self.image},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A song"},
                        {"type": "audio", "url": self.audio},
                    ],
                },
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": self.image},
                        {"type": "audio", "url": self.audio},
                    ],
                },
            ],
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        embeddings = self._pool(model, inputs)

        expected = self._load_expected("multiple_modalities")
        torch.testing.assert_close(embeddings.cpu(), expected["embeddings"], rtol=self.RTOL, atol=self.ATOL)

    def test_model_manual_placeholders(self):
        """Placeholders written in the text interleave the media with it, in the order the media items are given."""
        model = EmbeddingGemma2Model.from_pretrained(self.checkpoint_name, device_map=torch_device)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": self.image},
                    {"type": "image", "url": self.image_2},
                    {"type": "audio", "url": self.audio},
                    {"type": "text", "text": "A jacket similar to <|image|> or <|image|> featured in <|audio|>"},
                ],
            },
        ]
        self.assertEqual(
            self.processor.apply_chat_template(messages, tokenize=False),
            "A jacket similar to <|image|> or <|image|> featured in <|audio|>",
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        embeddings = self._pool(model, inputs)

        expected = self._load_expected("manual_placeholders")
        torch.testing.assert_close(embeddings.cpu(), expected["embeddings"], rtol=self.RTOL, atol=self.ATOL)

    # ==== Batching ====

    def test_model_mixed_modality_batch(self):
        """A batch mixing text, image, text + image and audio embeds each row as if it were alone."""
        model = EmbeddingGemma2Model.from_pretrained(self.checkpoint_name, device_map=torch_device)

        conversations = [
            [
                {"role": "system", "content": "title: none | text: "},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A photo of a cat"},
                    ],
                },
            ],
            [
                {"role": "system", "content": "title: none | text: "},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": self.image},
                    ],
                },
            ],
            [
                {"role": "system", "content": "title: none | text: "},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "A photo of a cat"},
                        {"type": "image", "url": self.image},
                    ],
                },
            ],
            [
                {"role": "system", "content": "title: none | text: "},
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "url": self.audio},
                    ],
                },
            ],
        ]
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        batched = self._pool(model, inputs)

        expected = self._load_expected("mixed_modality_batch")
        torch.testing.assert_close(batched.cpu(), expected["embeddings"], rtol=self.RTOL, atol=self.ATOL)

        for idx, conversation in enumerate(conversations):
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            torch.testing.assert_close(
                batched[idx : idx + 1], self._pool(model, inputs), rtol=self.BATCH_RTOL, atol=self.BATCH_ATOL
            )
