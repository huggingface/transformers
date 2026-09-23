# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Testing suite for the PyTorch NemotronH_Omni model."""

import unittest

from transformers import (
    NemotronH_Omni_Reasoning_V3_Config,
    ParakeetFeatureExtractor,
    PreTrainedTokenizerFast,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)
from transformers.video_utils import load_video

from ...generation.test_utils import GenerationTesterMixin
from ...test_modeling_common import ModelTesterMixin, floats_tensor, ids_tensor, random_attention_mask


if is_torch_available():
    import torch

    from transformers import (
        NemotronH_Omni_Reasoning_V3,
        NemotronH_Omni_Reasoning_V3ImageProcessor,
        NemotronH_Omni_Reasoning_V3Processor,
        NemotronH_Omni_Reasoning_V3VideoProcessor,
    )


class NemotronHOmniVisionText2TextModelTester:
    """Builds a tiny NemotronH_Omni model and coupled multimodal inputs.

    The image branch is sized so a single image yields exactly one `img_context` token after
    the RADIO patch-embed + pixel-shuffle:
        num_image_token = (force_image_size // patch_size) ** 2 * downsample_ratio ** 2
                        = (32 // 16) ** 2 * 0.5 ** 2 = 1
    so each sequence must contain exactly one `image_token_id`. Likewise each audio clip of
    `num_audio_frames` mel frames yields `num_audio_token` embeddings after the 8x conv subsampling.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        force_image_size=32,
        patch_size=16,
        downsample_ratio=0.5,
        vision_hidden_size=32,
        projector_hidden_size=64,
        image_token_id=1,
        audio_token_id=2,
        num_audio_frames=16,
        video_temporal_patch_size=2,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.force_image_size = force_image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.vision_hidden_size = vision_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.num_audio_frames = num_audio_frames
        self.video_temporal_patch_size = video_temporal_patch_size
        self.is_training = is_training

        self.vocab_size = 99
        self.hidden_size = 32
        self.text_config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "layers_block_type": ["linear_attention", "moe", "full_attention", "moe"],
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 40,
            "moe_intermediate_size": 40,
            "moe_shared_expert_intermediate_size": 40,
            "mlp_hidden_act": "relu2",
            "mamba_hidden_act": "silu",
            "ssm_state_size": 16,
            "mamba_num_heads": 8,
            "mamba_n_groups": 2,
            "mamba_head_dim": 8,
            "mamba_d_conv": 4,
            "mamba_expand": 2,
            "mamba_chunk_size": 8,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "use_mamba_kernels": False,
        }
        self.vision_config = {
            "hidden_size": self.vision_hidden_size,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "mlp_ratio": 2.0,
            "patch_size": self.patch_size,
            "image_size": self.force_image_size,
            "max_img_size": 64,
            "num_channels": 3,
            # >= 2 cls tokens so the default summary_idxs=[0, 1] is in-bounds
            "num_cls_tokens": 2,
            "num_registers": 1,
            # must match the top-level value; the tower builds its video patch projection from it
            "video_temporal_patch_size": video_temporal_patch_size,
        }
        # The video path reuses the same RADIO tower, so its expected layer counts are the vision ones.
        self.video_config = self.vision_config
        # Tiny Parakeet encoder + projection. Kept enabled so the `audio_tower` / `embed_audio`
        # weight renames in the conversion mapping have matching keys to check.
        self.audio_config = {
            "model_type": "parakeet",
            "hidden_size": 32,
            "num_attention_heads": 2,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "conv_kernel_size": 9,
            "convolution_bias": False,
            "subsampling_conv_channels": 16,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "subsampling_factor": 8,
            "num_mel_bins": 32,
            "projection_hidden_size": 64,
            "projection_bias": False,
            "sampling_rate": 16000,
        }
        self.num_hidden_layers = len(self.text_config["layers_block_type"])
        self.num_attention_heads = self.text_config["num_attention_heads"]
        self.num_image_token = int((force_image_size // patch_size) ** 2 * (downsample_ratio**2))
        self.num_audio_token = num_audio_frames // self.audio_config["subsampling_factor"]

    def get_config(self):
        return NemotronH_Omni_Reasoning_V3_Config(
            vision_config=self.vision_config,
            text_config=self.text_config,
            audio_config=self.audio_config,
            force_image_size=self.force_image_size,
            downsample_ratio=self.downsample_ratio,
            vision_hidden_size=self.vision_hidden_size,
            projector_hidden_size=self.projector_hidden_size,
            image_token_id=self.image_token_id,
            audio_token_id=self.audio_token_id,
            video_temporal_patch_size=self.video_temporal_patch_size,
            attn_implementation="eager",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # ids in [3, vocab) so they never collide with image_token_id (1) or audio_token_id (2)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 3) + 3
        audio_start = 1 + self.num_image_token
        audio_end = audio_start + self.num_audio_token
        input_ids[:, 1:audio_start] = self.image_token_id
        input_ids[:, audio_start:audio_end] = self.audio_token_id
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        attention_mask[:, :audio_end] = 1  # keep multimodal tokens unmasked
        grid_size = self.force_image_size // self.patch_size
        pixel_values = floats_tensor([self.batch_size * grid_size**2, 3 * self.patch_size**2])
        image_grid_hw = torch.tensor([[grid_size, grid_size]] * self.batch_size)
        input_features = floats_tensor([self.batch_size, self.num_audio_frames, self.audio_config["num_mel_bins"]])
        input_features_mask = torch.ones(self.batch_size, self.num_audio_frames, dtype=torch.long)
        return config, input_ids, attention_mask, pixel_values, image_grid_hw, input_features, input_features_mask

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, pixel_values, image_grid_hw, input_features, input_features_mask = (
            self.prepare_config_and_inputs()
        )
        inputs_dict = {
            "pixel_values": pixel_values,
            "image_grid_hw": image_grid_hw,
            "input_features": input_features,
            "input_features_mask": input_features_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class NemotronHOmniModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (NemotronH_Omni_Reasoning_V3,) if is_torch_available() else ()
    all_generative_model_classes = (NemotronH_Omni_Reasoning_V3,) if is_torch_available() else ()
    _is_composite = True
    # The video path packs `video_temporal_patch_dim` frames into a single tower pass, so the
    # vision batch dim is deliberately smaller than `pixel_values_videos.shape[0]`.
    skip_test_video_features_output_shape = True
    # packed image patches have no batch dimension
    skip_test_image_features_output_shape = True
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = NemotronHOmniVisionText2TextModelTester(self)

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = super().prepare_config_and_inputs_for_generate(batch_size=batch_size)
        # packed patches cannot be sliced per sample like the other inputs; keep the patches of the kept images
        grid_size = self.model_tester.force_image_size // self.model_tester.patch_size
        inputs_dict["pixel_values"] = floats_tensor(
            [len(inputs_dict["image_grid_hw"]) * grid_size**2, 3 * self.model_tester.patch_size**2]
        )
        return config, inputs_dict

    def _video_features_prepare_config_and_inputs(self):
        config = self.model_tester.get_config()
        size = self.model_tester.force_image_size
        return config, {"pixel_values_videos": floats_tensor([self.model_tester.batch_size, 3, size, size])}

    @unittest.skip(reason="Mixed Mamba/attention stack does not expose uniform per-layer outputs")
    def test_attention_outputs(self):
        pass

    @unittest.skip(reason="Mixed Mamba/attention stack does not expose uniform per-layer outputs")
    def test_hidden_states_output(self):
        pass

    @unittest.skip(reason="Mixed Mamba/attention stack does not expose uniform per-layer outputs")
    def test_retain_grad_hidden_states_attentions(self):
        pass

    @unittest.skip(reason="Language model needs at least one block of each mixed layer type")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip(reason="Composite attention-implementation dispatch not wired for sub-models")
    def test_attn_implementation_composite_models(self):
        pass

    @unittest.skip(reason="Composite attention-implementation dispatch not wired for sub-models")
    def test_can_set_attention_dynamically_composite_model(self):
        pass

    @unittest.skip(reason="Composite attention-implementation dispatch not wired for sub-models")
    def test_config_attn_implementation_setter(self):
        pass

    @unittest.skip(reason="Composite attention-implementation dispatch not wired for sub-models")
    def test_sdpa_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="Composite attention-implementation dispatch not wired for sub-models")
    def test_flash_attn_2_can_dispatch_composite_models(self):
        pass

    @unittest.skip(reason="device_map offload not supported (RADIO summary_idxs buffer / Mamba state)")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="device_map offload not supported (RADIO summary_idxs buffer / Mamba state)")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="device_map offload not supported (RADIO summary_idxs buffer / Mamba state)")
    def test_disk_offload_safetensors(self):
        pass

    @unittest.skip(reason="device_map offload not supported (RADIO summary_idxs buffer / Mamba state)")
    def test_model_parallelism(self):
        pass

    @unittest.skip(reason="device_map offload not supported (RADIO summary_idxs buffer / Mamba state)")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="NemotronH hybrid Mamba cache is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search_0_random(self):
        pass

    @unittest.skip(reason="NemotronH hybrid Mamba cache is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search_1_same(self):
        pass

    @unittest.skip(reason="NemotronH hybrid Mamba cache is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip(reason="NemotronH hybrid Mamba cache does not expose a standard past_key_values format")
    def test_past_key_values_format(self):
        pass

    @unittest.skip(reason="NemotronH hybrid Mamba cache: cached-generation hidden states are not uniform")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="NemotronH hybrid Mamba cache: cached-generation hidden states are not uniform")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="Composite model exposes no single base transformer via base_model_prefix")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(reason="RADIO config is @strict and rejects the scalar norm_std this test injects")
    def test_can_load_ignoring_mismatched_shapes(self):
        pass

    @unittest.skip(reason="RADIO summary_idxs is an int64 index buffer, exempt from dtype casting")
    def test_keep_in_fp32_modules(self):
        pass


@slow
@require_torch_gpu
@require_flash_attn
class NemotronH_Omni_Reasoning_V3IntegrationTest(unittest.TestCase):
    """End-to-end greedy generation against the released checkpoint, one test per modality.

    No expected text is pinned. The checkpoint runs in bfloat16 and its mamba/RADIO kernels are not
    bit-reproducible across GPU architectures. Each test asserts that the modality's full
    path (processor -> encoder -> projector -> LM) runs and decodes to non-empty text.
    """

    model_id = "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"
    num_video_frames = 8
    max_new_tokens = 12

    @classmethod
    def setUpClass(cls):
        # The Hub repo still advertises the pre-port `NemotronH_Nano_Omni_Reasoning_V3*` classes via
        # `auto_map`, so `AutoProcessor.from_pretrained` routes to remote code. Composing the native
        # processor from the same repo files avoids that and keeps this an end-to-end test of the port.
        tokenizer = PreTrainedTokenizerFast.from_pretrained(cls.model_id)
        cls.processor = NemotronH_Omni_Reasoning_V3Processor(
            image_processor=NemotronH_Omni_Reasoning_V3ImageProcessor.from_pretrained(cls.model_id),
            video_processor=NemotronH_Omni_Reasoning_V3VideoProcessor.from_pretrained(cls.model_id),
            feature_extractor=ParakeetFeatureExtractor(feature_size=128),
            tokenizer=tokenizer,
            chat_template=tokenizer.chat_template,
        )
        cls.model = NemotronH_Omni_Reasoning_V3.from_pretrained(
            cls.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
            # ParakeetEncoder has no flash-attention kernel
            attn_implementation={"": "flash_attention_2", "audio_config": "sdpa"},
        )

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def _generate(self, inputs):
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output[0, inputs["input_ids"].shape[-1] :]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True)

    def _chat(self, content, **kwargs):
        return self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
            **kwargs,
        )

    def test_image_generation(self):
        from huggingface_hub import hf_hub_download
        from PIL import Image

        image_path = hf_hub_download(repo_id=self.model_id, filename="media/example1a.jpeg")
        image = Image.open(image_path).convert("RGB")
        inputs = self._chat(
            [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image in detail."}],
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self._generate(inputs).strip())

    def test_audio_generation(self):
        from huggingface_hub import hf_hub_download

        audio_path = hf_hub_download(repo_id=self.model_id, filename="media/2414-165385-0000.wav")
        inputs = self._chat(
            [{"type": "audio", "audio": audio_path}, {"type": "text", "text": "Transcribe this audio."}],
            tokenize=True,
            return_dict=True,
        )
        self.assertTrue(self._generate(inputs).strip())

    def test_video_generation(self):
        import numpy as np
        from huggingface_hub import hf_hub_download
        from PIL import Image

        video_path = hf_hub_download(repo_id=self.model_id, filename="media/demo.mp4")

        def sample_indices_fn(metadata, **kwargs):
            return np.linspace(0, metadata.total_num_frames - 1, self.num_video_frames).round().astype(int)

        frames, _ = load_video(video_path, backend="decord", sample_indices_fn=sample_indices_fn)
        # The chat template only accepts pre-sampled frames, so the processor is called directly.
        text = self._chat(
            [{"type": "video", "video": None}, {"type": "text", "text": "Describe this video."}], tokenize=False
        )
        inputs = self.processor(text=text, videos=[[Image.fromarray(f) for f in frames]], return_tensors="pt")

        # 8 frames packed 2-per-temporal-patch -> 4 tower passes x 252 tokens after pixel shuffle.
        self.assertEqual(int((inputs["input_ids"] == self.model.image_token_id).sum()), 1008)

        self.model.video_pruning_rate = 0.0
        self.assertTrue(self._generate(inputs).strip())
