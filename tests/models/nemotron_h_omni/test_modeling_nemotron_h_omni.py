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

from transformers import AutoTokenizer, NemotronH_Omni_Reasoning_V3_Config, is_torch_available
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
    )


class NemotronHOmniVisionText2TextModelTester:
    """Builds a tiny NemotronH_Omni model and coupled multimodal inputs.

    The image branch is sized so a single image yields exactly one `img_context` token after
    the RADIO patch-embed + pixel-shuffle:
        num_image_token = (force_image_size // patch_size) ** 2 * downsample_ratio ** 2
                        = (32 // 16) ** 2 * 0.5 ** 2 = 1
    so each sequence must contain exactly one `img_context_token_id`.
    """

    def __init__(
        self,
        parent,
        batch_size=2,
        seq_length=8,
        force_image_size=32,
        patch_size=16,
        downsample_ratio=0.5,
        vit_hidden_size=32,
        projector_hidden_size=64,
        img_context_token_id=1,
        is_training=False,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.force_image_size = force_image_size
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.vit_hidden_size = vit_hidden_size
        self.projector_hidden_size = projector_hidden_size
        self.img_context_token_id = img_context_token_id
        self.is_training = is_training

        self.vocab_size = 99
        self.hidden_size = 32
        self.llm_config = {
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
            "hidden_size": self.vit_hidden_size,
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
        }
        # The video path reuses the same RADIO tower, so its expected layer counts are the vision ones.
        self.video_config = self.vision_config
        # Tiny Parakeet encoder + projection. Kept enabled so the `sound_encoder` / `sound_projection`
        # weight renames in the conversion mapping have matching keys to check.
        self.sound_config = {
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
        self.num_hidden_layers = len(self.llm_config["layers_block_type"])
        self.num_attention_heads = self.llm_config["num_attention_heads"]
        self.num_image_token = int((force_image_size // patch_size) ** 2 * (downsample_ratio**2))

    def get_config(self):
        return NemotronH_Omni_Reasoning_V3_Config(
            vision_config=self.vision_config,
            llm_config=self.llm_config,
            sound_config=self.sound_config,
            force_image_size=self.force_image_size,
            downsample_ratio=self.downsample_ratio,
            vit_hidden_size=self.vit_hidden_size,
            projector_hidden_size=self.projector_hidden_size,
            img_context_token_id=self.img_context_token_id,
            attn_implementation="eager",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        # ids in [3, vocab) so they never collide with img_context_token_id (1)
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size - 3) + 3
        input_ids[:, 1 : 1 + self.num_image_token] = self.img_context_token_id
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        attention_mask[:, : 1 + self.num_image_token] = 1  # keep image tokens unmasked
        pixel_values = floats_tensor([self.batch_size, 3, self.force_image_size, self.force_image_size])
        image_flags = torch.ones(self.batch_size, 1, dtype=torch.long)
        return config, input_ids, attention_mask, pixel_values, image_flags

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask, pixel_values, image_flags = self.prepare_config_and_inputs()
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
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
    test_pruning = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = NemotronHOmniVisionText2TextModelTester(self)

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

    @unittest.skip(reason="Model produces return_dict outputs only; the tuple path is unsupported")
    def test_model_outputs_equivalence(self):
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
        # processor from the same repo files avoids that and keeps this an end-to-end test of the
        # port; it yields byte-identical inputs to the remote processor for all three modalities.
        tokenizer = AutoTokenizer.from_pretrained(cls.model_id)
        image_processor = NemotronH_Omni_Reasoning_V3ImageProcessor.from_pretrained(cls.model_id)
        cls.processor = NemotronH_Omni_Reasoning_V3Processor(
            image_processor=image_processor, tokenizer=tokenizer, chat_template=tokenizer.chat_template
        )
        cls.model = NemotronH_Omni_Reasoning_V3.from_pretrained(
            cls.model_id, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2"
        ).eval()

    @classmethod
    def tearDownClass(cls):
        del cls.model
        cleanup(torch_device, gc_collect=True)

    def _generate(self, inputs):
        accepted = {
            "input_ids",
            "attention_mask",
            "pixel_values",
            "pixel_values_videos",
            "input_features",
            "input_features_mask",
        }
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items() if k in accepted}
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
        self.assertEqual(int((inputs["input_ids"] == self.model.img_context_token_id).sum()), 1008)

        self.model.video_pruning_rate = 0.0
        self.assertTrue(self._generate(inputs).strip())
