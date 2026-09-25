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

import copy
import re
import tempfile
import unittest

from transformers import (
    NemotronH_Omni_Reasoning_V3_Config,
    NemotronHConfig,
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    PreTrainedTokenizerFast,
    RadioConfig,
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

from ...alm_tester import ALMModelTest, ALMModelTester
from ...multimodal_tester import MultiModalModelTester
from ...test_modeling_common import floats_tensor, ids_tensor
from ...vlm_tester import VLMModelTest, VLMModelTester
from ..nemotron_h import test_modeling_nemotron_h


if is_torch_available():
    import torch

    from transformers import (
        NemotronH_Omni_Reasoning_V3,
        NemotronH_Omni_Reasoning_V3ImageProcessor,
        NemotronH_Omni_Reasoning_V3Processor,
        NemotronH_Omni_Reasoning_V3VideoProcessor,
    )


def get_tiny_text_config(tester) -> "NemotronHConfig":
    return NemotronHConfig(
        vocab_size=tester.vocab_size,
        hidden_size=tester.hidden_size,
        layers_block_type=["linear_attention", "moe", "full_attention", "moe"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=40,
        moe_intermediate_size=40,
        moe_shared_expert_intermediate_size=40,
        mlp_hidden_act="relu2",
        mamba_hidden_act="silu",
        ssm_state_size=16,
        mamba_num_heads=8,
        mamba_n_groups=2,
        mamba_head_dim=8,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_chunk_size=8,
        n_routed_experts=4,
        num_experts_per_tok=2,
        use_mamba_kernels=False,
        pad_token_id=tester.pad_token_id,
        bos_token_id=tester.bos_token_id,
        eos_token_id=tester.eos_token_id,
    )


TINY_VISION_LAYERS = 2
TINY_AUDIO_LAYERS = 2


def get_tiny_vision_config(tester) -> "RadioConfig":
    return RadioConfig(
        hidden_size=tester.vision_hidden_size,
        num_hidden_layers=TINY_VISION_LAYERS,
        num_attention_heads=4,
        mlp_ratio=2.0,
        patch_size=tester.patch_size,
        image_size=tester.image_size,
        max_img_size=64,
        num_channels=3,
        # >= 2 cls tokens so the default summary_idxs=[0, 1] is in-bounds
        num_cls_tokens=2,
        num_registers=1,
        video_temporal_patch_size=tester.video_temporal_patch_size,
    )


def get_tiny_audio_config(tester) -> "ParakeetEncoderConfig":
    return ParakeetEncoderConfig(
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=TINY_AUDIO_LAYERS,
        intermediate_size=64,
        conv_kernel_size=9,
        convolution_bias=False,
        subsampling_conv_channels=16,
        subsampling_conv_kernel_size=3,
        subsampling_conv_stride=2,
        subsampling_factor=tester.subsampling_factor,
        num_mel_bins=tester.num_mel_bins,
        attention_bias=False,
        scale_input=False,
        projection_hidden_size=64,
        projection_bias=False,
    )


def set_omni_tester_defaults(kwargs):
    """Sizes shared by the vision and audio testers. Both build the full model (vision, audio and a tiny hybrid
    NemotronH language model); they only differ in which modality they feed."""
    kwargs.setdefault("image_size", 32)
    kwargs.setdefault("patch_size", 16)
    kwargs.setdefault("downsample_ratio", 0.5)
    kwargs.setdefault("vision_hidden_size", 32)
    kwargs.setdefault("projector_hidden_size", 64)
    kwargs.setdefault("video_temporal_patch_size", 2)
    kwargs.setdefault("num_mel_bins", 32)
    # 64 mel frames subsample 8x into at most 8 audio tokens
    kwargs.setdefault("feat_seq_length", 64)
    kwargs.setdefault("subsampling_factor", 8)
    # the length of the tiny NemotronH's `layers_block_type`
    kwargs.setdefault("num_hidden_layers", 4)


class NemotronHOmniVision2TextModelTester(VLMModelTester):
    config_class = NemotronH_Omni_Reasoning_V3_Config
    text_config_class = NemotronHConfig
    vision_config_class = RadioConfig
    # the model is a single generative class with no separate base model
    base_model_class = None
    conditional_generation_class = NemotronH_Omni_Reasoning_V3
    _required_attributes = MultiModalModelTester._required_attributes + ("vision_config_class",)

    def __init__(self, parent, **kwargs):
        set_omni_tester_defaults(kwargs)
        # a (32 // 16) ** 2 patch grid pixel-shuffles 2x2 into a single image token
        kwargs.setdefault("num_image_tokens", 1)
        super().__init__(parent, **kwargs)

    @property
    def pipeline_model_mapping(self):
        return {"image-text-to-text": self.conditional_generation_class}

    def get_text_config(self):
        return get_tiny_text_config(self)

    def get_vision_config(self):
        return get_tiny_vision_config(self)

    def _build_modality_sub_configs(self):
        return {**super()._build_modality_sub_configs(), "audio_config": get_tiny_audio_config(self)}

    def _prepare_modality_inputs(self, input_ids, config):
        grid_size = self.image_size // self.patch_size
        pixel_values = floats_tensor([self.batch_size * grid_size**2, self.num_channels * self.patch_size**2])
        image_grid_hw = torch.tensor([[grid_size, grid_size]] * self.batch_size, device=torch_device)
        input_ids = self.place_image_tokens(input_ids, config)
        return input_ids, {"pixel_values": pixel_values, "image_grid_hw": image_grid_hw}


class NemotronHOmniAudio2TextModelTester(ALMModelTester):
    config_class = NemotronH_Omni_Reasoning_V3_Config
    text_config_class = NemotronHConfig
    audio_config_class = ParakeetEncoderConfig
    conditional_generation_class = NemotronH_Omni_Reasoning_V3
    audio_mask_key = "input_features_mask"

    def __init__(self, parent, **kwargs):
        set_omni_tester_defaults(kwargs)
        kwargs.setdefault("audio_token_id", 3)
        super().__init__(parent, **kwargs)

    @property
    def pipeline_model_mapping(self):
        # the ALM tester has no audio-text-to-text pipeline test yet
        return {}

    def get_text_config(self):
        return get_tiny_text_config(self)

    def get_audio_config(self):
        return get_tiny_audio_config(self)

    def _build_modality_sub_configs(self):
        # the vision tower is always built, so keep it tiny
        return {**super()._build_modality_sub_configs(), "vision_config": get_tiny_vision_config(self)}

    def create_audio_features(self, batch_size: int | None = None):
        # Parakeet takes `(batch, frames, mel_bins)`
        return floats_tensor([self.batch_size, self.feat_seq_length, self.num_mel_bins])

    def create_audio_mask(self, batch_size: int | None = None):
        # clips are right-padded; at least one uses every frame
        lengths = ids_tensor([self.batch_size], vocab_size=self.feat_seq_length).abs() + 1
        lengths[0] = self.feat_seq_length
        positions = torch.arange(self.feat_seq_length, device=torch_device)[None, :]
        return (positions < lengths[:, None]).long()

    def _subsampled_length(self, length):
        for _ in range(self.subsampling_factor.bit_length() - 1):
            length = (length - 1) // 2 + 1
        return length

    def get_audio_embeds_mask(self, audio_mask):
        # mirrors `get_audio_features`: each clip keeps `subsample(num_frames + 1)` tokens, capped by the padded length
        lengths = self._subsampled_length(audio_mask.sum(-1) + 1).clamp(
            max=self._subsampled_length(self.feat_seq_length)
        )
        positions = torch.arange(int(lengths.max()), device=audio_mask.device)[None, :]
        return (positions < lengths[:, None]).long()


class NemotronHOmniModelTestMixin:
    """Overrides shared by both test classes: the hybrid Mamba/attention language model, and `get_*_features`
    inputs for every modality so both classes exercise all three feature paths."""

    # each class feeds a single modality, so the other towers (and the video projection) get no gradient
    test_all_params_have_gradient = False
    # packed image patches have no batch dimension, and the video path packs `video_temporal_patch_size` frames
    # into one tower pass, so neither output keeps the input's leading dimension
    skip_test_image_features_output_shape = True
    skip_test_video_features_output_shape = True

    _get_conv_state_shape = test_modeling_nemotron_h.NemotronHModelTest._get_conv_state_shape
    _get_recurrent_state_shape = test_modeling_nemotron_h.NemotronHModelTest._get_recurrent_state_shape
    _check_past_key_values_for_generate = (
        test_modeling_nemotron_h.NemotronHModelTest._check_past_key_values_for_generate
    )

    def _image_features_prepare_config_and_inputs(self):
        tester = self.model_tester
        grid_size = tester.image_size // tester.patch_size
        pixel_values = floats_tensor([tester.batch_size * grid_size**2, 3 * tester.patch_size**2])
        image_grid_hw = torch.tensor([[grid_size, grid_size]] * tester.batch_size, device=torch_device)
        return tester.get_config(), {"pixel_values": pixel_values, "image_grid_hw": image_grid_hw}

    def _video_features_prepare_config_and_inputs(self):
        tester = self.model_tester
        pixel_values_videos = floats_tensor([tester.batch_size, 3, tester.image_size, tester.image_size])
        return tester.get_config(), {"pixel_values_videos": pixel_values_videos}

    def _audio_features_prepare_config_and_inputs(self):
        tester = self.model_tester
        input_features = floats_tensor([tester.batch_size, tester.feat_seq_length, tester.num_mel_bins])
        input_features_mask = torch.ones(
            tester.batch_size, tester.feat_seq_length, dtype=torch.long, device=torch_device
        )
        return tester.get_config(), {"input_features": input_features, "input_features_mask": input_features_mask}

    def _image_features_get_expected_num_attentions(self, model_tester=None):
        return TINY_VISION_LAYERS

    def _video_features_get_expected_num_attentions(self, model_tester=None):
        return TINY_VISION_LAYERS

    def _audio_features_get_expected_num_attentions(self, model_tester=None):
        return TINY_AUDIO_LAYERS

    def test_attention_outputs(self):
        # only the language model's attention layers return attention maps; the mamba and moe layers do not
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        text_config = config.get_text_config()
        num_attention_layers = text_config.layers_block_type.count("full_attention")
        seq_length = inputs_dict["input_ids"].shape[1]

        for model_class in self.all_model_classes:
            model = model_class._from_config(config, attn_implementation="eager").to(torch_device).eval()
            with torch.no_grad():
                outputs = model(**inputs_dict, output_attentions=True)
            self.assertEqual(len(outputs.attentions), num_attention_layers)
            self.assertListEqual(
                list(outputs.attentions[0].shape[-3:]), [text_config.num_attention_heads, seq_length, seq_length]
            )

    def test_keep_in_fp32_modules(self):
        # Same checks as the common test, except that integer tensors such as RADIO's `summary_idxs` index buffer
        # keep their dtype: only floating-point weights are cast to the requested dtype.
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            fp32_modules = model._keep_in_fp32_modules | model._keep_in_fp32_modules_strict
            if not fp32_modules:
                self.skipTest(reason=f"{model_class.__name__} has no `_keep_in_fp32_modules(_strict)`")
            original_dtypes = {name: tensor.dtype for name, tensor in model.state_dict().items()}

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                # fp16 upcasts both lists, bf16 only the strict one
                for dtype, upcast_modules in (
                    (torch.float16, fp32_modules),
                    (torch.bfloat16, model._keep_in_fp32_modules_strict),
                ):
                    reloaded = model_class.from_pretrained(tmpdirname, dtype=dtype)
                    for name, tensor in reloaded.state_dict().items():
                        if not tensor.is_floating_point():
                            self.assertEqual(tensor.dtype, original_dtypes[name], f"{name} changed dtype")
                        elif any(re.search(rf"(?:^|\.){module}(?:\.|$)", name) for module in upcast_modules):
                            self.assertEqual(tensor.dtype, torch.float32, f"{name} not upcasted to fp32")
                        else:
                            self.assertEqual(tensor.dtype, dtype, f"{name} was upcasted but it should NOT be")

    @unittest.skip(reason="NemotronH needs at least 3 layers to test (mamba, moe, attention)")
    def test_num_layers_is_small(self):
        pass

    @unittest.skip(reason="The model is a single generative class; there is no separate base model to expose")
    def test_model_base_model_prefix(self):
        pass

    @unittest.skip(
        reason="`_config_zero_init` sets every `*_std` config field to a scalar, which RADIO's strict `norm_std` "
        "(a per-channel list) rejects"
    )
    def test_can_load_ignoring_mismatched_shapes(self):
        pass


@require_torch
class NemotronHOmniVision2TextModelTest(NemotronHOmniModelTestMixin, VLMModelTest, unittest.TestCase):
    model_tester_class = NemotronHOmniVision2TextModelTester
    test_torch_exportable = False  # packed image patches use data-dependent shapes in RadioModel._forward_packed

    def test_mismatching_num_image_tokens(self):
        # packed `pixel_values` have no batch dimension, so images are dropped or added by their patches
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        patches_per_image = int(input_dict["image_grid_hw"][0].prod())
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device).eval()
            _ = model(**input_dict)

            # one image fewer than the image tokens in the text
            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][patches_per_image:]
            curr_input_dict["image_grid_hw"] = curr_input_dict["image_grid_hw"][1:]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # patches that do not add up to the image grids
            curr_input_dict = copy.deepcopy(input_dict)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][:-1]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # two prompts with image tokens but a single image
            curr_input_dict = {
                "input_ids": torch.cat([input_dict["input_ids"][:1]] * 2),
                "attention_mask": torch.cat([input_dict["attention_mask"][:1]] * 2),
                "pixel_values": input_dict["pixel_values"][:patches_per_image],
                "image_grid_hw": input_dict["image_grid_hw"][:1],
            }
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # two prompts with two images
            curr_input_dict["pixel_values"] = torch.cat([curr_input_dict["pixel_values"]] * 2)
            curr_input_dict["image_grid_hw"] = torch.cat([curr_input_dict["image_grid_hw"]] * 2)
            _ = model(**curr_input_dict)

    def test_mtp_checkpoints_normalize_vision_features(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = NemotronH_Omni_Reasoning_V3(config).to(torch_device).eval()
        self.assertIsNone(model.vision_final_layernorm)

        config.text_config.num_nextn_predict_layers = 1
        mtp_model = NemotronH_Omni_Reasoning_V3(config).to(torch_device).eval()
        self.assertIsInstance(mtp_model.vision_final_layernorm, torch.nn.LayerNorm)
        mtp_model.load_state_dict(model.state_dict(), strict=False)

        pixel_values = inputs_dict["pixel_values"].to(torch_device)
        image_grid_hw = inputs_dict["image_grid_hw"].to(torch_device)
        with torch.no_grad():
            features = model.vision_model(pixel_values, image_grid_hw=image_grid_hw).features
            expected = model.get_image_features(pixel_values, image_grid_hw).pooler_output
            normalized = mtp_model.get_image_features(pixel_values, image_grid_hw).pooler_output
            # the norm is the only difference, so feeding pre-normalized features reproduces its output
            normalized_features = torch.nn.functional.layer_norm(
                features, (features.shape[-1],), eps=config.vision_config.layer_norm_eps
            )
            manual = mtp_model.multi_modal_projector(
                torch.cat(
                    [
                        mtp_model.pixel_shuffle(f.view(1, h, w, -1), scale_factor=config.downsample_ratio).flatten(
                            0, 2
                        )
                        for f, (h, w) in zip(
                            normalized_features.split(image_grid_hw.prod(-1).tolist()), image_grid_hw.tolist()
                        )
                    ]
                )
            )

        self.assertFalse(torch.allclose(expected, normalized))
        torch.testing.assert_close(normalized, manual)


@require_torch
class NemotronHOmniAudio2TextModelTest(NemotronHOmniModelTestMixin, ALMModelTest, unittest.TestCase):
    model_tester_class = NemotronHOmniAudio2TextModelTester


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
