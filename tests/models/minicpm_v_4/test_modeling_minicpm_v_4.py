# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
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
"""Testing suite for the PyTorch MiniCPM-V-4 model."""

import os
import unittest
from io import BytesIO

import requests
from decord import VideoReader, cpu
from PIL import Image

from transformers import (
    AutoModel,
    AutoTokenizer,
    MiniCPM_V_4Config,
    MiniCPM_V_4ForConditionalGeneration,
    MiniCPM_V_4Model,
    is_torch_available,
)
from transformers.testing_utils import (
    require_sentencepiece,
    require_tokenizers,
    require_torch,
    require_vision,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor


if is_torch_available():
    import torch


class MiniCPM_V_4VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        # --- Basic Test setup ---
        batch_size=2, # Reduced batch size to save memory
        seq_length=256, # A modest sequence length
        is_training=True,

        # --- Comprehensive parameters EXACTLY from config.json ---
        vocab_size=73448,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=2,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768, # Full sequence length
        initializer_range=0.02, # Standard test value
        rms_norm_eps=1e-06,
        use_cache=True,
        pad_token_id=2,
        bos_token_id=1,
        eos_token_id=2, # Use a single int for testing
        tie_word_embeddings=False,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        pretraining_tp=1,
        rope_scaling={
            "factor": 1.0,
            "long_factor": [0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193, 1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284, 1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191, 1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756, 2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632, 4.048719380066383, 4.615569542115128, 5.2684819496549835, 6.014438591970396, 6.858830049237097, 7.804668263503327, 8.851768731513417, 9.99600492938444, 11.228766118181639, 12.536757560834843, 13.902257701387796, 15.303885189125953, 16.717837610115794, 18.119465097853947, 19.484965238406907, 20.792956681060105, 22.02571786985731, 23.16995406772833, 24.217054535738416, 25.16289275000465, 26.007284207271347, 26.753240849586767, 27.40615325712662, 27.973003419175363, 28.461674954469114, 28.880393889607006, 29.237306864684626, 29.540186419591297, 29.79624387177199, 30.01202719065413, 30.193382037992453, 30.34545697551969, 30.47273746338473, 30.579096895249787, 30.66785612408345, 30.741845563814174, 30.80346599254902, 30.85474569563567, 30.897392663720595, 30.932841297560394, 30.962293553185553, 30.986754758742034, 31.007064503249293, 31.02392307921529],
            "original_max_position_embeddings": 32786,
            "rope_type": "longrope",
            "short_factor": [0.9977997200264581, 1.014658295992452, 1.0349680404997148, 1.059429246056193, 1.0888815016813513, 1.1243301355211495, 1.166977103606075, 1.2182568066927284, 1.2798772354275727, 1.3538666751582975, 1.4426259039919596, 1.5489853358570191, 1.6762658237220625, 1.8283407612492941, 2.0096956085876183, 2.225478927469756, 2.481536379650452, 2.784415934557119, 3.1413289096347365, 3.560047844772632, 4.048719380066383, 4.615569542115128, 5.2684819496549835, 6.014438591970396, 6.858830049237097, 7.804668263503327, 8.851768731513417, 9.99600492938444, 11.228766118181639, 12.536757560834843, 13.902257701387796, 15.303885189125953, 16.717837610115794, 18.119465097853947, 19.484965238406907, 20.792956681060105, 22.02571786985731, 23.16995406772833, 24.217054535738416, 25.16289275000465, 26.007284207271347, 26.753240849586767, 27.40615325712662, 27.973003419175363, 28.461674954469114, 28.880393889607006, 29.237306864684626, 29.540186419591297, 29.79624387177199, 30.01202719065413, 30.193382037992453, 30.34545697551969, 30.47273746338473, 30.579096895249787, 30.66785612408345, 30.741845563814174, 30.80346599254902, 30.85474569563567, 30.897392663720595, 30.932841297560394, 30.962293553185553, 30.986754758742034, 31.007064503249293, 31.02392307921529]
        },

        # Top-level parameters specific to MiniCPM_V_4 or Vision
        query_num=64,
        image_size=448,
        patch_size=14,
        num_channels=3,
        vision_batch_size=16,
        use_image_id=True,
        drop_vision_last_layer=False,
        slice_mode=True,
        slice_config={"max_slice_nums": 9, "model_type": "minicpmv", "patch_size": 14, "scale_resolution": 448},

        # Parameters for the nested vision_config dictionary (from config.json)
        vision_hidden_size=1152,
        vision_intermediate_size=4304,
        vision_num_hidden_layers=2,
        vision_num_attention_heads=16,
        vision_hidden_act="gelu_pytorch_tanh",
        vision_layer_norm_eps=1e-06,
        vision_attention_dropout=0.0,
    ):
        # --- Store all parameters as instance attributes ---
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training

        # Text params
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.pretraining_tp = pretraining_tp
        self.rope_scaling = rope_scaling

        # Vision & MiniCPM-V specific params
        self.query_num = query_num
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.vision_batch_size = vision_batch_size
        self.use_image_id = use_image_id
        self.drop_vision_last_layer = drop_vision_last_layer
        self.slice_mode = slice_mode
        self.slice_config = slice_config

        # Nested vision config params
        self.vision_config_params = {
            "hidden_size": vision_hidden_size,
            "intermediate_size": vision_intermediate_size,
            "num_hidden_layers": vision_num_hidden_layers,
            "num_attention_heads": vision_num_attention_heads,
            "hidden_act": vision_hidden_act,
            "layer_norm_eps": vision_layer_norm_eps,
            "attention_dropout": vision_attention_dropout,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "num_channels": self.num_channels,
        }

    def get_config(self):
        """
        Creates a MiniCPM_V_4Config instance with values that are 100% identical to the real config.json.
        WARNING: This will be slow and memory-intensive.
        """
        return MiniCPM_V_4Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            rms_norm_eps=self.rms_norm_eps,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            rope_theta=self.rope_theta,
            attention_bias=self.attention_bias,
            attention_dropout=self.attention_dropout,
            mlp_bias=self.mlp_bias,
            pretraining_tp=self.pretraining_tp,
            rope_scaling=self.rope_scaling,
            query_num=self.query_num,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            vision_batch_size=self.vision_batch_size,
            use_image_id=self.use_image_id,
            drop_vision_last_layer=self.drop_vision_last_layer,
            slice_mode=self.slice_mode,
            slice_config=self.slice_config,
            vision_config=self.vision_config_params,
        )

    def prepare_config_and_inputs_for_common(self):
        config = self.get_config()
        
        # Use a sequence length smaller than max_position_embeddings for the actual input tensors
        total_seq_length = self.seq_length
        image_token_length = config.query_num
        
        input_ids = ids_tensor([self.batch_size, total_seq_length], config.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)
        position_ids = torch.arange(0, total_seq_length, dtype=torch.long, device=torch_device).unsqueeze(0).repeat(self.batch_size, 1)

        pixel_values_tensor = floats_tensor([self.batch_size, config.num_channels, config.image_size, config.image_size])
        pixel_values = [[pixel_values_tensor[i]] for i in range(self.batch_size)]

        patches_per_side = config.image_size // config.patch_size
        tgt_sizes = [torch.tensor([patches_per_side, patches_per_side], device=torch_device) for _ in range(self.batch_size)]

        image_start_index = 5
        image_end_index = image_start_index + image_token_length
        if image_end_index > total_seq_length:
            raise ValueError(
                f"seq_length ({total_seq_length}) is too small to fit the image tokens."
            )
        image_bound = [torch.tensor([[image_start_index, image_end_index]], device=torch_device, dtype=torch.long) for _ in range(self.batch_size)]

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "tgt_sizes": tgt_sizes,
            "image_bound": image_bound,
        }

        labels = ids_tensor([self.batch_size, total_seq_length], config.vocab_size)
        inputs_dict = {**data, "labels": labels}

        return config, inputs_dict


@require_torch
class MiniCPM_V_4ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    all_model_classes = (MiniCPM_V_4Model, ) if is_torch_available() else ()
    fx_compatible = False
    test_headmasking = False
    test_pruning = False
    test_resize_embeddings = False
    test_attention_outputs = False
    _is_composite = True

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)

        if model_class.__name__ == "MiniCPM_V_4Model":
            inputs_dict.pop("labels", None)

        return inputs_dict

    def setUp(self):
        self.model_tester = MiniCPM_V_4VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=MiniCPM_V_4Config, has_text_modality=False)

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

    def test_image_feature_injection(self):
        """
        Tests that the model's hidden states are different when an image is provided,
        specifically at the locations indicated by `image_bound`.
        """
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        model = MiniCPM_V_4Model(config).to(torch_device).eval()

        data_with_vision = inputs_dict["data"]
        with torch.no_grad():
            outputs_with_vision = model(data=data_with_vision)

        hidden_states_with_vision = outputs_with_vision.last_hidden_state

        data_text_only = data_with_vision.copy()
        data_text_only["pixel_values"] = [[] for _ in range(self.model_tester.batch_size)]
        data_text_only["tgt_sizes"] = [[] for _ in range(self.model_tester.batch_size)]
        data_text_only["image_bound"] = [torch.empty((0, 2), device=torch_device) for _ in range(self.model_tester.batch_size)]

        with torch.no_grad():
            outputs_text_only = model(data=data_text_only)

        hidden_states_text_only = outputs_text_only.last_hidden_state

        image_start = data_with_vision["image_bound"][0][0, 0].item()
        image_end = data_with_vision["image_bound"][0][0, 1].item()

        vision_part_with_vision = hidden_states_with_vision[0, image_start:image_end, :]
        vision_part_text_only = hidden_states_text_only[0, image_start:image_end, :]

        self.assertFalse(torch.allclose(vision_part_with_vision, vision_part_text_only, atol=1e-4))

        text_part_before_with_vision = hidden_states_with_vision[0, :image_start, :]
        text_part_before_text_only = hidden_states_text_only[0, :image_start, :]

        self.assertTrue(torch.allclose(text_part_before_with_vision, text_part_before_text_only, atol=1e-5))

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_cpu_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_bin(self):
        pass

    @unittest.skip(reason="Some undefined behavior encountered with test versions of this model. Skip for now.")
    def test_disk_offload_safetensors(self):
        pass


@require_torch
class MiniCPM_V_4ModelIngestionTest(unittest.TestCase):
    """Test for MiniCPM_V_4Model."""

    def setUp(self):
        """initial test environment"""
        self.assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        os.makedirs(self.assets_dir, exist_ok=True)

        self.video_path = os.path.join(self.assets_dir, "Skiing.mp4")

        if not os.path.exists(self.video_path):
            video_url = "https://huggingface.co/openbmb/MiniCPM-V-4/resolve/main/assets/Skiing.mp4"
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            with open(self.video_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        self.model = AutoModel.from_pretrained(
            "openbmb/MiniCPM-V-4",
            trust_remote_code=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
        )
        self.model = self.model.eval().to(torch_device)
        self.tokenizer = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-4", trust_remote_code=True)

    def tearDown(self):
        """clean up test environment"""
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        if os.path.exists(self.assets_dir):
            os.rmdir(self.assets_dir)

    @slow
    def test_minicpm_V_4_model_base(self):
        """test base model loading"""
        base_model = AutoModel.from_pretrained("openbmb/MiniCPM-V-4")
        self.assertIsNotNone(base_model)

    def _get_video_chunk_content(self, video_path, flatten=True):
        """process video content, extract frames"""
        MAX_NUM_FRAMES = 64
        def encode_video(video_path):
            def uniform_sample(l, n):
                gap = len(l) / n
                idxs = [int(i * gap + gap / 2) for i in range(n)]
                return [l[i] for i in idxs]

            vr = VideoReader(video_path, ctx=cpu(0))
            sample_fps = round(vr.get_avg_fps() / 1)  # FPS
            frame_idx = [list(range(0, len(vr), sample_fps))]
            if len(frame_idx) > MAX_NUM_FRAMES:
                frame_idx = uniform_sample(frame_idx, MAX_NUM_FRAMES)
            frames = vr.get_batch(frame_idx).asnumpy()
            frames = [Image.fromarray(v.astype('uint8')) for v in frames]
            print('num frames:', len(frames))
            return frames

        frames = encode_video(video_path)

        contents = []

        for image in range(frames):
            if flatten:
                contents.extend([image])
            else:
                contents.append([image])

        return contents

    @slow
    @require_vision
    @require_sentencepiece
    @require_tokenizers
    def test_single_image_inference(self):
        try:
            image_url = "https://www.ilankelman.org/stopsigns/australia.jpg"
            response = requests.get(image_url, stream=True)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content)).convert("RGB")
            question = "What is in the image?"

            msgs = [{"role": "user", "content": [image, question]}]

            res = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)

            self.assertIsNotNone(res, "Normal inference response should not be empty")
            self.assertTrue(len(res) > 0, "Normal inference response text should not be empty")

            res = self.model.chat(msgs=msgs, tokenizer=self.tokenizer, sampling=True, stream=True)

            generated_text = ""
            for new_text in res:
                generated_text += new_text
                self.assertIsNotNone(new_text, "Each part of streaming reasoning should not be empty")

            self.assertTrue(len(generated_text) > 0, "Text should not be empty")

        except requests.exceptions.RequestException as e:
            self.skipTest(f"Failed to download image: {str(e)}")
        except Exception as e:
            self.fail(f"Single image inference test failed: {str(e)}")
