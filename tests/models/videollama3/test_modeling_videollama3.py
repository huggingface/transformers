"""Testing suite for the PyTorch VideoLLaMA3 model."""

import copy
import gc
import tempfile
import unittest

import requests

from transformers import (
    AutoProcessor,
    Videollama3Config,
    Videollama3ForConditionalGeneration,
    Videollama3Model,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import (
    Expectations,
    backend_empty_cache,
    require_flash_attn,
    require_torch,
    require_torch_gpu,
    slow,
    torch_device,
)

from ...generation.test_utils import GenerationTesterMixin
from ...test_configuration_common import ConfigTester
from ...test_modeling_common import (
    ModelTesterMixin,
    _config_zero_init,
    floats_tensor,
    ids_tensor,
)


if is_torch_available():
    import torch


if is_vision_available():
    from PIL import Image


class Videollama3VisionText2TextModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        seq_length=7,
        num_channels=3,
        image_size=14,
        is_training=True,
        text_config={
            "attention_dropout": 0.0,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 32,
            "intermediate_size": 37,
            "max_position_embeddings": 512,
            "max_window_layers": 3,
            "model_type": "qwen2",
            "num_attention_heads": 4,
            "num_hidden_layers": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_scaling": None,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "vocab_size": 99
        },
        vision_config={
            "attention_dropout": 0.0,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 32,
            "intermediate_size": 64,
            "layer_norm_eps": 1e-06,
            "model_type": "videollama3_vision",
            "num_attention_heads": 4,
            "num_channels": 3,
            "num_hidden_layers": 2,
            "patch_size": 14,
        },
        use_token_compression=True,
        image_token_id=3,
        video_token_id=4,
    ):
        self.parent = parent
        self.hidden_size = text_config["hidden_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.patch_size = vision_config["patch_size"]
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.text_config = text_config
        self.vision_config = vision_config
        self.use_token_compression = use_token_compression
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.num_image_tokens = 32
        self.seq_length = seq_length + self.num_image_tokens

    def get_config(self):
        return Videollama3Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            use_token_compression=self.use_token_compression,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()
        patch_size = config.vision_config.patch_size
        pixel_values = floats_tensor(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2),
            ]
        )

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_tensor([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=torch_device)

        input_ids[:, -1] = config.text_config.pad_token_id
        attention_mask[:, -1] = 0
        input_ids[input_ids == self.video_token_id] = config.text_config.pad_token_id
        input_ids[input_ids == self.image_token_id] = config.text_config.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "grid_sizes": torch.tensor([[1, 1, 1]] * self.batch_size, device=torch_device),
            "merge_sizes": torch.tensor([1] * self.batch_size, device=torch_device),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


@require_torch
class Videollama3ModelTest(ModelTesterMixin, GenerationTesterMixin, unittest.TestCase):
    """
    Model tester for `Videollama3ForConditionalGeneration`.
    """

    all_model_classes = (
        (
            Videollama3Model,
            Videollama3ForConditionalGeneration,
        )
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = {"image-text-to-text": Videollama3ForConditionalGeneration}
    test_pruning = False
    test_head_masking = False
    _is_composite = True

    def setUp(self):
        self.model_tester = Videollama3VisionText2TextModelTester(self)
        self.config_tester = ConfigTester(self, config_class=Videollama3Config, has_text_modality=False)

    def _prepare_for_class(self, inputs_dict, model_class, return_labels=False):
        inputs_dict = super()._prepare_for_class(inputs_dict, model_class, return_labels=return_labels)
        if "pixel_values" not in inputs_dict:
            input_ids = inputs_dict["input_ids"]
            num_image_tokens = torch.sum(input_ids == self.model_tester.image_token_id)
            num_video_tokens = torch.sum(input_ids == self.model_tester.video_token_id)
            patch_size = self.model_tester.patch_size
            num_pixels = self.model_tester.num_channels * patch_size ** 2
            if num_image_tokens > 0:
                inputs_dict.update(
                    {
                        "pixel_values": floats_tensor([num_image_tokens, num_pixels]),
                        "grid_sizes": torch.tensor([[1, num_image_tokens, 1]], device=input_ids.device),
                        "merge_sizes": torch.tensor([[1]], device=input_ids.device),
                    }
                )
            if num_video_tokens > 0:
                inputs_dict.update(
                    {
                        "pixel_values_videos": floats_tensor([num_video_tokens, num_pixels]),
                        "grid_sizes_videos": torch.tensor([[1, num_video_tokens, 1]], device=input_ids.device),
                        "merge_sizes_videos": torch.tensor([[1]], device=input_ids.device),
                    }
                )
        return inputs_dict

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
        Also we need to test multi-image cases when one prompt has multiple image tokens.
        """
        config, input_dict = self.model_tester.prepare_config_and_inputs_for_common()
        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)
            curr_input_dict = copy.deepcopy(input_dict)
            _ = model(**curr_input_dict)  # successfull forward with no modifications

            # remove one image but leave the image token in text
            patch_size = config.vision_config.patch_size
            one_img_length = (self.model_tester.image_size**2) // (patch_size**2)
            curr_input_dict["pixel_values"] = curr_input_dict["pixel_values"][-one_img_length:, ...]
            curr_input_dict["grid_sizes"] = curr_input_dict["grid_sizes"][-1:, ...]
            curr_input_dict["merge_sizes"] = curr_input_dict["merge_sizes"][-1:]
            with self.assertRaises(ValueError):
                _ = model(**curr_input_dict)

            # simulate multi-image case by concatenating inputs where each has exactly one image/image-token
            input_ids = curr_input_dict["input_ids"][:1]
            pixel_values = curr_input_dict["pixel_values"][:one_img_length]
            grid_sizes = curr_input_dict["grid_sizes"][:1]
            merge_sizes = curr_input_dict["merge_sizes"][:1]
            input_ids = torch.cat([input_ids, input_ids], dim=0)

            # one image and two image tokens raise an error
            with self.assertRaises(ValueError):
                _ = model(input_ids=input_ids, pixel_values=pixel_values, grid_sizes=grid_sizes, merge_sizes=merge_sizes)

            # two images and two image tokens don't raise an error
            pixel_values = torch.cat([pixel_values, pixel_values], dim=0)
            grid_sizes = torch.cat([grid_sizes, grid_sizes], dim=0)
            merge_sizes = torch.cat([merge_sizes, merge_sizes], dim=0)
            _ = model(input_ids=input_ids, pixel_values=pixel_values, grid_sizes=grid_sizes, merge_sizes=merge_sizes)

    def attention_mask_padding_matches_padding_free_with_position_ids(
        self, attn_implementation: str, fa_kwargs: bool = False
    ):
        max_new_tokens = 30
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

            dummy_input = inputs_dict[model_class.main_input_name]
            if dummy_input.dtype in [torch.float32, torch.float16]:
                dummy_input = dummy_input.to(torch.bfloat16)

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + dummy_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                if 0 in inputs_dict["attention_mask"][:, -1]:
                    inputs_dict["attention_mask"] = inputs_dict["attention_mask"].flip(1)
                dummy_attention_mask = inputs_dict["attention_mask"]
                inputs_dict["input_ids"][~dummy_attention_mask.bool()] = config.get_text_config().pad_token_id

                model = (
                    model_class.from_pretrained(
                        tmpdirname,
                        torch_dtype=torch.bfloat16,
                        attn_implementation=attn_implementation,
                    )
                    .to(torch_device)
                    .eval()
                )

                # flatten
                padfree_positions = torch.cat(
                    [torch.arange(length) for length in dummy_attention_mask.sum(1).tolist()]
                )
                padfree_positions = padfree_positions.long().unsqueeze(0).to(torch_device)
                padfree_inputs_dict = {
                    "pixel_values": inputs_dict["pixel_values"],
                    "grid_sizes": inputs_dict["grid_sizes"],
                    "merge_sizes": inputs_dict["merge_sizes"],
                    "input_ids": inputs_dict["input_ids"][dummy_attention_mask.bool()].unsqueeze(0),
                    "position_ids": padfree_positions,
                }

                if fa_kwargs:
                    cu_seq_lens = [0] + dummy_attention_mask.sum(1).tolist()
                    cu_seq_lens = torch.tensor(cu_seq_lens, device=torch_device)
                    max_length = cu_seq_lens.diff().max().item()
                    padfree_inputs_dict.update(
                        {
                            "cu_seq_lens_q": cu_seq_lens.cumsum(-1).to(dtype=torch.int32),
                            "cu_seq_lens_k": cu_seq_lens.cumsum(-1).to(dtype=torch.int32),
                            "max_length_q": max_length,
                            "max_length_k": max_length,
                        }
                    )

                # We need to do simple forward without cache in roder to trigger packed SDPA/FLEX/EAGER path
                res_padded = model(**inputs_dict, use_cache=False)
                res_padfree = model(**padfree_inputs_dict, use_cache=False)

                logits_padded = res_padded.logits[inputs_dict["attention_mask"].bool()]
                logits_padfree = res_padfree.logits[0]

                # acceptable numerical instability
                tol = torch.finfo(torch.bfloat16).eps
                torch.testing.assert_close(logits_padded, logits_padfree, rtol=tol, atol=tol)

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

    @unittest.skip(reason="Compile not yet supported because in Videollama3 models")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip(reason="Got `CUDA error: misaligned address` with PyTorch 2.0.0.")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    @unittest.skip(reason="We cannot configure to output a smaller model.")
    def test_model_is_small(self):
        pass
