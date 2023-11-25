# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
""" Testing suite for the PyTorch SEGGPT model. """

import inspect
import unittest

import numpy as np
import requests
from PIL import Image
from datasets import load_dataset
from packaging import version

from transformers.models.auto import get_values
from transformers.models.seggpt import SegGPTConfig
from transformers.testing_utils import require_torch, require_torch_multi_gpu, require_vision, slow, torch_device
from transformers.utils import cached_property, is_torch_available, is_vision_available

from ...test_configuration_common import ConfigTester
from ...test_modeling_common import ModelTesterMixin, _config_zero_init, floats_tensor, ids_tensor
from ...test_pipeline_mixin import PipelineTesterMixin
import torch.nn.functional as F

if is_torch_available():
    import torch
    from torch import nn

    from transformers import (
        MODEL_MAPPING,
        SegGPTModel, SegGPTImageProcessor
    )
    from transformers.models.seggpt.modeling_seggpt import SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST, \
    SegGPTForInstanceSegmentation, SegGPTForSemanticSegmentation


# if is_vision_available():
#     import PIL
#     from PIL import Image
#
#     from transformers import BeitImageProcessor


class SegGPTModelTester:
    def __init__(
            self,
            parent,
            num_channels=3,
            image_size=[896, 448],
            patch_size=16,
            embed_dim=1024,
            num_heads=16,
            drop_path_rate=0.1,
            window_size=14,
            qkv_bias=True,
            mlp_ratio=4.,
            layer_norm_eps=1e-6,
            num_group_blocks=4,
            num_blocks_in_group=6,
            use_rel_pos=True,
            out_feature="last_feat",
            decoder_embed_dim=64,
            pretrain_img_size=224,
            num_labels=2,
            type_sequence_label_size=10,
            is_training=True,
            use_labels=True,
            batch_size=1,
    ):
        self.parent = parent
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_attention_heads = num_heads
        self.drop_path_rate = drop_path_rate
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.layer_norm_eps = layer_norm_eps
        self.num_group_blocks = num_group_blocks
        self.num_blocks_in_group = num_blocks_in_group
        self.use_rel_pos = use_rel_pos
        self.out_feature = out_feature
        self.decoder_embed_dim = decoder_embed_dim
        self.pretrain_img_size = pretrain_img_size
        self.batch_size = 1
        self.num_labels = num_labels
        self.use_labels = use_labels
        self.is_training = is_training
        self.batch_size = batch_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_hidden_layers = self.num_group_blocks * self.num_blocks_in_group

    def prepare_config_and_inputs(self):
        pixel_values = floats_tensor([2 * self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])
        prompts = floats_tensor([2 * self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        pixel_labels = None
        if self.use_labels:
            labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            pixel_labels = ids_tensor([self.batch_size, self.image_size[0], self.image_size[1]], self.num_labels)

        config = self.get_config()

        return config, pixel_values, prompts, labels, pixel_labels

    def get_config(self):
        return SegGPTConfig(
            # num_channels=self.num_channels,
            # image_size=self.image_size,
            # patch_size=self.patch_size,
            # embed_dim=self.embed_dim,
            # num_heads=self.num_attention_heads,
            # drop_path_rate=self.drop_path_rate,
            # window_size=self.window_size,
            # qkv_bias=self.qkv_bias,
            # mlp_ratio=self.mlp_ratio,
            # layer_norm_eps=self.layer_norm_eps,
            # num_group_blocks=self.num_group_blocks,
            # num_blocks_in_group=self.num_blocks_in_group,
            # use_rel_pos=self.use_rel_pos,
            # out_feature=self.out_feature,
            # decoder_embed_dim=self.decoder_embed_dim,
            # pretrain_img_size=self.pretrain_img_size,
        )

    def create_and_check_for_semantic_segmentation(self, config, pixel_values, prompts, labels, pixel_labels):
        config.num_labels = self.num_labels
        model = SegGPTModel(config)
        model.to(torch_device)
        model.eval()
        result = model(pixel_values, prompts)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )
        result = model(pixel_values, labels=pixel_labels)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.num_labels, self.image_size * 2, self.image_size * 2)
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, prompts, labels, pixel_labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "prompt_pixel_values": prompts}
        return config, inputs_dict


@require_torch
class BeitModelTest(ModelTesterMixin, PipelineTesterMixin, unittest.TestCase):
    """
    Here we also overwrite some of the tests of test_modeling_common.py, as BEiT does not use input_ids, inputs_embeds,
    attention_mask and seq_length.
    """

    all_model_classes = (
        (SegGPTForInstanceSegmentation,SegGPTForSemanticSegmentation)
        if is_torch_available()
        else ()
    )
    pipeline_model_mapping = ()
    #     {
    #         "feature-extraction": BeitModel,
    #         "image-classification": BeitForImageClassification,
    #         "image-segmentation": BeitForSemanticSegmentation,
    #     }
    #     if is_torch_available()
    #     else {}
    # )

    test_pruning = False
    test_resize_embeddings = False
    test_head_masking = False

    def setUp(self):
        self.model_tester = SegGPTModelTester(self)
        self.config_tester = ConfigTester(self, config_class=SegGPTConfig, has_text_modality=False, hidden_size=37)

    def test_config(self):
        self.config_tester.run_common_tests()

    @unittest.skip(reason="SEGGPT does not use inputs_embeds")
    def test_inputs_embeds(self):
        pass

    @require_torch_multi_gpu
    @unittest.skip(reason="SEGGPT has some layers using `add_module` which doesn't work well with `nn.DataParallel`")
    def test_multi_gpu_data_parallel_forward(self):
        pass

    def test_model_common_attributes(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            self.assertIsInstance(model.get_input_embeddings(), (nn.Module))
            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_forward_signature(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()

        for model_class in self.all_model_classes:
            model = model_class(config)
            signature = inspect.signature(model.forward)
            # signature.parameters is an OrderedDict => so arg_names order is deterministic
            arg_names = [*signature.parameters.keys()]

            expected_arg_names = ["pixel_values"]
            self.assertListEqual(arg_names[:1], expected_arg_names)

    def test_for_semantic_segmentation(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_for_semantic_segmentation(*config_and_inputs)

    # @slow
    # def test_model_from_pretrained(self):
    #     for model_name in BEIT_PRETRAINED_MODEL_ARCHIVE_LIST[:1]:
    #         model = BeitModel.from_pretrained(model_name)
    #         self.assertIsNotNone(model)


# We will verify our results on an image of cute cats
# def prepare_img():
#     image = Image.open("./tests/fixtures/tests_samples/COCO/000000039769.png")
#     return image


@require_torch
@require_vision
class BeitModelIntegrationTest(unittest.TestCase):
    # @cached_property
    # def default_image_processor(self):
    #     return SegGPTForSemanticSegmentation.from_pretrained("microsoft/beit-base-patch16-224") if is_vision_available() else None

    @slow
    def test_post_processing_semantic_segmentation_original(self):
        model = SegGPTModel.from_pretrained("Raghavan/seggpt_semantic_segmentation")
        model = model.to(torch_device)

        def inference_image(model, device, img_path, img2_paths, tgt2_paths, out_path):
            res, hres = 448, 448
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])

            image = Image.open(img_path).convert("RGB")
            input_image = np.array(image)
            size = image.size
            image = np.array(image.resize((res, hres))) / 255.

            image_batch, target_batch = [], []
            for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
                img2 = Image.open(img2_path).convert("RGB")
                img2 = img2.resize((res, hres))
                img2 = np.array(img2) / 255.

                tgt2 = Image.open(tgt2_path).convert("RGB")
                tgt2 = tgt2.resize((res, hres), Image.NEAREST)
                tgt2 = np.array(tgt2) / 255.

                tgt = tgt2  # tgt is not available
                tgt = np.concatenate((tgt2, tgt), axis=0)
                img = np.concatenate((img2, image), axis=0)

                assert img.shape == (2 * res, res, 3), f'{img.shape}'
                # normalize by ImageNet mean and std
                img = img - imagenet_mean
                img = img / imagenet_std

                assert tgt.shape == (2 * res, res, 3), f'{img.shape}'
                # normalize by ImageNet mean and std
                tgt = tgt - imagenet_mean
                tgt = tgt / imagenet_std

                image_batch.append(img)
                target_batch.append(tgt)

            img = np.stack(image_batch, axis=0)
            tgt = np.stack(target_batch, axis=0)
            """### Run SegGPT on the image"""
            # make random mask reproducible (comment out to make it change)
            torch.manual_seed(2)
            output = run_one_image(img, tgt, model, device)
            output = F.interpolate(
                output[None, ...].permute(0, 3, 1, 2),
                size=[size[1], size[0]],
                mode='nearest',
            ).permute(0, 2, 3, 1)[0].numpy()
            output = Image.fromarray((input_image * (0.6 * output / 255 + 0.4)).astype(np.uint8))
            output.save(out_path)

        def run_one_image(img, tgt, model, device):
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            x = torch.tensor(img)
            # make it a batch-like
            x = torch.einsum('nhwc->nchw', x)

            tgt = torch.tensor(tgt)
            # make it a batch-like
            tgt = torch.einsum('nhwc->nchw', tgt)

            _, y, mask = model(x.float().to(device), tgt.float().to(device))
            y = model.unpatchify(y)
            y = torch.einsum('nchw->nhwc', y).detach().cpu()

            output = y[0, y.shape[1] // 2:, :, :]
            output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
            return output

        image = Image.open("/Users/eaxxkra/Projects/myplay/Painter/SegGPT/SegGPT_inference/examples/hmbb_3.jpg")

        prompts = ['/Users/eaxxkra/Projects/myplay/Painter/SegGPT/SegGPT_inference/examples/hmbb_1.jpg',
                   '/Users/eaxxkra/Projects/myplay/Painter/SegGPT/SegGPT_inference/examples/hmbb_2.jpg']
        targets = ['/Users/eaxxkra/Projects/myplay/Painter/SegGPT/SegGPT_inference/examples/hmbb_1_target.png',
                   '/Users/eaxxkra/Projects/myplay/Painter/SegGPT/SegGPT_inference/examples/hmbb_2_target.png']

        prompt_images = [Image.open(a) for a in prompts]
        target_images = [Image.open(a) for a in targets]

        processor = SegGPTImageProcessor(size={"shortest_edge": 448})
        # batch = processor.pre_process_semantic_segmenation(image,prompt_images,target_images,return_tensors='pt')

        inference_image(model, 'cpu',
                        "/Users/eaxxkra/Projects/myplay/Painter/SegGPT/SegGPT_inference/examples/hmbb_3.jpg",
                        prompts, targets, './output_hmbb_3.png')

    # @slow
    def test_post_processing_semantic_segmentation(self):

        prompts = ['https://huggingface.co/datasets/Raghavan/seggpt_samples/resolve/main/hmbb_1.jpg',
                   'https://huggingface.co/datasets/Raghavan/seggpt_samples/resolve/main/hmbb_2.jpg']
        targets = ['https://huggingface.co/datasets/Raghavan/seggpt_samples/resolve/main/hmbb_1_target.png',
                   'https://huggingface.co/datasets/Raghavan/seggpt_samples/resolve/main/hmbb_2_target.png']

        def prepare_img(image_url):
            im = Image.open(requests.get(image_url, stream=True).raw)
            return im

        image = prepare_img("https://huggingface.co/datasets/Raghavan/seggpt_samples/resolve/main/hmbb_3.jpg")

        prompt_images = [prepare_img(a) for a in prompts]
        target_images = [prepare_img(a) for a in targets]

        processor = SegGPTImageProcessor(size={"shortest_edge": 448})
        inputs = processor.pre_process_semantic_segmenation(image, prompt_images, target_images, return_tensors='pt')

        model = SegGPTForInstanceSegmentation.from_pretrained("Raghavan/seggpt_semantic_segmentation")

        loss, output,_ = model(**inputs)

        self.assertEqual(0.1255, round(float(loss.detach().numpy()), 4))
        np.array_equal([-2.1193, -2.0402, -1.8096, -2.1195], output.detach().numpy()[1, 2, 6:10])