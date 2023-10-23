# coding=utf-8
# Copyright 2023 Xiaomi Corporation and The HuggingFace Inc. team. All rights reserved.
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
""" CED model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

CED_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "xiaomi/ced-tiny": "https://huggingface.co/xiaomi/ced-tiny/resolve/main/config.json",
}


class CedConfig(PretrainedConfig):
    r"""
    TODO: Add docstring
    """

    def __init__(
        self,
        name=None,
        attn_drop_rate=0.0,
        depth=12,
        drop_path_rate=0.0,
        drop_rate=0.0,
        embed_dim=768,
        eval_avg="mean",
        init_values=None,
        mlp_ratio=4.0,
        num_heads=12,
        outputdim=527,
        patch_size=16,
        patch_stride=16,
        pooling="mean",
        qkv_bias=True,
        target_length=1012,
        **kwargs,
    ):
        r"""
        TODO: Add docstring
        """

        super().__init__(**kwargs)

        if name == "ced-tiny":
            patch_size = 16
            embed_dim = 192
            depth = 12
            num_heads = 3
            mlp_ratio = 4
        elif name == "ced-mini":
            patch_size = 16
            embed_dim = 256
            depth = 12
            num_heads = 4
            mlp_ratio = 4
        elif name == "ced-small":
            patch_size = 16
            embed_dim = 384
            depth = 12
            num_heads = 6
            mlp_ratio = 4
        elif name == "ced-base":
            patch_size = 16
            embed_dim = 768
            depth = 12
            num_heads = 12
            mlp_ratio = 4
        else:
            logger.info("No model name specified for CedConfig, use default settings.")

        assert pooling in ("mean", "token", "dm", "logit")
        self.name = name
        self.attn_drop_rate = attn_drop_rate
        self.center = kwargs.get("center", True)
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.embed_dim = embed_dim
        self.eval_avg = eval_avg
        self.f_max = kwargs.get("f_max", 8000)
        self.f_min = kwargs.get("f_min", 0)
        self.hop_size = kwargs.get("hop_size", 160)
        self.init_values = init_values
        self.mlp_ratio = mlp_ratio
        self.n_fft = kwargs.get("n_fft", 512)
        self.n_mels = kwargs.get("n_mels", 64)
        self.n_mels = kwargs.get("n_mels", 64)
        self.num_heads = num_heads
        self.outputdim = outputdim
        self.pad_last = kwargs.get("pad_last", True)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.pooling = pooling
        self.qkv_bias = qkv_bias
        self.target_length = target_length
        self.win_size = kwargs.get("win_size", 512)
