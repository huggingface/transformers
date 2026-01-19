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

from ...configuration_utils import PretrainedConfig


class SnacConfig(PretrainedConfig):
    """
    This is the configuration class for SNAC models.

    It is intentionally minimal: it exists to allow Transformers' AutoConfig
    to recognize checkpoints with `model_type="snac"` (e.g. ONNX exports).
    """

    model_type = "snac"

    def __init__(
        self,
        sampling_rate: int = 24000,
        encoder_dim: int | None = None,
        decoder_dim: int | None = None,
        encoder_rates=None,
        decoder_rates=None,
        codebook_size: int | None = None,
        codebook_dim: int | None = None,
        vq_strides=None,
        attn_window_size: int | None = None,
        noise: bool | None = None,
        depthwise: bool | None = None,
        **kwargs,
    ):
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_rates = decoder_rates
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.noise = noise
        self.depthwise = depthwise
        super().__init__(**kwargs)
