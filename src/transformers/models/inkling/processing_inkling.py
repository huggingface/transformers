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


from ...processing_utils import ProcessingKwargs, ProcessorMixin
from ...utils import auto_docstring, is_torch_available, logging
from ...utils.import_utils import requires


if is_torch_available():
    import torch


logger = logging.get_logger(__name__)


class InklingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {}


@auto_docstring
@requires(backends=("torch",))
class InklingProcessor(ProcessorMixin):
    valid_processor_kwargs = InklingProcessorKwargs

    def __init__(
        self,
        feature_extractor=None,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<|unused_200054|>",
        audio_token="<|unused_200053|>",
        image_bos_token="<|content_image|>",
        audio_bos_token="<|content_audio_input|>",
        num_dmel_bins=16,
        dmel_min_value=-7.0,
        dmel_max_value=2.0,
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<|unused_200054|>"`):
            Placeholder token for each image soft-token slot (replaced by image features).
        audio_token (`str`, *optional*, defaults to `"<|unused_200053|>"`):
            Placeholder token for each audio soft-token slot (replaced by audio features).
        image_bos_token (`str`, *optional*, defaults to `"<|content_image|>"`):
            Marker token that begins an image span (kept as an ordinary embedded token).
        audio_bos_token (`str`, *optional*, defaults to `"<|content_audio_input|>"`):
            Marker token that begins an audio span (kept as an ordinary embedded token).
        num_dmel_bins (`int`, *optional*, defaults to 16):
            Number of discrete bins each (clamped) log-mel value is quantized into.
        dmel_min_value (`float`, *optional*, defaults to -7.0):
            Lower clamp bound, in log10 space, used for dMel quantization.
        dmel_max_value (`float`, *optional*, defaults to 2.0):
            Upper clamp bound, in log10 space, used for dMel quantization.
        """
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.image_token_id = tokenizer.encode(self.image_token, add_special_tokens=False)[0]
        self.audio_token = tokenizer.audio_token if hasattr(tokenizer, "audio_token") else audio_token
        self.audio_token_id = tokenizer.encode(self.audio_token, add_special_tokens=False)[0]
        self.image_bos_token = image_bos_token
        self.image_bos_token_id = tokenizer.encode(self.image_bos_token, add_special_tokens=False)[0]
        self.audio_bos_token = audio_bos_token
        self.audio_bos_token_id = tokenizer.encode(self.audio_bos_token, add_special_tokens=False)[0]

        # dMel
        self.num_dmel_bins = num_dmel_bins
        self.dmel_min_value = dmel_min_value
        self.dmel_max_value = dmel_max_value
        self.bin_centers = torch.linspace(dmel_min_value, dmel_max_value, num_dmel_bins, dtype=torch.float64)

        super().__init__(feature_extractor, image_processor, tokenizer, chat_template=chat_template)

    def _extract_dmel_bins(self, input_features: "torch.Tensor") -> "torch.Tensor":
        bin_centers = self.bin_centers.to(input_features.device)
        mel = input_features.to(torch.float64).clamp(min=self.dmel_min_value, max=self.dmel_max_value)
        return (mel.unsqueeze(-1) - bin_centers).abs().argmin(dim=-1).to(torch.int32)

    def _process_audio(self, audio, **kwargs):
        audio_inputs = self.feature_extractor(audio, **kwargs)

        processed_audio = {
            "audio_input_ids": self._extract_dmel_bins(audio_inputs["input_features"]),
            "audio_input_ids_mask": audio_inputs.get("input_features_mask"),
        }
        audio_replacements = [self.replace_audio_token(processed_audio, audio_idx=idx) for idx in range(len(audio))]
        return processed_audio, audio_replacements

    def replace_image_token(self, image_inputs: dict, image_idx: int) -> str:
        num_soft_tokens = image_inputs["num_patches"][image_idx]
        return self.image_token * num_soft_tokens

    def replace_audio_token(self, audio_inputs: dict, audio_idx: int) -> str:
        audio_input_ids_mask = audio_inputs.get("audio_input_ids_mask")

        if audio_input_ids_mask is not None:
            num_soft_tokens = int(audio_input_ids_mask[audio_idx].sum())
        else:
            num_soft_tokens = int(audio_inputs["audio_input_ids"][audio_idx].shape[-2])
        return self.audio_token * num_soft_tokens

    @property
    def unused_input_names(self) -> list[str]:
        return ["num_patches"]

    @property
    def model_input_names(self) -> list[str]:
        names = [
            "audio_input_ids",
            "audio_input_ids_mask",
            *self.image_processor.model_input_names,
            *self.tokenizer.model_input_names,
        ]
        return [name for name in dict.fromkeys(names) if name not in self.unused_input_names]


__all__ = ["InklingProcessor"]
