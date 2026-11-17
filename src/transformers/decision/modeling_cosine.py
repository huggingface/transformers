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
"""A reusable sparse span readout for dynamic decision options."""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from ..utils import cached_file


class SparseCosineDecisionHead(nn.Module):
    """Score dynamic options against their questions using projected span means.

    The caller supplies the backbone's final hidden states and CPU-native span metadata.
    Only tokens inside those spans are gathered for pooling. This avoids a full-sequence
    prefix-sum buffer; the backbone and its dense gradients still scale with input length.

    Args:
        hidden_size (`int`):
            Backbone hidden dimension.
        width (`int`, *optional*, defaults to 512):
            Dimension of the independent question and option projections.
        initial_scale (`float`, *optional*, defaults to 12.0):
            Initial positive multiplier of cosine similarities; the learned multiplier is capped at 100.
        head_type (`str`, *optional*, defaults to `"cosine_span_v1"`):
            Checkpoint format identifier, shared with the notebook's dense-pooling reference.

    This module is a head, not a complete `pipeline("decision")` model. Save the encoder
    and tokenizer separately. Exclude `logit_scale` from optimizer weight decay.
    """

    def __init__(
        self, hidden_size: int, width: int = 512, initial_scale: float = 12.0, head_type: str = "cosine_span_v1"
    ):
        super().__init__()
        if not isinstance(hidden_size, int) or not isinstance(width, int) or hidden_size < 1 or width < 1:
            raise ValueError("hidden_size and width must be positive integers")
        if not math.isfinite(initial_scale) or not 0 < initial_scale <= 100.0:
            raise ValueError("initial_scale must be finite and satisfy 0 < initial_scale <= 100")
        if head_type != "cosine_span_v1":
            raise ValueError(f"Unsupported decision head type: {head_type}")
        self.config = {
            "hidden_size": hidden_size,
            "width": width,
            "initial_scale": initial_scale,
            "head_type": head_type,
        }
        self.question_projection = nn.Linear(hidden_size, width)
        self.question_norm = nn.LayerNorm(width)
        self.option_projection = nn.Linear(hidden_size, width)
        self.option_norm = nn.LayerNorm(width)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_scale), dtype=torch.float32))
        self._max_log_scale = math.log(100.0)

    @staticmethod
    def _indices(values, device):
        indices = torch.tensor(values, dtype=torch.long)
        if device.type == "cuda":
            indices = indices.pin_memory()
        return indices.to(device, non_blocking=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        question_spans: Sequence[tuple[int, int, int]],
        option_spans: Sequence[tuple[int, int, int]],
        option_question_indices: Sequence[int],
    ) -> torch.Tensor:
        """Return one fp32 logit per option, in `option_spans` order.

        Args:
            hidden_states (`torch.Tensor` of shape `(batch, sequence_length, hidden_size)`):
                Final normalized backbone representations.
            question_spans (`Sequence[tuple[int, int, int]]`):
                CPU-native `(batch_index, start, end)` triples, with exclusive ends.
            option_spans (`Sequence[tuple[int, int, int]]`):
                CPU-native triples for the candidate option spans.
            option_question_indices (`Sequence[int]`):
                For each option, the index of its owner in `question_spans`.

        Options are dynamic and need not have single-token labels. Represent a boolean
        question with two option spans, and score questions with ordered option spans.
        The caller groups logits by question and applies softmax or a supervised loss.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.config["hidden_size"]:
            raise ValueError("hidden_states must have shape (batch, sequence_length, hidden_size)")
        if any(torch.is_tensor(x) for x in (question_spans, option_spans, option_question_indices)):
            raise TypeError("Span metadata must be CPU-native sequences, not tensors")
        if not question_spans or not option_spans or len(option_spans) != len(option_question_indices):
            raise ValueError("Nonempty question/option spans and one owner per option are required")
        batch, length, dimension = hidden_states.shape
        positions, sizes = [], []
        questions = list(question_spans)
        options = list(option_spans)
        for span in questions + options:
            if len(span) != 3 or any(not isinstance(x, int) for x in span):
                raise ValueError("Each span must contain three integer indices")
            row, start, end = span
            if not 0 <= row < batch or not 0 <= start < end <= length:
                raise ValueError(f"Span is outside hidden_states or empty: {span}")
            positions.extend(range(row * length + start, row * length + end))
            sizes.append(end - start)
        for span, owner in zip(options, option_question_indices):
            if not isinstance(owner, int) or not 0 <= owner < len(questions):
                raise ValueError("Option owner is not a valid question index")
            if span[0] != questions[owner][0]:
                raise ValueError("An option and its question must belong to the same batch record")
        indices = self._indices(positions, hidden_states.device)
        lengths = self._indices(sizes, hidden_states.device)
        owners = self._indices(option_question_indices, hidden_states.device)
        selected = hidden_states.reshape(-1, dimension).index_select(0, indices)
        if selected.dtype != torch.float64:
            selected = selected.float()
        # CPU metadata was checked above; unsafe avoids a redundant GPU validation sync.
        pooled = torch.segment_reduce(selected, reduce="mean", lengths=lengths, unsafe=True)
        pooled = pooled.to(self.question_projection.weight.dtype)
        count = len(questions)
        query = self.question_norm(self.question_projection(pooled[:count]))
        option = self.option_norm(self.option_projection(pooled[count:]))
        query = F.normalize(query.float(), dim=-1)
        option = F.normalize(option.float(), dim=-1)
        scale = self.logit_scale.float().clamp(max=self._max_log_scale).exp()
        return (query[owners] * option).sum(-1) * scale

    def save_pretrained(self, save_directory: str | Path):
        """Save head weights/config; an encoder and tokenizer are not included."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        weights = {name: value.detach().cpu().contiguous() for name, value in self.state_dict().items()}
        save_file(weights, str(path / "joint_head.safetensors"))
        (path / "joint_head_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **hub_kwargs):
        """Load a local or Hub head checkpoint into fp32 parameters on CPU.

        Hub options such as `revision`, `token`, `cache_dir` and `local_files_only` are
        forwarded to `cached_file`. Move/cast the returned head with `head.to(...)`.
        """
        config_file = cached_file(pretrained_model_name_or_path, "joint_head_config.json", **hub_kwargs)
        weights_file = cached_file(pretrained_model_name_or_path, "joint_head.safetensors", **hub_kwargs)
        if config_file is None or weights_file is None:
            raise OSError("Both joint_head_config.json and joint_head.safetensors are required")
        config = json.loads(Path(config_file).read_text(encoding="utf-8"))
        if config.get("head_type") != "cosine_span_v1":
            raise ValueError("The checkpoint does not contain a cosine_span_v1 head")
        with torch.device("cpu"):
            head = cls(**config).float()
        head.load_state_dict(load_file(weights_file), strict=True)
        return head


__all__ = ["SparseCosineDecisionHead"]
