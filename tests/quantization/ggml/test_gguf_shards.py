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

from pathlib import Path

import numpy as np
import pytest

from transformers import AutoModelForCausalLM, GgufConfig
from transformers.testing_utils import require_accelerate


torch = pytest.importorskip("torch")
gguf = pytest.importorskip("gguf")


def _write(directory, parts, architecture):
    directory.mkdir()
    paths = []
    for index, tensors in enumerate(parts):
        filename = f"tiny-{index + 1:05d}-of-{len(parts):05d}.gguf" if len(parts) > 1 else "tiny.gguf"
        path = directory / filename
        writer = gguf.GGUFWriter(path, architecture)
        writer.add_context_length(32)
        writer.add_embedding_length(32)
        writer.add_block_count(1)
        writer.add_feed_forward_length(32)
        writer.add_head_count(2)
        writer.add_head_count_kv([0] if architecture == "lfm2" else 2)
        writer.add_layer_norm_rms_eps(1e-5)
        writer.add_rope_freq_base(10000)
        writer.add_rope_dimension_count(16)
        writer.add_uint32(f"{architecture}.shortconv.l_cache", 3)
        writer.add_token_list([f"token_{token}" for token in range(16)])
        if len(parts) > 1:
            writer.add_uint16("split.no", index)
            writer.add_uint16("split.count", len(parts))
            writer.add_int32("split.tensors.count", sum(map(len, parts)))
        if index:
            writer.kv_data[0].pop("general.architecture")
        for name, value in tensors.items():
            data, kind = value if isinstance(value, tuple) else (value, None)
            writer.add_tensor(name, data, raw_dtype=kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        paths.append(str(path))
    return paths


def _weights(architecture):
    shapes = {"token_embd.weight": (16, 32), "blk.0.attn_norm.weight": (32,), "blk.0.ffn_norm.weight": (32,)}
    shapes.update({f"blk.0.ffn_{name}.weight": (32, 32) for name in ("gate", "up", "down")})
    if architecture == "llama":
        shapes.update({f"blk.0.attn_{name}.weight": (32, 32) for name in ("q", "k", "v", "output")})
        shapes["output_norm.weight"] = (32,)
    else:
        shapes.update({"token_embd_norm.weight": (32,), "blk.0.shortconv.conv.weight": (32, 3)})
        # The current converter accepts this direct name, but has no GGUF alias for the kernel bias.
        shapes["blk.0.conv.conv.bias"] = (32,)
        for name, width in (("in", 96), ("out", 32)):
            shapes[f"blk.0.shortconv.{name}_proj.weight"] = (width, 32)
            shapes[f"blk.0.shortconv.{name}_proj.bias"] = (width,)
    shapes["output.weight"] = (16, 32)
    names = sorted(shapes, key=lambda name: name in {"output.weight", "blk.0.shortconv.in_proj.bias"})
    return {
        name: (np.arange(np.prod(shapes[name]), dtype=np.float32).reshape(shapes[name]) % 17 + index) / 100
        for index, name in enumerate(names, 1)
    }


@require_accelerate
@pytest.mark.parametrize("architecture", ["llama", "lfm2"])
def test_checkpoint_matches_single(tmp_path, architecture):
    tensors = _weights(architecture)
    if architecture == "lfm2":
        kind = gguf.GGMLQuantizationType.Q8_0
        tensors["output.weight"] = (gguf.quants.quantize(tensors["output.weight"], kind), kind)
    items = list(tensors.items())
    parts = [dict(items[: len(items) // 2]), dict(items[len(items) // 2 :])]
    paths = [
        _write(tmp_path / "single", [tensors], architecture)[0],
        _write(tmp_path / "checkpoint", parts, architecture)[0],
    ]
    models = []
    for path in paths:
        model, info = AutoModelForCausalLM.from_pretrained(
            str(Path(path).parent),
            gguf_file=Path(path).name,
            local_files_only=True,
            quantization_config=GgufConfig(dequantize=True),
            dtype=torch.float32,
            attn_implementation="eager",
            output_loading_info=True,
        )
        assert not info["missing_keys"] and not info["unexpected_keys"]
        assert not model.config.tie_word_embeddings
        if architecture == "lfm2":
            assert model.config.conv_bias
        models.append(model)
    torch.testing.assert_close(models[0].state_dict(), models[1].state_dict(), atol=0, rtol=0)
    with torch.no_grad():
        ids = torch.tensor([[1, 3, 5]])
        logits = [model(ids, use_cache=False).logits for model in models]
        torch.testing.assert_close(logits[0], logits[1], atol=0, rtol=0)
