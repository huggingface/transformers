# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""
Model loading, caching, and lifecycle management.
"""

from __future__ import annotations

import gc
import json
import threading
from functools import lru_cache
from typing import TYPE_CHECKING

from huggingface_hub import scan_cache_dir
from tqdm import tqdm

import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizerBase

from ...utils import logging
from .protocol import Modality
from .utils import reset_torch_cache


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


logger = logging.get_logger(__name__)


class TimedModel:
    """Wraps a model + processor and auto-deletes them after a period of inactivity.

    Args:
        model: The loaded model.
        timeout_seconds: Seconds of inactivity before auto-deletion. Use -1 to disable.
        processor: The associated processor or tokenizer.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        timeout_seconds: int,
        processor: ProcessorMixin | PreTrainedTokenizerFast | None = None,
    ):
        self.model = model
        self._name_or_path = str(model.name_or_path)
        self.processor = processor
        self.timeout_seconds = timeout_seconds
        self._timer = threading.Timer(self.timeout_seconds, self._timeout_reached)
        self._timer.start()

    def reset_timer(self) -> None:
        """Reset the inactivity timer (called on each request)."""
        self._timer.cancel()
        self._timer = threading.Timer(self.timeout_seconds, self._timeout_reached)
        self._timer.start()

    def delete_model(self) -> None:
        """Delete the model and processor, free GPU memory."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            gc.collect()
            reset_torch_cache()
            self._timer.cancel()

    def _timeout_reached(self) -> None:
        if self.timeout_seconds > 0:
            self.delete_model()
            logger.info(f"{self._name_or_path} was removed from memory after {self.timeout_seconds}s of inactivity")

    def is_deleted(self) -> bool:
        """Check if the model has been deleted (by timeout or manually)."""
        return not hasattr(self, "model") or self.model is None


class ModelManager:
    """Loads, caches, and manages the lifecycle of models.

    Handlers receive a reference to this and call `load_model_and_processor()`
    to get a model ready for inference.

    Args:
        device: Device to place models on (e.g. "auto", "cuda", "cpu").
        dtype: Torch dtype override. "auto" derives from model weights.
        trust_remote_code: Whether to trust remote code when loading models.
        attn_implementation: Attention implementation override (e.g. "flash_attention_2").
        quantization: Quantization method ("bnb-4bit" or "bnb-8bit").
        model_timeout: Seconds before an idle model is unloaded. -1 disables.
        force_model: If set, preload this model at init time.
        processor_id: Override processor/tokenizer model ID. Needed for GGUF models
            whose repos don't include tokenizer files.
    """

    def __init__(
        self,
        device: str = "auto",
        dtype: str | None = "auto",
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
        quantization: str | None = None,
        model_timeout: int = 300,
        force_model: str | None = None,
        # TODO: auto-detect from GGUF base_model metadata
        processor_id: str | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.quantization = quantization
        self.model_timeout = model_timeout
        self.processor_id = processor_id

        self.loaded_models: dict[str, TimedModel] = {}

        if force_model is not None:
            self.load_model_and_processor(self.process_model_name(force_model))

    @staticmethod
    def process_model_name(model_id: str) -> str:
        """Canonicalize to `'model_id@revision'` format. Defaults to `@main`."""
        if "@" in model_id:
            return model_id
        return f"{model_id}@main"

    def get_quantization_config(self) -> BitsAndBytesConfig | None:
        """Return a BitsAndBytesConfig based on the `quantization` setting, or None."""
        if self.quantization == "bnb-4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "bnb-8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def _load_processor(self, model_id_and_revision: str) -> ProcessorMixin | PreTrainedTokenizerFast:
        """Load a processor, trying AutoProcessor first then AutoTokenizer.

        If `processor_id` was set (e.g. for GGUF models), uses that instead of `model_id`.
        Expects `model_id_and_revision` in `'model_id@revision'` format (from `process_model_name`).
        """
        from transformers import AutoProcessor

        if self.processor_id:
            model_id, revision = self.processor_id, "main"
        else:
            model_id, revision = model_id_and_revision.split("@", 1)
        try:
            return AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=self.trust_remote_code)
        except OSError:
            try:
                return AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=self.trust_remote_code)
            except OSError:
                raise OSError(f"Failed to load processor for {model_id} with AutoProcessor and AutoTokenizer.")

    def _load_model(self, model_id_and_revision: str) -> PreTrainedModel:
        """Load a model. GGUF files are detected by the `.gguf` extension and loaded via llama.cpp."""
        import torch

        from transformers import AutoConfig

        model_id, revision = model_id_and_revision.split("@", 1)

        if model_id.endswith(".gguf"):
            from llama_cpp_transformers import LlamaCppTransformersModel

            flash_attn = True if self.attn_implementation == "flash_attention_2" else "auto"
            return LlamaCppTransformersModel.from_pretrained(
                model_id, revision=revision, n_gpu_layers=-1, flash_attn=flash_attn,
            )

        dtype = self.dtype if self.dtype in ["auto", None] else getattr(torch, self.dtype)
        model_kwargs = {
            "revision": revision,
            "attn_implementation": self.attn_implementation,
            "dtype": dtype,
            "device_map": self.device,
            "trust_remote_code": self.trust_remote_code,
            "quantization_config": self.get_quantization_config(),
        }

        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        architecture = getattr(transformers, config.architectures[0])
        return architecture.from_pretrained(model_id, **model_kwargs)

    def load_model_and_processor(
        self, model_id_and_revision: str
    ) -> tuple[PreTrainedModel, ProcessorMixin | PreTrainedTokenizerFast]:
        """Load a model (or return it from cache), resetting its inactivity timer."""
        if model_id_and_revision not in self.loaded_models or self.loaded_models[model_id_and_revision].is_deleted():
            processor = self._load_processor(model_id_and_revision)
            model = self._load_model(model_id_and_revision)
            self.loaded_models[model_id_and_revision] = TimedModel(
                model, timeout_seconds=self.model_timeout, processor=processor
            )
        else:
            self.loaded_models[model_id_and_revision].reset_timer()
            model = self.loaded_models[model_id_and_revision].model
            processor = self.loaded_models[model_id_and_revision].processor
        return model, processor

    def shutdown(self) -> None:
        """Delete all loaded models and free resources."""
        for timed in self.loaded_models.values():
            timed.delete_model()
        self.loaded_models.clear()

    @staticmethod
    def get_model_modality(model: PreTrainedModel, processor=None) -> Modality:
        """Detect whether a model is an LLM or VLM based on its architecture."""
        if processor is not None and isinstance(processor, PreTrainedTokenizerBase):
            return Modality.LLM

        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )

        model_classname = model.__class__.__name__
        if model_classname in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values():
            return Modality.VLM
        elif model_classname in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            return Modality.LLM
        else:
            raise ValueError(f"Unknown modality for: {model_classname}")

    @staticmethod
    @lru_cache
    def get_gen_models(cache_dir: str | None = None) -> list[dict]:
        """List generative models (LLMs and VLMs) available in the HuggingFace cache."""
        from transformers.models.auto.modeling_auto import (
            MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
            MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
        )

        generative_models = []
        logger.warning("Scanning the cache directory for LLMs and VLMs.")

        for repo in tqdm(scan_cache_dir(cache_dir).repos):
            if repo.repo_type != "model":
                continue

            for ref, revision_info in repo.refs.items():
                config_path = next((f.file_path for f in revision_info.files if f.file_name == "config.json"), None)
                if not config_path:
                    continue

                config = json.loads(config_path.open().read())
                if not (isinstance(config, dict) and "architectures" in config):
                    continue

                architectures = config["architectures"]
                llms = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
                vlms = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.values()

                if any(arch for arch in architectures if arch in [*llms, *vlms]):
                    author = repo.repo_id.split("/") if "/" in repo.repo_id else ""
                    repo_handle = repo.repo_id + (f"@{ref}" if ref != "main" else "")
                    generative_models.append(
                        {
                            "owned_by": author,
                            "id": repo_handle,
                            "object": "model",
                            "created": repo.last_modified,
                        }
                    )

        return generative_models
