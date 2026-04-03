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
"""
Model loading, caching, and lifecycle management.
"""

import asyncio
import gc
import json
import threading
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING

from huggingface_hub import scan_cache_dir
from tqdm import tqdm

import transformers
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase

from ...utils import logging
from .utils import Modality, make_progress_tqdm_class, reset_torch_cache


if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerFast, ProcessorMixin


logger = logging.get_logger(__name__)


class TimedModel:
    """Wraps a model + processor and auto-unloads them after a period of inactivity.

    Args:
        model: The loaded model.
        timeout_seconds: Seconds of inactivity before auto-unload. Use -1 to disable.
        processor: The associated processor or tokenizer.
        on_unload: Optional callback invoked after the model is unloaded from memory.
    """

    def __init__(
        self,
        model: "PreTrainedModel",
        timeout_seconds: int,
        processor: "ProcessorMixin | PreTrainedTokenizerFast | None" = None,
        on_unload: "Callable | None" = None,
    ):
        self.model = model
        self._name_or_path = str(model.name_or_path)
        self.processor = processor
        self.timeout_seconds = timeout_seconds
        self._on_unload = on_unload
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
            if self._on_unload is not None:
                self._on_unload()

    def _timeout_reached(self) -> None:
        if self.timeout_seconds > 0:
            self.delete_model()
            logger.info(f"{self._name_or_path} was removed from memory after {self.timeout_seconds}s of inactivity")


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
    ):
        self.loaded_models: dict[str, TimedModel] = {}

        # Thread-safety for concurrent load_model_and_processor calls
        self._model_locks: dict[str, threading.Lock] = {}
        self._model_locks_guard = threading.Lock()

        # Tracks in-flight loads for fan-out to multiple SSE subscribers (used by load_model_streaming)
        self._loading_subscribers: dict[str, list[asyncio.Queue[str | None]]] = {}
        self._loading_tasks: dict[str, asyncio.Task] = {}

        # Convert numeric device strings (e.g. "0") to int so device_map works correctly
        self.device = int(device) if device.isdigit() else device
        self.dtype = self._resolve_dtype(dtype)
        self.trust_remote_code = trust_remote_code
        self.attn_implementation = attn_implementation
        self.quantization = quantization
        self.model_timeout = model_timeout
        self.force_model = force_model

        self._validate_args()

        # Preloaded models should never be auto-unloaded
        if force_model is not None:
            self.model_timeout = -1

        # Preload the forced model after all state is initialized
        if force_model is not None:
            self.load_model_and_processor(self.process_model_name(force_model))

    @staticmethod
    def _resolve_dtype(dtype: str | None):
        import torch

        if dtype in ("auto", None):
            return dtype
        resolved = getattr(torch, dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(
                f"Unsupported dtype: '{dtype}'. Must be 'auto' or a valid torch dtype (e.g. 'float16', 'bfloat16')."
            )
        return resolved

    def _validate_args(self):
        if self.quantization is not None and self.quantization not in ("bnb-4bit", "bnb-8bit"):
            raise ValueError(
                f"Unsupported quantization method: '{self.quantization}'. Must be 'bnb-4bit' or 'bnb-8bit'."
            )
        VALID_ATTN_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"}
        is_kernels_community = self.attn_implementation is not None and self.attn_implementation.startswith(
            "kernels-community/"
        )
        if (
            self.attn_implementation is not None
            and not is_kernels_community
            and self.attn_implementation not in VALID_ATTN_IMPLEMENTATIONS
        ):
            raise ValueError(
                f"Unsupported attention implementation: '{self.attn_implementation}'. "
                f"Must be one of {VALID_ATTN_IMPLEMENTATIONS} or a kernels-community kernel (e.g. 'kernels-community/flash-attn2')."
            )

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

    def _load_processor(self, model_id_and_revision: str) -> "ProcessorMixin | PreTrainedTokenizerFast":
        """Load a processor for the given model.

        Args:
            model_id_and_revision: Model ID in ``'model_id@revision'`` format.
        """
        from transformers import AutoProcessor

        model_id, revision = model_id_and_revision.split("@", 1)
        return AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=self.trust_remote_code)

    def _load_model(
        self, model_id_and_revision: str, tqdm_class: type | None = None, progress_callback: Callable | None = None
    ) -> "PreTrainedModel":
        """Load a model.

        Args:
            model_id_and_revision (`str`): Model ID in ``'model_id@revision'`` format.
            tqdm_class (*optional*): tqdm subclass for progress bars during ``from_pretrained``.
            progress_callback (`Callable`, *optional*): Called with progress dicts during loading.

        Returns:
            `PreTrainedModel`: The loaded model.
        """
        from transformers import AutoConfig

        model_id, revision = model_id_and_revision.split("@", 1)

        model_kwargs = {
            "revision": revision,
            "attn_implementation": self.attn_implementation,
            "dtype": self.dtype,
            "device_map": self.device,
            "trust_remote_code": self.trust_remote_code,
            "quantization_config": self.get_quantization_config(),
            "tqdm_class": tqdm_class,
        }

        if progress_callback is not None:
            progress_callback({"status": "loading", "model": model_id_and_revision, "stage": "config"})
        config = AutoConfig.from_pretrained(model_id, **model_kwargs)
        architecture = getattr(transformers, config.architectures[0])

        return architecture.from_pretrained(model_id, **model_kwargs)

    def load_model_and_processor(
        self,
        model_id_and_revision: str,
        progress_callback: Callable | None = None,
        tqdm_class: type | None = None,
    ) -> "tuple[PreTrainedModel, ProcessorMixin | PreTrainedTokenizerFast]":
        """Load a model (or return it from cache), resetting its inactivity timer.

        Args:
            model_id_and_revision: Model ID in ``'model_id@revision'`` format.
            progress_callback: If provided, called with dicts like
                ``{"status": "loading", "model": ..., "stage": ...}`` during loading.
            tqdm_class: Optional tqdm subclass for progress bars during ``from_pretrained``.
        """
        # Per-model lock prevents duplicate loads when concurrent requests arrive
        with self._model_locks_guard:
            lock = self._model_locks.setdefault(model_id_and_revision, threading.Lock())

        with lock:
            if model_id_and_revision not in self.loaded_models:
                logger.warning(f"Loading {model_id_and_revision}")
                if progress_callback is not None:
                    progress_callback({"status": "loading", "model": model_id_and_revision, "stage": "processor"})
                processor = self._load_processor(model_id_and_revision)
                model = self._load_model(
                    model_id_and_revision, tqdm_class=tqdm_class, progress_callback=progress_callback
                )
                self.loaded_models[model_id_and_revision] = TimedModel(
                    model,
                    timeout_seconds=self.model_timeout,
                    processor=processor,
                    on_unload=lambda key=model_id_and_revision: self.loaded_models.pop(key, None),
                )
                if progress_callback is not None:
                    progress_callback({"status": "ready", "model": model_id_and_revision, "cached": False})
            else:
                self.loaded_models[model_id_and_revision].reset_timer()
                model = self.loaded_models[model_id_and_revision].model
                processor = self.loaded_models[model_id_and_revision].processor
                if progress_callback is not None:
                    progress_callback({"status": "ready", "model": model_id_and_revision, "cached": True})
        return model, processor

    async def load_model_streaming(self, model_id_and_revision: str):
        """Load a model and stream progress as SSE events.

        Handles three cases:
        1. Model already cached → single ``ready`` event
        2. Load already in progress → join existing subscriber stream
        3. First request → start loading, broadcast to all subscribers

        Args:
            model_id_and_revision (`str`): Model ID in ``'model_id@revision'`` format.

        Yields:
            `str`: SSE ``data: ...`` lines with progress updates.
        """
        mid = model_id_and_revision
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        # Case 1: already cached
        if mid in self.loaded_models:
            self.loaded_models[mid].reset_timer()
            yield f"data: {json.dumps({'status': 'ready', 'model': mid, 'cached': True})}\n\n"
            return

        # Case 2: load in progress — join existing subscribers
        if mid in self._loading_tasks:
            self._loading_subscribers[mid].append(queue)
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
            return

        # Case 3: first request — start the load
        self._loading_subscribers[mid] = [queue]
        loop = asyncio.get_running_loop()

        def enqueue(payload: dict):
            msg = f"data: {json.dumps(payload)}\n\n"

            def broadcast():
                for q in self._loading_subscribers.get(mid, []):
                    q.put_nowait(msg)

            loop.call_soon_threadsafe(broadcast)

        tqdm_class = make_progress_tqdm_class(enqueue, mid)

        def _tqdm_hook(factory, args, kwargs):
            return tqdm_class(*args, **kwargs)

        async def run_load():
            try:
                # Install a global tqdm hook so the "Loading weights" bar in
                # core_model_loading.py (which uses logging.tqdm) routes through
                # our ProgressTqdm. The tqdm_class kwarg only covers download bars.
                previous_hook = logging.set_tqdm_hook(_tqdm_hook)
                try:
                    await asyncio.to_thread(
                        self.load_model_and_processor,
                        mid,
                        progress_callback=enqueue,
                        tqdm_class=tqdm_class,
                    )
                finally:
                    logging.set_tqdm_hook(previous_hook)
            except Exception as e:
                logger.error(f"Failed to load {mid}: {e}", exc_info=True)
                enqueue({"status": "error", "model": mid, "message": str(e)})
            finally:

                def _send_sentinel():
                    for q in self._loading_subscribers.pop(mid, []):
                        q.put_nowait(None)
                    self._loading_tasks.pop(mid, None)

                loop.call_soon_threadsafe(_send_sentinel)

        self._loading_tasks[mid] = asyncio.create_task(run_load())

        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    def shutdown(self) -> None:
        """Delete all loaded models and free resources."""
        for timed in list(self.loaded_models.values()):
            timed.delete_model()

    @staticmethod
    def get_model_modality(
        model: "PreTrainedModel", processor: "ProcessorMixin | PreTrainedTokenizerFast | None" = None
    ) -> Modality:
        """Detect whether a model is an LLM or VLM based on its architecture.

        Args:
            model (`PreTrainedModel`): The loaded model.
            processor (`ProcessorMixin | PreTrainedTokenizerFast`, *optional*):
                If a plain tokenizer (not a multi-modal processor), short-circuits to LLM.

        Returns:
            `Modality`: The detected modality (``Modality.LLM`` or ``Modality.VLM``).
        """
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
        """List generative models (LLMs and VLMs) available in the HuggingFace cache.

        Args:
            cache_dir (`str`, *optional*): Path to the HuggingFace cache directory.
                Defaults to the standard cache location.

        Returns:
            `list[dict]`: OpenAI-compatible model list entries with ``id``, ``object``, etc.
        """
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
