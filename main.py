# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from transformers import AutoImageProcessor
from transformers.models.omnivinci.configuration_omnivinci import OmniVinciConfig
from transformers.models.omnivinci.convert_omnivinci_to_hf import convert_omnivinci_to_hf
from transformers.models.omnivinci.modeling_omnivinci import OmniVinciForCausalLM
from transformers.models.omnivinci.processing_omnivinci import OmniVinciProcessor
from transformers.models.qwen2 import Qwen2TokenizerFast


os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NVOmniVideoInference:
    """A class to handle NVOmni video model inference."""

    def __init__(
        self,
        model_path: str,
        torch_dtype="torch.float16",
        device_map="auto",
    ):
        self.model_path = str(Path(model_path).resolve())
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.image_processor = None
        self.config = None
        self.device = None

        self.load_model()

    def validate_paths(self, model_path: str, video_path: str = None) -> bool:
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return False

        if video_path and not Path(video_path).exists():
            logger.error(f"Video path does not exist: {video_path}")
            return False

        return True

    @staticmethod
    def _has_top_level_weights(model_dir: Path) -> bool:
        candidates = (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
        return any((model_dir / name).is_file() for name in candidates)

    def _maybe_convert_legacy_checkpoint(self) -> None:
        model_dir = Path(self.model_path)
        if self._has_top_level_weights(model_dir):
            return

        required_components = ("llm", "vision_tower", "mm_projector")
        if not all((model_dir / name).is_dir() for name in required_components):
            return

        logger.warning(
            "Top-level HF weights were not found in %s. Running legacy-to-HF conversion in place.",
            model_dir,
        )
        convert_omnivinci_to_hf(model_dir)

        if not self._has_top_level_weights(model_dir):
            raise OSError(
                f"Conversion completed but no top-level checkpoint was produced in {model_dir}."
            )

    def _populate_config_from_tokenizer(self, tokenizer) -> None:
        self.config.padding_side = getattr(tokenizer, "padding_side", "left")

        tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
        if tokenizer_max_length is None or tokenizer_max_length > 10_000_000:
            llm_cfg = getattr(self.config, "llm_cfg", None)
            if isinstance(llm_cfg, dict):
                tokenizer_max_length = llm_cfg.get("model_max_length")
            elif llm_cfg is not None:
                tokenizer_max_length = getattr(llm_cfg, "model_max_length", None)
        if tokenizer_max_length is None:
            tokenizer_max_length = getattr(self.config, "model_max_length", 2048)
        self.config.model_max_length = int(tokenizer_max_length)

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.config.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id

        media_token_ids = {}
        for name, token in self.config.media_tokens.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id < 0:
                tokenized = tokenizer(token, add_special_tokens=False).input_ids
                if len(tokenized) != 1:
                    raise ValueError(f"Media token `{token}` must map to a single id.")
                token_id = tokenized[0]
            media_token_ids[name] = int(token_id)
        self.config.media_token_ids = media_token_ids

    def load_model(self) -> bool:
        if not self.validate_paths(self.model_path):
            return False

        self._maybe_convert_legacy_checkpoint()

        logger.info("Loading model configuration...")
        self.config = OmniVinciConfig.from_pretrained(self.model_path)
        self.config._name_or_path = str(self.model_path)

        default_attn_impl = "sdpa"
        attn_implementation = os.environ.get("OMNIVINCI_ATTN_IMPLEMENTATION", default_attn_impl).strip() or default_attn_impl
        if attn_implementation == "flash_attention_2":
            logger.warning("FlashAttention is disabled in this setup; forcing SDPA.")
            attn_implementation = "sdpa"
        self.config._attn_implementation = attn_implementation
        logger.info(f"Using attention implementation: {attn_implementation}")

        logger.info("Loading tokenizer and image processor...")
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(self.model_path)
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path, use_fast=False, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self._populate_config_from_tokenizer(self.tokenizer)

        logger.info("Loading model...")
        start_time = time.time()
        load_dtype = self.torch_dtype
        if isinstance(load_dtype, str) and load_dtype != "auto":
            load_dtype = eval(load_dtype, {"torch": torch})

        self.model = OmniVinciForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            dtype=load_dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.config = self.model.config
        self._populate_config_from_tokenizer(self.tokenizer)
        self.model.tokenizer = self.tokenizer

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        logger.info("Constructing processor from loaded components...")
        self.processor = OmniVinciProcessor(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            config=self.config,
            padding_side=self.tokenizer.padding_side,
        )

        if hasattr(self.model, "device"):
            self.device = self.model.device
        else:
            self.device = next(self.model.parameters()).device if self.model.parameters() else torch.device("cpu")

        logger.info(f"Model successfully loaded on device: {self.device}")
        self._print_model_info()
        return True

    def _print_model_info(self) -> None:
        logger.info("=" * 50)
        logger.info("MODEL INFORMATION")
        logger.info("=" * 50)

        if self.config:
            logger.info(f"Model type: {getattr(self.config, 'model_type', 'Unknown')}")
            logger.info(f"Hidden size: {getattr(self.config, 'hidden_size', 'Unknown')}")
            logger.info(f"Config class file: {inspect.getfile(type(self.config))}")

        if self.model:
            logger.info(f"Model class file: {inspect.getfile(type(self.model))}")

        if self.processor:
            logger.info(f"Processor class file: {inspect.getfile(type(self.processor))}")

        if self.model and torch.cuda.is_available():
            logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def create_conversation(self, video_path: str, text_prompt: str) -> List[Dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

    @torch.inference_mode()
    def generate_response(
        self,
        video_path: str,
        text_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = None,
        top_p: float = None,
        do_sample: bool = False,
        num_video_frames: int = -1,
        load_audio_in_video: bool = True,
        audio_length: Union[int, str] = "max_3600",
    ) -> Optional[str]:
        if not self.model or not self.processor:
            logger.error("Model or processor not loaded. Please initialize the model first.")
            return None

        if not self.validate_paths(self.model_path, video_path):
            return None

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Text prompt: {text_prompt}")

        conversation = self.create_conversation(video_path, text_prompt)
        text = self.processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        logger.info("Chat template applied")

        self.model.config.load_audio_in_video = load_audio_in_video
        self.processor.config.load_audio_in_video = load_audio_in_video
        if num_video_frames > 0:
            self.model.config.num_video_frames = num_video_frames
            self.processor.config.num_video_frames = num_video_frames
        if audio_length != -1:
            self.model.config.audio_chunk_length = audio_length
            self.processor.config.audio_chunk_length = audio_length
        logger.info(
            "Model config - load_audio_in_video: %s, num_video_frames: %s, audio_chunk_length: %s",
            self.model.config.load_audio_in_video,
            self.model.config.num_video_frames,
            self.model.config.audio_chunk_length,
        )

        start_time = time.time()
        inputs = self.processor([text])

        if hasattr(inputs, "input_ids") and inputs.input_ids is not None:
            inputs.input_ids = inputs.input_ids.to(self.device)

        processing_time = time.time() - start_time
        logger.info(f"Input processing completed in {processing_time:.2f} seconds")

        logger.info("Generating response...")
        start_time = time.time()

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "max_length": 99999999,
            "do_sample": bool(do_sample),
            "num_beams": 1,
        }
        if do_sample and top_p is not None:
            generation_kwargs["top_p"] = top_p
        if do_sample and temperature is not None:
            generation_kwargs["temperature"] = temperature

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.update(**generation_kwargs)
        if not generation_config.do_sample:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        logger.info(f"Generation config: {generation_config.to_dict()}")

        with torch.no_grad():
            # Build multimodal prefill embeddings before `generate` so HF initializes cache_position
            # with the full multimodal sequence length (not just raw text token length).
            prefill_inputs_embeds, _, prefill_attention_mask = self.model._embed(
                inputs.input_ids,
                getattr(inputs, "media", None),
                getattr(inputs, "media_config", None),
                None,
                getattr(inputs, "attention_mask", None),
            )

            output_ids = self.model.generate(
                input_ids=inputs.input_ids,
                inputs_embeds=prefill_inputs_embeds,
                attention_mask=prefill_attention_mask,
                generation_config=generation_config,
            )

        generation_time = time.time() - start_time
        logger.info(f"Generation completed in {generation_time:.2f} seconds")

        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        response = self.processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def batch_generate(self, video_text_pairs: List[tuple], **generation_kwargs) -> List[Optional[str]]:
        responses: List[Optional[str]] = []
        for i, (video_path, text_prompt) in enumerate(video_text_pairs):
            logger.info(f"Processing batch item {i + 1}/{len(video_text_pairs)}")
            response = self.generate_response(video_path, text_prompt, **generation_kwargs)
            responses.append(response)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return responses


def main() -> None:
    model_path = os.environ.get("OMNIVINCI_MODEL_PATH", "/fs/nexus-projects/JSALT_workshop/lasha/Dev/comni")
    video_path = os.environ.get("OMNIVINCI_VIDEO_PATH", "/nfshomes/lasha/Dev/omnivinci/nvidia.mp4")
    text_prompt = os.environ.get(
        "OMNIVINCI_TEXT_PROMPT",
        "Assess the video, followed by a detailed description of it's video and audio contents.",
    )

    num_video_frames = int(os.environ.get("OMNIVINCI_NUM_VIDEO_FRAMES", "128"))
    audio_length: Union[int, str] = os.environ.get("OMNIVINCI_AUDIO_LENGTH", "max_3600")
    load_audio_in_video = os.environ.get("OMNIVINCI_LOAD_AUDIO_IN_VIDEO", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }

    requested_device_map = os.environ.get("OMNIVINCI_DEVICE_MAP")
    if requested_device_map:
        device_map = requested_device_map
    else:
        device_map = "auto" if torch.cuda.is_available() else "cpu"

    if device_map in {"auto", "cuda"} and not torch.cuda.is_available():
        logger.warning("CUDA is not available; forcing device_map=cpu.")
        device_map = "cpu"

    logger.info("Initializing NVOmni Video Inference...")
    inferencer = NVOmniVideoInference(
        model_path,
        torch_dtype="torch.float16",
        device_map=device_map,
    )

    if inferencer.model is None:
        logger.error("Failed to initialize model. Exiting.")
        return

    logger.info("Starting inference...")
    response = inferencer.generate_response(
        video_path=video_path,
        text_prompt=text_prompt,
        num_video_frames=num_video_frames,
        load_audio_in_video=load_audio_in_video,
        audio_length=audio_length,
        max_new_tokens=1024,
        do_sample=False,
    )

    if response:
        print("\n" + "=" * 60)
        print("GENERATED RESPONSE")
        print("=" * 60)
        print(response)
        print("=" * 60)
    else:
        logger.error("Failed to generate response")


if __name__ == "__main__":
    main()
