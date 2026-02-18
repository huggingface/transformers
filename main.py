# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import inspect
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from transformers.models.omnivinci.configuration_omnivinci import OmniVinciConfig
from transformers.models.omnivinci.modeling_omnivinci import VILAForCausalLM
from transformers.models.omnivinci.processing_omnivinci import OmniVinciProcessor


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

    def load_model(self) -> bool:
        if not self.validate_paths(self.model_path):
            return False

        logger.info("Loading model configuration...")
        self.config = OmniVinciConfig.from_pretrained(self.model_path)
        self.config._name_or_path = str(self.model_path)
        if getattr(self.config, "resume_path", None) is None or not Path(str(self.config.resume_path)).exists():
            self.config.resume_path = str(self.model_path)

        default_attn_impl = "sdpa"
        attn_implementation = os.environ.get("OMNIVINCI_ATTN_IMPLEMENTATION", default_attn_impl).strip() or default_attn_impl
        if attn_implementation == "flash_attention_2":
            logger.warning("FlashAttention is disabled in this setup; forcing SDPA.")
            attn_implementation = "sdpa"
        self.config._attn_implementation = attn_implementation
        logger.info(f"Using attention implementation: {attn_implementation}")

        logger.info("Loading model...")
        start_time = time.time()
        self.model = VILAForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")

        logger.info("Loading processor...")
        self.processor = OmniVinciProcessor.from_pretrained(self.model_path)

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

        generation_config = self.model.default_generation_config
        generation_config.update(**generation_kwargs)

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
    model_path = os.environ.get("OMNIVINCI_MODEL_PATH", "/fs/nexus-projects/JSALT_workshop/lasha/Dev/omnivinci/")
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
