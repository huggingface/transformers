# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import warnings
from io import BytesIO
from typing import Any, Union, overload

import httpx

from ..generation import GenerationConfig
from ..utils import (
    add_end_docstrings,
    is_av_available,
    is_torch_available,
    logging,
    requires_backends,
)
from .base import Pipeline, build_pipeline_init_args


if is_av_available():
    import av
    import numpy as np

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_image_processor=True))
class VideoToTextPipeline(Pipeline):
    """
    Video To Text pipeline using a `AutoModelForImageTextToText`. This pipeline predicts a caption for a given video.

    Unless the model you're using explicitly sets these generation parameters in its configuration files
    (`generation_config.json`), the following default values will be used:
    - max_new_tokens: 256

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline("video-to-text", model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("path/to/video.mp4")
    [{'generated_text': 'a person is setting a table'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This video to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "video-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=video-to-text).
    """

    _pipeline_calls_generate = True
    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = True
    # Make sure the docstring is updated when the default generation config is changed
    _default_generation_config = GenerationConfig(
        max_new_tokens=256,
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "av")
        self.check_model_type(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)

    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, num_frames=None, frame_sampling_rate=None, timeout=None, system_prompt=None):
        forward_params = {}
        preprocess_params = {}

        if timeout is not None:
            preprocess_params["timeout"] = timeout
        if num_frames is not None:
            preprocess_params["num_frames"] = num_frames
        if frame_sampling_rate is not None:
            preprocess_params["frame_sampling_rate"] = frame_sampling_rate

        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens
        if system_prompt is not None:
            forward_params["system_prompt"] = system_prompt
        if generate_kwargs is not None:
            if max_new_tokens is not None and "max_new_tokens" in generate_kwargs:
                raise ValueError(
                    "`max_new_tokens` is defined both as an argument and inside `generate_kwargs` argument, please use"
                    " only 1 version"
                )
            forward_params.update(generate_kwargs)

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, {}

    @overload
    def __call__(self, inputs: str, **kwargs: Any) -> list[dict[str, Any]]: ...

    @overload
    def __call__(self, inputs: list[str], **kwargs: Any) -> list[list[dict[str, Any]]]: ...

    def __call__(self, inputs: str | list[str] | None = None, **kwargs):
        """
        Generate text captions for the video(s) passed as inputs.

        Args:
            inputs (`str`, `list[str]`):
                The pipeline handles two types of videos:

                - A string containing a http link pointing to a video
                - A string containing a local path to a video

                The pipeline accepts either a single video or a batch of videos, which must then be passed as a string.
                Videos in a batch must all be in the same format: all as http links or all as local paths.
            max_new_tokens (`int`, *optional*):
                The amount of maximum tokens to generate. By default it will use `generate` default.
            num_frames (`int`, *optional*):
                The number of frames sampled from the video to run the generation on. If not provided, will be
                calculated as a function of video duration (1 frame per second, min 8, max 128). If video duration
                is unavailable, will default to the number of frames specified in the model configuration.
            frame_sampling_rate (`int`, *optional*, defaults to 1):
                Currently unused - frames are time-spaced based on video duration.
            generate_kwargs (`Dict`, *optional*):
                Pass it to send all of these arguments directly to `generate` allowing full control of this function.
            system_prompt (`str`, *optional*):
                A system prompt to guide the model's generation. This will be tokenized and passed to the model
                to influence the style and detail of the generated description.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching videos from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        if "videos" in kwargs:
            warnings.warn(
                "The `videos` argument has been renamed to `inputs`. In version 5 of Transformers, `videos` will no longer be accepted",
                FutureWarning,
            )
            inputs = kwargs.pop("videos")
        if inputs is None:
            raise ValueError("Cannot call the video-to-text pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, video, num_frames=None, frame_sampling_rate=1, timeout=None):
        if video.startswith("http://") or video.startswith("https://"):
            video = BytesIO(httpx.get(video, follow_redirects=True, timeout=timeout).content)

        container = av.open(video)
        
        # Get video metadata for logging
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames if video_stream.frames else 0
        fps = float(video_stream.average_rate) if video_stream.average_rate else 0
        duration = container.duration / av.time_base if container.duration else 0
        
        # Calculate num_frames as a function of video length
        # Default: 1 frame per second, minimum 8, maximum 128
        if num_frames is None:
            if duration > 0:
                # 1 frame per second, with min/max bounds
                num_frames = max(8, min(128, int(duration)))
            else:
                # Fallback: try to get from model config, otherwise use default
                if hasattr(self.model.config, "num_frames"):
                    num_frames = self.model.config.num_frames
                else:
                    num_frames = 64  # Default fallback
        
        logger.info(f"Video metadata: duration={duration:.2f}s, fps={fps:.2f}, total_frames={total_frames}")
        logger.info(f"Frame selection: num_frames={num_frames} (calculated from duration)")

        # Use time-spaced frames (time-based sampling instead of frame-based)
        # Sample frames evenly spaced in time
        if duration > 0 and fps > 0:
            # Calculate time points evenly spaced across the video duration
            # Use endpoint=True to include the last frame
            time_points = np.linspace(0, duration, num=num_frames, endpoint=True)
            
            # Convert time points to frame indices
            indices = (time_points * fps).astype(np.int64)
            # Ensure indices don't exceed total frames
            if total_frames > 0:
                indices = np.clip(indices, 0, total_frames - 1)
            # Remove duplicates and sort to maintain temporal order
            indices = np.unique(indices)
            logger.info(f"Time-spaced sampling selected {len(indices)} frame indices: {indices.tolist()}")
        else:
            # Fallback to frame-based linear sampling if duration/fps unavailable
            start_idx = 0
            end_idx = total_frames - 1 if total_frames > 0 else num_frames - 1
            indices = np.linspace(start_idx, end_idx, num=num_frames, dtype=np.int64)
            logger.info(f"Frame-based linear sampling selected {len(indices)} frame indices: {indices.tolist()}")
        
        # Log temporal gaps between selected frames
        if len(indices) > 1 and fps > 0:
            gaps = []
            for i in range(len(indices) - 1):
                gap_frames = indices[i + 1] - indices[i]
                gap_seconds = gap_frames / fps if fps > 0 else 0
                gaps.append(f"{gap_frames} frames ({gap_seconds:.2f}s)")
            logger.info(f"Temporal gaps between selected frames: {gaps}")

        video_frames = read_video_pyav(container, indices)
        video_frames = list(video_frames)
        logger.info(f"Extracted {len(video_frames)} frames")

        # Process video frames through image processor
        logger.info(f"Processing {len(video_frames)} individual frames")
        model_inputs = self.image_processor(video_frames, return_tensors="pt")
        
        model_inputs = model_inputs.to(self.dtype)

        # Some models like GIT need input_ids set to None
        if self.model.config.model_type == "git":
            model_inputs["input_ids"] = None

        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        # Git model sets `model_inputs["input_ids"] = None` in `preprocess`. In batch model, the
        # pipeline will group them into a list of `None`, which fail `_forward`. Avoid this by checking it first.
        if (
            "input_ids" in model_inputs
            and isinstance(model_inputs["input_ids"], list)
            and all(x is None for x in model_inputs["input_ids"])
        ):
            model_inputs["input_ids"] = None

        # Handle system prompt if provided
        system_prompt = generate_kwargs.pop("system_prompt", None)
        if system_prompt is not None:
            # Tokenize the system prompt
            if self.model.config.model_type == "git":
                # For GIT models, we can pass the prompt as input_ids
                # Tokenize and add to model_inputs
                prompt_ids = self.tokenizer(system_prompt, return_tensors="pt", add_special_tokens=True)
                prompt_ids = prompt_ids["input_ids"].to(self.device)
                # If input_ids is None, set it to the prompt; otherwise prepend
                if model_inputs.get("input_ids") is None:
                    model_inputs["input_ids"] = prompt_ids
                else:
                    # Prepend system prompt to existing input_ids
                    if isinstance(model_inputs["input_ids"], torch.Tensor):
                        model_inputs["input_ids"] = torch.cat([prompt_ids, model_inputs["input_ids"]], dim=1)
            else:
                # For other models, add as input_ids or pass through generate_kwargs
                prompt_ids = self.tokenizer(system_prompt, return_tensors="pt", add_special_tokens=True)
                prompt_ids = prompt_ids["input_ids"].to(self.device)
                if "input_ids" not in model_inputs or model_inputs["input_ids"] is None:
                    model_inputs["input_ids"] = prompt_ids
                else:
                    # Prepend system prompt to existing input_ids
                    if isinstance(model_inputs["input_ids"], torch.Tensor):
                        model_inputs["input_ids"] = torch.cat([prompt_ids, model_inputs["input_ids"]], dim=1)

        # User-defined `generation_config` passed to the pipeline call take precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        inputs = model_inputs.pop(self.model.main_input_name)
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        return model_outputs

    def postprocess(self, model_outputs):
        records = []
        seen_texts = set()
        all_texts = []
        
        logger.info(f"Postprocessing {len(model_outputs)} model outputs")
        
        for idx, output_ids in enumerate(model_outputs):
            text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
            all_texts.append(text)
            logger.info(f"Generated text #{idx + 1}: '{text}'")
            
            # Deduplicate: only add if we haven't seen this text before
            if text not in seen_texts:
                seen_texts.add(text)
                record = {"generated_text": text}
                records.append(record)
                logger.debug(f"Added unique text: '{text}'")
            else:
                logger.debug(f"Deduplicated duplicate text: '{text}'")
        
        logger.info(f"Total generated texts: {len(all_texts)}, Unique texts after deduplication: {len(records)}")
        if len(all_texts) > len(records):
            duplicates = [t for t in all_texts if all_texts.count(t) > 1]
            logger.info(f"Duplicated texts: {set(duplicates)}")
        
        return records


def read_video_pyav(container, indices):
    """
    Read frames from video container in the order specified by indices.
    Maintains temporal order by reading frames in the exact order of the indices array.
    """
    # Ensure indices are sorted to maintain temporal order
    sorted_indices = np.sort(indices)
    frames = []
    container.seek(0)
    
    # Create a set for fast lookup, but iterate in sorted order
    indices_set = set(sorted_indices)
    frame_dict = {}
    
    # Read all needed frames in one pass
    for i, frame in enumerate(container.decode(video=0)):
        if i > sorted_indices[-1]:
            break
        if i in indices_set:
            frame_dict[i] = frame
    
    # Extract frames in the order specified by sorted_indices
    for idx in sorted_indices:
        if idx in frame_dict:
            frames.append(frame_dict[idx])
    
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

