# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import math

import numpy as np
import torch

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, MultiModalData, ProcessingKwargs, ProcessorMixin, Unpack, VideosKwargs
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...video_utils import VideoInput


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3Processor"]

# Audio input type - can be file paths, numpy arrays, or torch tensors
AudioInput = str | np.ndarray | torch.Tensor | list[str] | list[np.ndarray] | list[torch.Tensor]


class NemotronH_Nano_Omni_Reasoning_V3ImagesKwargs(ImagesKwargs):
    min_pixels: int | None
    max_pixels: int | None
    patch_size: int | None
    temporal_patch_size: int | None
    merge_size: int | None


class NemotronH_Nano_Omni_Reasoning_V3AudioKwargs(ProcessingKwargs, total=False):
    sampling_rate: int | None


class NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: NemotronH_Nano_Omni_Reasoning_V3ImagesKwargs
    videos_kwargs: VideosKwargs
    audio_kwargs: NemotronH_Nano_Omni_Reasoning_V3AudioKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class NemotronH_Nano_Omni_Reasoning_V3Processor(ProcessorMixin):
    r"""
    Constructs a Nemotron-3-Nano-Omni-30B-A3B-Reasoning processor which wraps an image processor, audio feature extractor,
    and a tokenizer into a single processor.
    [`NemotronH_Nano_Omni_Reasoning_V3Processor`] offers all the functionalities of the image processor, audio processor,
    and tokenizer. See the [`~NemotronH_Nano_Omni_Reasoning_V3Processor.__call__`] and [`~NemotronH_Nano_Omni_Reasoning_V3Processor.decode`]
    for more information.
    Args:
        image_processor ([`AutoImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`AutoTokenizer`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        audio_sampling_rate (`int`, *optional*): Sampling rate for audio processing (default: 16000).
        audio_subsampling_factor (`int`, *optional*): Subsampling factor for audio encoder (default: 8).
        audio_hop_length (`int`, *optional*): Hop length in samples for feature extraction (default: 160).
    """

    attributes = ["image_processor", "tokenizer"]

    image_processor_class = "AutoImageProcessor"
    video_processor_class = "AutoVideoProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        audio_sampling_rate: int = 16000,
        audio_subsampling_factor: int = 8,
        audio_hop_length: int = 160,
        video_temporal_patch_dim: int = 2,
        **kwargs,
    ):
        # Number of frames collapsed into a single temporal patch by the model's `video_embedder`.
        # The `<video>` expansion below issues one placeholder block per temporal patch.
        self.video_temporal_patch_dim = video_temporal_patch_dim
        self.image_token = "<image>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<video>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        self.audio_token = "<so_embedding>" if not hasattr(tokenizer, "audio_token") else tokenizer.audio_token
        self.audio_start_token = "<so_start>"
        self.audio_end_token = "<so_end>"
        self.image_start_token = (
            "<img>" if not hasattr(tokenizer, "image_start_token") else tokenizer.image_start_token
        )
        self.image_end_token = "</img>" if not hasattr(tokenizer, "image_end_token") else tokenizer.image_end_token
        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_token)
        )

        # Audio processing parameters
        self.audio_sampling_rate = audio_sampling_rate
        self.audio_subsampling_factor = audio_subsampling_factor
        self.audio_hop_length = audio_hop_length

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        videos: VideoInput = None,
        audio: AudioInput = None,
        **kwargs: Unpack[NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare multimodal inputs (text, images, videos, audio) for the model. This method processes
        text by replacing image/video/audio tokens with appropriate placeholder sequences, processes images and videos
        through the image processor, and tokenizes the final text.

        The method performs the following key operations:
        1. Processes images using the image processor to get pixel values and patch counts
        2. Processes videos using the image processor with max_num_tiles=1 to get video pixel values
        3. Processes audio to compute the number of audio tokens based on duration
        4. Replaces `<image>` tokens in text with `<img>` + image tokens + `</img>` sequences
        5. Replaces `<video>` tokens in text with frame-by-frame descriptions including timestamps (if metadata provided)
        6. Replaces `<audio>` tokens in text with repeated audio tokens based on duration
        7. Tokenizes the processed text and combines all outputs

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence should be a string. The text can contain
                special tokens `<image>`, `<video>`, and `<audio>` that will be replaced with appropriate token sequences.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The video or batch of videos to be prepared. Each video should be a 4D NumPy array or PyTorch
                tensor with shape (num_frames, channels, height, width). Both channels-first and channels-last formats
                are supported. Note: Currently only supports batch size of 1 for videos.
            audio (`str`, `np.ndarray`, `torch.Tensor`, `List[str]`, `List[np.ndarray]`, `List[torch.Tensor]`, *optional*):
                The audio or batch of audio clips to be prepared. Can be file paths, numpy arrays (waveforms),
                or torch tensors. Waveforms should be 1D arrays at the expected sampling rate.
            images_kwargs (`Dict`, *optional*):
                Additional keyword arguments for image processing, including:
                - `min_pixels` (`int`, *optional*): Minimum number of pixels for image processing
                - `max_pixels` (`int`, *optional*): Maximum number of pixels for image processing
                - `patch_size` (`int`, *optional*): Size of patches for image processing
                - `temporal_patch_size` (`int`, *optional*): Size of temporal patches
                - `merge_size` (`int`, *optional*): Size for merging patches
            videos_kwargs (`Dict`, *optional*):
                Additional keyword arguments for video processing, including:
                - `video_metadata` (`VideoMetadata`, *optional*): Metadata containing fps information for timestamp calculation
            audio_kwargs (`Dict`, *optional*):
                Additional keyword arguments for audio processing, including:
                - `sampling_rate` (`int`, *optional*): Target sampling rate for audio
            text_kwargs (`Dict`, *optional*):
                Additional keyword arguments for text tokenization, including:
                - `return_tensors` (`str` or [`~utils.TensorType`], *optional*): Framework for returned tensors ('tf', 'pt', 'np', 'jax')
                - `padding` (`bool`, *optional*): Whether to pad sequences (defaults to False)

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **num_patches** -- Number of patches per image. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **sound_clips** -- Raw audio waveforms to be fed to a model. Returned when `audio` is not `None`.

        Raises:
            AssertionError: If videos are provided with batch size > 1 (not currently supported).

        Note:
            - Image tokens `<image>` in text are replaced with `<img>` + repeated image tokens + `</img>`
            - Video tokens `<video>` in text are replaced with frame-by-frame descriptions
            - Audio tokens `<audio>` in text are replaced with repeated audio placeholder tokens
            - When video metadata with fps is provided, frame descriptions include timestamps
            - Videos are processed with max_num_tiles=1 regardless of the images setting
        """
        output_kwargs = self._merge_kwargs(
            NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        image_inputs, videos_inputs, audio_inputs = {}, {}, {}

        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_num_tokens = image_inputs["num_tokens"]

        if videos is not None:
            # One tile per frame, but sized with the **video** rule (`video_target_num_patches` +
            # aspect ratio) — different from the image rule. We flip a flag on the image processor
            # around the call rather than routing a new kwarg through `ImagesKwargs`, which is a
            # strict-dataclass and rejects unknown fields.
            self.image_processor._is_video_mode = True
            try:
                videos_inputs = self.image_processor(images=videos, **output_kwargs["images_kwargs"])
            finally:
                self.image_processor._is_video_mode = False
            video_num_patches = [sum(videos_inputs["num_patches"])]
            videos_inputs["pixel_values_videos"] = videos_inputs["pixel_values"]
            del videos_inputs["pixel_values"]

        # Process audio inputs
        audio_num_tokens = []
        if audio is not None:
            audio_clips, audio_num_tokens = self._process_audio(audio, output_kwargs.get("audio_kwargs", {}))
            # Keep as list of numpy arrays - don't let BatchFeature convert to tensor
            # The model's generate function will handle conversion
            audio_inputs["sound_clips"] = audio_clips

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place
        if images is not None:
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    n_tokens = image_num_tokens[index]
                    text[i] = text[i].replace(
                        self.image_token,
                        self.image_start_token + "<|placeholder|>" * n_tokens + self.image_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

        if videos is not None:
            assert len(text) == 1, "Video is not supported for batch size > 1"
            video_metadata = output_kwargs.get("videos_kwargs", {}).get("video_metadata", None)
            i = 0
            index = 0
            if self.video_token in text[i]:
                # Matches vLLM's `get_video_repl` (`vllm/transformers_utils/processors/
                # nano_nemotron_vl.py`): one `<img>…</img>` chunk per temporal patch (tubelet),
                # labeled with the per-frame timestamps joined by " and " — capitalized "Frame"
                # for the first frame in the tubelet, lowercase "frame" for the rest. No "This is
                # a video:\n" prefix is emitted by the processor — it's expected to come from the
                # client message, consistent with vLLM and training.
                tokens_per_tubelet = videos_inputs["num_tokens"][0]
                each_group = self.image_start_token + "<|placeholder|>" * tokens_per_tubelet + self.image_end_token
                T = self.video_temporal_patch_dim
                n_frames = video_num_patches[index]
                n_groups = (n_frames + T - 1) // T

                # vLLM formula: int(source_frame_idx) * int(1000 / source_fps) / 1000
                # Requires source fps and source frame indices from video_metadata.
                source_fps = video_metadata.fps if (video_metadata is not None and video_metadata.fps) else None
                frames_indices = video_metadata.frames_indices if video_metadata is not None else None
                if source_fps is not None:
                    frame_duration_ms = int(1000.0 / source_fps)

                frame_labels = []
                for g in range(n_groups):
                    parts = []
                    for j in range(T):
                        fi = g * T + j
                        if fi >= n_frames:
                            break  # last group may be short
                        prefix = "Frame" if j == 0 else "frame"
                        if source_fps is not None and frames_indices is not None and fi < len(frames_indices):
                            ts = int(frames_indices[fi]) * frame_duration_ms / 1000.0
                            parts.append(f"{prefix} {fi + 1} sampled at {ts:.2f} seconds")
                        elif source_fps is not None:
                            ts = fi / source_fps
                            parts.append(f"{prefix} {fi + 1} sampled at {ts:.2f} seconds")
                        else:
                            parts.append(f"{prefix} {fi + 1}")
                    frame_labels.append(" and ".join(parts) + ": ")

                video_prompt = ""
                for g, label in enumerate(frame_labels):
                    if g > 0:
                        video_prompt += "\n"
                    video_prompt += label + each_group

                text[i] = text[i].replace(self.video_token, video_prompt, 1)
            # The tokenizer has no real `<video>` token (the 131081 id in the config doesn't decode
            # to any printable string), so we reuse `<image>` (id 18) as the placeholder. The outer
            # model distinguishes image vs. video by which `pixel_values_*` arg was passed.
            text[i] = text[i].replace("<|placeholder|>", self.image_token)

        # Replace audio tokens with the correct number of placeholder tokens.
        # The expansion loop is per-row, so batch size > 1 is supported as long
        # as `audio_num_tokens` has one entry per `<so_embedding>` placeholder
        # across the batch (in row-major order).
        if audio is not None:
            index = 0
            for i in range(len(text)):
                while self.audio_token in text[i]:
                    num_tokens = audio_num_tokens[index] if index < len(audio_num_tokens) else 1
                    # Replace <audio> with repeated audio tokens
                    text[i] = text[i].replace(
                        self.audio_token,
                        self.audio_start_token + "<|audio_placeholder|>" * num_tokens + self.audio_end_token,
                        1,
                    )
                    index += 1
                text[i] = text[i].replace("<|audio_placeholder|>", self.audio_token)

        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        # Build output - exclude audio from tensor conversion since it's raw waveforms
        output_data = {**text_inputs, **image_inputs, **videos_inputs}
        result = BatchFeature(data=output_data, tensor_type=return_tensors)

        # Add audio clips separately (as list of numpy arrays, not tensors)
        if audio_inputs:
            result["sound_clips"] = audio_inputs["sound_clips"]

        return result

    def _process_audio(self, audio: AudioInput, audio_kwargs: dict) -> tuple:
        """Process audio inputs and compute the number of audio tokens.

        Args:
            audio: Audio input (file path, waveform array, or list thereof)
            audio_kwargs: Additional audio processing arguments

        Returns:
            Tuple of (audio_clips, num_tokens_per_clip)
        """
        # Get sampling rate from kwargs or use default
        sampling_rate = audio_kwargs.get("sampling_rate", self.audio_sampling_rate)

        # Normalize audio to list
        if not isinstance(audio, list):
            audio = [audio]

        audio_clips = []
        num_tokens = []

        for audio_item in audio:
            # Load audio if it's a file path
            if isinstance(audio_item, str):
                waveform = self._load_audio(audio_item, sampling_rate)
            elif isinstance(audio_item, torch.Tensor):
                waveform = audio_item.numpy() if audio_item.dim() == 1 else audio_item.squeeze().numpy()
            elif isinstance(audio_item, np.ndarray):
                waveform = audio_item.squeeze() if audio_item.ndim > 1 else audio_item
            else:
                raise ValueError(f"Unsupported audio type: {type(audio_item)}")

            audio_clips.append(waveform)

            n_tokens = self._estimate_audio_num_embeddings(len(waveform))
            num_tokens.append(max(1, n_tokens))  # At least 1 token

        return audio_clips, num_tokens

    def _estimate_audio_num_embeddings(self, audio_length_samples: int) -> int:
        """Predict the exact number of `<so_embedding>` tokens that the sound
        encoder will produce for an audio clip of `audio_length_samples` raw
        samples. Replaces the previous heuristic, which under-counted by 1 for
        certain lengths and tripped a shape mismatch in `modeling.py::generate`.

        Mirrors `ParakeetFeatureExtractor` (center-padded STFT → ``1 + L // hop``
        mel frames) followed by `ParakeetEncoder._get_subsampling_output_length`
        (``log2(subsampling_factor)`` stages of stride-2 conv with kernel-size
        ``subsampling_conv_kernel_size``, symmetric padding ``(kernel-1)//2`` on
        each side).
        """
        # Mel frame count (center=True is HF Parakeet's default).
        n_frames = 1 + audio_length_samples // self.audio_hop_length
        # Conv-subsampling: ``log2(subsampling_factor)`` stages, stride 2,
        # kernel ``subsampling_conv_kernel_size``, symmetric padding.
        kernel_size = getattr(self, "audio_subsampling_conv_kernel_size", 3)
        stride = getattr(self, "audio_subsampling_conv_stride", 2)
        num_layers = int(math.log2(self.audio_subsampling_factor))
        all_paddings = (kernel_size - 1) // 2 * 2
        add_pad = all_paddings - kernel_size  # kernel=3, sym pad=1 → -1
        L = n_frames
        for _ in range(num_layers):
            L = (L + add_pad) // stride + 1
        return L

    def _load_audio(self, audio_path: str, target_sr: int) -> np.ndarray:
        """Load audio from file and resample if necessary.

        Args:
            audio_path: Path to audio file
            target_sr: Target sampling rate

        Returns:
            Audio waveform as numpy array
        """
        try:
            import librosa

            waveform, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return waveform
        except ImportError:
            pass

        try:
            import soundfile as sf

            waveform, sr = sf.read(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # Convert to mono
            if sr != target_sr:
                # Simple resampling using numpy
                import scipy.signal

                num_samples = int(len(waveform) * target_sr / sr)
                waveform = scipy.signal.resample(waveform, num_samples)
            return waveform.astype(np.float32)
        except ImportError:
            pass

        raise ImportError(
            "Audio loading requires either librosa or soundfile. Install with: pip install librosa soundfile"
        )

    def _get_num_multimodal_tokens(self, image_sizes=None, video_sizes=None, **kwargs):
        """
        Computes the number of placeholder tokens needed for multimodal inputs with the given sizes.
        Args:
            image_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (height, width) per each image.
            video_sizes (`list[list[int]]`, *optional*):
                The input sizes formatted as (num_frames, height, width) per each video.
        Returns:
            `MultiModalData`: A `MultiModalData` object holding number of tokens per each of the provided
            input modalities, along with other useful data.
        """

        vision_data = {}
        if image_sizes is not None:
            images_kwargs = NemotronH_Nano_Omni_Reasoning_V3ProcessorKwargs._defaults.get("images_kwargs", {})
            images_kwargs.update(kwargs)
            merge_size = images_kwargs.get("merge_size", None) or self.image_processor.merge_size

            num_image_patches = [
                self.image_processor.get_number_of_image_patches(*image_size, images_kwargs)
                for image_size in image_sizes
            ]
            num_image_tokens = [(num_patches // merge_size**2) for num_patches in num_image_patches]
            vision_data.update({"num_image_tokens": num_image_tokens, "num_image_patches": num_image_patches})
        return MultiModalData(**vision_data)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to the tokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        """
        Post-process the output of the model to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.
            skip_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to remove special tokens in the output. Argument passed to the tokenizer's `batch_decode` method.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the tokenization spaces. Argument passed to the tokenizer's `batch_decode` method.
            **kwargs:
                Additional arguments to be passed to the tokenizer's `batch_decode method`.

        Returns:
            `list[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        names_from_processor = list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
        return names_from_processor + ["second_per_grid_ts"]


__all__ = ["NemotronH_Nano_Omni_Reasoning_V3Processor"]
