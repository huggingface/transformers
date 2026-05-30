# Copyright 2024 The HuggingFace Inc. team.
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
Processor class for LocateAnything.
"""

import base64
import importlib.util
import math
import os
import pickle
import re
import time
import warnings
from functools import lru_cache
from io import BytesIO

import numpy as np
import requests
import torch
import torchvision
from packaging import version
from PIL import Image
from torchvision import io, transforms
from torchvision.transforms import InterpolationMode

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput


try:
    from ...image_utils import VideoInput
except ImportError:
    VideoInput = None

from ...processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import auto_docstring, logging


try:
    import cv2
except ImportError:
    cv2 = None

try:
    import lmdb
except ImportError:
    lmdb = None

logger = logging.get_logger(__name__)

__all__ = ["LocateAnythingProcessor"]

FPS = 2.0
MAX_FRAMES = 64
VIDEO_TOTAL_PIXELS = int(float(os.environ.get("VIDEO_MAX_PIXELS", 32000 * 28 * 28 * 0.9)))
logger.debug("set VIDEO_TOTAL_PIXELS: %s", VIDEO_TOTAL_PIXELS)


def to_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
        return white_background
    else:
        return pil_image.convert("RGB")


def read_img_from_lmdb_v2(image_data):
    if lmdb is None or cv2 is None:
        raise ImportError("Reading LocateAnything LMDB images requires both `lmdb` and `opencv-python`.")
    # special case for AgiBotWorld
    lmdb_file, lmdb_key = image_data["lmdb_file"], image_data["lmdb_key"]
    key = lmdb_key.encode("ascii")
    env = lmdb.open(lmdb_file, max_readers=10240, readonly=True, lock=False, readahead=False, meminit=False)
    txn = env.begin()
    value = txn.get(key)
    if value is None:
        logger.warning("Key %s not found in LMDB file %s.", key, lmdb_file)
        return None
    record = pickle.loads(value)
    image_bgr = cv2.imdecode(np.frombuffer(record["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)

    return image


def parse_lmdb_image_data(image_data):
    if lmdb is None or cv2 is None:
        raise ImportError("Reading LocateAnything LMDB images requires both `lmdb` and `opencv-python`.")
    lmdb_file = image_data["lmdb_file"]
    if not os.path.exists(lmdb_file):
        if "/home/zhidingy/workspace/libs/eagle/Eagle2/" in lmdb_file:
            image_data["lmdb_file"] = lmdb_file.replace("/home/zhidingy/workspace/libs/eagle/Eagle2/", "")
        else:
            raise ValueError(f"LMDB file {lmdb_file} does not exist")
    # special case for AgiBotWorld
    if "AgiBotWorld" in image_data["lmdb_file"]:
        return read_img_from_lmdb_v2(image_data)

    try:
        env = lmdb.open(image_data["lmdb_file"], readonly=True, lock=False, max_readers=10240)
    except Exception as e:
        logger.error("Failed to open LMDB file %s. Error message: %s", image_data["lmdb_file"], e)
        raise e

    with env.begin(write=False) as txn:
        try:
            image_bin = txn.get(image_data["lmdb_key"].encode("ascii"))
            buf = BytesIO(image_bin)
        except Exception as e:
            logger.error("Failed to get image from LMDB file %s. Error message: %s", image_data["lmdb_file"], e)
            raise e
    try:
        image = Image.open(buf)
    except Exception:
        image_np = np.frombuffer(image_bin, dtype=np.uint8)
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
    return image


def fetch_image(ele: dict[str, str | Image.Image]) -> Image.Image:
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]
    image_obj = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, dict) and "lmdb_file" in image:
        image_obj = parse_lmdb_image_data(image)
    elif image.startswith("http://") or image.startswith("https://"):
        response = requests.get(image, stream=True)
        image_obj = Image.open(BytesIO(response.content))
    elif image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif image.startswith("data:image"):
        if "base64," in image:
            _, base64_data = image.split("base64,", 1)
            data = base64.b64decode(base64_data)
            image_obj = Image.open(BytesIO(data))
    else:
        image_obj = Image.open(image)
    if image_obj is None:
        raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
    image = to_rgb(image_obj)

    return image


def get_video_frame_indices(
    ele: dict,
    total_frames: int,
    video_fps: int | float,
) -> tuple[torch.Tensor, float]:
    target_fps = ele.get("fps", FPS)
    max_frames = ele.get("max_frames", MAX_FRAMES)

    nframes = (total_frames / video_fps) * target_fps
    nframes = int(round(nframes))
    nframes = max(1, nframes)

    if nframes > max_frames:
        nframes = max_frames

    nframes = min(nframes, total_frames)

    if nframes == total_frames:
        idx = torch.arange(total_frames).long()
    else:
        idx = torch.linspace(0, total_frames - 1, nframes).round().long()

    sample_fps = nframes / max(total_frames, 1e-6) * video_fps

    return idx, sample_fps


def _read_video_torchvision(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using torchvision.io.read_video and return also per-frame timestamps"""
    video_path = ele["video"]
    if version.parse(torchvision.__version__) < version.parse("0.19.0"):
        if "http://" in video_path or "https://" in video_path:
            warnings.warn("torchvision < 0.19.0 does not support http/https video path, please upgrade to 0.19.0.")
        if "file://" in video_path:
            video_path = video_path[7:]
    st = time.time()

    video, audio, info = io.read_video(
        video_path,
        start_pts=ele.get("video_start", 0.0),
        end_pts=ele.get("video_end"),
        pts_unit="sec",
        output_format="TCHW",
    )
    total_frames, video_fps = video.size(0), info["video_fps"]
    logger.debug(
        "torchvision: video_path=%s, total_frames=%s, video_fps=%s, time=%.3fs",
        video_path,
        total_frames,
        video_fps,
        time.time() - st,
    )

    idx, sample_fps = get_video_frame_indices(ele, total_frames, video_fps)

    start_time = ele.get("video_start", 0.0)
    timestamps = (start_time + idx.to(torch.float32) / video_fps).tolist()

    video = video[idx]
    return video, sample_fps, timestamps


def is_decord_available() -> bool:
    return importlib.util.find_spec("decord") is not None


def _read_video_decord(
    ele: dict,
) -> (torch.Tensor, float, list):
    """read video using decord.VideoReader and return also per-frame timestamps"""
    if not is_decord_available():
        raise ImportError("Reading LocateAnything videos with the decord backend requires `decord`.")
    import decord

    video_path = ele["video"]
    st = time.time()
    vr = decord.VideoReader(video_path)

    total_frames, video_fps = len(vr), vr.get_avg_fps()
    logger.debug(
        "decord: video_path=%s, total_frames=%s, video_fps=%s, time=%.3fs",
        video_path,
        total_frames,
        video_fps,
        time.time() - st,
    )

    idx_tensor, sample_fps = get_video_frame_indices(ele, total_frames, video_fps)
    idx = idx_tensor.tolist()

    start_time = ele.get("video_start", 0.0)
    timestamps = [start_time + i / video_fps for i in idx]

    video = vr.get_batch(idx).asnumpy()
    video = torch.tensor(video).permute(0, 3, 1, 2)  # Convert to TCHW format

    return video, sample_fps, timestamps


VIDEO_READER_BACKENDS = {
    "decord": _read_video_decord,
    "torchvision": _read_video_torchvision,
}


@lru_cache(maxsize=1)
def get_video_reader_backend() -> str:
    if is_decord_available():
        video_reader_backend = "decord"
    else:
        video_reader_backend = "torchvision"
    return video_reader_backend


def fetch_video(
    ele: dict, return_video_sample_fps: bool = False, video_reader_backend: str = "torchvision"
) -> torch.Tensor | list[Image.Image]:
    """
    Fetches video, samples frames, resizes based on video_total_pixels, and returns as Tensor (TCHW).
    """
    if isinstance(ele["video"], str):
        video_reader_backend = video_reader_backend if video_reader_backend is not None else get_video_reader_backend()
        try:
            video, sample_fps, timestamps = VIDEO_READER_BACKENDS[video_reader_backend](ele)
        except Exception as e:
            logger.warning(f"video_reader_backend {video_reader_backend} error, use torchvision as default, msg: {e}")
            video, sample_fps, timestamps = VIDEO_READER_BACKENDS["torchvision"](ele)

        nframes, _, height, width = video.shape

        video_total_pixels = ele.get("video_total_pixels", VIDEO_TOTAL_PIXELS)
        current_pixels = nframes * height * width

        if current_pixels > video_total_pixels:
            scale_factor = math.sqrt(video_total_pixels / current_pixels)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)

            video = transforms.functional.resize(
                video,
                [new_height, new_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()
        else:
            video = video.float()

        if return_video_sample_fps:
            return video, sample_fps, timestamps
        return video

    else:
        if not isinstance(ele["video"], (list, tuple)):
            raise ValueError("`video` must be a path/URL, a tensor, or a list/tuple of image-like frames.")
        process_info = ele.copy()
        process_info.pop("type", None)
        process_info.pop("video", None)

        images = [fetch_image({"image": video_element, **process_info}) for video_element in ele["video"]]

        nframes = len(images)
        timestamps = [-1 for i in range(nframes)]

        # For list of images, we return list of PIL images directly,
        # the processor will handle conversion to tensor later.
        if return_video_sample_fps:
            return images, process_info.get("fps", 2.0), timestamps
        return images


class LocateAnythingProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {},
        "videos_kwargs": {},
    }


@auto_docstring
class LocateAnythingProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    valid_processor_kwargs = LocateAnythingProcessorKwargs

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        image_token="<IMG_CONTEXT>",
        video_token="<IMG_CONTEXT>",
        merge_kernel_size=None,
        image_placeholder="image",
        video_placeholder="video",
        image_start_token="<img>",
        image_end_token="</img>",
        **kwargs,
    ):
        r"""
        image_token (`str`, *optional*, defaults to `"<IMG_CONTEXT>"`):
            Token repeated in the prompt for each visual feature.
        video_token (`str`, *optional*, defaults to `"<IMG_CONTEXT>"`):
            Token repeated in the prompt for each video-frame visual feature.
        merge_kernel_size (`list[int]`, *optional*, defaults to `[2, 2]`):
            Spatial merge kernel used to compute the number of visual placeholder tokens.
        image_placeholder (`str`, *optional*, defaults to `"image"`):
            Placeholder prefix used in chat templates before image expansion.
        video_placeholder (`str`, *optional*, defaults to `"video"`):
            Placeholder prefix used in chat templates before video expansion.
        image_start_token (`str`, *optional*, defaults to `"<img>"`):
            Token inserted before expanded image placeholders.
        image_end_token (`str`, *optional*, defaults to `"</img>"`):
            Token inserted after expanded image placeholders.
        """
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
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
        self.image_placeholder = image_placeholder
        self.video_placeholder = video_placeholder
        self.merge_kernel_size = merge_kernel_size if merge_kernel_size is not None else [2, 2]
        self.image_start_token = image_start_token
        self.image_end_token = image_end_token
        if "auto_map" in kwargs:
            self.auto_map = kwargs["auto_map"]
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def replace_media_placeholder(self, text, image_list, video_list, timestamps_list, fps_list, **output_kwargs):
        num_of_images_in_this_sample = 0
        num_of_videos_in_this_sample = 0
        pattern = re.compile(rf"<({self.image_placeholder}|{self.video_placeholder})-(\d+)>")
        unified_frame_list = []

        def replace_in_text(text):
            def repl(match):
                nonlocal unified_frame_list
                nonlocal num_of_images_in_this_sample
                nonlocal num_of_videos_in_this_sample
                media_type = match.group(1)
                idx_in_list = int(match.group(2)) - 1
                idx_mapper = {
                    0: "first",
                    1: "second",
                    2: "third",
                    3: "fourth",
                    4: "fifth",
                    5: "sixth",
                    6: "seventh",
                    7: "eighth",
                    8: "ninth",
                    9: "tenth",
                }

                if media_type == "image":
                    # Call LocateAnythingImageProcessor with a single image in a list
                    image_inputs = self.image_processor(
                        images=[image_list[idx_in_list]], **output_kwargs["images_kwargs"]
                    )

                    num_of_tokens_list = [
                        int(h * w)
                        // (self.image_processor.merge_kernel_size[0] * self.image_processor.merge_kernel_size[1])
                        for h, w in image_inputs["image_grid_hws"]
                    ]

                    special_placeholder = f"<image {idx_in_list + 1}>{self.image_start_token}{self.image_token * num_of_tokens_list[0]}{self.image_end_token}"
                    unified_frame_list.append(image_inputs)
                    num_of_images_in_this_sample += 1

                elif media_type == "video":
                    video_obj = video_list[idx_in_list]

                    # Convert Tensor TCHW to list of PIL Images for the ImageProcessor
                    if isinstance(video_obj, torch.Tensor):
                        # video_obj is [T, C, H, W], float, likely 0-255 or standardized
                        # LocateAnythingImageProcessor expects PIL or 0-255 inputs usually.
                        # We need to convert back to PIL or List[Tensor] compatible with make_list_of_images
                        video_frames = []
                        for i in range(video_obj.shape[0]):
                            frame = video_obj[i]  # [C, H, W]
                            # Assuming fetch_video returns float tensors.
                            # If they are 0-255, convert to uint8.
                            if frame.dtype.is_floating_point and frame.max() > 1.0:
                                frame = frame.byte()
                            elif frame.dtype.is_floating_point:
                                frame = (frame * 255).byte()

                            img = transforms.ToPILImage()(frame)
                            video_frames.append(img)
                    elif isinstance(video_obj, list):
                        # Already list of PIL images
                        video_frames = video_obj
                    else:
                        raise ValueError("Unsupported video format")

                    # Call ImageProcessor with list of frames
                    video_inputs = self.image_processor(images=video_frames, **output_kwargs["videos_kwargs"])

                    # Calculate tokens per frame
                    num_of_tokens_list = [
                        int(h * w)
                        // (self.image_processor.merge_kernel_size[0] * self.image_processor.merge_kernel_size[1])
                        for h, w in video_inputs["image_grid_hws"]
                    ]

                    if timestamps_list is not None and -1 not in timestamps_list:
                        frame_timestamps = timestamps_list[idx_in_list]
                    else:
                        frame_timestamps = None
                    sampled_fps = fps_list[idx_in_list] if fps_list is not None else None

                    if frame_timestamps is not None:
                        # Ensure lengths match (sometimes rounding might cause off-by-one if not careful, but usually safe here)
                        if len(frame_timestamps) != len(num_of_tokens_list):
                            logger.warning(f"Timestamp mismatch: {len(frame_timestamps)} vs {len(num_of_tokens_list)}")
                            min_len = min(len(frame_timestamps), len(num_of_tokens_list))
                            frame_timestamps = frame_timestamps[:min_len]
                            num_of_tokens_list = num_of_tokens_list[:min_len]

                        special_placeholder = [
                            f"Frame-{i + 1}-{frame_timestamps[i]:.2f}s: {self.image_start_token}{self.image_token * num_of_tokens}{self.image_end_token}"
                            for i, num_of_tokens in enumerate(num_of_tokens_list)
                        ]
                    else:
                        special_placeholder = [
                            f"Frame-{i + 1}: {self.image_start_token}{self.image_token * num_of_tokens}{self.image_end_token}"
                            for i, num_of_tokens in enumerate(num_of_tokens_list)
                        ]

                    if sampled_fps is not None:
                        special_placeholder = (
                            f"The {idx_mapper[idx_in_list]} video sampled with {sampled_fps:.2f} fps: "
                            + "".join(special_placeholder)
                        )
                    else:
                        special_placeholder = f"The {idx_mapper[idx_in_list]} video: " + "".join(special_placeholder)

                    unified_frame_list.append(video_inputs)
                    num_of_videos_in_this_sample += 1
                else:
                    raise ValueError(f"Unknown media type: {media_type}")
                return special_placeholder

            return pattern.sub(repl, text)

        text = replace_in_text(text)

        if len(unified_frame_list) > 0:
            # Concatenate all pixel values from all images/videos in this sample
            pixel_values = torch.cat([frame["pixel_values"] for frame in unified_frame_list], dim=0)
            # Concatenate grid hws
            image_grid_hws = np.concatenate([frame["image_grid_hws"] for frame in unified_frame_list], axis=0)
        else:
            pixel_values = torch.empty(0)
            image_grid_hws = np.empty(0)

        return text, pixel_values, image_grid_hws, num_of_images_in_this_sample, num_of_videos_in_this_sample

    def __call__(
        self,
        images: ImageInput = None,
        text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] = None,
        audio=None,
        videos: VideoInput = None,
        **kwargs: Unpack[LocateAnythingProcessorKwargs],
    ) -> BatchFeature:
        output_kwargs = self._merge_kwargs(
            LocateAnythingProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if text is None:
            raise ValueError("You have to specify `text`.")
        if isinstance(text, str):
            text_list = [text]
        elif isinstance(text, list) and text and all(isinstance(sample, str) for sample in text):
            text_list = text
        else:
            raise TypeError("Invalid input text. Please provide a string, or a non-empty list of strings.")

        if images is None:
            images = []
        if videos is None:
            videos = []

        pixel_values_list = []
        image_grid_hws_list = []
        new_sample_list = []
        image_start_idx = 0
        video_start_idx = 0
        timestamps_batch = output_kwargs["videos_kwargs"].pop("timestamps", None)
        fps_batch = output_kwargs["videos_kwargs"].pop("fps", None)

        for sample in text_list:
            timestamps_list = timestamps_batch[video_start_idx:] if timestamps_batch is not None else None
            fps_list = fps_batch[video_start_idx:] if fps_batch is not None else None

            sample, pixel_values, image_grid_hws, num_of_images_in_this_sample, num_of_videos_in_this_sample = (
                self.replace_media_placeholder(
                    sample,
                    images[image_start_idx:],
                    videos[video_start_idx:],
                    timestamps_list,
                    fps_list,
                    **output_kwargs,
                )
            )
            new_sample_list.append(sample)

            if pixel_values.numel() > 0:
                pixel_values_list.append(pixel_values)
                image_grid_hws_list.append(image_grid_hws)

            image_start_idx += num_of_images_in_this_sample
            video_start_idx += num_of_videos_in_this_sample

        image_inputs = {}
        if len(pixel_values_list) > 0:
            # Concatenate across the batch
            image_inputs["pixel_values"] = torch.cat(pixel_values_list, dim=0)
            image_inputs["image_grid_hws"] = np.concatenate(image_grid_hws_list, axis=0)

        video_inputs = {}  # Video data is merged into image_inputs now
        text_inputs = self.tokenizer(new_sample_list, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **video_inputs})

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def save_pretrained(self, save_directory, **kwargs):
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
        os.makedirs(save_directory, exist_ok=True)
        outputs = super().save_pretrained(save_directory, **kwargs)
        return outputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        if isinstance(processor, tuple):
            processor = processor[0]
        return processor

    def process_vision_info(
        self,
        conversations: list[dict] | list[list[dict]],
        return_video_kwargs: bool = False,
        video_reader_backend: str = "torchvision",
    ) -> tuple[list[Image.Image] | None, list[torch.Tensor | list[Image.Image]] | None, dict | None]:
        vision_infos = self.extract_vision_info(conversations)
        image_inputs = []
        video_inputs = []
        video_sample_fps_list = []
        video_timestamps_list = []

        for vision_info in vision_infos:
            if "image" in vision_info or "image_url" in vision_info:
                image_inputs.append(fetch_image(vision_info))
            elif "video" in vision_info:
                video_input, video_sample_fps, video_timestamps = fetch_video(
                    vision_info, return_video_sample_fps=True, video_reader_backend=video_reader_backend
                )
                video_sample_fps_list.append(video_sample_fps)
                video_inputs.append(video_input)
                video_timestamps_list.append(video_timestamps)
            else:
                raise ValueError("image, image_url or video should in content.")

        if len(image_inputs) == 0:
            image_inputs = None
        if len(video_inputs) == 0:
            video_inputs = None

        if return_video_kwargs:
            return image_inputs, video_inputs, {"fps": video_sample_fps_list, "timestamps": video_timestamps_list}
        return image_inputs, video_inputs

    def extract_vision_info(self, conversations: list[dict] | list[list[dict]]) -> list[dict]:
        vision_infos = []
        if isinstance(conversations[0], dict):
            conversations = [conversations]
        for conversation in conversations:
            for message in conversation:
                if isinstance(message["content"], list):
                    for ele in message["content"]:
                        if (
                            "image" in ele
                            or "image_url" in ele
                            or "video" in ele
                            or ele["type"] in ("image", "image_url", "video")
                        ):
                            vision_infos.append(ele)
        return vision_infos

    def py_apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise ValueError("`py_apply_chat_template` does not support `tokenize=True` yet.")
        result = ""
        image_count = 0
        video_count = 0

        message_text = ""
        for idx, message in enumerate(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message_text += content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        message_text += item["text"]
                    elif isinstance(item, str):
                        message_text += item

        for idx, message in enumerate(messages):
            if idx == 0 and message.get("role") != "system":
                result += "<|im_start|>system\n"
                result += "You are a helpful assistant.\n"
                result += "<|im_end|>\n"

            result += f"<|im_start|>{message.get('role', '')}\n"
            content = message.get("content")

            if isinstance(content, str):
                result += content
                result += "<|im_end|>\n"
            else:
                for item in content:
                    if isinstance(item, dict) and (
                        item.get("type") == "image" or "image" in item or "image_url" in item
                    ):
                        image_count += 1
                        candidate_token = f"<image-{image_count}>"
                        if candidate_token not in message_text:
                            result += candidate_token
                    elif isinstance(item, dict) and (item.get("type") == "video" or "video" in item):
                        video_count += 1
                        candidate_token = f"<video-{video_count}>"
                        if candidate_token not in message_text:
                            result += candidate_token
                    elif isinstance(item, dict) and "text" in item:
                        result += item["text"]
                    elif isinstance(item, str):
                        result += item
                result += "<|im_end|>\n"

        if add_generation_prompt:
            result += "<|im_start|>assistant\n"

        return result


__all__ = ["LocateAnythingProcessor"]
