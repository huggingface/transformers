from collections.abc import Sequence
from typing import Any, Union, cast

from ..tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ..video_utils import VideoMetadataType
from .generic import TensorType
from .import_utils import is_torch_available, is_vision_available


if is_vision_available():
    from ..image_utils import PILImageResampling

if is_torch_available():
    import torch


def positive_any_number(value: int | float | None = None):
    if value is not None and (not isinstance(value, (int, float)) or not value >= 0):
        raise ValueError(f"Value must be a positive integer or floating number, got {value}")


def positive_int(value: int | None = None):
    if value is not None and (not isinstance(value, int) or not value >= 0):
        raise ValueError(f"Value must be a positive integer, got {value}")


def padding_validator(value: bool | str | PaddingStrategy | None = None):
    possible_names = ["longest", "max_length", "do_not_pad"]
    if value is None:
        pass
    elif not isinstance(value, (bool, str, PaddingStrategy)):
        raise ValueError("Value for padding must be either a boolean, a string or a `PaddingStrategy`")
    elif isinstance(value, str) and value not in possible_names:
        raise ValueError(f"If padding is a string, the value must be one of {possible_names}")


def truncation_validator(value: bool | str | TruncationStrategy | None = None):
    possible_names = ["only_first", "only_second", "longest_first", "do_not_truncate"]
    if value is None:
        pass
    elif not isinstance(value, (bool, str, TruncationStrategy)):
        raise ValueError("Value for truncation must be either a boolean, a string or a `TruncationStrategy`")
    elif isinstance(value, str) and value not in possible_names:
        raise ValueError(f"If truncation is a string, value must be one of {possible_names}")


def image_size_validator(value: int | Sequence[int] | dict[str, int] | None = None):
    possible_keys = ["height", "width", "longest_edge", "shortest_edge", "max_height", "max_width"]
    if value is None:
        pass
    elif isinstance(value, dict) and any(k not in possible_keys for k in value.keys()):
        raise ValueError(f"Value for size must be a dict with keys {possible_keys} but got size={value}")


def device_validator(value: str | int | None = None):
    possible_names = ["cpu", "cuda", "xla", "xpu", "mps", "meta"]
    if value is None:
        pass
    elif is_torch_available() and isinstance(value, torch.device):
        # Convert torch.device to string for validation
        device_str = str(value)
        if device_str.split(":")[0] not in possible_names:
            raise ValueError(
                f"If device is a torch.device, the value must be one of {possible_names} but got device={value}"
            )
    elif isinstance(value, int) and value < 0:
        raise ValueError(
            f"If device is an integer, the value must be a strictly positive integer but got device={value}"
        )
    elif isinstance(value, str) and value.split(":")[0] not in possible_names:
        raise ValueError(f"If device is an string, the value must be one of {possible_names} but got device={value}")
    elif not isinstance(value, (int, str)):
        raise ValueError(
            f"Device must be either an integer device ID, a string (e.g., 'cpu', 'cuda:0'), or a torch.device object, but got device={value}"
        )


def resampling_validator(value: Union[int, "PILImageResampling"] | None = None):
    if value is None:
        pass
    elif isinstance(value, int) and value not in list(range(6)):
        raise ValueError(
            f"The resampling should be one of {list(range(6))} when provided as integer, but got resampling={value}"
        )
    elif is_vision_available() and not isinstance(value, (PILImageResampling, int)):
        raise ValueError(f"The resampling should an integer or `PIL.Image.Resampling`, but got resampling={value}")


def video_metadata_validator(value: VideoMetadataType | None = None):
    if value is None:
        return

    valid_keys = ["total_num_frames", "fps", "width", "height", "duration", "video_backend", "frames_indices"]

    def check_dict_keys(d: dict[str, Any]) -> bool:
        return all(key in valid_keys for key in d.keys())

    if isinstance(value, Sequence) and isinstance(value[0], Sequence) and isinstance(value[0][0], dict):
        for sublist in value:
            for item in sublist:
                if not check_dict_keys(item):
                    raise ValueError(
                        f"Invalid keys found in video metadata. Valid keys: {valid_keys} got: {list(item.keys())}"
                    )

    elif isinstance(value, Sequence) and isinstance(value[0], dict):
        for item in value:
            if not check_dict_keys(item):
                raise ValueError(
                    f"Invalid keys found in video metadata. Valid keys: {valid_keys} got: {list(cast(dict, item).keys())}"
                )

    elif isinstance(value, dict):
        if not check_dict_keys(value):
            raise ValueError(
                f"Invalid keys found in video metadata. Valid keys: {valid_keys}, got: {list(value.keys())}"
            )


def tensor_type_validator(value: str | TensorType | None = None):
    possible_names = ["pt", "np", "mlx"]
    if value is None:
        pass
    elif not isinstance(value, str) or value not in possible_names:
        raise ValueError(f"The tensor type should be one of {possible_names} but got tensor_type={value}")
