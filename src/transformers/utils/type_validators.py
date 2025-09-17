from collections.abc import Sequence
from dataclasses import MISSING, field, make_dataclass
from typing import Annotated, ForwardRef, Optional, TypedDict, Union, get_args, get_origin

from huggingface_hub.dataclasses import as_validated_field, strict

from ..tokenization_utils_base import PaddingStrategy, TruncationStrategy
from ..video_utils import VideoMetadataType
from .generic import TensorType
from .import_utils import is_vision_available


if is_vision_available():
    from ..image_utils import PILImageResampling


def unpack_annotated_type(type):
    if get_origin(type) is Annotated:
        base, *meta = get_args(type)
        return base, meta[0]
    return type, field(default=MISSING)


def get_type_hints_from_typed_dict(obj: type[TypedDict]):
    """
    Same as `typing.get_type_hints` but does not perform evaluation
    on the ForwardRefs. Evaluating might fails if the package is not imported
    or installed, therefore we will have our own "guarded" type validations.
    All `ForwardRef` will be ignored by the hub validator
    """
    raw_annots = obj.__dict__.get("__annotations__", {})
    type_hints = {}
    for name, value in raw_annots.items():
        if value is None:
            value = type(None)
        if isinstance(value, str):
            value = ForwardRef(value, is_argument=False)
        type_hints[name] = value
    return type_hints


# Minimalistic version of pydantic.TypeAdapter tailored for `TypedDict`
class TypedDictAdapter:
    """
    A utility class used to convert a TypedDict object to dataclass and attach
    a hub validator on top based on TypedDict annotations.

    We don't want to replace `TypedDict` by dataclasses in the codebase because
    with dataclasses we will lose typing hints that `Unpack[TypedDict]` gives.
    So this utility is a sweet spot to keep the balance between DevX and strong
    typing`validation.

    Args:
        type: The TypedDict object that needs to be validated.
    """

    def __init__(
        self,
        type: type[TypedDict],
    ):
        self.type = type
        self.dataclass = self.create_dataclass()
        self.dataclass = strict(self.dataclass)

    def validate_fields(self, **kwargs):
        # If not all kwargs are set, dataclass raises an error in python <= 3.9
        # In newer python we can bypass by creating a dataclass with `kw_only=True`
        for field in self.fields:
            if field[0] not in kwargs:
                kwargs[field[0]] = None
        self.dataclass(**kwargs)

    def create_dataclass(self):
        """
        Creates a dataclass object dynamically from `TypedDict`, so that
        we can use strict type validation from typing hints with `TypedDict`.

        Example:

        @as_validated_field
        def padding_validator(value: Union[bool, str, PaddingStrategy] = None):
            if value is None:
                return
            if not isinstance(value, (bool, str, PaddingStrategy)):
                raise ValueError(f"Value must be one of '[bool, string, PaddingStrategy]'")
            if isinstance(value, str) and value not in ["longest", "max_length", "do_not_pad"]:
                raise ValueError(f'Value for padding must be one of `["longest", "max_length", "do_not_pad"]`')

        class TokenizerKwargs(TypedDict, total=False):
            text: str
            padding: Annotated[Union[bool, str, PaddingStrategy], padding_validator()]

        # Now we can create a dataclass and warp it with hub validators for type constraints
        # The dataclass can also be used as a simple config class for easier kwarg management
        dataclass = dataclass_from_typed_dict(TokenizerKwargs)
        """
        hints = get_type_hints_from_typed_dict(self.type)
        fields = [(k, *unpack_annotated_type(v)) for k, v in hints.items()]
        self.fields = fields
        return make_dataclass(self.type.__name__ + "Config", fields)


@as_validated_field
def positive_any_number(value: Optional[Union[int, float]] = None):
    if value is not None and (not isinstance(value, (int, float)) or not value >= 0):
        raise ValueError(f"Value must be a positive integer or floating number, got {value}")


@as_validated_field
def positive_int(value: Optional[int] = None):
    if value is not None and (not isinstance(value, int) or not value >= 0):
        raise ValueError(f"Value must be a positive integer, got {value}")


@as_validated_field
def padding_validator(value: Optional[Union[bool, str, PaddingStrategy]] = None):
    possible_names = ["longest", "max_length", "do_not_pad"]
    if value is None:
        pass
    elif not isinstance(value, (bool, str, PaddingStrategy)):
        raise ValueError("Value for padding must be either a boolean, a string or a `PaddingStrategy`")
    elif isinstance(value, str) and value not in possible_names:
        raise ValueError(f"If padding is a string, the value must be one of {possible_names}")


@as_validated_field
def truncation_validator(value: Optional[Union[bool, str, TruncationStrategy]] = None):
    possible_names = ["only_first", "only_second", "longest_first", "do_not_truncate"]
    if value is None:
        pass
    elif not isinstance(value, (bool, str, TruncationStrategy)):
        raise ValueError("Value for truncation must be either a boolean, a string or a `TruncationStrategy`")
    elif isinstance(value, str) and value not in possible_names:
        raise ValueError(f"If truncation is a string, value must be one of {possible_names}")


@as_validated_field
def image_size_validator(value: Optional[dict[str, int]] = None):
    possible_keys = ["height", "width", "longest_edge", "shortest_edge", "max_height", "max_width"]
    if value is None:
        pass
    elif not isinstance(value, dict) or any(k not in possible_keys for k in value.keys()):
        raise ValueError(f"Value for size must be a dict with keys {possible_keys} but got size={value}")


@as_validated_field
def device_validator(value: Optional[Union[str, int]] = None):
    possible_names = ["cpu", "cuda", "xla", "xpu", "mps", "meta"]
    if value is None:
        pass
    elif isinstance(value, int) and value < 0:
        raise ValueError(
            f"If device is an integer, the value must be a strictly positive integer but got device={value}"
        )
    elif isinstance(value, str) and value.split(":")[0] not in possible_names:
        raise ValueError(f"If device is an string, the value must be one of {possible_names} but got device={value}")
    elif not isinstance(value, (int, str)):
        raise ValueError(
            f"Device must be either an integer device ID or a string (e.g., 'cpu', 'cuda:0'), but got device={value}"
        )


@as_validated_field
def resampling_validator(value: Optional[Union[int, "PILImageResampling"]] = None):
    if value is None:
        pass
    elif isinstance(value, int) and value not in list(range(6)):
        raise ValueError(
            f"The resampling should be one of {list(range(6))} when provided as integer, but got resampling={value}"
        )
    elif is_vision_available() and not isinstance(value, (PILImageResampling, int)):
        raise ValueError(f"The resampling should an integer or `PIL.Image.Resampling`, but got resampling={value}")


@as_validated_field
def video_metadata_validator(value: Optional[VideoMetadataType] = None):
    if value is None:
        return

    valid_keys = ["total_num_frames", "fps", "width", "height", "duration", "video_backend", "frames_indices"]

    def check_dict_keys(d: dict) -> bool:
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
                    f"Invalid keys found in video metadata. Valid keys: {valid_keys} got: {list(item.keys())}"
                )

    elif isinstance(value, dict):
        if not check_dict_keys(value):
            raise ValueError(
                f"Invalid keys found in video metadata. Valid keys: {valid_keys}, got: {list(value.keys())}"
            )


@as_validated_field
def tensor_type_validator(value: Optional[Union[str, TensorType]] = None):
    possible_names = ["pt", "np", "mlx"]
    if value is None:
        pass
    elif not isinstance(value, str) or value not in possible_names:
        raise ValueError(f"The tensor type should be one of {possible_names} but got tensor_type={value}")
