from transformers import Qwen2VLProcessor

if __name__ == "__main__":

    for i in range(1):
      processor = Qwen2VLProcessor.from_pretrained(pretrained_model_name_or_path="Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True)
      processor


from typing_extensions import Unpack
from transformers.tokenization_utils_base import PaddingStrategy
from typing import Union, TypeVar, Generic, get_type_hints, TypedDict, Literal, Annotated, Optional, get_origin, get_args
from dataclasses import make_dataclass, field

my_int = TypeVar('my_int', bound=int)


class Mixin:
    def mixin_method(self):
        return 0

class Stack(Mixin, Generic[my_int]):
    def __init__(self) -> None:
        # Create an empty list with items of type T
        self.items: list[my_int] = []

    def push(self, item: my_int) -> None:
        self.items.append(item)


class ModelStack(Stack[str]):
    pass

s = ModelStack()
s.push(0)



from dataclasses import dataclass, MISSING, fields
from huggingface_hub.dataclasses import as_validated_field, strict, validated_field

def positive_int(value: int):
    if not value >= 0:
        raise ValueError(f"Value must be positive, got {value}")


def multiple_of_64(value: int):
    if not value % 64 == 0:
        raise ValueError(f"Value must be a multiple of 64, got {value}")


@as_validated_field
def strictly_positive(value: int = None):
    if value is not None and not value > 0:
        raise ValueError(f"Value must be strictly positive, got {value}")

@as_validated_field
def padding_validator(value: Union[bool, str, PaddingStrategy] = None):
    if value is None:
        return

    if not isinstance(value, (bool, str, PaddingStrategy)):
        raise ValueError(f"Value must be padding")
    if isinstance(value, str) and value not in ["longest", "max_length", "do_not_pad"]:
        raise ValueError(f'Value for padding must be one of ["longest", "max_length", "do_not_pad"]')

@strict
@dataclass
class Config:
    model_type: str
    hidden_size: int = validated_field(validator=[positive_int, multiple_of_64])
    vocab_size: int = strictly_positive(default=16)


class AnotherKwargs(TypedDict, total=False):
    name: Union[str, list[str]]
    age: Annotated[Optional[int], strictly_positive()]
    padding: Annotated[Optional[Union[bool, str, PaddingStrategy]], padding_validator()]
    padding_side: Optional[Literal["right", "left"]]


def unpack_annotated_type(type):
    if get_origin(type) is Annotated:
        base, *meta = get_args(type)
        return base, meta[0]
    return type, field(default=MISSING)


def dataclass_from_typed_dict(td: type[TypedDict]):
    hints = get_type_hints(td, include_extras=True)
    dc_fields = [
        (k, *unpack_annotated_type(v))
        for k, v in hints.items()
    ]
    return make_dataclass(td.__name__ + "Config", dc_fields)


class HubTypeAdapter():
    def __init__(self, type: type[TypedDict]) -> None:
        self.type = type
        dataclass = dataclass_from_typed_dict(type)
        self.dataclass = strict(dataclass)
    
    def validate_fields(self, **kwargs):
        for f in fields(self.dataclass):
            if f.name not in kwargs:
                kwargs[f.name] = None
        self.dataclass(**kwargs)
    

config = Config(model_type="bert", vocab_size=30000, hidden_size=768)
print(config.__dataclass_fields__)
assert config.model_type == "bert"
assert config.vocab_size == 30000
assert config.hidden_size == 768

HubTypeAdapter(AnotherKwargs).validate_fields(name=["BOB", "MARY"], age=100, padding=None)
print(AnotherKwargs.__annotations__['age'])

