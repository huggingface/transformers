from transformers.models.lasr.processing_lasr import LasrProcessor
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import ProcessingKwargs
from transformers.utils.deprecation import deprecate_attr


class DeprecateAttrConfig(LlamaConfig):
    pass


class DeprecateAttrProcessor(LasrProcessor):
    pass


__getattr__ = deprecate_attr(
    __name__,
    {
        "DeprecateAttrOldConfig": (DeprecateAttrConfig, "6.0.0"),
        "DeprecateAttrOldProcessor": (DeprecateAttrProcessor, "6.0.0"),
        "DeprecateAttrOldProcessorKwargs": (ProcessingKwargs, "6.0.0"),
    },
)


__all__ = ["DeprecateAttrConfig", "DeprecateAttrProcessor"]
