from typing import TYPE_CHECKING

from ...utils import (
    _LazyModule,
)


if TYPE_CHECKING:
    from .configuration_deepseek_vl_v2 import (
        DeepseekVLV2Config,
        MlpProjectorConfig,
    )
    from .modeling_deepseek_vl_v2 import (
        DeepseekVLV2ForCausalLM,
        DeepseekVLV2Model,
        DeepseekVLV2PreTrainedModel,
    )
    from .processing_deepseek_vl_v2 import DeepseekVLV2Processor


else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        {
            "configuration_deepseek_vl_v2": [
                "DeepseekVLV2Config",
                "MlpProjectorConfig",
            ],
            "modeling_deepseek_vl_v2": [
                "DeepseekVLV2ForCausalLM",
                "DeepseekVLV2Model",
                "DeepseekVLV2PreTrainedModel",
            ],
            "processing_deepseek_vl_v2": ["DeepseekVLV2Processor"],
        },
        module_spec=__spec__,
    )
