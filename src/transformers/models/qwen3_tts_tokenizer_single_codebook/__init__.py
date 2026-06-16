from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure

_import_structure = {
    "configuration_qwen3_tts_tokenizer_single_codebook": [
        "Qwen3TTSTokenizerSingleCodebookConfig",
        "Qwen3TTSTokenizerSingleCodebookDiTConfig",
        "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANConfig",
        "Qwen3TTSTokenizerSingleCodebookDecoderConfig",
        "Qwen3TTSTokenizerSingleCodebookEncoderConfig",
    ],
    "feature_extraction_qwen3_tts_tokenizer_single_codebook": [
        "Qwen3TTSTokenizerSingleCodebookFeatureExtractor",
    ],
    "modeling_qwen3_tts_tokenizer_single_codebook": [
        "Qwen3TTSTokenizerSingleCodebookPreTrainedModel",
        "Qwen3TTSTokenizerSingleCodebookModel",
        "Qwen3TTSTokenizerSingleCodebookEncoderModel",
        "Qwen3TTSTokenizerSingleCodebookEncoder",
        "Qwen3TTSTokenizerSingleCodebookEncoderPreTrainedModel",
        "Qwen3TTSTokenizerSingleCodebookDecoder",
        "Qwen3TTSTokenizerSingleCodebookDecoderPreTrainedModel",
        "Qwen3TTSTokenizerSingleCodebookDecoderModel",
        "Qwen3TTSTokenizerSingleCodebookDecoderDiTModel",
        "Qwen3TTSTokenizerSingleCodebookDecoderBigVGANModel",
        "Qwen3TTSTokenizerSingleCodebookAMPBlock",
        "Qwen3TTSTokenizerSingleCodebookCausalConv1d",
        "Qwen3TTSTokenizerSingleCodebookVectorQuantization",
        "Qwen3TTSTokenizerSingleCodebookEuclideanCodebook",
    ],
}


if TYPE_CHECKING:
    from .configuration_qwen3_tts_tokenizer_single_codebook import *
    from .feature_extraction_qwen3_tts_tokenizer_single_codebook import *
    from .modeling_qwen3_tts_tokenizer_single_codebook import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, _import_structure, module_spec=__spec__)
