# Copyright 2023 The HuggingFace Inc. team.
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
Utility that checks all docstrings of public objects have an argument section matching their signature.

Use from the root of the repo with:

```bash
python utils/check_docstrings.py
```

for a check that will error in case of inconsistencies (used by `make check-repo`).

To auto-fix issues run:

```bash
python utils/check_docstrings.py --fix_and_overwrite
```

which is used by `make fix-repo` (note that this fills what it cans, you might have to manually fill information
like argument descriptions).
"""

import argparse
import ast
import enum
import glob
import inspect
import operator as op
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from check_repo import ignore_undocumented
from git import Repo

from transformers import logging
from transformers.utils import direct_transformers_import
from transformers.utils.auto_docstring import (
    ImageProcessorArgs,
    ModelArgs,
    ModelOutputArgs,
    ProcessorArgs,
    get_args_doc_from_source,
    parse_docstring,
    set_min_indent,
)


logger = logging.get_logger(__name__)


@dataclass
class DecoratedItem:
    """Information about a single @auto_docstring decorated function or class."""

    decorator_line: int  # 1-based line number of the decorator
    def_line: int  # 1-based line number of the def/class statement
    kind: str  # 'function' or 'class'
    body_start_line: (
        int  # 1-based line number where body starts (for functions) or __init__ body start (for classes with __init__)
    )
    args: list[str]  # List of argument names (excluding self, *args, **kwargs) - for classes, these are __init__ args
    custom_args_text: str | None = None  # custom_args string if provided in decorator

    # Class-specific fields (only populated when kind == 'class')
    has_init: bool = False  # Whether the class has an __init__ method
    init_def_line: int | None = None  # 1-based line number of __init__ def (if has_init)
    is_model_output: bool = False  # Whether the class inherits from ModelOutput
    is_processor: bool = False  # Whether the class inherits from ProcessorMixin


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
PATH_TO_TRANSFORMERS = Path("src").resolve() / "transformers"

# This is to make sure the transformers module imported is the one in the repo.
transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)

OPTIONAL_KEYWORD = "*optional*"
# Re pattern that catches args blocks in docstrings (with all variation around the name supported).
_re_args = re.compile(r"^\s*(Args?|Arguments?|Attributes?|Params?|Parameters?):\s*$")
# Re pattern that parses the start of an arg block: catches <name> (<description>) in those lines.
_re_parse_arg = re.compile(r"^(\s*)(\S+)\s+\((.+)\)(?:\:|$)")
# Re pattern that parses the end of a description of an arg (catches the default in *optional*, defaults to xxx).
_re_parse_description = re.compile(r"\*optional\*, defaults to (.*)$")
# Args that are always overridden in the docstring, for clarity we don't want to remove them from the docstring
ALWAYS_OVERRIDE = ["labels"]

# This is a temporary set of objects to ignore while we progressively fix them. Do not add anything here, fix the
# docstrings instead. If formatting should be ignored for the docstring, you can put a comment # no-format on the
# line before the docstring.
OBJECTS_TO_IGNORE = {
    "GlmMoeDsaConfig",
    "GlmAsrProcessor",
    "AudioFlamingo3Processor",
    "ApertusConfig",
    "Mxfp4Config",
    "Qwen3OmniMoeConfig",
    "Exaone4Config",
    "SmolLM3Config",
    "Gemma3nVisionConfig",
    "Llama4Processor",
    # Deprecated
    "InputExample",
    "InputFeatures",
    # Missing arguments in the docstring
    "ASTFeatureExtractor",
    "AlbertModel",
    "AlbertTokenizerFast",
    "AlignTextModel",
    "AlignVisionConfig",
    "AudioClassificationPipeline",
    "AutoformerConfig",
    "AutomaticSpeechRecognitionPipeline",
    "BarkCoarseConfig",
    "BarkConfig",
    "BarkFineConfig",
    "BarkSemanticConfig",
    "BartConfig",
    "BartTokenizerFast",
    "BarthezTokenizerFast",
    "BeitModel",
    "BertConfig",
    "BertJapaneseTokenizer",
    "CohereTokenizer",
    "DebertaTokenizer",
    "FNetTokenizer",
    "FunnelTokenizer",
    "GPT2Tokenizer",
    "GPTNeoXTokenizer",
    "GemmaTokenizer",
    "HerbertTokenizer",
    "LayoutLMv2Tokenizer",
    "LayoutLMv3Tokenizer",
    "LayoutXLMTokenizer",
    "LlamaTokenizer",
    "LlamaTokenizerFast",
    "MBart50Tokenizer",
    "NougatTokenizer",
    "OpenAIGPTTokenizer",
    "PythonBackend",
    "ReformerTokenizer",
    "SeamlessM4TTokenizer",
    "SentencePieceBackend",
    "SplinterTokenizer",
    "TokenizersBackend",
    "UdopTokenizer",
    "WhisperTokenizer",
    "XGLMTokenizer",
    "XLMRobertaTokenizer",
    "AlbertTokenizer",
    "BarthezTokenizer",
    "BigBirdTokenizer",
    "BlenderbotTokenizer",
    "CamembertTokenizer",
    "CodeLlamaTokenizer",
    "CodeLlamaTokenizerFast",
    "BertModel",
    "BertTokenizerFast",
    "BigBirdConfig",
    "BigBirdForQuestionAnswering",
    "BigBirdModel",
    "BigBirdPegasusConfig",
    "BigBirdTokenizerFast",
    "BitImageProcessor",
    "BlenderbotConfig",
    "BlenderbotSmallConfig",
    "BlenderbotSmallTokenizerFast",
    "BlenderbotTokenizerFast",
    "Blip2VisionConfig",
    "BlipTextConfig",
    "BlipVisionConfig",
    "BloomConfig",
    "BLTConfig",
    "BLTPatcherConfig",
    "BridgeTowerTextConfig",
    "BridgeTowerVisionConfig",
    "BrosModel",
    "CamembertConfig",
    "CamembertModel",
    "CamembertTokenizerFast",
    "CanineModel",
    "CanineTokenizer",
    "ChineseCLIPTextModel",
    "ClapTextConfig",
    "ConditionalDetrConfig",
    "ConditionalDetrImageProcessor",
    "ConvBertConfig",
    "ConvBertTokenizerFast",
    "ConvNextConfig",
    "ConvNextV2Config",
    "CpmAntTokenizer",
    "CvtConfig",
    "CvtModel",
    "DeiTImageProcessor",
    "DPRReaderTokenizer",
    "DPRReaderTokenizerFast",
    "DPTModel",
    "Data2VecAudioConfig",
    "Data2VecTextConfig",
    "Data2VecTextModel",
    "Data2VecVisionModel",
    "DataCollatorForLanguageModeling",
    "DebertaConfig",
    "DebertaV2Config",
    "DebertaV2Tokenizer",
    "DebertaV2TokenizerFast",
    "DecisionTransformerConfig",
    "DeformableDetrConfig",
    "DeformableDetrImageProcessor",
    "DeiTModel",
    "DepthEstimationPipeline",
    "DetaConfig",
    "DetaImageProcessor",
    "DetrConfig",
    "DetrImageProcessor",
    "DinatModel",
    "DINOv3ConvNextConfig",
    "DINOv3ViTConfig",
    "DistilBertConfig",
    "DistilBertTokenizerFast",
    "DocumentQuestionAnsweringPipeline",
    "DonutSwinModel",
    "EarlyStoppingCallback",
    "EfficientFormerConfig",
    "EfficientFormerImageProcessor",
    "EfficientNetConfig",
    "ElectraConfig",
    "ElectraTokenizerFast",
    "EncoderDecoderModel",
    "ErnieMModel",
    "ErnieModel",
    "ErnieMTokenizer",
    "EsmConfig",
    "EsmModel",
    "FNetConfig",
    "FNetModel",
    "FNetTokenizerFast",
    "FSMTConfig",
    "FeatureExtractionPipeline",
    "FillMaskPipeline",
    "FlaubertConfig",
    "FlavaConfig",
    "FlavaForPreTraining",
    "FlavaImageModel",
    "FlavaImageProcessor",
    "FlavaMultimodalModel",
    "FlavaTextConfig",
    "FlavaTextModel",
    "FocalNetModel",
    "FunnelTokenizerFast",
    "GPTBigCodeConfig",
    "GPTJConfig",
    "GPTNeoXConfig",
    "GPTNeoXJapaneseConfig",
    "GPTNeoXTokenizerFast",
    "GPTSanJapaneseConfig",
    "GitConfig",
    "GitVisionConfig",
    "Glm4vVisionConfig",
    "Glm4vMoeVisionConfig",
    "GraphormerConfig",
    "GroupViTTextConfig",
    "GroupViTVisionConfig",
    "HerbertTokenizerFast",
    "HubertConfig",
    "HubertForCTC",
    "IBertConfig",
    "IBertModel",
    "IdeficsConfig",
    "IdeficsProcessor",
    "IJepaModel",
    "ImageClassificationPipeline",
    "ImageFeatureExtractionPipeline",
    "ImageGPTConfig",
    "ImageSegmentationPipeline",
    "ImageTextToTextPipeline",
    "AnyToAnyPipeline",
    "ImageToImagePipeline",
    "InformerConfig",
    "JukeboxPriorConfig",
    "JukeboxTokenizer",
    "LEDConfig",
    "LEDTokenizerFast",
    "LasrEncoderConfig",
    "LasrFeatureExtractor",
    "LasrTokenizer",
    "LayoutLMForQuestionAnswering",
    "LayoutLMTokenizerFast",
    "LayoutLMv2Config",
    "LayoutLMv2ForQuestionAnswering",
    "LayoutLMv2TokenizerFast",
    "LayoutLMv3Config",
    "LayoutLMv3ImageProcessor",
    "LayoutLMv3TokenizerFast",
    "LayoutXLMTokenizerFast",
    "LevitConfig",
    "LiltConfig",
    "LiltModel",
    "LongT5Config",
    "LongformerConfig",
    "LongformerModel",
    "LongformerTokenizerFast",
    "LukeModel",
    "LukeTokenizer",
    "LxmertTokenizerFast",
    "M2M100Config",
    "M2M100Tokenizer",
    "MarkupLMProcessor",
    "MaskGenerationPipeline",
    "MBart50TokenizerFast",
    "MBartConfig",
    "MCTCTFeatureExtractor",
    "MPNetConfig",
    "MPNetModel",
    "MPNetTokenizerFast",
    "MT5Config",
    "MT5TokenizerFast",
    "MarianConfig",
    "MarianTokenizer",
    "MarkupLMConfig",
    "MarkupLMModel",
    "MarkupLMTokenizer",
    "MarkupLMTokenizerFast",
    "Mask2FormerConfig",
    "MaskFormerConfig",
    "MaxTimeCriteria",
    "MegaConfig",
    "MegaModel",
    "MegatronBertConfig",
    "MegatronBertForPreTraining",
    "MegatronBertModel",
    "MLCDVisionConfig",
    "MobileBertConfig",
    "MobileBertModel",
    "MobileBertTokenizerFast",
    "MobileNetV1ImageProcessor",
    "MobileNetV1Model",
    "MobileNetV2ImageProcessor",
    "MobileNetV2Model",
    "MobileViTModel",
    "MobileViTV2Model",
    "MLukeTokenizer",
    "MraConfig",
    "MusicgenDecoderConfig",
    "MusicgenForConditionalGeneration",
    "MusicgenMelodyForConditionalGeneration",
    "MvpConfig",
    "MvpTokenizerFast",
    "MT5Tokenizer",
    "NatModel",
    "NerPipeline",
    "NezhaConfig",
    "NezhaModel",
    "NllbMoeConfig",
    "NllbTokenizer",
    "NllbTokenizerFast",
    "NystromformerConfig",
    "OPTConfig",
    "ObjectDetectionPipeline",
    "OneFormerProcessor",
    "OpenAIGPTTokenizerFast",
    "OpenLlamaConfig",
    "PLBartConfig",
    "ParakeetCTCConfig",
    "LasrCTCConfig",
    "PegasusConfig",
    "PegasusTokenizer",
    "PegasusTokenizerFast",
    "PegasusXConfig",
    "PerceiverImageProcessor",
    "PerceiverModel",
    "PerceiverTokenizer",
    "PersimmonConfig",
    "Pipeline",
    "Pix2StructConfig",
    "Pix2StructTextConfig",
    "PLBartTokenizer",
    "Pop2PianoConfig",
    "PreTrainedTokenizer",
    "PreTrainedTokenizerBase",
    "PreTrainedTokenizerFast",
    "PrefixConstrainedLogitsProcessor",
    "ProphetNetConfig",
    "QDQBertConfig",
    "QDQBertModel",
    "QuestionAnsweringPipeline",
    "RagConfig",
    "RagModel",
    "RagRetriever",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
    "ReformerConfig",
    "ReformerTokenizerFast",
    "RegNetConfig",
    "RemBertConfig",
    "RemBertModel",
    "RemBertTokenizer",
    "RemBertTokenizerFast",
    "RetriBertConfig",
    "RetriBertTokenizerFast",
    "RoCBertConfig",
    "RoCBertModel",
    "RoCBertTokenizer",
    "RoFormerConfig",
    "RobertaConfig",
    "RobertaModel",
    "RobertaPreLayerNormConfig",
    "RobertaPreLayerNormModel",
    "RobertaTokenizerFast",
    "SEWConfig",
    "SEWDConfig",
    "SEWDForCTC",
    "SEWForCTC",
    "SamConfig",
    "SamPromptEncoderConfig",
    "SamHQConfig",
    "SamHQPromptEncoderConfig",
    "SeamlessM4TConfig",  # use of unconventional markdown
    "SeamlessM4Tv2Config",  # use of unconventional markdown
    "Seq2SeqTrainingArguments",
    "Speech2Text2Config",
    "Speech2Text2Tokenizer",
    "Speech2TextTokenizer",
    "SpeechEncoderDecoderModel",
    "SpeechT5Config",
    "SpeechT5Model",
    "SplinterConfig",
    "SplinterTokenizerFast",
    "SqueezeBertTokenizerFast",
    "Swin2SRImageProcessor",
    "Swinv2Model",
    "SwitchTransformersConfig",
    "T5Config",
    "T5Tokenizer",
    "T5TokenizerFast",
    "TableQuestionAnsweringPipeline",
    "TableTransformerConfig",
    "TapasConfig",
    "TapasModel",
    "TapasTokenizer",
    "TextClassificationPipeline",
    "TextGenerationPipeline",
    "TimeSeriesTransformerConfig",
    "TokenClassificationPipeline",
    "TrOCRConfig",
    "Phi4MultimodalProcessor",
    "TrainerState",
    "TrainingArguments",
    "TrajectoryTransformerConfig",
    "TvltImageProcessor",
    "UMT5Config",
    "UperNetConfig",
    "UperNetForSemanticSegmentation",
    "ViTHybridImageProcessor",
    "ViTHybridModel",
    "ViTMSNModel",
    "ViTModel",
    "VideoClassificationPipeline",
    "ViltConfig",
    "ViltForImagesAndTextClassification",
    "ViltModel",
    "VisionEncoderDecoderModel",
    "VisionTextDualEncoderModel",
    "VisualBertConfig",
    "VisualBertModel",
    "VisualQuestionAnsweringPipeline",
    "VitMatteForImageMatting",
    "VitsTokenizer",
    "VivitModel",
    "Wav2Vec2BertForCTC",
    "Wav2Vec2CTCTokenizer",
    "Wav2Vec2Config",
    "Wav2Vec2ConformerConfig",
    "Wav2Vec2ConformerForCTC",
    "Wav2Vec2FeatureExtractor",
    "Wav2Vec2PhonemeCTCTokenizer",
    "WavLMConfig",
    "WavLMForCTC",
    "WhisperConfig",
    "WhisperFeatureExtractor",
    "WhisperForAudioClassification",
    "XCLIPTextConfig",
    "XCLIPVisionConfig",
    "XGLMConfig",
    "XGLMModel",
    "XGLMTokenizerFast",
    "XLMConfig",
    "XLMProphetNetConfig",
    "XLMRobertaConfig",
    "XLMRobertaModel",
    "XLMRobertaTokenizerFast",
    "XLMRobertaXLConfig",
    "XLMRobertaXLModel",
    "XLNetConfig",
    "XLNetTokenizerFast",
    "XmodConfig",
    "XmodModel",
    "YolosImageProcessor",
    "YolosModel",
    "YosoConfig",
    "ZeroShotAudioClassificationPipeline",
    "ZeroShotClassificationPipeline",
    "ZeroShotImageClassificationPipeline",
    "ZeroShotObjectDetectionPipeline",
    "Llama4TextConfig",
    "BltConfig",
    "BltPatcherConfig",
    "HiggsAudioV2Config",
    "HiggsAudioV2TokenizerConfig",
    "MoonshineStreamingConfig",
    "MoonshineStreamingEncoderConfig",
    "VoxtralRealtimeFeatureExtractor",
    "VoxtralRealtimeEncoderConfig",
}
# In addition to the objects above, we also ignore objects with certain prefixes. If you add an item to the list
# below, make sure to add a comment explaining why.
OBJECT_TO_IGNORE_PREFIXES = [
    "_",  # Private objects are not documented
]

# Supported math operations when interpreting the value of defaults.
MATH_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
}


def has_auto_docstring_decorator(obj) -> bool:
    try:
        # Get the source lines for the object
        source_lines = inspect.getsourcelines(obj)[0]

        # Check the lines before the definition for @auto_docstring decorator
        for line in source_lines[:10]:  # Check first 10 lines (decorators come before def/class)
            line = line.strip()
            if line.startswith("@auto_docstring"):
                return True
    except (TypeError, OSError):
        # Some objects don't have source code available
        pass

    return False


def find_indent(line: str) -> int:
    """
    Returns the number of spaces that start a line indent.
    """
    search = re.search(r"^(\s*)(?:\S|$)", line)
    if search is None:
        return 0
    return len(search.groups()[0])


def stringify_default(default: Any) -> str:
    """
    Returns the string representation of a default value, as used in docstring: numbers are left as is, all other
    objects are in backtiks.

    Args:
        default (`Any`): The default value to process

    Returns:
        `str`: The string representation of that default.
    """
    if isinstance(default, bool):
        # We need to test for bool first as a bool passes isinstance(xxx, (int, float))
        return f"`{default}`"
    elif isinstance(default, enum.Enum):
        # We need to test for enum first as an enum with int values will pass isinstance(xxx, (int, float))
        return f"`{str(default)}`"
    elif isinstance(default, int):
        return str(default)
    elif isinstance(default, float):
        result = str(default)
        return str(round(default, 2)) if len(result) > 6 else result
    elif isinstance(default, str):
        return str(default) if default.isnumeric() else f'`"{default}"`'
    elif isinstance(default, type):
        return f"`{default.__name__}`"
    else:
        return f"`{default}`"


def eval_math_expression(expression: str) -> float | int | None:
    # Mainly taken from the excellent https://stackoverflow.com/a/9558001
    """
    Evaluate (safely) a mathematial expression and returns its value.

    Args:
        expression (`str`): The expression to evaluate.

    Returns:
        `Optional[Union[float, int]]`: Returns `None` if the evaluation fails in any way and the value computed
        otherwise.

    Example:

    ```py
    >>> eval_expr('2^6')
    4
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4^5) / (6 + -7)')
    -5.0
    ```
    """
    try:
        return eval_node(ast.parse(expression, mode="eval").body)
    except TypeError:
        return


def eval_node(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
        return node.value
    elif isinstance(node, ast.BinOp):  # <left> <operator> <right>
        return MATH_OPERATORS[type(node.op)](eval_node(node.left), eval_node(node.right))
    elif isinstance(node, ast.UnaryOp):  # <operator> <operand> e.g., -1
        return MATH_OPERATORS[type(node.op)](eval_node(node.operand))
    else:
        raise TypeError(node)


def replace_default_in_arg_description(description: str, default: Any) -> str:
    """
    Catches the default value in the description of an argument inside a docstring and replaces it by the value passed.

    Args:
        description (`str`): The description of an argument in a docstring to process.
        default (`Any`): The default value that would be in the docstring of that argument.

    Returns:
       `str`: The description updated with the new default value.
    """
    # Lots of docstrings have `optional` or **opational** instead of *optional* so we do this fix here.
    description = description.replace("`optional`", OPTIONAL_KEYWORD)
    description = description.replace("**optional**", OPTIONAL_KEYWORD)
    if default is inspect._empty:
        # No default, make sure the description doesn't have any either
        idx = description.find(OPTIONAL_KEYWORD)
        if idx != -1:
            description = description[:idx].rstrip()
            if description.endswith(","):
                description = description[:-1].rstrip()
    elif default is None:
        # Default None are not written, we just set `*optional*`. If there is default that is not None specified in the
        # description, we do not erase it (as sometimes we set the default to `None` because the default is a mutable
        # object).
        idx = description.find(OPTIONAL_KEYWORD)
        if idx == -1:
            description = f"{description}, {OPTIONAL_KEYWORD}"
        elif re.search(r"defaults to `?None`?", description) is not None:
            len_optional = len(OPTIONAL_KEYWORD)
            description = description[: idx + len_optional]
    else:
        str_default = None
        # For numbers we may have a default that is given by a math operation (1/255 is really popular). We don't
        # want to replace those by their actual values.
        if isinstance(default, (int, float)) and re.search("defaults to `?(.*?)(?:`|$)", description) is not None:
            # Grab the default and evaluate it.
            current_default = re.search("defaults to `?(.*?)(?:`|$)", description).groups()[0]
            if default == eval_math_expression(current_default):
                try:
                    # If it can be directly converted to the type of the default, it's a simple value
                    str_default = str(type(default)(current_default))
                except Exception:
                    # Otherwise there is a math operator so we add a code block.
                    str_default = f"`{current_default}`"
            elif isinstance(default, enum.Enum) and default.name == current_default.split(".")[-1]:
                # When the default is an Enum (this is often the case for PIL.Image.Resampling), and the docstring
                # matches the enum name, keep the existing docstring rather than clobbering it with the enum value.
                str_default = f"`{current_default}`"

        if str_default is None:
            str_default = stringify_default(default)
        # Make sure default match
        if OPTIONAL_KEYWORD not in description:
            description = f"{description}, {OPTIONAL_KEYWORD}, defaults to {str_default}"
        elif _re_parse_description.search(description) is None:
            idx = description.find(OPTIONAL_KEYWORD)
            len_optional = len(OPTIONAL_KEYWORD)
            description = f"{description[: idx + len_optional]}, defaults to {str_default}"
        else:
            description = _re_parse_description.sub(rf"*optional*, defaults to {str_default}", description)

    return description


def get_default_description(arg: inspect.Parameter) -> str:
    """
    Builds a default description for a parameter that was not documented.

    Args:
        arg (`inspect.Parameter`): The argument in the signature to generate a description for.

    Returns:
        `str`: The description.
    """
    if arg.annotation is inspect._empty:
        arg_type = "<fill_type>"
    elif hasattr(arg.annotation, "__name__"):
        arg_type = arg.annotation.__name__
    else:
        arg_type = str(arg.annotation)

    if arg.default is inspect._empty:
        return f"`{arg_type}`"
    elif arg.default is None:
        return f"`{arg_type}`, {OPTIONAL_KEYWORD}"
    else:
        str_default = stringify_default(arg.default)
        return f"`{arg_type}`, {OPTIONAL_KEYWORD}, defaults to {str_default}"


def find_source_file(obj: Any) -> Path:
    """
    Finds the source file of an object.

    Args:
        obj (`Any`): The object whose source file we are looking for.

    Returns:
        `Path`: The source file.
    """
    module = obj.__module__
    obj_file = PATH_TO_TRANSFORMERS
    for part in module.split(".")[1:]:
        obj_file = obj_file / part
    return obj_file.with_suffix(".py")


def match_docstring_with_signature(obj: Any) -> tuple[str, str] | None:
    """
    Matches the docstring of an object with its signature.

    Args:
        obj (`Any`): The object to process.

    Returns:
        `Optional[Tuple[str, str]]`: Returns `None` if there is no docstring or no parameters documented in the
        docstring, otherwise returns a tuple of two strings: the current documentation of the arguments in the
        docstring and the one matched with the signature.
    """
    if len(getattr(obj, "__doc__", "")) == 0:
        # Nothing to do, there is no docstring.
        return

    # Read the docstring in the source code to see if there is a special command to ignore this object.
    try:
        source, _ = inspect.getsourcelines(obj)
    except OSError:
        source = []

    # Find the line where the docstring starts
    idx = 0
    while idx < len(source) and '"""' not in source[idx]:
        idx += 1

    ignore_order = False
    if idx < len(source):
        line_before_docstring = source[idx - 1]
        # Match '# no-format' (allowing surrounding whitespaces)
        if re.search(r"^\s*#\s*no-format\s*$", line_before_docstring):
            # This object is ignored by the auto-docstring tool
            return
        # Match '# ignore-order' (allowing surrounding whitespaces)
        elif re.search(r"^\s*#\s*ignore-order\s*$", line_before_docstring):
            ignore_order = True

    # Read the signature. Skip on `TypedDict` objects for now. Inspect cannot
    # parse their signature ("no signature found for builtin type <class 'dict'>")
    if issubclass(obj, dict) and hasattr(obj, "__annotations__"):
        return

    signature = inspect.signature(obj).parameters

    obj_doc_lines = obj.__doc__.split("\n")
    # Get to the line where we start documenting arguments
    idx = 0
    while idx < len(obj_doc_lines) and _re_args.search(obj_doc_lines[idx]) is None:
        idx += 1

    if idx == len(obj_doc_lines):
        # Nothing to do, no parameters are documented.
        return

    if "kwargs" in signature and signature["kwargs"].annotation != inspect._empty:
        # Inspecting signature with typed kwargs is not supported yet.
        return

    indent = find_indent(obj_doc_lines[idx])
    arguments = {}
    current_arg = None
    idx += 1
    start_idx = idx
    # Keep going until the arg section is finished (nonempty line at the same indent level) or the end of the docstring.
    while idx < len(obj_doc_lines) and (
        len(obj_doc_lines[idx].strip()) == 0 or find_indent(obj_doc_lines[idx]) > indent
    ):
        if find_indent(obj_doc_lines[idx]) == indent + 4:
            # New argument -> let's generate the proper doc for it
            re_search_arg = _re_parse_arg.search(obj_doc_lines[idx])
            if re_search_arg is not None:
                _, name, description = re_search_arg.groups()
                current_arg = name
                if name in signature:
                    default = signature[name].default
                    if signature[name].kind is inspect._ParameterKind.VAR_KEYWORD:
                        default = None
                    new_description = replace_default_in_arg_description(description, default)
                else:
                    new_description = description
                init_doc = _re_parse_arg.sub(rf"\1\2 ({new_description}):", obj_doc_lines[idx])
                arguments[current_arg] = [init_doc]
        elif current_arg is not None:
            arguments[current_arg].append(obj_doc_lines[idx])

        idx += 1

    # We went too far by one (perhaps more if there are a lot of new lines)
    idx -= 1
    if current_arg:
        while len(obj_doc_lines[idx].strip()) == 0:
            arguments[current_arg] = arguments[current_arg][:-1]
            idx -= 1
    # And we went too far by one again.
    idx += 1

    old_doc_arg = "\n".join(obj_doc_lines[start_idx:idx])

    old_arguments = list(arguments.keys())
    arguments = {name: "\n".join(doc) for name, doc in arguments.items()}
    # Add missing arguments with a template
    for name in set(signature.keys()) - set(arguments.keys()):
        arg = signature[name]
        # We ignore private arguments or *args/**kwargs (unless they are documented by the user)
        if name.startswith("_") or arg.kind in [
            inspect._ParameterKind.VAR_KEYWORD,
            inspect._ParameterKind.VAR_POSITIONAL,
        ]:
            arguments[name] = ""
        else:
            arg_desc = get_default_description(arg)
            arguments[name] = " " * (indent + 4) + f"{name} ({arg_desc}): <fill_docstring>"

    # Arguments are sorted by the order in the signature unless a special comment is put.
    if ignore_order:
        new_param_docs = [arguments[name] for name in old_arguments if name in signature]
        missing = set(signature.keys()) - set(old_arguments)
        new_param_docs.extend([arguments[name] for name in missing if len(arguments[name]) > 0])
    else:
        new_param_docs = [arguments[name] for name in signature if len(arguments[name]) > 0]
    new_doc_arg = "\n".join(new_param_docs)

    return old_doc_arg, new_doc_arg


def fix_docstring(obj: Any, old_doc_args: str, new_doc_args: str):
    """
    Fixes the docstring of an object by replacing its arguments documentation by the one matched with the signature.

    Args:
        obj (`Any`):
            The object whose dostring we are fixing.
        old_doc_args (`str`):
            The current documentation of the parameters of `obj` in the docstring (as returned by
            `match_docstring_with_signature`).
        new_doc_args (`str`):
            The documentation of the parameters of `obj` matched with its signature (as returned by
            `match_docstring_with_signature`).
    """
    # Read the docstring in the source code and make sure we have the right part of the docstring
    source, line_number = inspect.getsourcelines(obj)

    # Get to the line where we start documenting arguments
    idx = 0
    while idx < len(source) and _re_args.search(source[idx]) is None:
        idx += 1

    if idx == len(source):
        # Args are not defined in the docstring of this object. This can happen when the docstring is inherited.
        # In this case, we are not trying to fix it on the child object.
        return

    # Get to the line where we stop documenting arguments
    indent = find_indent(source[idx])
    idx += 1
    start_idx = idx
    while idx < len(source) and (len(source[idx].strip()) == 0 or find_indent(source[idx]) > indent):
        idx += 1

    idx -= 1
    while len(source[idx].strip()) == 0:
        idx -= 1
    idx += 1

    # `old_doc_args` is built from `obj.__doc__`, which may have
    # different indentation than the raw source from `inspect.getsourcelines`.
    # We use `inspect.cleandoc` to remove indentation uniformly from both
    # strings before comparing them.
    source_args_as_str = "".join(source[start_idx:idx])
    if inspect.cleandoc(source_args_as_str) != inspect.cleandoc(old_doc_args):
        # Args are not fully defined in the docstring of this object
        obj_file = find_source_file(obj)
        actual_args_section = source_args_as_str.rstrip()
        raise ValueError(
            f"Cannot fix docstring of {obj.__name__} in {obj_file} because the argument section in the source code "
            f"does not match the expected format. This usually happens when:\n"
            f"1. The argument section is not properly indented\n"
            f"2. The argument section contains unexpected formatting\n"
            f"3. The docstring parsing failed to correctly identify the argument boundaries\n\n"
            f"Expected argument section:\n{repr(old_doc_args)}\n\n"
            f"Actual argument section found:\n{repr(actual_args_section)}\n\n"
        )

    obj_file = find_source_file(obj)
    with open(obj_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace content
    lines = content.split("\n")
    prev_line_indentation = find_indent(lines[line_number + start_idx - 2])
    # Now increase the indentation of every line in new_doc_args by prev_line_indentation
    new_doc_args = "\n".join([f"{' ' * prev_line_indentation}{line}" for line in new_doc_args.split("\n")])

    lines = lines[: line_number + start_idx - 1] + [new_doc_args] + lines[line_number + idx - 1 :]

    print(f"Fixing the docstring of {obj.__name__} in {obj_file}.")
    with open(obj_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _find_docstring_end_line(lines, docstring_start_line):
    """Find the line number where a docstring ends. Only handles triple double quotes."""
    if docstring_start_line is None or docstring_start_line < 0 or docstring_start_line >= len(lines):
        return None
    start_line = lines[docstring_start_line]
    if '"""' not in start_line:
        return None
    # Check if docstring starts and ends on the same line
    if start_line.count('"""') >= 2:
        return docstring_start_line
    # Find the closing triple quotes on subsequent lines
    for idx in range(docstring_start_line + 1, len(lines)):
        if '"""' in lines[idx]:
            return idx
    return len(lines) - 1


def _is_auto_docstring_decorator(dec):
    """Return True if the decorator expression corresponds to `@auto_docstring`."""
    # Handle @auto_docstring(...) - unwrap the Call to get the function
    target = dec.func if isinstance(dec, ast.Call) else dec
    # Check if it's named "auto_docstring"
    return isinstance(target, ast.Name) and target.id == "auto_docstring"


def _extract_function_args(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract argument names from a function node, excluding 'self', *args, **kwargs."""
    all_args = (func_node.args.posonlyargs or []) + func_node.args.args + func_node.args.kwonlyargs
    return [a.arg for a in all_args if a.arg != "self"]


def find_matching_model_files(check_all: bool = False):
    """
    Find all model files in the transformers repo that should be checked for @auto_docstring,
    excluding files with certain substrings.
    Returns:
        List of file paths.
    """
    module_diff_files = None
    if not check_all:
        module_diff_files = set()
        repo = Repo(PATH_TO_REPO)
        # Diff from index to unstaged files
        for modified_file_diff in repo.index.diff(None):
            if modified_file_diff.a_path.startswith("src/transformers"):
                module_diff_files.add(os.path.join(PATH_TO_REPO, modified_file_diff.a_path))
        # Diff from index to `main`
        for modified_file_diff in repo.index.diff(repo.refs.main.commit):
            if modified_file_diff.a_path.startswith("src/transformers"):
                module_diff_files.add(os.path.join(PATH_TO_REPO, modified_file_diff.a_path))
        # quick escape route: if there are no module files in the diff, skip this check
        if len(module_diff_files) == 0:
            return None

    modeling_glob_pattern = os.path.join(PATH_TO_TRANSFORMERS, "models/**/modeling_**")
    potential_files = glob.glob(modeling_glob_pattern)
    image_processing_glob_pattern = os.path.join(PATH_TO_TRANSFORMERS, "models/**/image_processing_*_fast.py")
    potential_files += glob.glob(image_processing_glob_pattern)
    processing_glob_pattern = os.path.join(PATH_TO_TRANSFORMERS, "models/**/processing_*.py")
    potential_files += glob.glob(processing_glob_pattern)
    matching_files = []
    for file_path in potential_files:
        if os.path.isfile(file_path):
            matching_files.append(file_path)
    if not check_all:
        # intersect with module_diff_files
        matching_files = sorted([file for file in matching_files if file in module_diff_files])

    return matching_files


def find_files_with_auto_docstring(matching_files, decorator="@auto_docstring"):
    """
    From a list of files, return those that contain the @auto_docstring decorator.
    Fast path: simple substring presence check.
    """
    auto_docstrings_files = []
    for file_path in matching_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except OSError:
            continue
        if decorator in source:
            auto_docstrings_files.append(file_path)
    return auto_docstrings_files


def get_args_in_dataclass(lines, dataclass_content):
    dataclass_content = [line.split("#")[0] for line in dataclass_content]
    dataclass_content = "\n".join(dataclass_content)
    args_in_dataclass = re.findall(r"^    (\w+)(?:\s*:|\s*=|\s*$)", dataclass_content, re.MULTILINE)
    if "self" in args_in_dataclass:
        args_in_dataclass.remove("self")
    return args_in_dataclass


def generate_new_docstring_for_signature(
    lines,
    args_in_signature,
    sig_end_line,
    docstring_start_line,
    arg_indent="    ",
    output_docstring_indent=8,
    custom_args_dict={},
    source_args_doc=[ModelArgs, ImageProcessorArgs],
    is_model_output=False,
):
    """
    Generalized docstring generator for a function or class signature.
    Args:
        lines: List of lines from the file.
        sig_start_line: Line index where the signature starts.
        sig_end_line: Line index where the signature ends.
        docstring_line: Line index where the docstring starts (or None if not present).
        arg_indent: Indentation for missing argument doc entries.
        is_model_output: Whether this is a ModelOutput dataclass (inherited args should be kept)
    Returns:
        new_docstring, sig_end_line, docstring_end (last docstring line index)
    """
    # Extract and clean signature
    missing_docstring_args = []
    docstring_args_ro_remove = []
    fill_docstring_args = []

    # Parse docstring if present
    args_docstring_dict = {}
    remaining_docstring = ""
    if docstring_start_line is not None:
        docstring_end_line = _find_docstring_end_line(lines, docstring_start_line)
        docstring_content = lines[docstring_start_line : docstring_end_line + 1]
        parsed_docstring, remaining_docstring = parse_docstring("\n".join(docstring_content))
        args_docstring_dict.update(parsed_docstring)
    else:
        docstring_end_line = None

    # Remove pre-existing entries for *args and untyped **kwargs from the docstring
    # (No longer needed since *args are excluded from args_in_signature)

    # Remove args that are the same as the ones in the source args doc OR have placeholders
    for arg in args_docstring_dict:
        if arg in get_args_doc_from_source(source_args_doc) and arg not in ALWAYS_OVERRIDE:
            source_arg_doc = get_args_doc_from_source(source_args_doc)[arg]
            arg_doc = args_docstring_dict[arg]

            # Check if this arg has placeholders
            has_placeholder = "<fill_type>" in arg_doc.get("type", "") or "<fill_docstring>" in arg_doc.get(
                "description", ""
            )

            # Remove if has placeholder (source will provide the real doc)
            if has_placeholder:
                docstring_args_ro_remove.append(arg)
            # Or remove if description matches source exactly
            elif source_arg_doc["description"].strip("\n ") == arg_doc["description"].strip("\n "):
                if source_arg_doc.get("shape") is not None and arg_doc.get("shape") is not None:
                    if source_arg_doc.get("shape").strip("\n ") == arg_doc.get("shape").strip("\n "):
                        docstring_args_ro_remove.append(arg)
                elif source_arg_doc.get("additional_info") is not None and arg_doc.get("additional_info") is not None:
                    if source_arg_doc.get("additional_info").strip("\n ") == arg_doc.get("additional_info").strip(
                        "\n "
                    ):
                        docstring_args_ro_remove.append(arg)
                else:
                    docstring_args_ro_remove.append(arg)

    # For regular methods/functions (not ModelOutput), also remove args not in signature
    if not is_model_output:
        for arg in list(args_docstring_dict.keys()):
            if (
                arg not in args_in_signature
                and arg not in get_args_doc_from_source(source_args_doc)
                and arg not in custom_args_dict
            ):
                docstring_args_ro_remove.append(arg)

    args_docstring_dict = {
        arg: args_docstring_dict[arg] for arg in args_docstring_dict if arg not in docstring_args_ro_remove
    }

    # Fill missing args
    for arg in args_in_signature:
        if (
            arg not in args_docstring_dict
            and arg not in get_args_doc_from_source(source_args_doc)
            and arg not in custom_args_dict
        ):
            missing_docstring_args.append(arg)
            args_docstring_dict[arg] = {
                "type": "<fill_type>",
                "optional": False,
                "shape": None,
                "description": "\n    <fill_docstring>",
                "default": None,
                "additional_info": None,
            }

    # Handle docstring of inherited args (for dataclasses like ModelOutput)
    # For regular methods, this will be empty since we removed args not in signature above
    ordered_args_docstring_dict = OrderedDict(
        (arg, args_docstring_dict[arg]) for arg in args_docstring_dict if arg not in args_in_signature
    )
    # Add args in the order of the signature
    ordered_args_docstring_dict.update(
        (arg, args_docstring_dict[arg]) for arg in args_in_signature if arg in args_docstring_dict
    )
    # Build new docstring
    new_docstring = ""
    if len(ordered_args_docstring_dict) > 0 or remaining_docstring:
        new_docstring += 'r"""\n'
        for arg in ordered_args_docstring_dict:
            additional_info = ordered_args_docstring_dict[arg]["additional_info"] or ""
            custom_arg_description = ordered_args_docstring_dict[arg]["description"]
            if "<fill_docstring>" in custom_arg_description and arg not in missing_docstring_args:
                fill_docstring_args.append(arg)
            if custom_arg_description.endswith('"""'):
                custom_arg_description = "\n".join(custom_arg_description.split("\n")[:-1])
            new_docstring += (
                f"{arg} ({ordered_args_docstring_dict[arg]['type']}{additional_info}):{custom_arg_description}\n"
            )
        close_docstring = True
        if remaining_docstring:
            if remaining_docstring.endswith('"""'):
                close_docstring = False
            end_docstring = "\n" if close_docstring else ""
            new_docstring += f"{set_min_indent(remaining_docstring, 0)}{end_docstring}"
        if close_docstring:
            new_docstring += '"""'
        new_docstring = set_min_indent(new_docstring, output_docstring_indent)

    return (
        new_docstring,
        sig_end_line,
        docstring_end_line if docstring_end_line is not None else sig_end_line - 1,
        missing_docstring_args,
        fill_docstring_args,
        docstring_args_ro_remove,
    )


def generate_new_docstring_for_function(
    lines,
    item: DecoratedItem,
    custom_args_dict,
):
    """
    Wrapper for function docstring generation using the generalized helper.
    """
    sig_end_line = item.body_start_line - 1  # Convert to 0-based
    args_in_signature = item.args
    docstring_start_line = sig_end_line if '"""' in lines[sig_end_line] else None

    # Use ProcessorArgs for processor methods
    if item.is_processor:
        source_args_doc = [ModelArgs, ImageProcessorArgs, ProcessorArgs]
    else:
        source_args_doc = [ModelArgs, ImageProcessorArgs]

    return generate_new_docstring_for_signature(
        lines,
        args_in_signature,
        sig_end_line,
        docstring_start_line,
        arg_indent="    ",
        custom_args_dict=custom_args_dict,
        source_args_doc=source_args_doc,
        is_model_output=False,  # Functions are never ModelOutput
    )


def generate_new_docstring_for_class(
    lines,
    item: DecoratedItem,
    custom_args_dict,
    source: str,
):
    """
    Wrapper for class docstring generation (via __init__) using the generalized helper.
    Returns the new docstring and relevant signature/docstring indices.
    """
    # Use pre-extracted information from DecoratedItem (no need to search or re-parse!)
    if item.has_init:
        # Class has an __init__ method - use its args and body start
        sig_end_line = item.body_start_line - 1  # Convert from body start to sig end (0-based)
        args_in_signature = item.args
        output_docstring_indent = 8
        # Add ProcessorArgs for Processor classes
        if item.is_processor:
            source_args_doc = [ModelArgs, ImageProcessorArgs, ProcessorArgs]
        else:
            source_args_doc = [ModelArgs, ImageProcessorArgs]
    elif item.is_model_output:
        # ModelOutput class - extract args from dataclass attributes
        current_line_end = item.def_line - 1  # Convert to 0-based
        sig_end_line = current_line_end + 1
        docstring_end = _find_docstring_end_line(lines, sig_end_line)
        model_output_class_start = docstring_end + 1 if docstring_end is not None else sig_end_line - 1
        model_output_class_end = model_output_class_start
        while model_output_class_end < len(lines) and (
            lines[model_output_class_end].startswith("    ") or lines[model_output_class_end] == ""
        ):
            model_output_class_end += 1
        dataclass_content = lines[model_output_class_start : model_output_class_end - 1]
        args_in_signature = get_args_in_dataclass(lines, dataclass_content)
        output_docstring_indent = 4
        source_args_doc = [ModelOutputArgs]
    else:
        # Class has no __init__ and is not a ModelOutput - nothing to document
        return "", None, None, [], [], []

    docstring_start_line = sig_end_line if '"""' in lines[sig_end_line] else None

    return generate_new_docstring_for_signature(
        lines,
        args_in_signature,
        sig_end_line,
        docstring_start_line,
        arg_indent="",
        custom_args_dict=custom_args_dict,
        output_docstring_indent=output_docstring_indent,
        source_args_doc=source_args_doc,
        is_model_output=item.is_model_output,
    )


def _build_ast_indexes(source: str) -> list[DecoratedItem]:
    """Parse source once and return list of all @auto_docstring decorated items.

    Returns:
        List of DecoratedItem objects, one for each @auto_docstring decorated function or class.
    """
    tree = ast.parse(source)
    # First pass: collect top-level string variables (for resolving custom_args variable references)
    var_to_string: dict[str, str] = {}
    for node in tree.body:
        # Handle: ARGS = "some string"
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_to_string[target.id] = node.value.value
        # Handle: ARGS: str = "some string"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str) and isinstance(node.target, ast.Name):
                var_to_string[node.target.id] = node.value.value
    # Second pass: find all @auto_docstring decorated functions/classes
    # First, identify processor classes to track method context
    processor_classes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and ("ProcessorMixin" in base.id or "Processor" in base.id):
                    processor_classes.add(node.name)
                    break

    decorated_items: list[DecoratedItem] = []

    # Helper function to process decorated items
    def process_node(node, parent_class_name=None):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        # Find @auto_docstring decorator and extract custom_args if present
        decorator_line = None
        custom_args_text = None
        for dec in node.decorator_list:
            if not _is_auto_docstring_decorator(dec):
                continue
            decorator_line = dec.lineno
            # Extract custom_args from @auto_docstring(custom_args=...)
            if isinstance(dec, ast.Call):
                for kw in dec.keywords:
                    if kw.arg == "custom_args":
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            custom_args_text = kw.value.value.strip()
                        elif isinstance(kw.value, ast.Name):
                            custom_args_text = var_to_string.get(kw.value.id, "").strip()
            break
        if decorator_line is None:  # No @auto_docstring decorator found
            return
        # Extract info for this decorated item
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        body_start_line = node.body[0].lineno if node.body else node.lineno + 1
        # Extract function arguments (skip self, *args, **kwargs)
        arg_names = []
        has_init = False
        init_def_line = None
        is_model_output = False
        is_processor = False

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # For functions/methods, extract args directly
            arg_names = _extract_function_args(node)
            # Check if this method is inside a processor class
            if parent_class_name and parent_class_name in processor_classes:
                is_processor = True
        elif isinstance(node, ast.ClassDef):
            # For classes, look for __init__ method and check if it's a ModelOutput or Processor
            # Check if class inherits from ModelOutput or ProcessorMixin
            for base in node.bases:
                if isinstance(base, ast.Name):
                    if "ModelOutput" in base.id:
                        is_model_output = True
                    elif "ProcessorMixin" in base.id or "Processor" in base.id:
                        is_processor = True
            # Look for __init__ method in the class body
            for class_item in node.body:
                if isinstance(class_item, ast.FunctionDef) and class_item.name == "__init__":
                    has_init = True
                    init_def_line = class_item.lineno
                    arg_names = _extract_function_args(class_item)
                    # Update body_start_line to be the __init__ body start
                    body_start_line = class_item.body[0].lineno if class_item.body else class_item.lineno + 1
                    break

        decorated_items.append(
            DecoratedItem(
                decorator_line=decorator_line,
                def_line=node.lineno,
                kind=kind,
                body_start_line=body_start_line,
                args=arg_names,
                custom_args_text=custom_args_text,
                has_init=has_init,
                init_def_line=init_def_line,
                is_model_output=is_model_output,
                is_processor=is_processor,
            )
        )

    # Traverse tree with parent context
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # Check class itself
            process_node(node)
            # Check methods within the class
            for class_item in node.body:
                process_node(class_item, parent_class_name=node.name)
        else:
            # Top-level functions
            process_node(node)

    return sorted(decorated_items, key=lambda x: x.decorator_line)


def _extract_type_name(annotation) -> str | None:
    """
    Extract the type name from an AST annotation node.
    Handles: TypeName, Optional[TypeName], Union[TypeName, ...], list[TypeName], etc.
    Returns the base type name if found, or None.
    """
    if isinstance(annotation, ast.Name):
        # Simple type: TypeName
        return annotation.id
    elif isinstance(annotation, ast.Subscript):
        # Generic type: Optional[TypeName], list[TypeName], etc.
        # Try to extract from the subscript value
        if isinstance(annotation.value, ast.Name):
            # If it's Optional, Union, list, etc., look at the slice
            if isinstance(annotation.slice, ast.Name):
                return annotation.slice.id
            elif isinstance(annotation.slice, ast.Tuple):
                # Union[TypeName, None] - take first element
                if annotation.slice.elts and isinstance(annotation.slice.elts[0], ast.Name):
                    return annotation.slice.elts[0].id
    return None


def _find_typed_dict_classes(source: str) -> list[dict]:
    """
    Find all custom TypedDict kwargs classes in the source.

    Returns:
        List of dicts with TypedDict info: name, line, fields, all_fields, field_types, docstring info
        - fields: fields that need custom documentation (not in standard args, not nested TypedDicts)
        - all_fields: all fields including those in standard args (for redundancy checking)
    """
    tree = ast.parse(source)

    # Get standard args that are already documented in source classes
    standard_args = set()
    try:
        standard_args.update(get_args_doc_from_source([ModelArgs, ImageProcessorArgs, ProcessorArgs]).keys())
    except Exception as e:
        logger.debug(f"Could not get standard args from source: {e}")

    # Collect all TypedDict class names first (for excluding nested TypedDicts)
    typed_dict_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and ("TypedDict" in base.id or "Kwargs" in base.id):
                    typed_dict_names.add(node.name)
                    break

    typed_dicts = []

    # Check each TypedDict class
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if this is a TypedDict
        is_typed_dict = False
        for base in node.bases:
            if isinstance(base, ast.Name) and ("TypedDict" in base.id or "Kwargs" in base.id):
                is_typed_dict = True
                break

        if not is_typed_dict:
            continue

        # Skip standard kwargs classes
        if node.name in ["TextKwargs", "ImagesKwargs", "VideosKwargs", "AudioKwargs", "ProcessingKwargs"]:
            continue

        # Extract fields and their types (in declaration order)
        fields = []  # Fields that need custom documentation
        all_fields = []  # All fields including those in standard args
        field_types = {}
        for class_item in node.body:
            if isinstance(class_item, ast.AnnAssign) and isinstance(class_item.target, ast.Name):
                field_name = class_item.target.id
                if not field_name.startswith("_"):
                    # Extract type and check if it's a nested TypedDict
                    if class_item.annotation:
                        type_name = _extract_type_name(class_item.annotation)
                        if type_name:
                            field_types[field_name] = type_name
                            # Skip nested TypedDicts
                            if type_name in typed_dict_names or type_name.endswith("Kwargs"):
                                continue
                    # Track all fields for redundancy checking
                    all_fields.append(field_name)
                    # Only add to fields if not in standard args (needs custom documentation)
                    if field_name not in standard_args:
                        fields.append(field_name)

        # Skip if no fields at all (including standard args)
        if not all_fields:
            continue

        # Extract docstring info
        docstring = None
        docstring_start_line = None
        docstring_end_line = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value
            docstring_start_line = node.body[0].lineno
            docstring_end_line = node.body[0].end_lineno

        typed_dicts.append(
            {
                "name": node.name,
                "line": node.lineno,
                "fields": fields,
                "all_fields": all_fields,
                "field_types": field_types,
                "docstring": docstring,
                "docstring_start_line": docstring_start_line,
                "docstring_end_line": docstring_end_line,
            }
        )

    return typed_dicts


def _process_typed_dict_docstrings(
    candidate_file: str,
    overwrite: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    Check and optionally fix TypedDict docstrings.
    Runs as a separate pass after @auto_docstring processing.

    Args:
        candidate_file: Path to the file to process
        overwrite: Whether to fix issues by writing to the file

    Returns:
        Tuple of (missing_warnings, fill_warnings, redundant_warnings)
    """
    with open(candidate_file, "r", encoding="utf-8") as f:
        content = f.read()

    typed_dicts = _find_typed_dict_classes(content)
    if not typed_dicts:
        return [], [], []

    # Get source args for comparison
    source_args_doc = get_args_doc_from_source([ModelArgs, ImageProcessorArgs, ProcessorArgs])

    missing_warnings = []
    fill_warnings = []
    redundant_warnings = []

    # Process each TypedDict
    for td in typed_dicts:
        # Parse existing docstring
        documented_fields = {}
        remaining_docstring = ""
        if td["docstring"]:
            try:
                documented_fields, remaining_docstring = parse_docstring(td["docstring"])
            except Exception as e:
                logger.debug(f"Could not parse docstring for {td.get('name', 'unknown')}: {e}")

        # Find missing, fill, and redundant fields
        missing_fields = []
        fill_fields = []
        redundant_fields = []

        # Check fields that need custom documentation (not in source args)
        for field in td["fields"]:
            if field not in documented_fields:
                missing_fields.append(field)
            else:
                field_doc = documented_fields[field]
                desc = field_doc.get("description", "")
                type_str = field_doc.get("type", "")
                has_placeholder = "<fill_type>" in type_str or "<fill_docstring>" in desc
                if has_placeholder:
                    fill_fields.append(field)

        # Check ALL documented fields (including those in source args) for redundancy
        for field in documented_fields:
            if field in source_args_doc:
                field_doc = documented_fields[field]
                desc = field_doc.get("description", "")
                type_str = field_doc.get("type", "")
                has_placeholder = "<fill_type>" in type_str or "<fill_docstring>" in desc

                source_doc = source_args_doc[field]
                source_desc = source_doc.get("description", "").strip("\n ")
                field_desc = desc.strip("\n ")

                # Mark as redundant if has placeholder OR description matches source
                if has_placeholder or source_desc == field_desc:
                    redundant_fields.append(field)

        if missing_fields:
            field_list = ", ".join(sorted(missing_fields))
            missing_warnings.append(f"    - {td['name']} (line {td['line']}): undocumented fields: {field_list}")

        if fill_fields:
            field_list = ", ".join(sorted(fill_fields))
            fill_warnings.append(f"    - {td['name']} (line {td['line']}): fields with placeholders: {field_list}")

        if redundant_fields:
            field_list = ", ".join(sorted(redundant_fields))
            redundant_warnings.append(
                f"    - {td['name']} (line {td['line']}): redundant fields (in source): {field_list}"
            )

    # If overwrite mode, fix missing fields and remove redundant ones
    if overwrite and (missing_warnings or redundant_warnings):
        lines = content.split("\n")

        # Process TypedDicts in reverse order to avoid line number shifts
        for td in sorted(typed_dicts, key=lambda x: x["line"], reverse=True):
            # Parse existing docstring
            documented_fields = {}
            remaining_docstring = ""
            if td["docstring"]:
                try:
                    documented_fields, remaining_docstring = parse_docstring(td["docstring"])
                except Exception as e:
                    logger.debug(f"Could not parse docstring for {td.get('name', 'unknown')}: {e}")

            # Determine which fields to remove (redundant with source)
            fields_to_remove = set()
            for field in documented_fields:
                if field in source_args_doc:
                    field_doc = documented_fields[field]
                    desc = field_doc.get("description", "")
                    type_str = field_doc.get("type", "")
                    has_placeholder = "<fill_type>" in type_str or "<fill_docstring>" in desc

                    source_doc = source_args_doc[field]
                    source_desc = source_doc.get("description", "").strip("\n ")
                    field_desc = desc.strip("\n ")

                    # Remove if has placeholder OR description matches source
                    if has_placeholder or source_desc == field_desc:
                        fields_to_remove.add(field)

            # Check if any fields are missing or need removal
            has_missing = any(f not in documented_fields for f in td["fields"])
            has_changes = has_missing or len(fields_to_remove) > 0

            if not has_changes:
                continue

            # Build new docstring dict (preserving existing, removing redundant, adding missing)
            # We iterate over documented_fields first to preserve order, then add missing fields
            new_doc_dict = OrderedDict()

            # First, add documented fields that should be kept (not redundant)
            for field in documented_fields:
                if field not in fields_to_remove:
                    # Only keep fields that are either:
                    # 1. In td["fields"] (needs custom documentation)
                    # 2. Not in source_args_doc (might be inherited or custom)
                    if field in td["fields"] or field not in source_args_doc:
                        new_doc_dict[field] = documented_fields[field]

            # Then, add missing fields from td["fields"]
            for field in td["fields"]:
                if field not in documented_fields and field not in new_doc_dict:
                    # Add placeholder for missing field
                    new_doc_dict[field] = {
                        "type": "`<fill_type>`",
                        "optional": False,
                        "shape": None,
                        "description": "\n    <fill_docstring>",
                        "default": None,
                        "additional_info": None,
                    }

            # Build new docstring text
            class_line_idx = td["line"] - 1
            class_line = lines[class_line_idx]
            indent = len(class_line) - len(class_line.lstrip())

            # If all fields were removed, remove the docstring entirely
            if not new_doc_dict and not remaining_docstring:
                if td["docstring"] is not None:
                    doc_start_idx = td["docstring_start_line"] - 1
                    doc_end_idx = td["docstring_end_line"]
                    lines = lines[:doc_start_idx] + lines[doc_end_idx:]
                continue

            # Build docstring content (without indentation first)
            docstring_content = '"""\n'
            for field_name, field_doc in new_doc_dict.items():
                additional_info = field_doc.get("additional_info", "") or ""
                description = field_doc["description"]
                if description.endswith('"""'):
                    description = "\n".join(description.split("\n")[:-1])
                docstring_content += f"{field_name} ({field_doc['type']}{additional_info}):{description}\n"

            # Add remaining docstring content if any
            close_docstring = True
            if remaining_docstring:
                if remaining_docstring.endswith('"""'):
                    close_docstring = False
                end_str = "\n" if close_docstring else ""
                docstring_content += f"{set_min_indent(remaining_docstring, 0)}{end_str}"
            if close_docstring:
                docstring_content += '"""'

            # Apply proper indentation
            docstring_content = set_min_indent(docstring_content, indent + 4)
            docstring_lines = docstring_content.split("\n")

            # Replace in lines
            if td["docstring"] is None:
                # Insert new docstring after class definition
                insert_idx = class_line_idx + 1
                lines = lines[:insert_idx] + docstring_lines + lines[insert_idx:]
            else:
                # Replace existing docstring
                doc_start_idx = td["docstring_start_line"] - 1
                doc_end_idx = td["docstring_end_line"]  # end_lineno is 1-based, we want to include this line
                lines = lines[:doc_start_idx] + docstring_lines + lines[doc_end_idx:]

        # Write updated content
        with open(candidate_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return missing_warnings, fill_warnings, redundant_warnings


def update_file_with_new_docstrings(
    candidate_file,
    lines,
    decorated_items: list[DecoratedItem],
    source: str,
    overwrite=False,
):
    """
    For a given file, update the docstrings for all @auto_docstring candidates and write the new content.
    """
    if not decorated_items:
        return [], [], []

    missing_docstring_args_warnings = []
    fill_docstring_args_warnings = []
    docstring_args_ro_remove_warnings = []

    # Build new file content by processing decorated items and unchanged sections
    content_base_file_new_lines = []
    last_line_added = 0  # Track the last line we've already added to output (0-based)

    for index, item in enumerate(decorated_items):
        def_line_0 = item.def_line - 1  # Convert to 0-based

        # Parse custom_args if present
        custom_args_dict = {}
        if item.custom_args_text:
            custom_args_dict, _ = parse_docstring(item.custom_args_text)

        # Generate new docstring based on kind
        if item.kind == "function":
            (
                new_docstring,
                sig_line_end,
                docstring_end,
                missing_docstring_args,
                fill_docstring_args,
                docstring_args_ro_remove,
            ) = generate_new_docstring_for_function(lines, item, custom_args_dict)
        else:  # class
            (
                new_docstring,
                sig_line_end,
                docstring_end,
                missing_docstring_args,
                fill_docstring_args,
                docstring_args_ro_remove,
            ) = generate_new_docstring_for_class(lines, item, custom_args_dict, source)

        # If sig_line_end is None, this item couldn't be processed (e.g., class with no __init__)
        # In this case, we don't modify anything and just continue to the next item
        if sig_line_end is None:
            continue

        # Add all lines from last processed line up to current def line
        content_base_file_new_lines += lines[last_line_added:def_line_0]

        # Collect warnings
        for arg in missing_docstring_args:
            missing_docstring_args_warnings.append(f"    - {arg} line {def_line_0}")
        for arg in fill_docstring_args:
            fill_docstring_args_warnings.append(f"    - {arg} line {def_line_0}")
        for arg in docstring_args_ro_remove:
            docstring_args_ro_remove_warnings.append(f"    - {arg} line {def_line_0}")

        # Add lines from current def through signature
        content_base_file_new_lines += lines[def_line_0:sig_line_end]

        # Add new docstring if generated
        if new_docstring:
            content_base_file_new_lines += new_docstring.split("\n")

        # Update last_line_added to skip the old docstring
        last_line_added = (docstring_end + 1) if docstring_end is not None else sig_line_end

    # Add any remaining lines after the last decorated item
    content_base_file_new_lines += lines[last_line_added:]

    content_base_file_new = "\n".join(content_base_file_new_lines)
    if overwrite:
        with open(candidate_file, "w", encoding="utf-8") as f:
            f.write(content_base_file_new)

    return (
        missing_docstring_args_warnings,
        fill_docstring_args_warnings,
        docstring_args_ro_remove_warnings,
    )


def check_auto_docstrings(overwrite: bool = False, check_all: bool = False):
    """
    Check docstrings of all public objects that are decorated with `@auto_docstrings`.
    This function orchestrates the process by finding relevant files, scanning for decorators,
    generating new docstrings, and updating files as needed.
    """
    # 1. Find all model files to check
    matching_files = find_matching_model_files(check_all)
    if matching_files is None:
        return
    # 2. Find files that contain the @auto_docstring decorator
    auto_docstrings_files = find_files_with_auto_docstring(matching_files)

    # Collect all errors before raising
    has_errors = False

    # 3. For each file, update docstrings for all candidates
    for candidate_file in auto_docstrings_files:
        with open(candidate_file, "r", encoding="utf-8") as f:
            content = f.read()
        lines = content.split("\n")

        # Parse file once to find all @auto_docstring decorated items
        decorated_items = _build_ast_indexes(content)

        missing_docstring_args_warnings = []
        fill_docstring_args_warnings = []
        docstring_args_ro_remove_warnings = []

        # Process @auto_docstring decorated items
        if decorated_items:
            missing_docstring_args_warnings, fill_docstring_args_warnings, docstring_args_ro_remove_warnings = (
                update_file_with_new_docstrings(
                    candidate_file,
                    lines,
                    decorated_items,
                    content,
                    overwrite=overwrite,
                )
            )

        # Process TypedDict kwargs (separate pass to avoid line number conflicts)
        # This runs AFTER @auto_docstring processing is complete
        typed_dict_missing_warnings, typed_dict_fill_warnings, typed_dict_redundant_warnings = (
            _process_typed_dict_docstrings(candidate_file, overwrite=overwrite)
        )

        # Report TypedDict errors
        if typed_dict_missing_warnings:
            has_errors = True
            if not overwrite:
                print(
                    "Some TypedDict fields are undocumented. Run `make fix-copies` or "
                    "`python utils/check_docstrings.py --fix_and_overwrite` to generate placeholders."
                )
            print(f"[ERROR] Undocumented fields in custom TypedDict kwargs in {candidate_file}:")
            for warning in typed_dict_missing_warnings:
                print(warning)
        if typed_dict_redundant_warnings:
            has_errors = True
            if not overwrite:
                print(
                    "Some TypedDict fields are redundant (same as source or have placeholders). "
                    "Run `make fix-copies` or `python utils/check_docstrings.py --fix_and_overwrite` to remove them."
                )
            print(f"[ERROR] Redundant TypedDict docstrings in {candidate_file}:")
            for warning in typed_dict_redundant_warnings:
                print(warning)
        if typed_dict_fill_warnings:
            has_errors = True
            print(f"[ERROR] TypedDict docstrings need to be filled in {candidate_file}:")
            for warning in typed_dict_fill_warnings:
                print(warning)
        if missing_docstring_args_warnings:
            has_errors = True
            if not overwrite:
                print(
                    "Some docstrings are missing. Run `make fix-repo` or `python utils/check_docstrings.py --fix_and_overwrite` to generate the docstring templates where needed."
                )
            print(f"[ERROR] Missing docstring for the following arguments in {candidate_file}:")
            for warning in missing_docstring_args_warnings:
                print(warning)
        if docstring_args_ro_remove_warnings:
            has_errors = True
            if not overwrite:
                print(
                    "Some docstrings are redundant with the ones in `auto_docstring.py` and will be removed. Run `make fix-repo` or `python utils/check_docstrings.py --fix_and_overwrite` to remove the redundant docstrings."
                )
            print(f"[ERROR] Redundant docstring for the following arguments in {candidate_file}:")
            for warning in docstring_args_ro_remove_warnings:
                print(warning)
        if fill_docstring_args_warnings:
            has_errors = True
            print(f"[ERROR] Docstring needs to be filled for the following arguments in {candidate_file}:")
            for warning in fill_docstring_args_warnings:
                print(warning)

    # Raise error after processing all files
    if has_errors:
        raise ValueError(
            "There was at least one problem when checking docstrings of objects decorated with @auto_docstring."
        )


def check_docstrings(overwrite: bool = False, check_all: bool = False):
    """
    Check docstrings of all public objects that are callables and are documented. By default, only checks the diff.

    Args:
        overwrite (`bool`, *optional*, defaults to `False`):
            Whether to fix inconsistencies or not.
        check_all (`bool`, *optional*, defaults to `False`):
            Whether to check all files.
    """
    module_diff_files = None
    if not check_all:
        module_diff_files = set()
        repo = Repo(PATH_TO_REPO)
        # Diff from index to unstaged files
        for modified_file_diff in repo.index.diff(None):
            if modified_file_diff.a_path.startswith("src/transformers"):
                module_diff_files.add(modified_file_diff.a_path)
        # Diff from index to `main`
        for modified_file_diff in repo.index.diff(repo.refs.main.commit):
            if modified_file_diff.a_path.startswith("src/transformers"):
                module_diff_files.add(modified_file_diff.a_path)
        # quick escape route: if there are no module files in the diff, skip this check
        if len(module_diff_files) == 0:
            return

    failures = []
    hard_failures = []
    to_clean = []
    for name in dir(transformers):
        # Skip objects that are private or not documented.
        if (
            any(name.startswith(prefix) for prefix in OBJECT_TO_IGNORE_PREFIXES)
            or ignore_undocumented(name)
            or name in OBJECTS_TO_IGNORE
        ):
            continue

        obj = getattr(transformers, name)
        if not callable(obj) or not isinstance(obj, type) or getattr(obj, "__doc__", None) is None:
            continue

        # Skip objects decorated with @auto_docstring - they have auto-generated documentation
        if has_auto_docstring_decorator(obj):
            continue

        # If we are checking against the diff, we skip objects that are not part of the diff.
        if module_diff_files is not None:
            object_file = find_source_file(getattr(transformers, name))
            object_file_relative_path = "src/" + str(object_file).split("/src/")[1]
            if object_file_relative_path not in module_diff_files:
                continue

        # Check docstring
        try:
            result = match_docstring_with_signature(obj)
            if result is not None:
                old_doc, new_doc = result
            else:
                old_doc, new_doc = None, None
        except Exception as e:
            print(e)
            hard_failures.append(name)
            continue
        if old_doc != new_doc:
            if overwrite:
                fix_docstring(obj, old_doc, new_doc)
            else:
                failures.append(name)
        elif not overwrite and new_doc is not None and ("<fill_type>" in new_doc or "<fill_docstring>" in new_doc):
            to_clean.append(name)

    # Deal with errors
    error_message = ""
    if len(hard_failures) > 0:
        error_message += (
            "The argument part of the docstrings of the following objects could not be processed, check they are "
            "properly formatted."
        )
        error_message += "\n" + "\n".join([f"- {name}" for name in hard_failures])
    if len(failures) > 0:
        error_message += (
            "The following objects docstrings do not match their signature. Run `make fix-repo` to fix this. "
            "In some cases, this error may be raised incorrectly by the docstring checker. If you think this is the "
            "case, you can manually check the docstrings and then add the object name to `OBJECTS_TO_IGNORE` in "
            "`utils/check_docstrings.py`."
        )
        error_message += "\n" + "\n".join([f"- {name}" for name in failures])
    if len(to_clean) > 0:
        error_message += (
            "The following objects docstrings contain templates you need to fix: search for `<fill_type>` or "
            "`<fill_docstring>`."
        )
        error_message += "\n" + "\n".join([f"- {name}" for name in to_clean])

    if len(error_message) > 0:
        error_message = "There was at least one problem when checking docstrings of public objects.\n" + error_message
        raise ValueError(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    parser.add_argument(
        "--check_all", action="store_true", help="Whether to check all files. By default, only checks the diff"
    )
    args = parser.parse_args()
    check_auto_docstrings(overwrite=args.fix_and_overwrite, check_all=args.check_all)
    check_docstrings(overwrite=args.fix_and_overwrite, check_all=args.check_all)
