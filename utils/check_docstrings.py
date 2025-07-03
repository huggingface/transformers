# coding=utf-8
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

for a check that will error in case of inconsistencies (used by `make repo-consistency`).

To auto-fix issues run:

```bash
python utils/check_docstrings.py --fix_and_overwrite
```

which is used by `make fix-copies` (note that this fills what it cans, you might have to manually fill information
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
from pathlib import Path
from typing import Any, Optional, Union

from check_repo import ignore_undocumented
from git import Repo

from transformers.utils import direct_transformers_import
from transformers.utils.args_doc import (
    ImageProcessorArgs,
    ModelArgs,
    ModelOutputArgs,
    get_args_doc_from_source,
    parse_docstring,
    set_min_indent,
)


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

# This is a temporary list of objects to ignore while we progressively fix them. Do not add anything here, fix the
# docstrings instead. If formatting should be ignored for the docstring, you can put a comment # no-format on the
# line before the docstring.
OBJECTS_TO_IGNORE = [
    "Gemma3nVisionConfig",
    "Llama4Processor",
    # Deprecated
    "InputExample",
    "InputFeatures",
    # Signature is *args/**kwargs
    "TFSequenceSummary",
    "TFBertTokenizer",
    "TFGPT2Tokenizer",
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
    "BloomTokenizerFast",
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
    "FlaxAlbertForMaskedLM",
    "FlaxAlbertForMultipleChoice",
    "FlaxAlbertForPreTraining",
    "FlaxAlbertForQuestionAnswering",
    "FlaxAlbertForSequenceClassification",
    "FlaxAlbertForTokenClassification",
    "FlaxAlbertModel",
    "FlaxBartForCausalLM",
    "FlaxBartForConditionalGeneration",
    "FlaxBartForQuestionAnswering",
    "FlaxBartForSequenceClassification",
    "FlaxBartModel",
    "FlaxBeitForImageClassification",
    "FlaxBeitForMaskedImageModeling",
    "FlaxBeitModel",
    "FlaxBertForCausalLM",
    "FlaxBertForMaskedLM",
    "FlaxBertForMultipleChoice",
    "FlaxBertForNextSentencePrediction",
    "FlaxBertForPreTraining",
    "FlaxBertForQuestionAnswering",
    "FlaxBertForSequenceClassification",
    "FlaxBertForTokenClassification",
    "FlaxBertModel",
    "FlaxBigBirdForCausalLM",
    "FlaxBigBirdForMaskedLM",
    "FlaxBigBirdForMultipleChoice",
    "FlaxBigBirdForPreTraining",
    "FlaxBigBirdForQuestionAnswering",
    "FlaxBigBirdForSequenceClassification",
    "FlaxBigBirdForTokenClassification",
    "FlaxBigBirdModel",
    "FlaxBlenderbotForConditionalGeneration",
    "FlaxBlenderbotModel",
    "FlaxBlenderbotSmallForConditionalGeneration",
    "FlaxBlenderbotSmallModel",
    "FlaxBloomForCausalLM",
    "FlaxBloomModel",
    "FlaxCLIPModel",
    "FlaxDinov2ForImageClassification",
    "FlaxDinov2Model",
    "FlaxDistilBertForMaskedLM",
    "FlaxDistilBertForMultipleChoice",
    "FlaxDistilBertForQuestionAnswering",
    "FlaxDistilBertForSequenceClassification",
    "FlaxDistilBertForTokenClassification",
    "FlaxDistilBertModel",
    "FlaxElectraForCausalLM",
    "FlaxElectraForMaskedLM",
    "FlaxElectraForMultipleChoice",
    "FlaxElectraForPreTraining",
    "FlaxElectraForQuestionAnswering",
    "FlaxElectraForSequenceClassification",
    "FlaxElectraForTokenClassification",
    "FlaxElectraModel",
    "FlaxEncoderDecoderModel",
    "FlaxGPT2LMHeadModel",
    "FlaxGPT2Model",
    "FlaxGPTJForCausalLM",
    "FlaxGPTJModel",
    "FlaxGPTNeoForCausalLM",
    "FlaxGPTNeoModel",
    "FlaxLlamaForCausalLM",
    "FlaxLlamaModel",
    "FlaxGemmaForCausalLM",
    "FlaxGemmaModel",
    "FlaxMBartForConditionalGeneration",
    "FlaxMBartForQuestionAnswering",
    "FlaxMBartForSequenceClassification",
    "FlaxMBartModel",
    "FlaxMarianMTModel",
    "FlaxMarianModel",
    "FlaxMistralForCausalLM",
    "FlaxMistralModel",
    "FlaxOPTForCausalLM",
    "FlaxPegasusForConditionalGeneration",
    "FlaxPegasusModel",
    "FlaxRegNetForImageClassification",
    "FlaxRegNetModel",
    "FlaxResNetForImageClassification",
    "FlaxResNetModel",
    "FlaxRoFormerForMaskedLM",
    "FlaxRoFormerForMultipleChoice",
    "FlaxRoFormerForQuestionAnswering",
    "FlaxRoFormerForSequenceClassification",
    "FlaxRoFormerForTokenClassification",
    "FlaxRoFormerModel",
    "FlaxRobertaForCausalLM",
    "FlaxRobertaForMaskedLM",
    "FlaxRobertaForMultipleChoice",
    "FlaxRobertaForQuestionAnswering",
    "FlaxRobertaForSequenceClassification",
    "FlaxRobertaForTokenClassification",
    "FlaxRobertaModel",
    "FlaxRobertaPreLayerNormForCausalLM",
    "FlaxRobertaPreLayerNormForMaskedLM",
    "FlaxRobertaPreLayerNormForMultipleChoice",
    "FlaxRobertaPreLayerNormForQuestionAnswering",
    "FlaxRobertaPreLayerNormForSequenceClassification",
    "FlaxRobertaPreLayerNormForTokenClassification",
    "FlaxRobertaPreLayerNormModel",
    "FlaxSpeechEncoderDecoderModel",
    "FlaxViTForImageClassification",
    "FlaxViTModel",
    "FlaxVisionEncoderDecoderModel",
    "FlaxVisionTextDualEncoderModel",
    "FlaxWav2Vec2ForCTC",
    "FlaxWav2Vec2ForPreTraining",
    "FlaxWav2Vec2Model",
    "FlaxWhisperForAudioClassification",
    "FlaxWhisperForConditionalGeneration",
    "FlaxWhisperModel",
    "FlaxWhisperTimeStampLogitsProcessor",
    "FlaxXGLMForCausalLM",
    "FlaxXGLMModel",
    "FlaxXLMRobertaForCausalLM",
    "FlaxXLMRobertaForMaskedLM",
    "FlaxXLMRobertaForMultipleChoice",
    "FlaxXLMRobertaForQuestionAnswering",
    "FlaxXLMRobertaForSequenceClassification",
    "FlaxXLMRobertaForTokenClassification",
    "FlaxXLMRobertaModel",
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
    "ImageToImagePipeline",
    "ImageToTextPipeline",
    "InformerConfig",
    "JukeboxPriorConfig",
    "JukeboxTokenizer",
    "LEDConfig",
    "LEDTokenizerFast",
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
    "RealmConfig",
    "RealmForOpenQA",
    "RealmScorer",
    "RealmTokenizerFast",
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
    "SpecialTokensMixin",
    "Speech2Text2Config",
    "Speech2Text2Tokenizer",
    "Speech2TextTokenizer",
    "SpeechEncoderDecoderModel",
    "SpeechT5Config",
    "SpeechT5Model",
    "SplinterConfig",
    "SplinterTokenizerFast",
    "SqueezeBertTokenizerFast",
    "SummarizationPipeline",
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
    "Text2TextGenerationPipeline",
    "TextClassificationPipeline",
    "TextGenerationPipeline",
    "TFBartForConditionalGeneration",
    "TFBartForSequenceClassification",
    "TFBartModel",
    "TFBertModel",
    "TFConvNextModel",
    "TFData2VecVisionModel",
    "TFDeiTModel",
    "TFEncoderDecoderModel",
    "TFEsmModel",
    "TFMobileViTModel",
    "TFRagModel",
    "TFRagSequenceForGeneration",
    "TFRagTokenForGeneration",
    "TFRepetitionPenaltyLogitsProcessor",
    "TFSwinModel",
    "TFViTModel",
    "TFVisionEncoderDecoderModel",
    "TFVisionTextDualEncoderModel",
    "TFXGLMForCausalLM",
    "TFXGLMModel",
    "TimeSeriesTransformerConfig",
    "TokenClassificationPipeline",
    "TrOCRConfig",
    "Phi4MultimodalProcessor",
    "TrainerState",
    "TrainingArguments",
    "TrajectoryTransformerConfig",
    "TranslationPipeline",
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


def eval_math_expression(expression: str) -> Optional[Union[float, int]]:
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
    if isinstance(node, ast.Num):  # <number>
        return node.n
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


def match_docstring_with_signature(obj: Any) -> Optional[tuple[str, str]]:
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

    idx = 0
    while idx < len(source) and '"""' not in source[idx]:
        idx += 1

    ignore_order = False
    if idx < len(source):
        line_before_docstring = source[idx - 1]
        if re.search(r"^\s*#\s*no-format\s*$", line_before_docstring):
            # This object is ignored
            return
        elif re.search(r"^\s*#\s*ignore-order\s*$", line_before_docstring):
            ignore_order = True

    # Read the signature
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
        new_param_docs = [arguments[name] for name in signature.keys() if len(arguments[name]) > 0]
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
        # Args are not defined in the docstring of this object
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

    if "".join(source[start_idx:idx])[:-1] != old_doc_args:
        # Args are not fully defined in the docstring of this object
        return

    obj_file = find_source_file(obj)
    with open(obj_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace content
    lines = content.split("\n")
    lines = lines[: line_number + start_idx - 1] + [new_doc_args] + lines[line_number + idx - 1 :]

    print(f"Fixing the docstring of {obj.__name__} in {obj_file}.")
    with open(obj_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _find_sig_line(lines, line_end):
    parenthesis_count = 0
    sig_line_end = line_end
    found_sig = False
    while not found_sig:
        for char in lines[sig_line_end]:
            if char == "(":
                parenthesis_count += 1
            elif char == ")":
                parenthesis_count -= 1
                if parenthesis_count == 0:
                    found_sig = True
                    break
        sig_line_end += 1
    return sig_line_end


def _find_docstring_end_line(lines, docstring_start_line):
    if '"""' not in lines[docstring_start_line]:
        return None
    docstring_end = docstring_start_line
    if docstring_start_line is not None:
        docstring_end = docstring_start_line
        if not lines[docstring_start_line].count('"""') >= 2:
            docstring_end += 1
            while '"""' not in lines[docstring_end]:
                docstring_end += 1
    return docstring_end


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
    exclude_substrings = ["modeling_tf_", "modeling_flax_"]
    matching_files = []
    for file_path in potential_files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            is_excluded = any(exclude in filename for exclude in exclude_substrings)
            if not is_excluded:
                matching_files.append(file_path)
    if not check_all:
        # intersect with module_diff_files
        matching_files = sorted([file for file in matching_files if file in module_diff_files])

    print("    Checking auto_docstrings in the following files:" + "\n    - " + "\n    - ".join(matching_files))

    return matching_files


def find_files_with_auto_docstring(matching_files, decorator="@auto_docstring"):
    """
    From a list of files, return those that contain the @auto_docstring decorator.
    """
    auto_docstrings_files = []
    for file_path in matching_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content_base_file = f.read()
            if decorator in content_base_file:
                lines = content_base_file.split("\n")
                line_numbers = [i for i, line in enumerate(lines) if decorator in line]
                for line_number in line_numbers:
                    line_end = line_number
                    end_patterns = ["class ", "    def"]
                    stop_condition = False
                    while line_end < len(lines) and not stop_condition:
                        line_end += 1
                        stop_condition = any(lines[line_end].startswith(end_pattern) for end_pattern in end_patterns)
                    candidate_patterns = ["class ", "    def"]
                    candidate = any(
                        lines[line_end].startswith(candidate_pattern) for candidate_pattern in candidate_patterns
                    )
                    if stop_condition and candidate:
                        auto_docstrings_files.append(file_path)
                        break
    return auto_docstrings_files


def get_auto_docstring_candidate_lines(lines):
    """
    For a file's lines, find the start and end line indices of all @auto_docstring candidates.
    Returns two lists: starts and ends.
    """
    line_numbers = [i for i, line in enumerate(lines) if "@auto_docstring" in line]
    line_starts_candidates = []
    line_ends_candidates = []
    for line_number in line_numbers:
        line_end = line_number
        end_patterns = ["class ", "    def"]
        stop_condition = False
        while line_end < len(lines) and not stop_condition:
            line_end += 1
            stop_condition = any(lines[line_end].startswith(end_pattern) for end_pattern in end_patterns)
        candidate_patterns = ["class ", "    def"]
        candidate = any(lines[line_end].startswith(candidate_pattern) for candidate_pattern in candidate_patterns)
        if stop_condition and candidate:
            line_ends_candidates.append(line_end)
            line_starts_candidates.append(line_number)
    return line_starts_candidates, line_ends_candidates


def get_args_in_signature(lines, signature_content):
    signature_content = [line.split("#")[0] for line in signature_content]
    signature_content = "".join(signature_content)
    signature_content = "".join(signature_content.split(")")[:-1])
    args_in_signature = re.findall(r"[,(]\s*(\w+)\s*(?=:|=|,|\))", signature_content)
    if "self" in args_in_signature:
        args_in_signature.remove("self")
    return args_in_signature


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
):
    """
    Generalized docstring generator for a function or class signature.
    Args:
        lines: List of lines from the file.
        sig_start_line: Line index where the signature starts.
        sig_end_line: Line index where the signature ends.
        docstring_line: Line index where the docstring starts (or None if not present).
        arg_indent: Indentation for missing argument doc entries.
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

    # Remove args that are the same as the ones in the source args doc
    for arg in args_docstring_dict:
        if arg in get_args_doc_from_source(source_args_doc) and arg not in ALWAYS_OVERRIDE:
            source_arg_doc = get_args_doc_from_source(source_args_doc)[arg]
            if source_arg_doc["description"].strip("\n ") == args_docstring_dict[arg]["description"].strip("\n "):
                if source_arg_doc.get("shape") is not None and args_docstring_dict[arg].get("shape") is not None:
                    if source_arg_doc.get("shape").strip("\n ") == args_docstring_dict[arg].get("shape").strip("\n "):
                        docstring_args_ro_remove.append(arg)
                elif (
                    source_arg_doc.get("additional_info") is not None
                    and args_docstring_dict[arg].get("additional_info") is not None
                ):
                    if source_arg_doc.get("additional_info").strip("\n ") == args_docstring_dict[arg].get(
                        "additional_info"
                    ).strip("\n "):
                        docstring_args_ro_remove.append(arg)
                else:
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

    # Handle docstring of inherited args (for dataclasses)
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


def generate_new_docstring_for_function(lines, current_line_end, custom_args_dict):
    """
    Wrapper for function docstring generation using the generalized helper.
    """
    sig_end_line = _find_sig_line(lines, current_line_end)
    signature_content = lines[current_line_end:sig_end_line]
    args_in_signature = get_args_in_signature(lines, signature_content)
    docstring_start_line = sig_end_line if '"""' in lines[sig_end_line] else None
    return generate_new_docstring_for_signature(
        lines,
        args_in_signature,
        sig_end_line,
        docstring_start_line,
        arg_indent="    ",
        custom_args_dict=custom_args_dict,
    )


def generate_new_docstring_for_class(lines, current_line_end, custom_args_dict):
    """
    Wrapper for class docstring generation (via __init__) using the generalized helper.
    Returns the new docstring and relevant signature/docstring indices.
    """
    sig_start_line = current_line_end
    found_init_method = False
    found_model_output = False
    while sig_start_line < len(lines) - 1 and not found_init_method:
        sig_start_line += 1
        if "    def __init__" in lines[sig_start_line]:
            found_init_method = True
        elif lines[sig_start_line].startswith("class ") or lines[sig_start_line].startswith("def "):
            break
    if not found_init_method:
        if "ModelOutput" in lines[current_line_end]:
            found_model_output = True
            sig_start_line = current_line_end
        else:
            return "", None, None, [], [], []

    if found_init_method:
        sig_end_line = _find_sig_line(lines, sig_start_line)
        signature_content = lines[sig_start_line:sig_end_line]
        args_in_signature = get_args_in_signature(lines, signature_content)
    else:
        # we have a ModelOutput class, the class attributes are the args
        sig_end_line = sig_start_line + 1
        docstring_end = _find_docstring_end_line(lines, sig_end_line)
        model_output_class_start = docstring_end + 1 if docstring_end is not None else sig_end_line - 1
        model_output_class_end = model_output_class_start
        while model_output_class_end < len(lines) and (
            lines[model_output_class_end].startswith("    ") or lines[model_output_class_end] == ""
        ):
            model_output_class_end += 1
        dataclass_content = lines[model_output_class_start : model_output_class_end - 1]
        args_in_signature = get_args_in_dataclass(lines, dataclass_content)

    docstring_start_line = sig_end_line if '"""' in lines[sig_end_line] else None

    return generate_new_docstring_for_signature(
        lines,
        args_in_signature,
        sig_end_line,
        docstring_start_line,
        arg_indent="",
        custom_args_dict=custom_args_dict,
        output_docstring_indent=4 if found_model_output else 8,
        source_args_doc=[ModelArgs, ImageProcessorArgs] if not found_model_output else [ModelOutputArgs],
    )


def find_custom_args_with_details(file_content: str, custom_args_var_name: str) -> list[dict]:
    """
    Find the given custom args variable in the file content and return its content.

    Args:
        file_content: The string content of the Python file.
        custom_args_var_name: The name of the custom args variable.
    """
    # Escape the variable_name to handle any special regex characters it might contain
    escaped_variable_name = re.escape(custom_args_var_name)

    # Construct the regex pattern dynamically with the specific variable name
    # This regex looks for:
    # ^\s* : Start of a line with optional leading whitespace.
    # ({escaped_variable_name}) : Capture the exact variable name.
    # \s*=\s* : An equals sign, surrounded by optional whitespace.
    # (r?\"\"\")               : Capture the opening triple quotes (raw or normal string).
    # (.*?)                    : Capture the content (non-greedy).
    # (\"\"\")                  : Match the closing triple quotes.
    regex_pattern = rf"^\s*({escaped_variable_name})\s*=\s*(r?\"\"\")(.*?)(\"\"\")"

    flags = re.MULTILINE | re.DOTALL

    # Use re.search to find the first match
    match = re.search(regex_pattern, file_content, flags)

    if match:
        # match.group(1) will be the variable_name itself
        # match.group(3) will be the content inside the triple quotes
        content = match.group(3).strip()
        return content
    return None


def update_file_with_new_docstrings(
    candidate_file, lines, line_starts_candidates, line_ends_candidates, overwrite=False
):
    """
    For a given file, update the docstrings for all @auto_docstring candidates and write the new content.
    """
    content_base_file_new_lines = lines[: line_ends_candidates[0]]
    current_line_start = line_starts_candidates[0]
    current_line_end = line_ends_candidates[0]
    index = 1
    missing_docstring_args_warnings = []
    fill_docstring_args_warnings = []
    docstring_args_ro_remove_warnings = []

    while index <= len(line_starts_candidates):
        custom_args_dict = {}
        auto_docstring_signature_content = "".join(lines[current_line_start:current_line_end])
        match = re.findall(r"custom_args=(\w+)", auto_docstring_signature_content)
        if match:
            custom_args_var_name = match[0]
            custom_args_var_content = find_custom_args_with_details("\n".join(lines), custom_args_var_name)
            if custom_args_var_content:
                custom_args_dict, _ = parse_docstring(custom_args_var_content)
        new_docstring = ""
        modify_class_docstring = False
        # Function
        if "    def" in lines[current_line_end]:
            (
                new_docstring,
                sig_line_end,
                docstring_end,
                missing_docstring_args,
                fill_docstring_args,
                docstring_args_ro_remove,
            ) = generate_new_docstring_for_function(lines, current_line_end, custom_args_dict)
        # Class
        elif "class " in lines[current_line_end]:
            (
                new_docstring,
                class_sig_line_end,
                class_docstring_end_line,
                missing_docstring_args,
                fill_docstring_args,
                docstring_args_ro_remove,
            ) = generate_new_docstring_for_class(lines, current_line_end, custom_args_dict)
            modify_class_docstring = class_sig_line_end is not None
        # Add warnings if needed
        if missing_docstring_args:
            for arg in missing_docstring_args:
                missing_docstring_args_warnings.append(f"    - {arg} line {current_line_end}")
        if fill_docstring_args:
            for arg in fill_docstring_args:
                fill_docstring_args_warnings.append(f"    - {arg} line {current_line_end}")
        if docstring_args_ro_remove:
            for arg in docstring_args_ro_remove:
                docstring_args_ro_remove_warnings.append(f"    - {arg} line {current_line_end}")
        # Write new lines
        if index >= len(line_ends_candidates) or line_ends_candidates[index] > current_line_end:
            if "    def" in lines[current_line_end]:
                content_base_file_new_lines += lines[current_line_end:sig_line_end]
                if new_docstring != "":
                    content_base_file_new_lines += new_docstring.split("\n")
                if index < len(line_ends_candidates):
                    content_base_file_new_lines += lines[docstring_end + 1 : line_ends_candidates[index]]
                else:
                    content_base_file_new_lines += lines[docstring_end + 1 :]
            elif modify_class_docstring:
                content_base_file_new_lines += lines[current_line_end:class_sig_line_end]
                if new_docstring != "":
                    content_base_file_new_lines += new_docstring.split("\n")
                if index < len(line_ends_candidates):
                    content_base_file_new_lines += lines[class_docstring_end_line + 1 : line_ends_candidates[index]]
                else:
                    content_base_file_new_lines += lines[class_docstring_end_line + 1 :]
            elif index < len(line_ends_candidates):
                content_base_file_new_lines += lines[current_line_end : line_ends_candidates[index]]
            else:
                content_base_file_new_lines += lines[current_line_end:]
            if index < len(line_ends_candidates):
                current_line_end = line_ends_candidates[index]
                current_line_start = line_starts_candidates[index]
        index += 1
    content_base_file_new = "\n".join(content_base_file_new_lines)
    if overwrite:
        with open(candidate_file, "w", encoding="utf-8") as f:
            f.write(content_base_file_new)

    return (
        missing_docstring_args_warnings,
        fill_docstring_args_warnings,
        docstring_args_ro_remove_warnings,
    )


# TODO (Yoni): The functions in check_auto_docstrings rely on direct code parsing, which is prone to
# failure on edge cases and not robust to code changes. While this approach is significantly faster
# than using inspect (like in check_docstrings) and allows parsing any object including non-public
# ones, it may need to be refactored in the future to use a more robust parsing method. Note that
# we still need auto_docstring for some non-public objects since their docstrings are included in the
# docs of public objects (e.g. ModelOutput classes).
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
    # 3. For each file, update docstrings for all candidates
    for candidate_file in auto_docstrings_files:
        with open(candidate_file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        line_starts_candidates, line_ends_candidates = get_auto_docstring_candidate_lines(lines)
        missing_docstring_args_warnings, fill_docstring_args_warnings, docstring_args_ro_remove_warnings = (
            update_file_with_new_docstrings(
                candidate_file, lines, line_starts_candidates, line_ends_candidates, overwrite=overwrite
            )
        )
        if missing_docstring_args_warnings:
            if not overwrite:
                print(
                    "Some docstrings are missing. Run `make fix-copies` or `python utils/check_docstrings.py --fix_and_overwrite` to generate the docstring templates where needed."
                )
            print(f"🚨 Missing docstring for the following arguments in {candidate_file}:")
            for warning in missing_docstring_args_warnings:
                print(warning)
        if docstring_args_ro_remove_warnings:
            if not overwrite:
                print(
                    "Some docstrings are redundant with the ones in `args_doc.py` and will be removed. Run `make fix-copies` or `python utils/check_docstrings.py --fix_and_overwrite` to remove the redundant docstrings."
                )
            print(f"🚨 Redundant docstring for the following arguments in {candidate_file}:")
            for warning in docstring_args_ro_remove_warnings:
                print(warning)
        if fill_docstring_args_warnings:
            print(f"🚨 Docstring needs to be filled for the following arguments in {candidate_file}:")
            for warning in fill_docstring_args_warnings:
                print(warning)


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
        print("    Checking docstrings in the following files:" + "\n    - " + "\n    - ".join(module_diff_files))

    failures = []
    hard_failures = []
    to_clean = []
    for name in dir(transformers):
        # Skip objects that are private or not documented.
        if name.startswith("_") or ignore_undocumented(name) or name in OBJECTS_TO_IGNORE:
            continue

        obj = getattr(transformers, name)
        if not callable(obj) or not isinstance(obj, type) or getattr(obj, "__doc__", None) is None:
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
            "The following objects docstrings do not match their signature. Run `make fix-copies` to fix this. "
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
