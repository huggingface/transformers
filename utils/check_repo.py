# Copyright 2020 The HuggingFace Inc. team.
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
Utility that performs several consistency checks on the repo. This includes:
- checking all models are properly defined in the __init__ of models/
- checking all models are in the main __init__
- checking all models are properly tested
- checking all object in the main __init__ are documented
- checking all models are in at least one auto class
- checking all the auto mapping are properly defined (no typos, importable)
- checking the list of deprecated models is up to date

Use from the root of the repo with (as used in `make check-repo`):

```bash
python utils/check_repo.py
```

It has no auto-fix mode.
"""

import ast
import os
import re
import types
import warnings
from collections import OrderedDict
from difflib import get_close_matches
from pathlib import Path

from transformers import is_torch_available
from transformers.models.auto.auto_factory import get_values
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.feature_extraction_auto import FEATURE_EXTRACTOR_MAPPING_NAMES
from transformers.models.auto.image_processing_auto import IMAGE_PROCESSOR_MAPPING_NAMES
from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
from transformers.testing_utils import _COMMON_MODEL_NAMES_MAP
from transformers.utils import ENV_VARS_TRUE_VALUES, direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_repo.py
PATH_TO_TRANSFORMERS = "src/transformers"
PATH_TO_TESTS = "tests"
PATH_TO_DOC = "docs/source/en"

# Update this list with models that are supposed to be private.
PRIVATE_MODELS = [
    "AltRobertaModel",
    "DPRSpanPredictor",
    "UdopStack",
    "LongT5Stack",
    "RealmBertModel",
    "T5Stack",
    "MT5Stack",
    "UMT5Stack",
    "Pop2PianoStack",
    "Qwen2AudioEncoder",
    "Qwen2VisionTransformerPretrainedModel",
    "Qwen2_5_VisionTransformerPretrainedModel",
    "Qwen3VLVisionModel",
    "Qwen3VLMoeVisionModel",
    "Qwen3_5VisionModel",
    "Qwen3_5MoeVisionModel",
    "SwitchTransformersStack",
    "SiglipTextTransformer",
    "Siglip2TextTransformer",
    "MaskFormerSwinModel",
    "MaskFormerSwinPreTrainedModel",
    "BridgeTowerTextModel",
    "BridgeTowerVisionModel",
    "Kosmos2TextModel",
    "Kosmos2TextForCausalLM",
    "Kosmos2VisionModel",
    "SeamlessM4Tv2TextToUnitModel",
    "SeamlessM4Tv2CodeHifiGan",
    "SeamlessM4Tv2TextToUnitForConditionalGeneration",
    "Idefics2PerceiverResampler",
    "Idefics2VisionTransformer",
    "Idefics3VisionTransformer",
    "Kosmos2_5TextModel",
    "Kosmos2_5TextForCausalLM",
    "Kosmos2_5VisionModel",
    "SmolVLMVisionTransformer",
    "SiglipVisionTransformer",
    "Siglip2VisionTransformer",
    "AriaTextForCausalLM",
    "AriaTextModel",
    "Phi4MultimodalAudioModel",
    "Phi4MultimodalVisionModel",
    "Glm4vVisionModel",
    "Glm4vMoeVisionModel",
    "GlmImageVisionModel",
    "GlmOcrVisionModel",
    "EvollaSaProtPreTrainedModel",
    "BltLocalEncoder",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
    "BltLocalDecoder",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
    "BltGlobalTransformer",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
    "Ovis2VisionModel",
    "PeAudioPreTrainedModel",
    "PeAudioVideoPreTrainedModel",
    "PeVideoPreTrainedModel",
    # the following models should have been PreTrainedModels
    "Owlv2TextTransformer",
    "Owlv2VisionTransformer",
    "OwlViTTextTransformer",
    "OwlViTVisionTransformer",
    "XCLIPTextTransformer",
    "CLIPSegTextTransformer",
    "DetrDecoder",
    "GroupViTTextTransformer",
    "CLIPTextTransformer",
    "CLIPVisionTransformer",
    "MetaClip2TextTransformer",
    "MetaClip2VisionTransformer",
    "MLCDVisionTransformer",
    # end of should have beens
    "VoxtralRealtimeTextModel",
    "VoxtralRealtimeTextForCausalLM",
    "VoxtralRealtimeTextPreTrainedModel",
]

# Update this list for models that are not tested with a comment explaining the reason it should not be.
# Being in this list is an exception and should **not** be the rule.
IGNORE_NON_TESTED = (
    PRIVATE_MODELS.copy()
    + [
        # models to ignore for not tested
        "RecurrentGemmaModel",  # Building part of bigger (tested) model.
        "FuyuForCausalLM",  # Not tested fort now
        "InstructBlipQFormerModel",  # Building part of bigger (tested) model.
        "InstructBlipVideoQFormerModel",  # Building part of bigger (tested) model.
        "UMT5EncoderModel",  # Building part of bigger (tested) model.
        "Blip2QFormerModel",  # Building part of bigger (tested) model.
        "ErnieMForInformationExtraction",
        "FastSpeech2ConformerHifiGan",  # Already tested by SpeechT5HifiGan (# Copied from)
        "FastSpeech2ConformerWithHifiGan",  # Built with two smaller (tested) models.
        "GlmImageVQVAE",  # Building part of bigger (tested) model.
        "GraphormerDecoderHead",  # Building part of bigger (tested) model.
        "JukeboxVQVAE",  # Building part of bigger (tested) model.
        "JukeboxPrior",  # Building part of bigger (tested) model.
        "DecisionTransformerGPT2Model",  # Building part of bigger (tested) model.
        "SegformerDecodeHead",  # Building part of bigger (tested) model.
        "MgpstrModel",  # Building part of bigger (tested) model.
        "BertLMHeadModel",  # Needs to be setup as decoder.
        "MegatronBertLMHeadModel",  # Building part of bigger (tested) model.
        "RealmBertModel",  # Building part of bigger (tested) model.
        "RealmReader",  # Not regular model.
        "RealmScorer",  # Not regular model.
        "RealmForOpenQA",  # Not regular model.
        "ReformerForMaskedLM",  # Needs to be setup as decoder.
        "SeparableConv1D",  # Building part of bigger (tested) model.
        "OPTDecoderWrapper",
        "AltRobertaModel",  # Building part of bigger (tested) model.
        "BlipTextLMHeadModel",  # No need to test it as it is tested by BlipTextVision models
        "BridgeTowerTextModel",  # No need to test it as it is tested by BridgeTowerModel model.
        "BridgeTowerVisionModel",  # No need to test it as it is tested by BridgeTowerModel model.
        "BarkCausalModel",  # Building part of bigger (tested) model.
        "BarkModel",  # Does not have a forward signature - generation tested with integration tests.
        "Sam2HieraDetModel",  # Building part of bigger (tested) model.
        "Sam3TrackerVideoModel",  # Partly tested in Sam3TrackerModel, not regular model.
        "Sam2VideoModel",  # Partly tested in Sam2Model, not regular model.
        "Sam3ViTModel",  # Building part of bigger (tested) model.
        "Sam3VideoModel",  # Partly tested in Sam3Model, not regular model.
        "EdgeTamVisionModel",  # Building part of bigger (tested) model.
        "EdgeTamVideoModel",  # Partly tested in EdgeTamModel, not regular model.
        "SeamlessM4TTextToUnitModel",  # Building part of bigger (tested) model.
        "SeamlessM4TCodeHifiGan",  # Building part of bigger (tested) model.
        "SeamlessM4TTextToUnitForConditionalGeneration",  # Building part of bigger (tested) model.
        "ChameleonVQVAE",  # VQVAE here is used only for encoding (discretizing) and is tested as part of bigger model
        "PPDocLayoutV3Model",  # Building part of bigger (tested) model. Tested implicitly through PPDocLayoutV3ForObjectDetection.
        "PaddleOCRVLModel",  # Building part of bigger (tested) model. Tested implicitly through PaddleOCRVLForConditionalGeneration.
        "PaddleOCRVisionModel",  # Building part of bigger (tested) model. Tested implicitly through PaddleOCRVLForConditionalGeneration.
        "PaddleOCRVisionTransformer",  # Building part of bigger (tested) model. Tested implicitly through PaddleOCRVLForConditionalGeneration.
        "PaddleOCRTextModel",  # Building part of bigger (tested) model. Tested implicitly through PaddleOCRVLForConditionalGeneration.
        "Qwen2VLModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2VLForConditionalGeneration.
        "Qwen2_5_VLModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2_5_VLForConditionalGeneration.
        "Qwen3VLModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen3VLForConditionalGeneration.
        "Qwen3VLMoeModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen3VLMoeForConditionalGeneration.
        "Qwen3VLTextModel",  # Building part of bigger (tested) model.
        "Qwen3VLMoeTextModel",  # Building part of bigger (tested) model.
        "Qwen3_5TextModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen3_5ForConditionalGeneration.
        "Qwen3_5MoeTextModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen3_5MoeForConditionalGeneration.
        "Qwen2_5OmniForConditionalGeneration",  # Not a regular model. Testted in Qwen2_5OmniModelIntergrationTest
        "Qwen2_5OmniTalkerForConditionalGeneration",  #  Building part of bigger (tested) model. Tested implicitly through Qwen2_5OmniModelIntergrationTest.
        "Qwen2_5OmniTalkerModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2_5OmniModelIntergrationTest.
        "Qwen2_5OmniThinkerTextModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2_5OmniModelIntergrationTest.
        "Qwen2_5OmniToken2WavModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2_5OmniModelIntergrationTest.
        "Qwen2_5OmniToken2WavDiTModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2_5OmniModelIntergrationTest.
        "Qwen2_5OmniToken2WavBigVGANModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen2_5OmniModelIntergrationTest.
        "Qwen3OmniMoeCode2Wav",  # Building part of bigger (tested) model. Tested implicitly through Qwen3OmniMoeForConditionalGenerationIntegrationTest.
        "Qwen3OmniMoeCode2WavDecoderBlock",
        "Qwen3OmniMoeText2Wav",  # Building part of bigger (tested) model. Tested implicitly through Qwen3OmniMoeForConditionalGenerationIntegrationTest.
        "Qwen3OmniMoeTalkerCodePredictorModel",  # Building part of bigger (tested) model. Tested implicitly through Qwen3OmniMoeForConditionalGenerationIntegrationTest.
        "Qwen3OmniMoeCode2WavTransformerModel",
        "Qwen3OmniMoeTalkerForConditionalGeneration",
        "Qwen3OmniMoeTalkerModel",
        "Qwen3OmniMoeThinkerTextModel",
        "Qwen3OmniMoeForConditionalGeneration",  # Bigger model tested through Qwen3OmniMoeForConditionalGenerationIntegrationTest.
        "Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration",  # Building part of bigger (tested) model. Tested implicitly through Qwen3OmniMoeForConditionalGenerationIntegrationTest.
        "MllamaTextModel",  # Building part of bigger (tested) model. # TODO: add tests
        "MllamaVisionModel",  # Building part of bigger (tested) model. # TODO: add tests
        "Llama4TextModel",  # Building part of bigger (tested) model. # TODO: add tests
        "Llama4VisionModel",  # Building part of bigger (tested) model. # TODO: add tests
        "Emu3VQVAE",  # Building part of bigger (tested) model
        "Emu3TextModel",  # Building part of bigger (tested) model
        "Glm4vTextModel",  # Building part of bigger (tested) model
        "Glm4vMoeTextModel",  # Building part of bigger (tested) model
        "GlmImageTextModel",  # Building part of bigger (tested) model
        "GlmOcrTextModel",  # Building part of bigger (tested) model
        "Qwen2VLTextModel",  # Building part of bigger (tested) model
        "Qwen2_5_VLTextModel",  # Building part of bigger (tested) model
        "InternVLVisionModel",  # Building part of bigger (tested) model
        "JanusVisionModel",  # Building part of bigger (tested) model
        "PPDocLayoutV3Model",  # Building part of bigger (tested) model
        "TimesFmModel",  # Building part of bigger (tested) model
        "CsmDepthDecoderForCausalLM",  # Building part of bigger (tested) model. Tested implicitly through CsmForConditionalGenerationIntegrationTest.
        "CsmDepthDecoderModel",  # Building part of bigger (tested) model. Tested implicitly through CsmForConditionalGenerationIntegrationTest.
        "CsmBackboneModel",  # Building part of bigger (tested) model. Tested implicitly through CsmForConditionalGenerationIntegrationTest.
        "BltPatcher",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
        "BltLocalEncoder",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
        "BltLocalDecoder",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
        "BltGlobalTransformer",  # Building part of bigger (tested) model. Tested implicitly through BLTForCausalLM.
        "Florence2VisionBackbone",  # Building part of bigger (tested) model. Tested implicitly through Florence2ForConditionalGeneration.
        "HiggsAudioV2Model",  # Building part of bigger (tested) model. Tested implicitly through HiggsAudioV2ForConditionalGenerationIntegrationTest.
        "Ernie4_5_VL_MoeTextModel",  # Building part of bigger (tested) model
        "PeAudioFrameLevelModel",
        "PeAudioVideoModel",
    ]
)

# Update this list with test files that don't have a tester with a `all_model_classes` variable and which don't
# trigger the common tests.
TEST_FILES_WITH_NO_COMMON_TESTS = [
    "models/decision_transformer/test_modeling_decision_transformer.py",
    "models/camembert/test_modeling_camembert.py",
    "models/mbart/test_modeling_mbart.py",
    "models/mt5/test_modeling_mt5.py",
    "models/pegasus/test_modeling_pegasus.py",
    "models/xlm_prophetnet/test_modeling_xlm_prophetnet.py",
    "models/xlm_roberta/test_modeling_xlm_roberta.py",
    "models/vision_text_dual_encoder/test_modeling_vision_text_dual_encoder.py",
    "models/decision_transformer/test_modeling_decision_transformer.py",
    "models/bark/test_modeling_bark.py",
    "models/shieldgemma2/test_modeling_shieldgemma2.py",
    "models/llama4/test_modeling_llama4.py",
    "models/sam2_video/test_modeling_sam2_video.py",
    "models/sam3_tracker_video/test_modeling_sam3_tracker_video.py",
    "models/sam3_video/test_modeling_sam3_video.py",
    "models/edgetam_video/test_modeling_edgetam_video.py",
]

# Update this list for models that are not in any of the auto MODEL_XXX_MAPPING. Being in this list is an exception and
# should **not** be the rule.
IGNORE_NON_AUTO_CONFIGURED = PRIVATE_MODELS.copy() + [
    # models to ignore for model xxx mapping
    "Aimv2TextModel",
    "AlignTextModel",
    "AlignVisionModel",
    "ClapTextModel",
    "ClapTextModelWithProjection",
    "ClapAudioModel",
    "ClapAudioModelWithProjection",
    "Blip2TextModelWithProjection",
    "Blip2VisionModelWithProjection",
    "Blip2VisionModel",
    "ErnieMForInformationExtraction",
    "FastSpeech2ConformerHifiGan",
    "FastSpeech2ConformerWithHifiGan",
    "GitVisionModel",
    "GraphormerModel",
    "GraphormerForGraphClassification",
    "BlipForImageTextRetrieval",
    "BlipForQuestionAnswering",
    "BlipVisionModel",
    "BlipTextLMHeadModel",
    "BlipTextModel",
    "BrosSpadeEEForTokenClassification",
    "BrosSpadeELForTokenClassification",
    "Swin2SRForImageSuperResolution",
    "BridgeTowerForImageAndTextRetrieval",
    "BridgeTowerForMaskedLM",
    "BridgeTowerForContrastiveLearning",
    "CLIPSegForImageSegmentation",
    "CLIPSegVisionModel",
    "CLIPSegTextModel",
    "EsmForProteinFolding",
    "GPTSanJapaneseModel",
    "TimeSeriesTransformerForPrediction",
    "InformerForPrediction",
    "AutoformerForPrediction",
    "PatchTSTForPretraining",
    "PatchTSTForPrediction",
    "JukeboxVQVAE",
    "JukeboxPrior",
    "SamModel",
    "Sam2Model",
    "Sam2VideoModel",
    "EdgeTamModel",
    "EdgeTamVideoModel",
    "SamHQModel",
    "DPTForDepthEstimation",
    "DecisionTransformerGPT2Model",
    "GLPNForDepthEstimation",
    "ViltForImagesAndTextClassification",
    "ViltForImageAndTextRetrieval",
    "ViltForTokenClassification",
    "ViltForMaskedLM",
    "PerceiverForMultimodalAutoencoding",
    "PerceiverForOpticalFlow",
    "SegformerDecodeHead",
    "BeitForMaskedImageModeling",
    "ChineseCLIPTextModel",
    "ChineseCLIPVisionModel",
    "CLIPTextModelWithProjection",
    "CLIPVisionModelWithProjection",
    "ClvpForCausalLM",
    "ClvpModel",
    "GroupViTTextModel",
    "GroupViTVisionModel",
    "DetrForSegmentation",
    "Pix2StructVisionModel",
    "Pix2StructTextModel",
    "ConditionalDetrForSegmentation",
    "DPRReader",
    "FlaubertForQuestionAnswering",
    "FlavaImageCodebook",
    "FlavaTextModel",
    "FlavaImageModel",
    "FlavaMultimodalModel",
    "GlmImageForConditionalGeneration",
    "GPT2DoubleHeadsModel",
    "GPTSw3DoubleHeadsModel",
    "InstructBlipVisionModel",
    "InstructBlipQFormerModel",
    "InstructBlipVideoVisionModel",
    "InstructBlipVideoQFormerModel",
    "LayoutLMForQuestionAnswering",
    "LukeForMaskedLM",
    "LukeForEntityClassification",
    "LukeForEntityPairClassification",
    "LukeForEntitySpanClassification",
    "MgpstrModel",
    "OpenAIGPTDoubleHeadsModel",
    "OwlViTTextModel",
    "OwlViTVisionModel",
    "Owlv2TextModel",
    "Owlv2VisionModel",
    "OwlViTForObjectDetection",
    "PatchTSMixerForPrediction",
    "PatchTSMixerForPretraining",
    "RagModel",
    "RagSequenceForGeneration",
    "RagTokenForGeneration",
    "RealmEmbedder",
    "RealmForOpenQA",
    "RealmScorer",
    "RealmReader",
    "Wav2Vec2ForCTC",
    "HubertForCTC",
    "SEWForCTC",
    "SEWDForCTC",
    "XLMForQuestionAnswering",
    "XLNetForQuestionAnswering",
    "SeparableConv1D",
    "VisualBertForRegionToPhraseAlignment",
    "VisualBertForVisualReasoning",
    "VisualBertForQuestionAnswering",
    "VisualBertForMultipleChoice",
    "XCLIPVisionModel",
    "XCLIPTextModel",
    "AltCLIPTextModel",
    "AltCLIPVisionModel",
    "AltRobertaModel",
    "TvltForAudioVisualClassification",
    "BarkCausalModel",
    "BarkCoarseModel",
    "BarkFineModel",
    "BarkSemanticModel",
    "MusicgenMelodyModel",
    "MusicgenModel",
    "MusicgenForConditionalGeneration",
    "SpeechT5ForSpeechToSpeech",
    "SpeechT5ForTextToSpeech",
    "SpeechT5HifiGan",
    "VitMatteForImageMatting",
    "SeamlessM4TTextToUnitModel",
    "SeamlessM4TTextToUnitForConditionalGeneration",
    "SeamlessM4TCodeHifiGan",
    "SeamlessM4TForSpeechToSpeech",  # no auto class for speech-to-speech
    "TvpForVideoGrounding",
    "SeamlessM4Tv2NARTextToUnitModel",
    "SeamlessM4Tv2NARTextToUnitForConditionalGeneration",
    "SeamlessM4Tv2CodeHifiGan",
    "SeamlessM4Tv2ForSpeechToSpeech",  # no auto class for speech-to-speech
    "SegGptForImageSegmentation",
    "SiglipVisionModel",
    "SiglipTextModel",
    "SiglipVisionTransformer",
    "Siglip2VisionModel",
    "Siglip2VisionTransformer",
    "Siglip2TextModel",
    "ChameleonVQVAE",  # no autoclass for VQ-VAE models
    "VitPoseForPoseEstimation",
    "CLIPTextModel",
    "MetaClip2TextModel",
    "MetaClip2TextModelWithProjection",
    "MetaClip2VisionModel",
    "MetaClip2VisionModelWithProjection",
    "MoshiForConditionalGeneration",  # no auto class for speech-to-speech
    "Emu3VQVAE",  # no autoclass for VQ-VAE models
    "Emu3TextModel",  # Building part of bigger (tested) model
    "JanusVQVAE",  # no autoclass for VQ-VAE models
    "JanusVisionModel",  # Building part of bigger (tested) model
    "PaddleOCRVLModel",  # Building part of bigger (tested) model
    "PaddleOCRVisionModel",  # Building part of bigger (tested) model
    "PaddleOCRVisionTransformer",  # Building part of bigger (tested) model
    "PaddleOCRTextModel",  # Building part of bigger (tested) model
    "Qwen2_5OmniTalkerForConditionalGeneration",  # Building part of a bigger model
    "Qwen2_5OmniTalkerModel",  # Building part of a bigger model
    "Qwen2_5OmniThinkerForConditionalGeneration",  # Building part of a bigger model
    "Qwen2_5OmniThinkerTextModel",  # Building part of a bigger model
    "Qwen2_5OmniToken2WavModel",  # Building part of a bigger model
    "Qwen2_5OmniToken2WavBigVGANModel",  # Building part of a bigger model
    "Qwen2_5OmniToken2WavDiTModel",  # Building part of a bigger model
    "CsmBackboneModel",  # Building part of a bigger model
    "CsmDepthDecoderModel",  # Building part of a bigger model
    "CsmDepthDecoderForCausalLM",  # Building part of a bigger model
    "CsmForConditionalGeneration",  # Building part of a bigger model
    "BltPatcher",  # Building part of a bigger model, tested implicitly through BltForCausalLM
    "Florence2VisionBackbone",  # Building part of a bigger model
    "HiggsAudioV2Model",  # Building part of a bigger model
    "Qwen3OmniMoeCode2Wav",  # Building part of a bigger model
    "Qwen3OmniMoeCode2WavTransformerModel",  # Building part of a bigger model
    "Qwen3OmniMoeTalkerCodePredictorModel",  # Building part of a bigger model
    "Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration",  # Building part of a bigger model
    "Qwen3OmniMoeTalkerForConditionalGeneration",  # Building part of a bigger model
    "Qwen3OmniMoeTalkerModel",  # Building part of a bigger model
    "Qwen3OmniMoeThinkerForConditionalGeneration",  # Building part of a bigger model
    "Qwen3OmniMoeThinkerTextModel",  # Building part of a bigger model
    "Ernie4_5_VL_MoeTextModel",  # Building part of a bigger model
    "PeAudioFrameLevelModel",
]


# Update this list for models that have multiple model types for the same model doc.
MODEL_TYPE_TO_DOC_MAPPING = OrderedDict(
    [
        ("data2vec-text", "data2vec"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("kosmos-2.5", "kosmos2_5"),
        ("dinov3_convnext", "dinov3"),
        ("dinov3_vit", "dinov3"),
    ]
)


# This is to make sure the transformers module imported is the one in the repo.
transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)


def check_missing_backends():
    """
    Checks if all backends are installed (otherwise the check of this script is incomplete). Will error in the CI if
    that's not the case but only throw a warning for users running this.
    """
    missing_backends = []
    if not is_torch_available():
        missing_backends.append("PyTorch")

    if len(missing_backends) > 0:
        missing = ", ".join(missing_backends)
        if os.getenv("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
            raise Exception(
                "Full repo consistency checks require all backends to be installed (with `pip install -e '.[dev]'` in the "
                f"Transformers repo, the following are missing: {missing}."
            )
        else:
            warnings.warn(
                "Full repo consistency checks require all backends to be installed (with `pip install -e '.[dev]'` in the "
                f"Transformers repo, the following are missing: {missing}. While it's probably fine as long as you "
                "didn't make any change in one of those backends modeling files, you should probably execute the "
                "command above to be on the safe side."
            )


def check_model_list():
    """
    Checks the model listed as subfolders of `models` match the models available in `transformers.models`.
    """
    # Get the models from the directory structure of `src/transformers/models/`
    import transformers as tfrs

    models_dir = os.path.join(PATH_TO_TRANSFORMERS, "models")
    _models = []
    for model in os.listdir(models_dir):
        if model == "deprecated":
            continue
        model_dir = os.path.join(models_dir, model)
        if os.path.isdir(model_dir) and "__init__.py" in os.listdir(model_dir):
            # If the init is empty, and there are only two files, it's likely that there's just a conversion
            # script. Those should not be in the init.
            if (Path(model_dir) / "__init__.py").read_text().strip() == "":
                continue

            _models.append(model)

    # Get the models in the submodule `transformers.models`
    models = [model for model in dir(tfrs.models) if not model.startswith("__")]

    missing_models = sorted(set(_models).difference(models))
    if missing_models:
        raise Exception(
            f"The following models should be included in {models_dir}/__init__.py: {','.join(missing_models)}."
        )


# If some modeling modules should be ignored for all checks, they should be added in the nested list
# _ignore_modules of this function.
def get_model_modules() -> list[str]:
    """Get all the model modules inside the transformers library (except deprecated models)."""
    _ignore_modules = [
        "modeling_auto",
        "modeling_encoder_decoder",
        "modeling_marian",
        "modeling_retribert",
        "modeling_speech_encoder_decoder",
        "modeling_timm_backbone",
        "modeling_vision_encoder_decoder",
    ]
    modules = []
    for model in dir(transformers.models):
        # There are some magic dunder attributes in the dir, we ignore them
        if "deprecated" in model or model.startswith("__"):
            continue

        model_module = getattr(transformers.models, model)
        for submodule in dir(model_module):
            if submodule.startswith("modeling") and submodule not in _ignore_modules:
                modeling_module = getattr(model_module, submodule)
                modules.append(modeling_module)
    return modules


def get_models(module: types.ModuleType, include_pretrained: bool = False) -> list[tuple[str, type]]:
    """
    Get the objects in a module that are models.

    Args:
        module (`types.ModuleType`):
            The module from which we are extracting models.
        include_pretrained (`bool`, *optional*, defaults to `False`):
            Whether or not to include the `PreTrainedModel` subclass (like `BertPreTrainedModel`) or not.

    Returns:
        List[Tuple[str, type]]: List of models as tuples (class name, actual class).
    """
    models = []
    for attr_name in dir(module):
        if not include_pretrained and ("Pretrained" in attr_name or "PreTrained" in attr_name):
            continue
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, transformers.PreTrainedModel)
            and attr.__module__ == module.__name__
        ):
            models.append((attr_name, attr))
    return models


def is_building_block(model: str) -> bool:
    """
    Returns `True` if a model is a building block part of a bigger model.
    """
    if model.endswith("Wrapper"):
        return True
    if model.endswith("Encoder"):
        return True
    if model.endswith("Decoder"):
        return True
    if model.endswith("Prenet"):
        return True


def is_a_private_model(model: str) -> bool:
    """Returns `True` if the model should not be in the main init."""
    if model in PRIVATE_MODELS:
        return True
    return is_building_block(model)


def check_models_are_in_init():
    """Checks all models defined in the library are in the main init."""
    models_not_in_init = []
    dir_transformers = dir(transformers)
    for module in get_model_modules():
        models_not_in_init += [
            model[0] for model in get_models(module, include_pretrained=True) if model[0] not in dir_transformers
        ]

    # Remove private models
    models_not_in_init = [model for model in models_not_in_init if not is_a_private_model(model)]
    if len(models_not_in_init) > 0:
        raise Exception(f"The following models should be in the main init: {','.join(models_not_in_init)}.")


# If some test_modeling files should be ignored when checking models are all tested, they should be added in the
# nested list _ignore_files of this function.
def get_model_test_files() -> list[str]:
    """
    Get the model test files.

    Returns:
        `List[str]`: The list of test files. The returned files will NOT contain the `tests` (i.e. `PATH_TO_TESTS`
        defined in this script). They will be considered as paths relative to `tests`. A caller has to use
        `os.path.join(PATH_TO_TESTS, ...)` to access the files.
    """

    _ignore_files = [
        "test_modeling_common",
        "test_modeling_encoder_decoder",
        "test_modeling_marian",
    ]
    test_files = []
    model_test_root = os.path.join(PATH_TO_TESTS, "models")
    model_test_dirs = []
    for x in os.listdir(model_test_root):
        x = os.path.join(model_test_root, x)
        if os.path.isdir(x):
            model_test_dirs.append(x)

    for target_dir in [PATH_TO_TESTS] + model_test_dirs:
        for file_or_dir in os.listdir(target_dir):
            path = os.path.join(target_dir, file_or_dir)
            if os.path.isfile(path):
                filename = os.path.split(path)[-1]
                if "test_modeling" in filename and os.path.splitext(filename)[0] not in _ignore_files:
                    file = os.path.join(*path.split(os.sep)[1:])
                    test_files.append(file)

    return test_files


# This is a bit hacky but I didn't find a way to import the test_file as a module and read inside the tester class
# for the all_model_classes variable.
def find_tested_models(test_file: str) -> set[str]:
    """
    Parse the content of test_file to detect what's in `all_model_classes`. This detects the models that inherit from
    the common test class.

    Args:
        test_file (`str`): The path to the test file to check

    Returns:
        `Set[str]`: The set of models tested in that file.
    """
    with open(os.path.join(PATH_TO_TESTS, test_file), "r", encoding="utf-8", newline="\n") as f:
        content = f.read()

    model_tested = set()

    all_models = re.findall(r"all_model_classes\s+=\s+\(\s*\(([^\)]*)\)", content)
    # Check with one less parenthesis as well
    all_models += re.findall(r"all_model_classes\s+=\s+\(([^\)]*)\)", content)
    if len(all_models) > 0:
        for entry in all_models:
            for line in entry.split(","):
                name = line.strip()
                if len(name) > 0:
                    model_tested.add(name)

    # Models that inherit from `CausalLMModelTester` don't need to set `all_model_classes` -- it is built from other
    # attributes by default.
    if "CausalLMModelTester" in content:
        base_model_class = re.findall(r"base_model_class\s+=.*", content)  # Required attribute
        base_class = base_model_class[0].split("=")[1].strip()
        model_tested.add(base_class)

        model_name = base_class.replace("Model", "")
        # Optional attributes: if not set explicitly, the tester will attempt to infer and use the corresponding class
        for test_class_type in [
            "causal_lm_class",
            "sequence_classification_class",
            "question_answering_class",
            "token_classification_class",
        ]:
            tested_class = re.findall(rf"{test_class_type}\s+=.*", content)
            if tested_class:
                tested_class = tested_class[0].split("=")[1].strip()
            else:
                tested_class = model_name + _COMMON_MODEL_NAMES_MAP[test_class_type]
            model_tested.add(tested_class)

    return model_tested


def should_be_tested(model_name: str) -> bool:
    """
    Whether or not a model should be tested.
    """
    if model_name in IGNORE_NON_TESTED:
        return False
    return not is_building_block(model_name)


def check_models_are_tested(module: types.ModuleType, test_file: str) -> list[str]:
    """Check models defined in a module are all tested in a given file.

    Args:
        module (`types.ModuleType`): The module in which we get the models.
        test_file (`str`): The path to the file where the module is tested.

    Returns:
        `List[str]`: The list of error messages corresponding to models not tested.
    """
    # XxxPreTrainedModel are not tested
    defined_models = get_models(module)
    tested_models = find_tested_models(test_file)
    if len(tested_models) == 0:
        if test_file.replace(os.path.sep, "/") in TEST_FILES_WITH_NO_COMMON_TESTS:
            return
        return [
            f"{test_file} should define `all_model_classes` or inherit from `CausalLMModelTester` (and fill in the "
            "model class attributes) to apply common tests to the models it tests. "
            "If this intentional, add the test filename to `TEST_FILES_WITH_NO_COMMON_TESTS` in the file "
            "`utils/check_repo.py`."
        ]
    failures = []
    for model_name, _ in defined_models:
        if model_name not in tested_models and should_be_tested(model_name):
            failures.append(
                f"{model_name} is defined in {module.__name__} but is not tested in "
                f"{os.path.join(PATH_TO_TESTS, test_file)}. Add it to the `all_model_classes` in that file or, if "
                "it inherits from `CausalLMModelTester`, fill in the model class attributes. "
                "If common tests should not applied to that model, add its name to `IGNORE_NON_TESTED`"
                "in the file `utils/check_repo.py`."
            )
    return failures


def check_all_models_are_tested():
    """Check all models are properly tested."""
    modules = get_model_modules()
    test_files = get_model_test_files()
    failures = []
    for module in modules:
        # Matches a module to its test file.
        test_file = [file for file in test_files if f"test_{module.__name__.split('.')[-1]}.py" in file]
        if len(test_file) == 0:
            failures.append(f"{module.__name__} does not have its corresponding test file {test_file}.")
        elif len(test_file) > 1:
            failures.append(f"{module.__name__} has several test files: {test_file}.")
        else:
            test_file = test_file[0]
            new_failures = check_models_are_tested(module, test_file)
            if new_failures is not None:
                failures += new_failures
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


def get_all_auto_configured_models() -> list[str]:
    """Return the list of all models in at least one auto class."""
    result = set()  # To avoid duplicates we concatenate all model classes in a set.
    if is_torch_available():
        for attr_name in dir(transformers.models.auto.modeling_auto):
            if attr_name.startswith("MODEL_") and attr_name.endswith("MAPPING_NAMES"):
                result = result | set(get_values(getattr(transformers.models.auto.modeling_auto, attr_name)))
    return list(result)


def ignore_unautoclassed(model_name: str) -> bool:
    """Rules to determine if a model should be in an auto class."""
    # Special white list
    if model_name in IGNORE_NON_AUTO_CONFIGURED:
        return True
    # Encoder and Decoder should be ignored
    if "Encoder" in model_name or "Decoder" in model_name:
        return True
    return False


def check_models_are_auto_configured(module: types.ModuleType, all_auto_models: list[str]) -> list[str]:
    """
    Check models defined in module are each in an auto class.

    Args:
        module (`types.ModuleType`):
            The module in which we get the models.
        all_auto_models (`List[str]`):
            The list of all models in an auto class (as obtained with `get_all_auto_configured_models()`).

    Returns:
        `List[str]`: The list of error messages corresponding to models not tested.
    """
    defined_models = get_models(module)
    failures = []
    for model_name, _ in defined_models:
        if model_name not in all_auto_models and not ignore_unautoclassed(model_name):
            failures.append(
                f"{model_name} is defined in {module.__name__} but is not present in any of the auto mapping. "
                "If that is intended behavior, add its name to `IGNORE_NON_AUTO_CONFIGURED` in the file "
                "`utils/check_repo.py`."
            )
    return failures


def check_all_models_are_auto_configured():
    """Check all models are each in an auto class."""
    # This is where we need to check we have all backends or the check is incomplete.
    check_missing_backends()
    modules = get_model_modules()
    all_auto_models = get_all_auto_configured_models()
    failures = []
    for module in modules:
        new_failures = check_models_are_auto_configured(module, all_auto_models)
        if new_failures is not None:
            failures += new_failures
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


def check_all_auto_object_names_being_defined():
    """Check all names defined in auto (name) mappings exist in the library."""
    # This is where we need to check we have all backends or the check is incomplete.
    check_missing_backends()

    failures = []
    mappings_to_check = {
        "TOKENIZER_MAPPING_NAMES": TOKENIZER_MAPPING_NAMES,
        "IMAGE_PROCESSOR_MAPPING_NAMES": IMAGE_PROCESSOR_MAPPING_NAMES,
        "FEATURE_EXTRACTOR_MAPPING_NAMES": FEATURE_EXTRACTOR_MAPPING_NAMES,
        "PROCESSOR_MAPPING_NAMES": PROCESSOR_MAPPING_NAMES,
    }

    module = getattr(transformers.models.auto, "modeling_auto")
    # all mappings in a single auto modeling file
    mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES")]
    mappings_to_check.update({name: getattr(module, name) for name in mapping_names})

    for name, mapping in mappings_to_check.items():
        for class_names in mapping.values():
            if not isinstance(class_names, tuple):
                class_names = (class_names,)
                for class_name in class_names:
                    if class_name is None:
                        continue
                    # dummy object is accepted
                    if not hasattr(transformers, class_name):
                        # If the class name is in a model name mapping, let's not check if there is a definition in any modeling
                        # module, if it's a private model defined in this file.
                        if name.endswith("MODEL_MAPPING_NAMES") and is_a_private_model(class_name):
                            continue
                        if name.endswith("MODEL_FOR_IMAGE_MAPPING_NAMES") and is_a_private_model(class_name):
                            continue
                        failures.append(
                            f"`{class_name}` appears in the mapping `{name}` but it is not defined in the library."
                        )
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


def check_all_auto_mapping_names_in_config_mapping_names():
    """Check all keys defined in auto mappings (mappings of names) appear in `CONFIG_MAPPING_NAMES`."""
    # This is where we need to check we have all backends or the check is incomplete.
    check_missing_backends()

    failures = []
    # `TOKENIZER_PROCESSOR_MAPPING_NAMES` and `AutoTokenizer` is special, and don't need to follow the rule.
    mappings_to_check = {
        "IMAGE_PROCESSOR_MAPPING_NAMES": IMAGE_PROCESSOR_MAPPING_NAMES,
        "FEATURE_EXTRACTOR_MAPPING_NAMES": FEATURE_EXTRACTOR_MAPPING_NAMES,
        "PROCESSOR_MAPPING_NAMES": PROCESSOR_MAPPING_NAMES,
    }

    module = getattr(transformers.models.auto, "modeling_auto")
    # all mappings in a single auto modeling file
    mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES")]
    mappings_to_check.update({name: getattr(module, name) for name in mapping_names})

    for name, mapping in mappings_to_check.items():
        for model_type in mapping:
            if model_type not in CONFIG_MAPPING_NAMES:
                failures.append(
                    f"`{model_type}` appears in the mapping `{name}` but it is not defined in the keys of "
                    "`CONFIG_MAPPING_NAMES`."
                )
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


def check_all_auto_mappings_importable():
    """Check all auto mappings can be imported."""
    # This is where we need to check we have all backends or the check is incomplete.
    check_missing_backends()

    failures = []
    mappings_to_check = {}

    module = getattr(transformers.models.auto, "modeling_auto")
    # all mappings in a single auto modeling file
    mapping_names = [x for x in dir(module) if x.endswith("_MAPPING_NAMES")]
    mappings_to_check.update({name: getattr(module, name) for name in mapping_names})

    for name in mappings_to_check:
        name = name.replace("_MAPPING_NAMES", "_MAPPING")
        if not hasattr(transformers, name):
            failures.append(f"`{name}`")
    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))


_re_decorator = re.compile(r"^\s*@(\S+)\s+$")


def check_decorator_order(filename: str) -> list[int]:
    """
    Check that in a given test file, the slow decorator is always last.

    Args:
        filename (`str`): The path to a test file to check.

    Returns:
        `List[int]`: The list of failures as a list of indices where there are problems.
    """
    with open(filename, "r", encoding="utf-8", newline="\n") as f:
        lines = f.readlines()
    decorator_before = None
    errors = []
    for i, line in enumerate(lines):
        search = _re_decorator.search(line)
        if search is not None:
            decorator_name = search.groups()[0]
            if decorator_before is not None and decorator_name.startswith("parameterized"):
                errors.append(i)
            decorator_before = decorator_name
        elif decorator_before is not None:
            decorator_before = None
    return errors


def check_all_decorator_order():
    """Check that in all test files, the slow decorator is always last."""
    errors = []
    for fname in os.listdir(PATH_TO_TESTS):
        if fname.endswith(".py"):
            filename = os.path.join(PATH_TO_TESTS, fname)
            new_errors = check_decorator_order(filename)
            errors += [f"- {filename}, line {i}" for i in new_errors]
    if len(errors) > 0:
        msg = "\n".join(errors)
        raise ValueError(
            "The parameterized decorator (and its variants) should always be first, but this is not the case in the"
            f" following files:\n{msg}"
        )


def find_all_documented_objects() -> list[str]:
    """
    Parse the content of all doc files to detect which classes and functions it documents.

    Returns:
        `List[str]`: The list of all object names being documented.
        `Dict[str, List[str]]`: A dictionary mapping the object name (full import path, e.g.
            `integrations.PeftAdapterMixin`) to its documented methods
    """
    documented_obj = []
    documented_methods_map = {}
    for doc_file in Path(PATH_TO_DOC).glob("**/*.md"):
        with open(doc_file, "r", encoding="utf-8", newline="\n") as f:
            content = f.read()
        raw_doc_objs = re.findall(r"\[\[autodoc\]\]\s+(\S+)\s+", content)
        documented_obj += [obj.split(".")[-1] for obj in raw_doc_objs]

        for obj in raw_doc_objs:
            obj_public_methods = re.findall(rf"\[\[autodoc\]\] {obj}((\n\s+-.*)+)", content)
            # Some objects have no methods documented
            if len(obj_public_methods) == 0:
                continue
            else:
                documented_methods_map[obj] = re.findall(r"(?<=-\s).*", obj_public_methods[0][0])

    return documented_obj, documented_methods_map


# One good reason for not being documented is to be deprecated. Put in this list deprecated objects.
DEPRECATED_OBJECTS = [
    "PretrainedConfig",  # deprecated in favor of PreTrainedConfig
    "BartPretrainedModel",
    "DataCollator",
    "DataCollatorForSOP",
    "GlueDataset",
    "GlueDataTrainingArguments",
    "NerPipeline",
    "OwlViTFeatureExtractor",
    "PretrainedBartModel",
    "PretrainedFSMTModel",
    "SingleSentenceClassificationProcessor",
    "SquadDataTrainingArguments",
    "SquadDataset",
    "SquadExample",
    "SquadFeatures",
    "SquadV1Processor",
    "SquadV2Processor",
    "glue_compute_metrics",
    "glue_convert_examples_to_features",
    "glue_output_modes",
    "glue_processors",
    "glue_tasks_num_labels",
    "shape_list",
    "squad_convert_examples_to_features",
    "xnli_compute_metrics",
    "xnli_output_modes",
    "xnli_processors",
    "xnli_tasks_num_labels",
]

# Exceptionally, some objects should not be documented after all rules passed.
# ONLY PUT SOMETHING IN THIS LIST AS A LAST RESORT!
UNDOCUMENTED_OBJECTS = [
    "AddedToken",  # This is a tokenizers class.
    "BasicTokenizer",  # Internal, should never have been in the main init.
    "CharacterTokenizer",  # Internal, should never have been in the main init.
    "DPRPretrainedReader",  # Like an Encoder.
    "DummyObject",  # Just picked by mistake sometimes.
    "MecabTokenizer",  # Internal, should never have been in the main init.
    "SqueezeBertModule",  # Internal building block (should have been called SqueezeBertLayer)
    "TransfoXLCorpus",  # Internal type.
    "WordpieceTokenizer",  # Internal, should never have been in the main init.
    "absl",  # External module
    "add_end_docstrings",  # Internal, should never have been in the main init.
    "add_start_docstrings",  # Internal, should never have been in the main init.
    "logger",  # Internal logger
    "logging",  # External module
    "requires_backends",  # Internal function
    "AltRobertaModel",  # Internal module
    "VitPoseBackbone",  # Internal module
    "VitPoseBackboneConfig",  # Internal module
    "get_values",  # Internal object
]

# This list should be empty. Objects in it should get their own doc page.
SHOULD_HAVE_THEIR_OWN_PAGE = [
    "AutoBackbone",
    "BeitBackbone",
    "BitBackbone",
    "ConvNextBackbone",
    "ConvNextV2Backbone",
    "DinatBackbone",
    "Dinov2Backbone",
    "Dinov2WithRegistersBackbone",
    "FocalNetBackbone",
    "HieraBackbone",
    "MaskFormerSwinBackbone",
    "MaskFormerSwinConfig",
    "MaskFormerSwinModel",
    "NatBackbone",
    "PvtV2Backbone",
    "ResNetBackbone",
    "SwinBackbone",
    "Swinv2Backbone",
    "TextNetBackbone",
    "TimmBackbone",
    "TimmBackboneConfig",
    "VitDetBackbone",
    "RoFormerTokenizerFast",  # An alias
]


def ignore_undocumented(name: str) -> bool:
    """Rules to determine if `name` should be undocumented (returns `True` if it should not be documented)."""
    # NOT DOCUMENTED ON PURPOSE.
    # Constants uppercase are not documented.
    if name.isupper():
        return True
    # PreTrainedModels / Encoders / Decoders / Layers / Embeddings / Attention are not documented.
    if (
        name.endswith("PreTrainedModel")
        or name.endswith("Decoder")
        or name.endswith("Encoder")
        or name.endswith("Layer")
        or name.endswith("Embeddings")
        or name.endswith("Attention")
    ):
        return True
    # Submodules are not documented.
    if os.path.isdir(os.path.join(PATH_TO_TRANSFORMERS, name)) or os.path.isfile(
        os.path.join(PATH_TO_TRANSFORMERS, f"{name}.py")
    ):
        return True
    # All load functions are not documented.
    if name.startswith("load_pytorch"):
        return True
    # is_xxx_available functions are not documented.
    if name.startswith("is_") and name.endswith("_available"):
        return True
    # Deprecated objects are not documented.
    if name in DEPRECATED_OBJECTS or name in UNDOCUMENTED_OBJECTS:
        return True
    # MMBT model does not really work.
    if name.startswith("MMBT"):
        return True
    # BLT models are internal building blocks, tested implicitly through BltForCausalLM
    if name.startswith("Blt"):
        return True
    if name in SHOULD_HAVE_THEIR_OWN_PAGE:
        return True
    return False


def check_all_objects_are_documented():
    """Check all models are properly documented."""
    documented_objs, documented_methods_map = find_all_documented_objects()
    modules = transformers._modules
    # the objects with the following prefixes are not required to be in the docs
    ignore_prefixes = [
        "_",  # internal objects
    ]
    objects = [c for c in dir(transformers) if c not in modules and not any(c.startswith(p) for p in ignore_prefixes)]
    undocumented_objs = [c for c in objects if c not in documented_objs and not ignore_undocumented(c)]
    if len(undocumented_objs) > 0:
        raise Exception(
            "The following objects are in the public init, but not in the docs:\n - " + "\n - ".join(undocumented_objs)
        )
    check_model_type_doc_match()
    check_public_method_exists(documented_methods_map)


def check_public_method_exists(documented_methods_map):
    """Check that all explicitly documented public methods are defined in the corresponding class."""
    failures = []
    for obj, methods in documented_methods_map.items():
        # Let's ensure there is no repetition
        if len(set(methods)) != len(methods):
            failures.append(f"Error in the documentation of {obj}: there are repeated documented methods.")

        # Navigates into the object, given the full import path
        nested_path = obj.split(".")
        submodule = transformers
        if len(nested_path) > 1:
            nested_submodules = nested_path[:-1]
            for submodule_name in nested_submodules:
                if submodule_name == "transformers":
                    continue

                try:
                    submodule = getattr(submodule, submodule_name)
                except AttributeError:
                    failures.append(f"Could not parse {submodule_name}. Are the required dependencies installed?")
                continue

        class_name = nested_path[-1]

        try:
            obj_class = getattr(submodule, class_name)
        except AttributeError:
            failures.append(f"Could not parse {class_name}. Are the required dependencies installed?")
            continue

        # Checks that all explicitly documented methods are defined in the class
        for method in methods:
            if method == "all":  # Special keyword to document all public methods
                continue
            try:
                if not hasattr(obj_class, method):
                    failures.append(
                        "The following public method is explicitly documented but not defined in the corresponding "
                        f"class. class: {obj}, method: {method}. If the method is defined, this error can be due to "
                        f"lacking dependencies."
                    )
            except ImportError:
                pass

    if len(failures) > 0:
        raise Exception("\n".join(failures))


def check_model_type_doc_match():
    """Check all doc pages have a corresponding model type."""
    model_doc_folder = Path(PATH_TO_DOC) / "model_doc"
    model_docs = [m.stem for m in model_doc_folder.glob("*.md")]

    model_types = list(transformers.models.auto.configuration_auto.MODEL_NAMES_MAPPING.keys())
    model_types = [MODEL_TYPE_TO_DOC_MAPPING.get(m, m) for m in model_types]

    errors = []
    for m in model_docs:
        if m not in model_types and m != "auto":
            close_matches = get_close_matches(m, model_types)
            error_message = f"{m} is not a proper model identifier."
            if len(close_matches) > 0:
                close_matches = "/".join(close_matches)
                error_message += f" Did you mean {close_matches}?"
            errors.append(error_message)

    if len(errors) > 0:
        raise ValueError(
            "Some model doc pages do not match any existing model type:\n"
            + "\n".join(errors)
            + "\nYou can add any missing model type to the `MODEL_NAMES_MAPPING` constant in "
            "models/auto/configuration_auto.py."
        )


def check_deprecated_constant_is_up_to_date():
    """
    Check if the constant `DEPRECATED_MODELS` in `models/auto/configuration_auto.py` is up to date.
    """
    deprecated_folder = os.path.join(PATH_TO_TRANSFORMERS, "models", "deprecated")
    deprecated_models = [m for m in os.listdir(deprecated_folder) if not m.startswith("_")]

    constant_to_check = transformers.models.auto.configuration_auto.DEPRECATED_MODELS
    message = []
    missing_models = sorted(set(deprecated_models) - set(constant_to_check))
    if len(missing_models) != 0:
        missing_models = ", ".join(missing_models)
        message.append(
            "The following models are in the deprecated folder, make sure to add them to `DEPRECATED_MODELS` in "
            f"`models/auto/configuration_auto.py`: {missing_models}."
        )

    extra_models = sorted(set(constant_to_check) - set(deprecated_models))
    if len(extra_models) != 0:
        extra_models = ", ".join(extra_models)
        message.append(
            "The following models are in the `DEPRECATED_MODELS` constant but not in the deprecated folder. Either "
            f"remove them from the constant or move to the deprecated folder: {extra_models}."
        )

    if len(message) > 0:
        raise Exception("\n".join(message))


def check_models_have_kwargs():
    """
    Checks that all model classes defined in modeling files accept **kwargs in their forward pass.
    Since we ast.parse() here, it might be a good idea to add other tests that inspect modeling code here rather than
    repeatedly ast.parsing() in each test!
    """
    models_dir = Path(PATH_TO_TRANSFORMERS) / "models"
    failing_classes = []
    for model_dir in models_dir.iterdir():
        if model_dir.name == "deprecated":
            continue
        if model_dir.is_dir() and (modeling_file := list(model_dir.glob("modeling_*.py"))):
            modeling_file = modeling_file[0]

            with open(modeling_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            # Map all classes in the file to their base classes
            class_bases = {}
            all_class_nodes = {}

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # We only care about base classes that are simple names
                    bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                    class_bases[node.name] = bases
                    all_class_nodes[node.name] = node

            inherits_from_pretrained = {"PreTrainedModel"}
            # Loop over classes and mark the ones that inherit from PreTrainedModel, or from
            # previously marked classes (which indicates indirect inheritance from PreTrainedModel)
            # Keep going until you go through the whole list without discovering a new child class, then break
            while True:
                for class_name, bases in class_bases.items():
                    if class_name in inherits_from_pretrained:
                        continue
                    if inherits_from_pretrained.intersection(bases):
                        inherits_from_pretrained.add(class_name)
                        break
                else:
                    break

            # 2. Iterate through classes and check conditions
            for class_name, class_def in all_class_nodes.items():
                if class_name not in inherits_from_pretrained:
                    continue

                forward_method = next(
                    (n for n in class_def.body if isinstance(n, ast.FunctionDef) and n.name == "forward"), None
                )
                if forward_method:
                    # 3. Check for **kwargs (represented as .kwarg in AST)
                    if forward_method.args.kwarg is None:
                        failing_classes.append(class_name)

    if failing_classes:
        raise Exception(
            "The following model classes do not accept **kwargs in their forward() method: \n"
            f"{', '.join(failing_classes)}."
        )


def check_repo_quality():
    """Check all models are tested and documented."""
    print("Repository-wide checks:")
    print("    - checking all models are included.")
    check_model_list()
    print("    - checking all models are public.")
    check_models_are_in_init()
    print("    - checking all models have tests.")
    check_all_decorator_order()
    check_all_models_are_tested()
    print("    - checking all objects have documentation.")
    check_all_objects_are_documented()
    print("    - checking all models are in at least one auto class.")
    check_all_models_are_auto_configured()
    print("    - checking all names in auto name mappings are defined.")
    check_all_auto_object_names_being_defined()
    print("    - checking all keys in auto name mappings are defined in `CONFIG_MAPPING_NAMES`.")
    check_all_auto_mapping_names_in_config_mapping_names()
    print("    - checking all auto mappings could be imported.")
    check_all_auto_mappings_importable()
    print("    - checking the DEPRECATED_MODELS constant is up to date.")
    check_deprecated_constant_is_up_to_date()
    print("    - checking all models accept **kwargs in their call.")
    check_models_have_kwargs()


if __name__ == "__main__":
    check_repo_quality()
