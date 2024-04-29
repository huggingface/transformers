# coding=utf-8
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
from collections import OrderedDict

from ...utils import logging


logger = logging.get_logger(__name__)


class DeprecatedDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        logger.warning(
            "Archive maps are deprecated and will be removed in version v4.40.0 as they are no longer relevant. "
            "If looking to get all checkpoints for a given architecture, we recommend using `huggingface_hub` "
            "with the list_models method."
        )
        return self[item]


class DeprecatedList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, item):
        logger.warning_once(
            "Archive maps are deprecated and will be removed in version v4.40.0 as they are no longer relevant. "
            "If looking to get all checkpoints for a given architecture, we recommend using `huggingface_hub` "
            "with the `list_models` method."
        )
        return super().__getitem__(item)


ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "albert/albert-base-v1": "https://huggingface.co/albert/albert-base-v1/resolve/main/config.json",
        "albert/albert-large-v1": "https://huggingface.co/albert/albert-large-v1/resolve/main/config.json",
        "albert/albert-xlarge-v1": "https://huggingface.co/albert/albert-xlarge-v1/resolve/main/config.json",
        "albert/albert-xxlarge-v1": "https://huggingface.co/albert/albert-xxlarge-v1/resolve/main/config.json",
        "albert/albert-base-v2": "https://huggingface.co/albert/albert-base-v2/resolve/main/config.json",
        "albert/albert-large-v2": "https://huggingface.co/albert/albert-large-v2/resolve/main/config.json",
        "albert/albert-xlarge-v2": "https://huggingface.co/albert/albert-xlarge-v2/resolve/main/config.json",
        "albert/albert-xxlarge-v2": "https://huggingface.co/albert/albert-xxlarge-v2/resolve/main/config.json",
    }
)

ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "albert/albert-base-v1",
        "albert/albert-large-v1",
        "albert/albert-xlarge-v1",
        "albert/albert-xxlarge-v1",
        "albert/albert-base-v2",
        "albert/albert-large-v2",
        "albert/albert-xlarge-v2",
        "albert/albert-xxlarge-v2",
    ]
)

TF_ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "albert/albert-base-v1",
        "albert/albert-large-v1",
        "albert/albert-xlarge-v1",
        "albert/albert-xxlarge-v1",
        "albert/albert-base-v2",
        "albert/albert-large-v2",
        "albert/albert-xlarge-v2",
        "albert/albert-xxlarge-v2",
    ]
)

ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"kakaobrain/align-base": "https://huggingface.co/kakaobrain/align-base/resolve/main/config.json"}
)

ALIGN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["kakaobrain/align-base"])

ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"BAAI/AltCLIP": "https://huggingface.co/BAAI/AltCLIP/resolve/main/config.json"}
)

ALTCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["BAAI/AltCLIP"])

AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "MIT/ast-finetuned-audioset-10-10-0.4593": "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/resolve/main/config.json"
    }
)

AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["MIT/ast-finetuned-audioset-10-10-0.4593"]
)

AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "huggingface/autoformer-tourism-monthly": "https://huggingface.co/huggingface/autoformer-tourism-monthly/resolve/main/config.json"
    }
)

AUTOFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["huggingface/autoformer-tourism-monthly"])

BARK_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["suno/bark-small", "suno/bark"])

BART_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/bart-large"])

BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/beit-base-patch16-224-pt22k": "https://huggingface.co/microsoft/beit-base-patch16-224-pt22k/resolve/main/config.json"
    }
)

BEIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/beit-base-patch16-224"])

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json",
        "google-bert/bert-large-uncased": "https://huggingface.co/google-bert/bert-large-uncased/resolve/main/config.json",
        "google-bert/bert-base-cased": "https://huggingface.co/google-bert/bert-base-cased/resolve/main/config.json",
        "google-bert/bert-large-cased": "https://huggingface.co/google-bert/bert-large-cased/resolve/main/config.json",
        "google-bert/bert-base-multilingual-uncased": "https://huggingface.co/google-bert/bert-base-multilingual-uncased/resolve/main/config.json",
        "google-bert/bert-base-multilingual-cased": "https://huggingface.co/google-bert/bert-base-multilingual-cased/resolve/main/config.json",
        "google-bert/bert-base-chinese": "https://huggingface.co/google-bert/bert-base-chinese/resolve/main/config.json",
        "google-bert/bert-base-german-cased": "https://huggingface.co/google-bert/bert-base-german-cased/resolve/main/config.json",
        "google-bert/bert-large-uncased-whole-word-masking": "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking/resolve/main/config.json",
        "google-bert/bert-large-cased-whole-word-masking": "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking/resolve/main/config.json",
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad": "https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/config.json",
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad": "https://huggingface.co/google-bert/bert-large-cased-whole-word-masking-finetuned-squad/resolve/main/config.json",
        "google-bert/bert-base-cased-finetuned-mrpc": "https://huggingface.co/google-bert/bert-base-cased-finetuned-mrpc/resolve/main/config.json",
        "google-bert/bert-base-german-dbmdz-cased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-cased/resolve/main/config.json",
        "google-bert/bert-base-german-dbmdz-uncased": "https://huggingface.co/google-bert/bert-base-german-dbmdz-uncased/resolve/main/config.json",
        "cl-tohoku/bert-base-japanese": "https://huggingface.co/cl-tohoku/bert-base-japanese/resolve/main/config.json",
        "cl-tohoku/bert-base-japanese-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-whole-word-masking/resolve/main/config.json",
        "cl-tohoku/bert-base-japanese-char": "https://huggingface.co/cl-tohoku/bert-base-japanese-char/resolve/main/config.json",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking": "https://huggingface.co/cl-tohoku/bert-base-japanese-char-whole-word-masking/resolve/main/config.json",
        "TurkuNLP/bert-base-finnish-cased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-cased-v1/resolve/main/config.json",
        "TurkuNLP/bert-base-finnish-uncased-v1": "https://huggingface.co/TurkuNLP/bert-base-finnish-uncased-v1/resolve/main/config.json",
        "wietsedv/bert-base-dutch-cased": "https://huggingface.co/wietsedv/bert-base-dutch-cased/resolve/main/config.json",
    }
)

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google-bert/bert-base-uncased",
        "google-bert/bert-large-uncased",
        "google-bert/bert-base-cased",
        "google-bert/bert-large-cased",
        "google-bert/bert-base-multilingual-uncased",
        "google-bert/bert-base-multilingual-cased",
        "google-bert/bert-base-chinese",
        "google-bert/bert-base-german-cased",
        "google-bert/bert-large-uncased-whole-word-masking",
        "google-bert/bert-large-cased-whole-word-masking",
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
        "google-bert/bert-base-cased-finetuned-mrpc",
        "google-bert/bert-base-german-dbmdz-cased",
        "google-bert/bert-base-german-dbmdz-uncased",
        "cl-tohoku/bert-base-japanese",
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        "cl-tohoku/bert-base-japanese-char",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking",
        "TurkuNLP/bert-base-finnish-cased-v1",
        "TurkuNLP/bert-base-finnish-uncased-v1",
        "wietsedv/bert-base-dutch-cased",
    ]
)

TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google-bert/bert-base-uncased",
        "google-bert/bert-large-uncased",
        "google-bert/bert-base-cased",
        "google-bert/bert-large-cased",
        "google-bert/bert-base-multilingual-uncased",
        "google-bert/bert-base-multilingual-cased",
        "google-bert/bert-base-chinese",
        "google-bert/bert-base-german-cased",
        "google-bert/bert-large-uncased-whole-word-masking",
        "google-bert/bert-large-cased-whole-word-masking",
        "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad",
        "google-bert/bert-large-cased-whole-word-masking-finetuned-squad",
        "google-bert/bert-base-cased-finetuned-mrpc",
        "cl-tohoku/bert-base-japanese",
        "cl-tohoku/bert-base-japanese-whole-word-masking",
        "cl-tohoku/bert-base-japanese-char",
        "cl-tohoku/bert-base-japanese-char-whole-word-masking",
        "TurkuNLP/bert-base-finnish-cased-v1",
        "TurkuNLP/bert-base-finnish-uncased-v1",
        "wietsedv/bert-base-dutch-cased",
    ]
)

BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/bigbird-roberta-base": "https://huggingface.co/google/bigbird-roberta-base/resolve/main/config.json",
        "google/bigbird-roberta-large": "https://huggingface.co/google/bigbird-roberta-large/resolve/main/config.json",
        "google/bigbird-base-trivia-itc": "https://huggingface.co/google/bigbird-base-trivia-itc/resolve/main/config.json",
    }
)

BIG_BIRD_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["google/bigbird-roberta-base", "google/bigbird-roberta-large", "google/bigbird-base-trivia-itc"]
)

BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/bigbird-pegasus-large-arxiv": "https://huggingface.co/google/bigbird-pegasus-large-arxiv/resolve/main/config.json",
        "google/bigbird-pegasus-large-pubmed": "https://huggingface.co/google/bigbird-pegasus-large-pubmed/resolve/main/config.json",
        "google/bigbird-pegasus-large-bigpatent": "https://huggingface.co/google/bigbird-pegasus-large-bigpatent/resolve/main/config.json",
    }
)

BIGBIRD_PEGASUS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/bigbird-pegasus-large-arxiv",
        "google/bigbird-pegasus-large-pubmed",
        "google/bigbird-pegasus-large-bigpatent",
    ]
)

BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/biogpt": "https://huggingface.co/microsoft/biogpt/resolve/main/config.json"}
)

BIOGPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/biogpt", "microsoft/BioGPT-Large"])

BIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/bit-50": "https://huggingface.co/google/bit-50/resolve/main/config.json"}
)

BIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/bit-50"])

BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/blenderbot-3B": "https://huggingface.co/facebook/blenderbot-3B/resolve/main/config.json"}
)

BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/blenderbot-3B"])

BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/blenderbot_small-90M": "https://huggingface.co/facebook/blenderbot_small-90M/resolve/main/config.json",
    # See all BlenderbotSmall models at https://huggingface.co/models?filter=blenderbot_small
}

BLENDERBOT_SMALL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/blenderbot_small-90M"])

BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "Salesforce/blip-vqa-base": "https://huggingface.co/Salesforce/blip-vqa-base/resolve/main/config.json",
        "Salesforce/blip-vqa-capfit-large": "https://huggingface.co/Salesforce/blip-vqa-base-capfit/resolve/main/config.json",
        "Salesforce/blip-image-captioning-base": "https://huggingface.co/Salesforce/blip-image-captioning-base/resolve/main/config.json",
        "Salesforce/blip-image-captioning-large": "https://huggingface.co/Salesforce/blip-image-captioning-large/resolve/main/config.json",
        "Salesforce/blip-itm-base-coco": "https://huggingface.co/Salesforce/blip-itm-base-coco/resolve/main/config.json",
        "Salesforce/blip-itm-large-coco": "https://huggingface.co/Salesforce/blip-itm-large-coco/resolve/main/config.json",
        "Salesforce/blip-itm-base-flikr": "https://huggingface.co/Salesforce/blip-itm-base-flikr/resolve/main/config.json",
        "Salesforce/blip-itm-large-flikr": "https://huggingface.co/Salesforce/blip-itm-large-flikr/resolve/main/config.json",
    }
)

BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "Salesforce/blip-vqa-base",
        "Salesforce/blip-vqa-capfilt-large",
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-itm-base-coco",
        "Salesforce/blip-itm-large-coco",
        "Salesforce/blip-itm-base-flickr",
        "Salesforce/blip-itm-large-flickr",
    ]
)

TF_BLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "Salesforce/blip-vqa-base",
        "Salesforce/blip-vqa-capfilt-large",
        "Salesforce/blip-image-captioning-base",
        "Salesforce/blip-image-captioning-large",
        "Salesforce/blip-itm-base-coco",
        "Salesforce/blip-itm-large-coco",
        "Salesforce/blip-itm-base-flickr",
        "Salesforce/blip-itm-large-flickr",
    ]
)

BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"salesforce/blip2-opt-2.7b": "https://huggingface.co/salesforce/blip2-opt-2.7b/resolve/main/config.json"}
)

BLIP_2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Salesforce/blip2-opt-2.7b"])

BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "bigscience/bloom": "https://huggingface.co/bigscience/bloom/resolve/main/config.json",
        "bigscience/bloom-560m": "https://huggingface.co/bigscience/bloom-560m/blob/main/config.json",
        "bigscience/bloom-1b1": "https://huggingface.co/bigscience/bloom-1b1/blob/main/config.json",
        "bigscience/bloom-1b7": "https://huggingface.co/bigscience/bloom-1b7/blob/main/config.json",
        "bigscience/bloom-3b": "https://huggingface.co/bigscience/bloom-3b/blob/main/config.json",
        "bigscience/bloom-7b1": "https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json",
    }
)

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "bigscience/bigscience-small-testing",
        "bigscience/bloom-560m",
        "bigscience/bloom-1b1",
        "bigscience/bloom-1b7",
        "bigscience/bloom-3b",
        "bigscience/bloom-7b1",
        "bigscience/bloom",
    ]
)

BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "BridgeTower/bridgetower-base": "https://huggingface.co/BridgeTower/bridgetower-base/blob/main/config.json",
        "BridgeTower/bridgetower-base-itm-mlm": "https://huggingface.co/BridgeTower/bridgetower-base-itm-mlm/blob/main/config.json",
    }
)

BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["BridgeTower/bridgetower-base", "BridgeTower/bridgetower-base-itm-mlm"]
)

BROS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "jinho8345/bros-base-uncased": "https://huggingface.co/jinho8345/bros-base-uncased/blob/main/config.json",
        "jinho8345/bros-large-uncased": "https://huggingface.co/jinho8345/bros-large-uncased/blob/main/config.json",
    }
)

BROS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["jinho8345/bros-base-uncased", "jinho8345/bros-large-uncased"])

CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "almanach/camembert-base": "https://huggingface.co/almanach/camembert-base/resolve/main/config.json",
        "umberto-commoncrawl-cased-v1": "https://huggingface.co/Musixmatch/umberto-commoncrawl-cased-v1/resolve/main/config.json",
        "umberto-wikipedia-uncased-v1": "https://huggingface.co/Musixmatch/umberto-wikipedia-uncased-v1/resolve/main/config.json",
    }
)

CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["almanach/camembert-base", "Musixmatch/umberto-commoncrawl-cased-v1", "Musixmatch/umberto-wikipedia-uncased-v1"]
)

TF_CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList([])

CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/canine-s": "https://huggingface.co/google/canine-s/resolve/main/config.json"}
)

CANINE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/canine-s", "google/canine-r"])

CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "OFA-Sys/chinese-clip-vit-base-patch16": "https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16/resolve/main/config.json"
    }
)

CHINESE_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["OFA-Sys/chinese-clip-vit-base-patch16"])

CLAP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["laion/clap-htsat-fused", "laion/clap-htsat-unfused"])

CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"openai/clip-vit-base-patch32": "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json"}
)

CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai/clip-vit-base-patch32"])

TF_CLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai/clip-vit-base-patch32"])

CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"CIDAS/clipseg-rd64": "https://huggingface.co/CIDAS/clipseg-rd64/resolve/main/config.json"}
)

CLIPSEG_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["CIDAS/clipseg-rd64-refined"])

CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"susnato/clvp_dev": "https://huggingface.co/susnato/clvp_dev/resolve/main/config.json"}
)

CLVP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["susnato/clvp_dev"])

CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "Salesforce/codegen-350M-nl": "https://huggingface.co/Salesforce/codegen-350M-nl/resolve/main/config.json",
        "Salesforce/codegen-350M-multi": "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/config.json",
        "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/config.json",
        "Salesforce/codegen-2B-nl": "https://huggingface.co/Salesforce/codegen-2B-nl/resolve/main/config.json",
        "Salesforce/codegen-2B-multi": "https://huggingface.co/Salesforce/codegen-2B-multi/resolve/main/config.json",
        "Salesforce/codegen-2B-mono": "https://huggingface.co/Salesforce/codegen-2B-mono/resolve/main/config.json",
        "Salesforce/codegen-6B-nl": "https://huggingface.co/Salesforce/codegen-6B-nl/resolve/main/config.json",
        "Salesforce/codegen-6B-multi": "https://huggingface.co/Salesforce/codegen-6B-multi/resolve/main/config.json",
        "Salesforce/codegen-6B-mono": "https://huggingface.co/Salesforce/codegen-6B-mono/resolve/main/config.json",
        "Salesforce/codegen-16B-nl": "https://huggingface.co/Salesforce/codegen-16B-nl/resolve/main/config.json",
        "Salesforce/codegen-16B-multi": "https://huggingface.co/Salesforce/codegen-16B-multi/resolve/main/config.json",
        "Salesforce/codegen-16B-mono": "https://huggingface.co/Salesforce/codegen-16B-mono/resolve/main/config.json",
    }
)

CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "Salesforce/codegen-350M-nl",
        "Salesforce/codegen-350M-multi",
        "Salesforce/codegen-350M-mono",
        "Salesforce/codegen-2B-nl",
        "Salesforce/codegen-2B-multi",
        "Salesforce/codegen-2B-mono",
        "Salesforce/codegen-6B-nl",
        "Salesforce/codegen-6B-multi",
        "Salesforce/codegen-6B-mono",
        "Salesforce/codegen-16B-nl",
        "Salesforce/codegen-16B-multi",
        "Salesforce/codegen-16B-mono",
    ]
)

CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/conditional-detr-resnet-50": "https://huggingface.co/microsoft/conditional-detr-resnet-50/resolve/main/config.json"
    }
)

CONDITIONAL_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/conditional-detr-resnet-50"])

CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "YituTech/conv-bert-base": "https://huggingface.co/YituTech/conv-bert-base/resolve/main/config.json",
        "YituTech/conv-bert-medium-small": "https://huggingface.co/YituTech/conv-bert-medium-small/resolve/main/config.json",
        "YituTech/conv-bert-small": "https://huggingface.co/YituTech/conv-bert-small/resolve/main/config.json",
    }
)

CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["YituTech/conv-bert-base", "YituTech/conv-bert-medium-small", "YituTech/conv-bert-small"]
)

TF_CONVBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["YituTech/conv-bert-base", "YituTech/conv-bert-medium-small", "YituTech/conv-bert-small"]
)

CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/convnext-tiny-224": "https://huggingface.co/facebook/convnext-tiny-224/resolve/main/config.json"}
)

CONVNEXT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/convnext-tiny-224"])

CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/convnextv2-tiny-1k-224": "https://huggingface.co/facebook/convnextv2-tiny-1k-224/resolve/main/config.json"
    }
)

CONVNEXTV2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/convnextv2-tiny-1k-224"])

CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"openbmb/cpm-ant-10b": "https://huggingface.co/openbmb/cpm-ant-10b/blob/main/config.json"}
)

CPMANT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openbmb/cpm-ant-10b"])

CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"Salesforce/ctrl": "https://huggingface.co/Salesforce/ctrl/resolve/main/config.json"}
)

CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Salesforce/ctrl"])

TF_CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Salesforce/ctrl"])

CVT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/cvt-13": "https://huggingface.co/microsoft/cvt-13/resolve/main/config.json"}
)

CVT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "microsoft/cvt-13",
        "microsoft/cvt-13-384",
        "microsoft/cvt-13-384-22k",
        "microsoft/cvt-21",
        "microsoft/cvt-21-384",
        "microsoft/cvt-21-384-22k",
    ]
)

TF_CVT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "microsoft/cvt-13",
        "microsoft/cvt-13-384",
        "microsoft/cvt-13-384-22k",
        "microsoft/cvt-21",
        "microsoft/cvt-21-384",
        "microsoft/cvt-21-384-22k",
    ]
)

DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/data2vec-text-base": "https://huggingface.co/data2vec/resolve/main/config.json"}
)

DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/data2vec-vision-base-ft": "https://huggingface.co/facebook/data2vec-vision-base-ft/resolve/main/config.json"
    }
)

DATA2VEC_AUDIO_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "facebook/data2vec-audio-base",
        "facebook/data2vec-audio-base-10m",
        "facebook/data2vec-audio-base-100h",
        "facebook/data2vec-audio-base-960h",
    ]
)

DATA2VEC_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/data2vec-text-base"])

DATA2VEC_VISION_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/data2vec-vision-base-ft1k"])

DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
        "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
        "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
        "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/config.json",
        "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/config.json",
        "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/config.json",
    }
)

DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "microsoft/deberta-base",
        "microsoft/deberta-large",
        "microsoft/deberta-xlarge",
        "microsoft/deberta-base-mnli",
        "microsoft/deberta-large-mnli",
        "microsoft/deberta-xlarge-mnli",
    ]
)

TF_DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["kamalkraj/deberta-base"])

DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
        "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
        "microsoft/deberta-v2-xlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json",
        "microsoft/deberta-v2-xxlarge-mnli": "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json",
    }
)

DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "microsoft/deberta-v2-xlarge",
        "microsoft/deberta-v2-xxlarge",
        "microsoft/deberta-v2-xlarge-mnli",
        "microsoft/deberta-v2-xxlarge-mnli",
    ]
)

TF_DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["kamalkraj/deberta-v2-xlarge"])

DECISION_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "edbeeching/decision-transformer-gym-hopper-medium": "https://huggingface.co/edbeeching/decision-transformer-gym-hopper-medium/resolve/main/config.json"
    }
)

DECISION_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["edbeeching/decision-transformer-gym-hopper-medium"]
)

DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"SenseTime/deformable-detr": "https://huggingface.co/sensetime/deformable-detr/resolve/main/config.json"}
)

DEFORMABLE_DETR_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["sensetime/deformable-detr"])

DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/deit-base-distilled-patch16-224": "https://huggingface.co/facebook/deit-base-patch16-224/resolve/main/config.json"
    }
)

DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/deit-base-distilled-patch16-224"])

TF_DEIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/deit-base-distilled-patch16-224"])

MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"speechbrain/m-ctc-t-large": "https://huggingface.co/speechbrain/m-ctc-t-large/resolve/main/config.json"}
)

MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["speechbrain/m-ctc-t-large"])

OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"s-JoL/Open-Llama-V1": "https://huggingface.co/s-JoL/Open-Llama-V1/blob/main/config.json"}
)

RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "yjernite/retribert-base-uncased": "https://huggingface.co/yjernite/retribert-base-uncased/resolve/main/config.json"
    }
)

RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["yjernite/retribert-base-uncased"])

TRAJECTORY_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "CarlCochet/trajectory-transformer-halfcheetah-medium-v2": "https://huggingface.co/CarlCochet/trajectory-transformer-halfcheetah-medium-v2/resolve/main/config.json"
    }
)

TRAJECTORY_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["CarlCochet/trajectory-transformer-halfcheetah-medium-v2"]
)

TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"transfo-xl/transfo-xl-wt103": "https://huggingface.co/transfo-xl/transfo-xl-wt103/resolve/main/config.json"}
)

TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["transfo-xl/transfo-xl-wt103"])

TF_TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["transfo-xl/transfo-xl-wt103"])

VAN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "Visual-Attention-Network/van-base": "https://huggingface.co/Visual-Attention-Network/van-base/blob/main/config.json"
    }
)

VAN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Visual-Attention-Network/van-base"])

DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "LiheYoung/depth-anything-small-hf": "https://huggingface.co/LiheYoung/depth-anything-small-hf/resolve/main/config.json"
    }
)

DEPTH_ANYTHING_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["LiheYoung/depth-anything-small-hf"])

DETA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"ut/deta": "https://huggingface.co/ut/deta/resolve/main/config.json"}
)

DETA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["jozhang97/deta-swin-large-o365"])

DETR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/detr-resnet-50": "https://huggingface.co/facebook/detr-resnet-50/resolve/main/config.json"}
)

DETR_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/detr-resnet-50"])

DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"shi-labs/dinat-mini-in1k-224": "https://huggingface.co/shi-labs/dinat-mini-in1k-224/resolve/main/config.json"}
)

DINAT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["shi-labs/dinat-mini-in1k-224"])

DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/dinov2-base": "https://huggingface.co/facebook/dinov2-base/resolve/main/config.json"}
)

DINOV2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/dinov2-base"])

DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "distilbert-base-uncased": "https://huggingface.co/distilbert-base-uncased/resolve/main/config.json",
        "distilbert-base-uncased-distilled-squad": "https://huggingface.co/distilbert-base-uncased-distilled-squad/resolve/main/config.json",
        "distilbert-base-cased": "https://huggingface.co/distilbert-base-cased/resolve/main/config.json",
        "distilbert-base-cased-distilled-squad": "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/main/config.json",
        "distilbert-base-german-cased": "https://huggingface.co/distilbert-base-german-cased/resolve/main/config.json",
        "distilbert-base-multilingual-cased": "https://huggingface.co/distilbert-base-multilingual-cased/resolve/main/config.json",
        "distilbert-base-uncased-finetuned-sst-2-english": "https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/main/config.json",
    }
)

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "distilbert-base-uncased",
        "distilbert-base-uncased-distilled-squad",
        "distilbert-base-cased",
        "distilbert-base-cased-distilled-squad",
        "distilbert-base-german-cased",
        "distilbert-base-multilingual-cased",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
)

TF_DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "distilbert-base-uncased",
        "distilbert-base-uncased-distilled-squad",
        "distilbert-base-cased",
        "distilbert-base-cased-distilled-squad",
        "distilbert-base-multilingual-cased",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]
)

DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"naver-clova-ix/donut-base": "https://huggingface.co/naver-clova-ix/donut-base/resolve/main/config.json"}
)

DONUT_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["naver-clova-ix/donut-base"])

DPR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/dpr-ctx_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/config.json",
        "facebook/dpr-question_encoder-single-nq-base": "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/config.json",
        "facebook/dpr-reader-single-nq-base": "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/config.json",
        "facebook/dpr-ctx_encoder-multiset-base": "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/config.json",
        "facebook/dpr-question_encoder-multiset-base": "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/config.json",
        "facebook/dpr-reader-multiset-base": "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/config.json",
    }
)

DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]
)

DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]
)

DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]
)

TF_DPR_CONTEXT_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/dpr-ctx_encoder-single-nq-base", "facebook/dpr-ctx_encoder-multiset-base"]
)

TF_DPR_QUESTION_ENCODER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/dpr-question_encoder-single-nq-base", "facebook/dpr-question_encoder-multiset-base"]
)

TF_DPR_READER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/dpr-reader-single-nq-base", "facebook/dpr-reader-multiset-base"]
)

DPT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"Intel/dpt-large": "https://huggingface.co/Intel/dpt-large/resolve/main/config.json"}
)

DPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Intel/dpt-large", "Intel/dpt-hybrid-midas"])

EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "snap-research/efficientformer-l1-300": "https://huggingface.co/snap-research/efficientformer-l1-300/resolve/main/config.json"
    }
)

EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["snap-research/efficientformer-l1-300"])

TF_EFFICIENTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["snap-research/efficientformer-l1-300"])

EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/efficientnet-b7": "https://huggingface.co/google/efficientnet-b7/resolve/main/config.json"}
)

EFFICIENTNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/efficientnet-b7"])

ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/electra-small-generator": "https://huggingface.co/google/electra-small-generator/resolve/main/config.json",
        "google/electra-base-generator": "https://huggingface.co/google/electra-base-generator/resolve/main/config.json",
        "google/electra-large-generator": "https://huggingface.co/google/electra-large-generator/resolve/main/config.json",
        "google/electra-small-discriminator": "https://huggingface.co/google/electra-small-discriminator/resolve/main/config.json",
        "google/electra-base-discriminator": "https://huggingface.co/google/electra-base-discriminator/resolve/main/config.json",
        "google/electra-large-discriminator": "https://huggingface.co/google/electra-large-discriminator/resolve/main/config.json",
    }
)

ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/electra-small-generator",
        "google/electra-base-generator",
        "google/electra-large-generator",
        "google/electra-small-discriminator",
        "google/electra-base-discriminator",
        "google/electra-large-discriminator",
    ]
)

TF_ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/electra-small-generator",
        "google/electra-base-generator",
        "google/electra-large-generator",
        "google/electra-small-discriminator",
        "google/electra-base-discriminator",
        "google/electra-large-discriminator",
    ]
)

ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/encodec_24khz": "https://huggingface.co/facebook/encodec_24khz/resolve/main/config.json",
        "facebook/encodec_48khz": "https://huggingface.co/facebook/encodec_48khz/resolve/main/config.json",
    }
)

ENCODEC_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/encodec_24khz", "facebook/encodec_48khz"])

ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "nghuyong/ernie-1.0-base-zh": "https://huggingface.co/nghuyong/ernie-1.0-base-zh/resolve/main/config.json",
        "nghuyong/ernie-2.0-base-en": "https://huggingface.co/nghuyong/ernie-2.0-base-en/resolve/main/config.json",
        "nghuyong/ernie-2.0-large-en": "https://huggingface.co/nghuyong/ernie-2.0-large-en/resolve/main/config.json",
        "nghuyong/ernie-3.0-base-zh": "https://huggingface.co/nghuyong/ernie-3.0-base-zh/resolve/main/config.json",
        "nghuyong/ernie-3.0-medium-zh": "https://huggingface.co/nghuyong/ernie-3.0-medium-zh/resolve/main/config.json",
        "nghuyong/ernie-3.0-mini-zh": "https://huggingface.co/nghuyong/ernie-3.0-mini-zh/resolve/main/config.json",
        "nghuyong/ernie-3.0-micro-zh": "https://huggingface.co/nghuyong/ernie-3.0-micro-zh/resolve/main/config.json",
        "nghuyong/ernie-3.0-nano-zh": "https://huggingface.co/nghuyong/ernie-3.0-nano-zh/resolve/main/config.json",
        "nghuyong/ernie-gram-zh": "https://huggingface.co/nghuyong/ernie-gram-zh/resolve/main/config.json",
        "nghuyong/ernie-health-zh": "https://huggingface.co/nghuyong/ernie-health-zh/resolve/main/config.json",
    }
)

ERNIE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "nghuyong/ernie-1.0-base-zh",
        "nghuyong/ernie-2.0-base-en",
        "nghuyong/ernie-2.0-large-en",
        "nghuyong/ernie-3.0-base-zh",
        "nghuyong/ernie-3.0-medium-zh",
        "nghuyong/ernie-3.0-mini-zh",
        "nghuyong/ernie-3.0-micro-zh",
        "nghuyong/ernie-3.0-nano-zh",
        "nghuyong/ernie-gram-zh",
        "nghuyong/ernie-health-zh",
    ]
)

ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "susnato/ernie-m-base_pytorch": "https://huggingface.co/susnato/ernie-m-base_pytorch/blob/main/config.json",
        "susnato/ernie-m-large_pytorch": "https://huggingface.co/susnato/ernie-m-large_pytorch/blob/main/config.json",
    }
)

ERNIE_M_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["susnato/ernie-m-base_pytorch", "susnato/ernie-m-large_pytorch"]
)

ESM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/esm-1b": "https://huggingface.co/facebook/esm-1b/resolve/main/config.json"}
)

ESM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D"])

FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "tiiuae/falcon-40b": "https://huggingface.co/tiiuae/falcon-40b/resolve/main/config.json",
        "tiiuae/falcon-7b": "https://huggingface.co/tiiuae/falcon-7b/resolve/main/config.json",
    }
)

FALCON_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "tiiuae/falcon-40b",
        "tiiuae/falcon-40b-instruct",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-7b-instruct",
        "tiiuae/falcon-rw-7b",
        "tiiuae/falcon-rw-1b",
    ]
)

FASTSPEECH2_CONFORMER_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "espnet/fastspeech2_conformer_hifigan": "https://huggingface.co/espnet/fastspeech2_conformer_hifigan/raw/main/config.json"
    }
)

FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"espnet/fastspeech2_conformer": "https://huggingface.co/espnet/fastspeech2_conformer/raw/main/config.json"}
)

FASTSPEECH2_CONFORMER_WITH_HIFIGAN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "espnet/fastspeech2_conformer_with_hifigan": "https://huggingface.co/espnet/fastspeech2_conformer_with_hifigan/raw/main/config.json"
    }
)

FASTSPEECH2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["espnet/fastspeech2_conformer"])

FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "flaubert/flaubert_small_cased": "https://huggingface.co/flaubert/flaubert_small_cased/resolve/main/config.json",
        "flaubert/flaubert_base_uncased": "https://huggingface.co/flaubert/flaubert_base_uncased/resolve/main/config.json",
        "flaubert/flaubert_base_cased": "https://huggingface.co/flaubert/flaubert_base_cased/resolve/main/config.json",
        "flaubert/flaubert_large_cased": "https://huggingface.co/flaubert/flaubert_large_cased/resolve/main/config.json",
    }
)

FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "flaubert/flaubert_small_cased",
        "flaubert/flaubert_base_uncased",
        "flaubert/flaubert_base_cased",
        "flaubert/flaubert_large_cased",
    ]
)

TF_FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList([])

FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/flava-full": "https://huggingface.co/facebook/flava-full/resolve/main/config.json"}
)

FLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/flava-full"])

FNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/fnet-base": "https://huggingface.co/google/fnet-base/resolve/main/config.json",
        "google/fnet-large": "https://huggingface.co/google/fnet-large/resolve/main/config.json",
    }
)

FNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/fnet-base", "google/fnet-large"])

FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/focalnet-tiny": "https://huggingface.co/microsoft/focalnet-tiny/resolve/main/config.json"}
)

FOCALNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/focalnet-tiny"])

FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict({})

FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "funnel-transformer/small": "https://huggingface.co/funnel-transformer/small/resolve/main/config.json",
        "funnel-transformer/small-base": "https://huggingface.co/funnel-transformer/small-base/resolve/main/config.json",
        "funnel-transformer/medium": "https://huggingface.co/funnel-transformer/medium/resolve/main/config.json",
        "funnel-transformer/medium-base": "https://huggingface.co/funnel-transformer/medium-base/resolve/main/config.json",
        "funnel-transformer/intermediate": "https://huggingface.co/funnel-transformer/intermediate/resolve/main/config.json",
        "funnel-transformer/intermediate-base": "https://huggingface.co/funnel-transformer/intermediate-base/resolve/main/config.json",
        "funnel-transformer/large": "https://huggingface.co/funnel-transformer/large/resolve/main/config.json",
        "funnel-transformer/large-base": "https://huggingface.co/funnel-transformer/large-base/resolve/main/config.json",
        "funnel-transformer/xlarge": "https://huggingface.co/funnel-transformer/xlarge/resolve/main/config.json",
        "funnel-transformer/xlarge-base": "https://huggingface.co/funnel-transformer/xlarge-base/resolve/main/config.json",
    }
)

FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "funnel-transformer/small",
        "funnel-transformer/small-base",
        "funnel-transformer/medium",
        "funnel-transformer/medium-base",
        "funnel-transformer/intermediate",
        "funnel-transformer/intermediate-base",
        "funnel-transformer/large",
        "funnel-transformer/large-base",
        "funnel-transformer/xlarge-base",
        "funnel-transformer/xlarge",
    ]
)

TF_FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "funnel-transformer/small",
        "funnel-transformer/small-base",
        "funnel-transformer/medium",
        "funnel-transformer/medium-base",
        "funnel-transformer/intermediate",
        "funnel-transformer/intermediate-base",
        "funnel-transformer/large",
        "funnel-transformer/large-base",
        "funnel-transformer/xlarge-base",
        "funnel-transformer/xlarge",
    ]
)

FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"adept/fuyu-8b": "https://huggingface.co/adept/fuyu-8b/resolve/main/config.json"}
)

GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict({})

GIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/git-base": "https://huggingface.co/microsoft/git-base/resolve/main/config.json"}
)

GIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/git-base"])

GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"vinvino02/glpn-kitti": "https://huggingface.co/vinvino02/glpn-kitti/resolve/main/config.json"}
)

GLPN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["vinvino02/glpn-kitti"])

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
        "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/config.json",
        "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/config.json",
        "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/config.json",
        "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/config.json",
    }
)

GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl",
        "distilbert/distilgpt2",
    ]
)

TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "openai-community/gpt2",
        "openai-community/gpt2-medium",
        "openai-community/gpt2-large",
        "openai-community/gpt2-xl",
        "distilbert/distilgpt2",
    ]
)

GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "bigcode/gpt_bigcode-santacoder": "https://huggingface.co/bigcode/gpt_bigcode-santacoder/resolve/main/config.json"
    }
)

GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["bigcode/gpt_bigcode-santacoder"])

GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"EleutherAI/gpt-neo-1.3B": "https://huggingface.co/EleutherAI/gpt-neo-1.3B/resolve/main/config.json"}
)

GPT_NEO_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["EleutherAI/gpt-neo-1.3B"])

GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json"}
)

GPT_NEOX_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["EleutherAI/gpt-neox-20b"])

GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"abeja/gpt-neox-japanese-2.7b": "https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json"}
)

GPT_NEOX_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["https://huggingface.co/abeja/gpt-neox-japanese-2.7b/resolve/main/config.json"]
)

GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"EleutherAI/gpt-j-6B": "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/config.json"}
)

GPTJ_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["EleutherAI/gpt-j-6B"])

GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "tanreinama/GPTSAN-2.8B-spout_is_uniform": "https://huggingface.co/tanreinama/GPTSAN-2.8B-spout_is_uniform/resolve/main/config.json"
    }
)

GPTSAN_JAPANESE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Tanrei/GPTSAN-japanese"])

GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"graphormer-base": "https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2/resolve/main/config.json"}
)

GRAPHORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["clefourrier/graphormer-base-pcqm4mv1", "clefourrier/graphormer-base-pcqm4mv2"]
)

GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"nvidia/groupvit-gcc-yfcc": "https://huggingface.co/nvidia/groupvit-gcc-yfcc/resolve/main/config.json"}
)

GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["nvidia/groupvit-gcc-yfcc"])

TF_GROUPVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["nvidia/groupvit-gcc-yfcc"])

HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/hubert-base-ls960": "https://huggingface.co/facebook/hubert-base-ls960/resolve/main/config.json"}
)

HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/hubert-base-ls960"])

TF_HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/hubert-base-ls960"])

IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "kssteven/ibert-roberta-base": "https://huggingface.co/kssteven/ibert-roberta-base/resolve/main/config.json",
        "kssteven/ibert-roberta-large": "https://huggingface.co/kssteven/ibert-roberta-large/resolve/main/config.json",
        "kssteven/ibert-roberta-large-mnli": "https://huggingface.co/kssteven/ibert-roberta-large-mnli/resolve/main/config.json",
    }
)

IBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["kssteven/ibert-roberta-base", "kssteven/ibert-roberta-large", "kssteven/ibert-roberta-large-mnli"]
)

IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "HuggingFaceM4/idefics-9b": "https://huggingface.co/HuggingFaceM4/idefics-9b/blob/main/config.json",
        "HuggingFaceM4/idefics-80b": "https://huggingface.co/HuggingFaceM4/idefics-80b/blob/main/config.json",
    }
)

IDEFICS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["HuggingFaceM4/idefics-9b", "HuggingFaceM4/idefics-80b"])

IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"openai/imagegpt-small": "", "openai/imagegpt-medium": "", "openai/imagegpt-large": ""}
)

IMAGEGPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["openai/imagegpt-small", "openai/imagegpt-medium", "openai/imagegpt-large"]
)

INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "huggingface/informer-tourism-monthly": "https://huggingface.co/huggingface/informer-tourism-monthly/resolve/main/config.json"
    }
)

INFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["huggingface/informer-tourism-monthly"])

INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "Salesforce/instruct-blip-flan-t5": "https://huggingface.co/Salesforce/instruct-blip-flan-t5/resolve/main/config.json"
    }
)

INSTRUCTBLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Salesforce/instructblip-flan-t5-xl"])

JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "openai/jukebox-5b-lyrics": "https://huggingface.co/openai/jukebox-5b-lyrics/blob/main/config.json",
        "openai/jukebox-1b-lyrics": "https://huggingface.co/openai/jukebox-1b-lyrics/blob/main/config.json",
    }
)

JUKEBOX_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai/jukebox-1b-lyrics", "openai/jukebox-5b-lyrics"])

KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/kosmos-2-patch14-224": "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/config.json"
    }
)

KOSMOS2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/kosmos-2-patch14-224"])

LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/layoutlm-base-uncased": "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/config.json",
        "microsoft/layoutlm-large-uncased": "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/config.json",
    }
)

LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["layoutlm-base-uncased", "layoutlm-large-uncased"])

TF_LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["microsoft/layoutlm-base-uncased", "microsoft/layoutlm-large-uncased"]
)

LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/config.json",
        "layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/config.json",
    }
)

LAYOUTLMV2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["microsoft/layoutlmv2-base-uncased", "microsoft/layoutlmv2-large-uncased"]
)

LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/layoutlmv3-base": "https://huggingface.co/microsoft/layoutlmv3-base/resolve/main/config.json"}
)

LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/layoutlmv3-base", "microsoft/layoutlmv3-large"])

TF_LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["microsoft/layoutlmv3-base", "microsoft/layoutlmv3-large"]
)

LED_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"allenai/led-base-16384": "https://huggingface.co/allenai/led-base-16384/resolve/main/config.json"}
)

LED_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["allenai/led-base-16384"])

LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/levit-128S": "https://huggingface.co/facebook/levit-128S/resolve/main/config.json"}
)

LEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/levit-128S"])

LILT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "SCUT-DLVCLab/lilt-roberta-en-base": "https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base/resolve/main/config.json"
    }
)

LILT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["SCUT-DLVCLab/lilt-roberta-en-base"])

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict({})

LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"llava-hf/llava-v1.5-7b": "https://huggingface.co/llava-hf/llava-v1.5-7b/resolve/main/config.json"}
)

LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["llava-hf/llava-1.5-7b-hf", "llava-hf/llava-1.5-13b-hf", "llava-hf/bakLlava-v1-hf"]
)

LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
        "allenai/longformer-large-4096": "https://huggingface.co/allenai/longformer-large-4096/resolve/main/config.json",
        "allenai/longformer-large-4096-finetuned-triviaqa": "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/config.json",
        "allenai/longformer-base-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/config.json",
        "allenai/longformer-large-4096-extra.pos.embd.only": "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/config.json",
    }
)

LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "allenai/longformer-base-4096",
        "allenai/longformer-large-4096",
        "allenai/longformer-large-4096-finetuned-triviaqa",
        "allenai/longformer-base-4096-extra.pos.embd.only",
        "allenai/longformer-large-4096-extra.pos.embd.only",
    ]
)

TF_LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "allenai/longformer-base-4096",
        "allenai/longformer-large-4096",
        "allenai/longformer-large-4096-finetuned-triviaqa",
        "allenai/longformer-base-4096-extra.pos.embd.only",
        "allenai/longformer-large-4096-extra.pos.embd.only",
    ]
)

LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/long-t5-local-base": "https://huggingface.co/google/long-t5-local-base/blob/main/config.json",
        "google/long-t5-local-large": "https://huggingface.co/google/long-t5-local-large/blob/main/config.json",
        "google/long-t5-tglobal-base": "https://huggingface.co/google/long-t5-tglobal-base/blob/main/config.json",
        "google/long-t5-tglobal-large": "https://huggingface.co/google/long-t5-tglobal-large/blob/main/config.json",
    }
)

LONGT5_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/long-t5-local-base",
        "google/long-t5-local-large",
        "google/long-t5-tglobal-base",
        "google/long-t5-tglobal-large",
    ]
)

LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "studio-ousia/luke-base": "https://huggingface.co/studio-ousia/luke-base/resolve/main/config.json",
        "studio-ousia/luke-large": "https://huggingface.co/studio-ousia/luke-large/resolve/main/config.json",
    }
)

LUKE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["studio-ousia/luke-base", "studio-ousia/luke-large"])

LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"unc-nlp/lxmert-base-uncased": "https://huggingface.co/unc-nlp/lxmert-base-uncased/resolve/main/config.json"}
)

TF_LXMERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["unc-nlp/lxmert-base-uncased"])

M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/m2m100_418M": "https://huggingface.co/facebook/m2m100_418M/resolve/main/config.json"}
)

M2M_100_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/m2m100_418M"])

MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"state-spaces/mamba-2.8b": "https://huggingface.co/state-spaces/mamba-2.8b/resolve/main/config.json"}
)

MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList([])

MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/config.json",
        "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/config.json",
    }
)

MARKUPLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/markuplm-base", "microsoft/markuplm-large"])

MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/mask2former-swin-small-coco-instance": "https://huggingface.co/facebook/mask2former-swin-small-coco-instance/blob/main/config.json"
    }
)

MASK2FORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/mask2former-swin-small-coco-instance"])

MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/maskformer-swin-base-ade": "https://huggingface.co/facebook/maskformer-swin-base-ade/blob/main/config.json"
    }
)

MASKFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/maskformer-swin-base-ade"])

MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"mnaylor/mega-base-wikitext": "https://huggingface.co/mnaylor/mega-base-wikitext/resolve/main/config.json"}
)

MEGA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["mnaylor/mega-base-wikitext"])

MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict({})

MEGATRON_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["nvidia/megatron-bert-cased-345m"])

MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"alibaba-damo/mgp-str-base": "https://huggingface.co/alibaba-damo/mgp-str-base/resolve/main/config.json"}
)

MGP_STR_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["alibaba-damo/mgp-str-base"])

MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "mistralai/Mistral-7B-v0.1": "https://huggingface.co/mistralai/Mistral-7B-v0.1/resolve/main/config.json",
        "mistralai/Mistral-7B-Instruct-v0.1": "https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/resolve/main/config.json",
    }
)

MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"mistral-ai/Mixtral-8x7B": "https://huggingface.co/mistral-ai/Mixtral-8x7B/resolve/main/config.json"}
)

MOBILEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/mobilebert-uncased": "https://huggingface.co/google/mobilebert-uncased/resolve/main/config.json"}
)

MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/mobilebert-uncased"])

TF_MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/mobilebert-uncased"])

MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/mobilenet_v1_1.0_224": "https://huggingface.co/google/mobilenet_v1_1.0_224/resolve/main/config.json",
        "google/mobilenet_v1_0.75_192": "https://huggingface.co/google/mobilenet_v1_0.75_192/resolve/main/config.json",
    }
)

MOBILENET_V1_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["google/mobilenet_v1_1.0_224", "google/mobilenet_v1_0.75_192"]
)

MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/mobilenet_v2_1.4_224": "https://huggingface.co/google/mobilenet_v2_1.4_224/resolve/main/config.json",
        "google/mobilenet_v2_1.0_224": "https://huggingface.co/google/mobilenet_v2_1.0_224/resolve/main/config.json",
        "google/mobilenet_v2_0.75_160": "https://huggingface.co/google/mobilenet_v2_0.75_160/resolve/main/config.json",
        "google/mobilenet_v2_0.35_96": "https://huggingface.co/google/mobilenet_v2_0.35_96/resolve/main/config.json",
    }
)

MOBILENET_V2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/mobilenet_v2_1.4_224",
        "google/mobilenet_v2_1.0_224",
        "google/mobilenet_v2_0.37_160",
        "google/mobilenet_v2_0.35_96",
    ]
)

MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "apple/mobilevit-small": "https://huggingface.co/apple/mobilevit-small/resolve/main/config.json",
        "apple/mobilevit-x-small": "https://huggingface.co/apple/mobilevit-x-small/resolve/main/config.json",
        "apple/mobilevit-xx-small": "https://huggingface.co/apple/mobilevit-xx-small/resolve/main/config.json",
        "apple/deeplabv3-mobilevit-small": "https://huggingface.co/apple/deeplabv3-mobilevit-small/resolve/main/config.json",
        "apple/deeplabv3-mobilevit-x-small": "https://huggingface.co/apple/deeplabv3-mobilevit-x-small/resolve/main/config.json",
        "apple/deeplabv3-mobilevit-xx-small": "https://huggingface.co/apple/deeplabv3-mobilevit-xx-small/resolve/main/config.json",
    }
)

MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "apple/mobilevit-small",
        "apple/mobilevit-x-small",
        "apple/mobilevit-xx-small",
        "apple/deeplabv3-mobilevit-small",
        "apple/deeplabv3-mobilevit-x-small",
        "apple/deeplabv3-mobilevit-xx-small",
    ]
)

TF_MOBILEVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "apple/mobilevit-small",
        "apple/mobilevit-x-small",
        "apple/mobilevit-xx-small",
        "apple/deeplabv3-mobilevit-small",
        "apple/deeplabv3-mobilevit-x-small",
        "apple/deeplabv3-mobilevit-xx-small",
    ]
)

MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"apple/mobilevitv2-1.0": "https://huggingface.co/apple/mobilevitv2-1.0/resolve/main/config.json"}
)

MOBILEVITV2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["apple/mobilevitv2-1.0-imagenet1k-256"])

MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/mpnet-base": "https://huggingface.co/microsoft/mpnet-base/resolve/main/config.json"}
)

MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/mpnet-base"])

TF_MPNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/mpnet-base"])

MPT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"mosaicml/mpt-7b": "https://huggingface.co/mosaicml/mpt-7b/resolve/main/config.json"}
)

MPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "mosaicml/mpt-7b",
        "mosaicml/mpt-7b-storywriter",
        "mosaicml/mpt-7b-instruct",
        "mosaicml/mpt-7b-8k",
        "mosaicml/mpt-7b-8k-instruct",
        "mosaicml/mpt-7b-8k-chat",
        "mosaicml/mpt-30b",
        "mosaicml/mpt-30b-instruct",
        "mosaicml/mpt-30b-chat",
    ]
)

MRA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"uw-madison/mra-base-512-4": "https://huggingface.co/uw-madison/mra-base-512-4/resolve/main/config.json"}
)

MRA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["uw-madison/mra-base-512-4"])

MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/musicgen-small": "https://huggingface.co/facebook/musicgen-small/resolve/main/config.json"}
)

MUSICGEN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/musicgen-small"])

MUSICGEN_MELODY_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/musicgen-melody": "https://huggingface.co/facebook/musicgen-melody/resolve/main/config.json"}
)

MUSICGEN_MELODY_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/musicgen-melody"])

MVP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "RUCAIBox/mvp",
        "RUCAIBox/mvp-data-to-text",
        "RUCAIBox/mvp-open-dialog",
        "RUCAIBox/mvp-question-answering",
        "RUCAIBox/mvp-question-generation",
        "RUCAIBox/mvp-story",
        "RUCAIBox/mvp-summarization",
        "RUCAIBox/mvp-task-dialog",
        "RUCAIBox/mtl-data-to-text",
        "RUCAIBox/mtl-multi-task",
        "RUCAIBox/mtl-open-dialog",
        "RUCAIBox/mtl-question-answering",
        "RUCAIBox/mtl-question-generation",
        "RUCAIBox/mtl-story",
        "RUCAIBox/mtl-summarization",
    ]
)

NAT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"shi-labs/nat-mini-in1k-224": "https://huggingface.co/shi-labs/nat-mini-in1k-224/resolve/main/config.json"}
)

NAT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["shi-labs/nat-mini-in1k-224"])

NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"sijunhe/nezha-cn-base": "https://huggingface.co/sijunhe/nezha-cn-base/resolve/main/config.json"}
)

NEZHA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["sijunhe/nezha-cn-base", "sijunhe/nezha-cn-large", "sijunhe/nezha-base-wwm", "sijunhe/nezha-large-wwm"]
)

NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/nllb-moe-54B": "https://huggingface.co/facebook/nllb-moe-54b/resolve/main/config.json"}
)

NLLB_MOE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/nllb-moe-54b"])

NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"uw-madison/nystromformer-512": "https://huggingface.co/uw-madison/nystromformer-512/resolve/main/config.json"}
)

NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["uw-madison/nystromformer-512"])

OLMO_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "allenai/OLMo-1B-hf": "https://huggingface.co/allenai/OLMo-1B-hf/resolve/main/config.json",
        "allenai/OLMo-7B-hf": "https://huggingface.co/allenai/OLMo-7B-hf/resolve/main/config.json",
        "allenai/OLMo-7B-Twin-2T-hf": "https://huggingface.co/allenai/OLMo-7B-Twin-2T-hf/resolve/main/config.json",
    }
)

ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "shi-labs/oneformer_ade20k_swin_tiny": "https://huggingface.co/shi-labs/oneformer_ade20k_swin_tiny/blob/main/config.json"
    }
)

ONEFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["shi-labs/oneformer_ade20k_swin_tiny"])

OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"openai-community/openai-gpt": "https://huggingface.co/openai-community/openai-gpt/resolve/main/config.json"}
)

OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai-community/openai-gpt"])

TF_OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai-community/openai-gpt"])

OPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
        "facebook/opt-2.7b",
        "facebook/opt-6.7b",
        "facebook/opt-13b",
        "facebook/opt-30b",
    ]
)

OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/owlv2-base-patch16": "https://huggingface.co/google/owlv2-base-patch16/resolve/main/config.json"}
)

OWLV2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/owlv2-base-patch16-ensemble"])

OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/owlvit-base-patch32": "https://huggingface.co/google/owlvit-base-patch32/resolve/main/config.json",
        "google/owlvit-base-patch16": "https://huggingface.co/google/owlvit-base-patch16/resolve/main/config.json",
        "google/owlvit-large-patch14": "https://huggingface.co/google/owlvit-large-patch14/resolve/main/config.json",
    }
)

OWLVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["google/owlvit-base-patch32", "google/owlvit-base-patch16", "google/owlvit-large-patch14"]
)

PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "ibm/patchtsmixer-etth1-pretrain": "https://huggingface.co/ibm/patchtsmixer-etth1-pretrain/resolve/main/config.json"
    }
)

PATCHTSMIXER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["ibm/patchtsmixer-etth1-pretrain"])

PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"ibm/patchtst-base": "https://huggingface.co/ibm/patchtst-base/resolve/main/config.json"}
)

PATCHTST_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["ibm/patchtst-etth1-pretrain"])

PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/pegasus-large": "https://huggingface.co/google/pegasus-large/resolve/main/config.json"}
)

PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/pegasus-x-base": "https://huggingface.co/google/pegasus-x-base/resolve/main/config.json",
        "google/pegasus-x-large": "https://huggingface.co/google/pegasus-x-large/resolve/main/config.json",
    }
)

PEGASUS_X_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/pegasus-x-base", "google/pegasus-x-large"])

PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"deepmind/language-perceiver": "https://huggingface.co/deepmind/language-perceiver/resolve/main/config.json"}
)

PERCEIVER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["deepmind/language-perceiver"])

PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"adept/persimmon-8b-base": "https://huggingface.co/adept/persimmon-8b-base/resolve/main/config.json"}
)

PHI_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/phi-1": "https://huggingface.co/microsoft/phi-1/resolve/main/config.json",
        "microsoft/phi-1_5": "https://huggingface.co/microsoft/phi-1_5/resolve/main/config.json",
        "microsoft/phi-2": "https://huggingface.co/microsoft/phi-2/resolve/main/config.json",
    }
)

PHI_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/phi-1", "microsoft/phi-1_5", "microsoft/phi-2"])

PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/pix2struct-textcaps-base": "https://huggingface.co/google/pix2struct-textcaps-base/resolve/main/config.json"
    }
)

PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/pix2struct-textcaps-base",
        "google/pix2struct-textcaps-large",
        "google/pix2struct-base",
        "google/pix2struct-large",
        "google/pix2struct-ai2d-base",
        "google/pix2struct-ai2d-large",
        "google/pix2struct-widget-captioning-base",
        "google/pix2struct-widget-captioning-large",
        "google/pix2struct-screen2words-base",
        "google/pix2struct-screen2words-large",
        "google/pix2struct-docvqa-base",
        "google/pix2struct-docvqa-large",
        "google/pix2struct-ocrvqa-base",
        "google/pix2struct-ocrvqa-large",
        "google/pix2struct-chartqa-base",
        "google/pix2struct-inforgraphics-vqa-base",
        "google/pix2struct-inforgraphics-vqa-large",
    ]
)

PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/config.json"}
)

PLBART_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["uclanlp/plbart-base", "uclanlp/plbart-cs-java", "uclanlp/plbart-multi_task-all"]
)

POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"sail/poolformer_s12": "https://huggingface.co/sail/poolformer_s12/resolve/main/config.json"}
)

POOLFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["sail/poolformer_s12"])

POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"sweetcocoa/pop2piano": "https://huggingface.co/sweetcocoa/pop2piano/blob/main/config.json"}
)

POP2PIANO_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["sweetcocoa/pop2piano"])

PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/prophetnet-large-uncased": "https://huggingface.co/microsoft/prophetnet-large-uncased/resolve/main/config.json"
    }
)

PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/prophetnet-large-uncased"])

PVT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict({"pvt-tiny-224": "https://huggingface.co/Zetatech/pvt-tiny-224"})

PVT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Zetatech/pvt-tiny-224"])

QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google-bert/bert-base-uncased": "https://huggingface.co/google-bert/bert-base-uncased/resolve/main/config.json"}
)

QDQBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google-bert/bert-base-uncased"])

QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"Qwen/Qwen2-7B-beta": "https://huggingface.co/Qwen/Qwen2-7B-beta/resolve/main/config.json"}
)

REALM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/realm-cc-news-pretrained-embedder": "https://huggingface.co/google/realm-cc-news-pretrained-embedder/resolve/main/config.json",
        "google/realm-cc-news-pretrained-encoder": "https://huggingface.co/google/realm-cc-news-pretrained-encoder/resolve/main/config.json",
        "google/realm-cc-news-pretrained-scorer": "https://huggingface.co/google/realm-cc-news-pretrained-scorer/resolve/main/config.json",
        "google/realm-cc-news-pretrained-openqa": "https://huggingface.co/google/realm-cc-news-pretrained-openqa/aresolve/main/config.json",
        "google/realm-orqa-nq-openqa": "https://huggingface.co/google/realm-orqa-nq-openqa/resolve/main/config.json",
        "google/realm-orqa-nq-reader": "https://huggingface.co/google/realm-orqa-nq-reader/resolve/main/config.json",
        "google/realm-orqa-wq-openqa": "https://huggingface.co/google/realm-orqa-wq-openqa/resolve/main/config.json",
        "google/realm-orqa-wq-reader": "https://huggingface.co/google/realm-orqa-wq-reader/resolve/main/config.json",
    }
)

REALM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/realm-cc-news-pretrained-embedder",
        "google/realm-cc-news-pretrained-encoder",
        "google/realm-cc-news-pretrained-scorer",
        "google/realm-cc-news-pretrained-openqa",
        "google/realm-orqa-nq-openqa",
        "google/realm-orqa-nq-reader",
        "google/realm-orqa-wq-openqa",
        "google/realm-orqa-wq-reader",
    ]
)

REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/reformer-crime-and-punishment": "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/config.json",
        "google/reformer-enwik8": "https://huggingface.co/google/reformer-enwik8/resolve/main/config.json",
    }
)

REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["google/reformer-crime-and-punishment", "google/reformer-enwik8"]
)

REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/regnet-y-040": "https://huggingface.co/facebook/regnet-y-040/blob/main/config.json"}
)

REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/regnet-y-040"])

TF_REGNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/regnet-y-040"])

REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/rembert": "https://huggingface.co/google/rembert/resolve/main/config.json"}
)

REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/rembert"])

TF_REMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/rembert"])

RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/resnet-50": "https://huggingface.co/microsoft/resnet-50/blob/main/config.json"}
)

RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/resnet-50"])

TF_RESNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/resnet-50"])

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "FacebookAI/roberta-base": "https://huggingface.co/FacebookAI/roberta-base/resolve/main/config.json",
        "FacebookAI/roberta-large": "https://huggingface.co/FacebookAI/roberta-large/resolve/main/config.json",
        "FacebookAI/roberta-large-mnli": "https://huggingface.co/FacebookAI/roberta-large-mnli/resolve/main/config.json",
        "distilbert/distilroberta-base": "https://huggingface.co/distilbert/distilroberta-base/resolve/main/config.json",
        "openai-community/roberta-base-openai-detector": "https://huggingface.co/openai-community/roberta-base-openai-detector/resolve/main/config.json",
        "openai-community/roberta-large-openai-detector": "https://huggingface.co/openai-community/roberta-large-openai-detector/resolve/main/config.json",
    }
)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "FacebookAI/roberta-large-mnli",
        "distilbert/distilroberta-base",
        "openai-community/roberta-base-openai-detector",
        "openai-community/roberta-large-openai-detector",
    ]
)

TF_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "FacebookAI/roberta-base",
        "FacebookAI/roberta-large",
        "FacebookAI/roberta-large-mnli",
        "distilbert/distilroberta-base",
    ]
)

ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "andreasmadsen/efficient_mlm_m0.40": "https://huggingface.co/andreasmadsen/efficient_mlm_m0.40/resolve/main/config.json"
    }
)

ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "andreasmadsen/efficient_mlm_m0.15",
        "andreasmadsen/efficient_mlm_m0.20",
        "andreasmadsen/efficient_mlm_m0.30",
        "andreasmadsen/efficient_mlm_m0.40",
        "andreasmadsen/efficient_mlm_m0.50",
        "andreasmadsen/efficient_mlm_m0.60",
        "andreasmadsen/efficient_mlm_m0.70",
        "andreasmadsen/efficient_mlm_m0.80",
    ]
)

TF_ROBERTA_PRELAYERNORM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "andreasmadsen/efficient_mlm_m0.15",
        "andreasmadsen/efficient_mlm_m0.20",
        "andreasmadsen/efficient_mlm_m0.30",
        "andreasmadsen/efficient_mlm_m0.40",
        "andreasmadsen/efficient_mlm_m0.50",
        "andreasmadsen/efficient_mlm_m0.60",
        "andreasmadsen/efficient_mlm_m0.70",
        "andreasmadsen/efficient_mlm_m0.80",
    ]
)

ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"weiweishi/roc-bert-base-zh": "https://huggingface.co/weiweishi/roc-bert-base-zh/resolve/main/config.json"}
)

ROC_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["weiweishi/roc-bert-base-zh"])

ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "junnyu/roformer_chinese_small": "https://huggingface.co/junnyu/roformer_chinese_small/resolve/main/config.json",
        "junnyu/roformer_chinese_base": "https://huggingface.co/junnyu/roformer_chinese_base/resolve/main/config.json",
        "junnyu/roformer_chinese_char_small": "https://huggingface.co/junnyu/roformer_chinese_char_small/resolve/main/config.json",
        "junnyu/roformer_chinese_char_base": "https://huggingface.co/junnyu/roformer_chinese_char_base/resolve/main/config.json",
        "junnyu/roformer_small_discriminator": "https://huggingface.co/junnyu/roformer_small_discriminator/resolve/main/config.json",
        "junnyu/roformer_small_generator": "https://huggingface.co/junnyu/roformer_small_generator/resolve/main/config.json",
    }
)

ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "junnyu/roformer_chinese_small",
        "junnyu/roformer_chinese_base",
        "junnyu/roformer_chinese_char_small",
        "junnyu/roformer_chinese_char_base",
        "junnyu/roformer_small_discriminator",
        "junnyu/roformer_small_generator",
    ]
)

TF_ROFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "junnyu/roformer_chinese_small",
        "junnyu/roformer_chinese_base",
        "junnyu/roformer_chinese_char_small",
        "junnyu/roformer_chinese_char_base",
        "junnyu/roformer_small_discriminator",
        "junnyu/roformer_small_generator",
    ]
)

RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "RWKV/rwkv-4-169m-pile": "https://huggingface.co/RWKV/rwkv-4-169m-pile/resolve/main/config.json",
        "RWKV/rwkv-4-430m-pile": "https://huggingface.co/RWKV/rwkv-4-430m-pile/resolve/main/config.json",
        "RWKV/rwkv-4-1b5-pile": "https://huggingface.co/RWKV/rwkv-4-1b5-pile/resolve/main/config.json",
        "RWKV/rwkv-4-3b-pile": "https://huggingface.co/RWKV/rwkv-4-3b-pile/resolve/main/config.json",
        "RWKV/rwkv-4-7b-pile": "https://huggingface.co/RWKV/rwkv-4-7b-pile/resolve/main/config.json",
        "RWKV/rwkv-4-14b-pile": "https://huggingface.co/RWKV/rwkv-4-14b-pile/resolve/main/config.json",
        "RWKV/rwkv-raven-1b5": "https://huggingface.co/RWKV/rwkv-raven-1b5/resolve/main/config.json",
        "RWKV/rwkv-raven-3b": "https://huggingface.co/RWKV/rwkv-raven-3b/resolve/main/config.json",
        "RWKV/rwkv-raven-7b": "https://huggingface.co/RWKV/rwkv-raven-7b/resolve/main/config.json",
        "RWKV/rwkv-raven-14b": "https://huggingface.co/RWKV/rwkv-raven-14b/resolve/main/config.json",
    }
)

RWKV_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "RWKV/rwkv-4-169m-pile",
        "RWKV/rwkv-4-430m-pile",
        "RWKV/rwkv-4-1b5-pile",
        "RWKV/rwkv-4-3b-pile",
        "RWKV/rwkv-4-7b-pile",
        "RWKV/rwkv-4-14b-pile",
        "RWKV/rwkv-raven-1b5",
        "RWKV/rwkv-raven-3b",
        "RWKV/rwkv-raven-7b",
        "RWKV/rwkv-raven-14b",
    ]
)

SAM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/sam-vit-huge": "https://huggingface.co/facebook/sam-vit-huge/resolve/main/config.json",
        "facebook/sam-vit-large": "https://huggingface.co/facebook/sam-vit-large/resolve/main/config.json",
        "facebook/sam-vit-base": "https://huggingface.co/facebook/sam-vit-base/resolve/main/config.json",
    }
)

SAM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/sam-vit-huge", "facebook/sam-vit-large", "facebook/sam-vit-base"]
)

TF_SAM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["facebook/sam-vit-huge", "facebook/sam-vit-large", "facebook/sam-vit-base"]
)

SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/config.json"
    }
)

SEAMLESS_M4T_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/hf-seamless-m4t-medium"])

SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"": "https://huggingface.co//resolve/main/config.json"}
)

SEAMLESS_M4T_V2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/seamless-m4t-v2-large"])

SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "nvidia/segformer-b0-finetuned-ade-512-512": "https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/config.json"
    }
)

SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["nvidia/segformer-b0-finetuned-ade-512-512"])

TF_SEGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["nvidia/segformer-b0-finetuned-ade-512-512"])

SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"BAAI/seggpt-vit-large": "https://huggingface.co/BAAI/seggpt-vit-large/resolve/main/config.json"}
)

SEGGPT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["BAAI/seggpt-vit-large"])

SEW_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"asapp/sew-tiny-100k": "https://huggingface.co/asapp/sew-tiny-100k/resolve/main/config.json"}
)

SEW_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["asapp/sew-tiny-100k", "asapp/sew-small-100k", "asapp/sew-mid-100k"]
)

SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"asapp/sew-d-tiny-100k": "https://huggingface.co/asapp/sew-d-tiny-100k/resolve/main/config.json"}
)

SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "asapp/sew-d-tiny-100k",
        "asapp/sew-d-small-100k",
        "asapp/sew-d-mid-100k",
        "asapp/sew-d-mid-k127-100k",
        "asapp/sew-d-base-100k",
        "asapp/sew-d-base-plus-100k",
        "asapp/sew-d-mid-400k",
        "asapp/sew-d-mid-k127-400k",
        "asapp/sew-d-base-plus-400k",
    ]
)

SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/siglip-base-patch16-224": "https://huggingface.co/google/siglip-base-patch16-224/resolve/main/config.json"
    }
)

SIGLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/siglip-base-patch16-224"])

SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/s2t-small-librispeech-asr": "https://huggingface.co/facebook/s2t-small-librispeech-asr/resolve/main/config.json"
    }
)

SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/s2t-small-librispeech-asr"])

TF_SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/s2t-small-librispeech-asr"])

SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/s2t-wav2vec2-large-en-de": "https://huggingface.co/facebook/s2t-wav2vec2-large-en-de/resolve/main/config.json"
    }
)

SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/speecht5_asr": "https://huggingface.co/microsoft/speecht5_asr/resolve/main/config.json",
        "microsoft/speecht5_tts": "https://huggingface.co/microsoft/speecht5_tts/resolve/main/config.json",
        "microsoft/speecht5_vc": "https://huggingface.co/microsoft/speecht5_vc/resolve/main/config.json",
    }
)

SPEECHT5_PRETRAINED_HIFIGAN_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/speecht5_hifigan": "https://huggingface.co/microsoft/speecht5_hifigan/resolve/main/config.json"}
)

SPEECHT5_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["microsoft/speecht5_asr", "microsoft/speecht5_tts", "microsoft/speecht5_vc"]
)

SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "tau/splinter-base": "https://huggingface.co/tau/splinter-base/resolve/main/config.json",
        "tau/splinter-base-qass": "https://huggingface.co/tau/splinter-base-qass/resolve/main/config.json",
        "tau/splinter-large": "https://huggingface.co/tau/splinter-large/resolve/main/config.json",
        "tau/splinter-large-qass": "https://huggingface.co/tau/splinter-large-qass/resolve/main/config.json",
    }
)

SPLINTER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["tau/splinter-base", "tau/splinter-base-qass", "tau/splinter-large", "tau/splinter-large-qass"]
)

SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "squeezebert/squeezebert-uncased": "https://huggingface.co/squeezebert/squeezebert-uncased/resolve/main/config.json",
        "squeezebert/squeezebert-mnli": "https://huggingface.co/squeezebert/squeezebert-mnli/resolve/main/config.json",
        "squeezebert/squeezebert-mnli-headless": "https://huggingface.co/squeezebert/squeezebert-mnli-headless/resolve/main/config.json",
    }
)

SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["squeezebert/squeezebert-uncased", "squeezebert/squeezebert-mnli", "squeezebert/squeezebert-mnli-headless"]
)

STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"stabilityai/stablelm-3b-4e1t": "https://huggingface.co/stabilityai/stablelm-3b-4e1t/resolve/main/config.json"}
)

STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict({})

SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"MBZUAI/swiftformer-xs": "https://huggingface.co/MBZUAI/swiftformer-xs/resolve/main/config.json"}
)

SWIFTFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["MBZUAI/swiftformer-xs"])

SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/swin-tiny-patch4-window7-224": "https://huggingface.co/microsoft/swin-tiny-patch4-window7-224/resolve/main/config.json"
    }
)

SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/swin-tiny-patch4-window7-224"])

TF_SWIN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/swin-tiny-patch4-window7-224"])

SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "caidas/swin2sr-classicalsr-x2-64": "https://huggingface.co/caidas/swin2sr-classicalsr-x2-64/resolve/main/config.json"
    }
)

SWIN2SR_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["caidas/swin2SR-classical-sr-x2-64"])

SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/swinv2-tiny-patch4-window8-256": "https://huggingface.co/microsoft/swinv2-tiny-patch4-window8-256/resolve/main/config.json"
    }
)

SWINV2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/swinv2-tiny-patch4-window8-256"])

SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/switch-base-8": "https://huggingface.co/google/switch-base-8/blob/main/config.json"}
)

SWITCH_TRANSFORMERS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/switch-base-8",
        "google/switch-base-16",
        "google/switch-base-32",
        "google/switch-base-64",
        "google/switch-base-128",
        "google/switch-base-256",
        "google/switch-large-128",
        "google/switch-xxl-128",
        "google/switch-c-2048",
    ]
)

T5_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google-t5/t5-small": "https://huggingface.co/google-t5/t5-small/resolve/main/config.json",
        "google-t5/t5-base": "https://huggingface.co/google-t5/t5-base/resolve/main/config.json",
        "google-t5/t5-large": "https://huggingface.co/google-t5/t5-large/resolve/main/config.json",
        "google-t5/t5-3b": "https://huggingface.co/google-t5/t5-3b/resolve/main/config.json",
        "google-t5/t5-11b": "https://huggingface.co/google-t5/t5-11b/resolve/main/config.json",
    }
)

T5_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["google-t5/t5-small", "google-t5/t5-base", "google-t5/t5-large", "google-t5/t5-3b", "google-t5/t5-11b"]
)

TF_T5_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["google-t5/t5-small", "google-t5/t5-base", "google-t5/t5-large", "google-t5/t5-3b", "google-t5/t5-11b"]
)

TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/table-transformer-detection": "https://huggingface.co/microsoft/table-transformer-detection/resolve/main/config.json"
    }
)

TABLE_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/table-transformer-detection"])

TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/tapas-base-finetuned-sqa": "https://huggingface.co/google/tapas-base-finetuned-sqa/resolve/main/config.json",
        "google/tapas-base-finetuned-wtq": "https://huggingface.co/google/tapas-base-finetuned-wtq/resolve/main/config.json",
        "google/tapas-base-finetuned-wikisql-supervised": "https://huggingface.co/google/tapas-base-finetuned-wikisql-supervised/resolve/main/config.json",
        "google/tapas-base-finetuned-tabfact": "https://huggingface.co/google/tapas-base-finetuned-tabfact/resolve/main/config.json",
    }
)

TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/tapas-large",
        "google/tapas-large-finetuned-sqa",
        "google/tapas-large-finetuned-wtq",
        "google/tapas-large-finetuned-wikisql-supervised",
        "google/tapas-large-finetuned-tabfact",
        "google/tapas-base",
        "google/tapas-base-finetuned-sqa",
        "google/tapas-base-finetuned-wtq",
        "google/tapas-base-finetuned-wikisql-supervised",
        "google/tapas-base-finetuned-tabfact",
        "google/tapas-small",
        "google/tapas-small-finetuned-sqa",
        "google/tapas-small-finetuned-wtq",
        "google/tapas-small-finetuned-wikisql-supervised",
        "google/tapas-small-finetuned-tabfact",
        "google/tapas-mini",
        "google/tapas-mini-finetuned-sqa",
        "google/tapas-mini-finetuned-wtq",
        "google/tapas-mini-finetuned-wikisql-supervised",
        "google/tapas-mini-finetuned-tabfact",
        "google/tapas-tiny",
        "google/tapas-tiny-finetuned-sqa",
        "google/tapas-tiny-finetuned-wtq",
        "google/tapas-tiny-finetuned-wikisql-supervised",
        "google/tapas-tiny-finetuned-tabfact",
    ]
)

TF_TAPAS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "google/tapas-large",
        "google/tapas-large-finetuned-sqa",
        "google/tapas-large-finetuned-wtq",
        "google/tapas-large-finetuned-wikisql-supervised",
        "google/tapas-large-finetuned-tabfact",
        "google/tapas-base",
        "google/tapas-base-finetuned-sqa",
        "google/tapas-base-finetuned-wtq",
        "google/tapas-base-finetuned-wikisql-supervised",
        "google/tapas-base-finetuned-tabfact",
        "google/tapas-small",
        "google/tapas-small-finetuned-sqa",
        "google/tapas-small-finetuned-wtq",
        "google/tapas-small-finetuned-wikisql-supervised",
        "google/tapas-small-finetuned-tabfact",
        "google/tapas-mini",
        "google/tapas-mini-finetuned-sqa",
        "google/tapas-mini-finetuned-wtq",
        "google/tapas-mini-finetuned-wikisql-supervised",
        "google/tapas-mini-finetuned-tabfact",
        "google/tapas-tiny",
        "google/tapas-tiny-finetuned-sqa",
        "google/tapas-tiny-finetuned-wtq",
        "google/tapas-tiny-finetuned-wikisql-supervised",
        "google/tapas-tiny-finetuned-tabfact",
    ]
)

TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "huggingface/time-series-transformer-tourism-monthly": "https://huggingface.co/huggingface/time-series-transformer-tourism-monthly/resolve/main/config.json"
    }
)

TIME_SERIES_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["huggingface/time-series-transformer-tourism-monthly"]
)

TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/timesformer": "https://huggingface.co/facebook/timesformer/resolve/main/config.json"}
)

TIMESFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/timesformer-base-finetuned-k400"])

TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/trocr-base-handwritten": "https://huggingface.co/microsoft/trocr-base-handwritten/resolve/main/config.json"
    }
)

TROCR_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/trocr-base-handwritten"])

TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"ZinengTang/tvlt-base": "https://huggingface.co/ZinengTang/tvlt-base/blob/main/config.json"}
)

TVLT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["ZinengTang/tvlt-base"])

TVP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"Intel/tvp-base": "https://huggingface.co/Intel/tvp-base/resolve/main/config.json"}
)

TVP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["Intel/tvp-base", "Intel/tvp-base-ANet"])

UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/udop-large": "https://huggingface.co/microsoft/udop-large/resolve/main/config.json"}
)

UDOP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/udop-large"])

UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/unispeech-large-1500h-cv": "https://huggingface.co/microsoft/unispeech-large-1500h-cv/resolve/main/config.json"
    }
)

UNISPEECH_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["microsoft/unispeech-large-1500h-cv", "microsoft/unispeech-large-multi-lingual-1500h-cv"]
)

UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/unispeech-sat-base-100h-libri-ft": "https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft/resolve/main/config.json"
    }
)

UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList([])

UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"dg845/univnet-dev": "https://huggingface.co/dg845/univnet-dev/resolve/main/config.json"}
)

UNIVNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["dg845/univnet-dev"])

VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"MCG-NJU/videomae-base": "https://huggingface.co/MCG-NJU/videomae-base/resolve/main/config.json"}
)

VIDEOMAE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["MCG-NJU/videomae-base"])

VILT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"dandelin/vilt-b32-mlm": "https://huggingface.co/dandelin/vilt-b32-mlm/blob/main/config.json"}
)

VILT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["dandelin/vilt-b32-mlm"])

VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"ybelkada/vip-llava-7b-hf": "https://huggingface.co/llava-hf/vip-llava-7b-hf/resolve/main/config.json"}
)

VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["llava-hf/vip-llava-7b-hf"])

VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "uclanlp/visualbert-vqa": "https://huggingface.co/uclanlp/visualbert-vqa/resolve/main/config.json",
        "uclanlp/visualbert-vqa-pre": "https://huggingface.co/uclanlp/visualbert-vqa-pre/resolve/main/config.json",
        "uclanlp/visualbert-vqa-coco-pre": "https://huggingface.co/uclanlp/visualbert-vqa-coco-pre/resolve/main/config.json",
        "uclanlp/visualbert-vcr": "https://huggingface.co/uclanlp/visualbert-vcr/resolve/main/config.json",
        "uclanlp/visualbert-vcr-pre": "https://huggingface.co/uclanlp/visualbert-vcr-pre/resolve/main/config.json",
        "uclanlp/visualbert-vcr-coco-pre": "https://huggingface.co/uclanlp/visualbert-vcr-coco-pre/resolve/main/config.json",
        "uclanlp/visualbert-nlvr2": "https://huggingface.co/uclanlp/visualbert-nlvr2/resolve/main/config.json",
        "uclanlp/visualbert-nlvr2-pre": "https://huggingface.co/uclanlp/visualbert-nlvr2-pre/resolve/main/config.json",
        "uclanlp/visualbert-nlvr2-coco-pre": "https://huggingface.co/uclanlp/visualbert-nlvr2-coco-pre/resolve/main/config.json",
    }
)

VISUAL_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "uclanlp/visualbert-vqa",
        "uclanlp/visualbert-vqa-pre",
        "uclanlp/visualbert-vqa-coco-pre",
        "uclanlp/visualbert-vcr",
        "uclanlp/visualbert-vcr-pre",
        "uclanlp/visualbert-vcr-coco-pre",
        "uclanlp/visualbert-nlvr2",
        "uclanlp/visualbert-nlvr2-pre",
        "uclanlp/visualbert-nlvr2-coco-pre",
    ]
)

VIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/vit-base-patch16-224": "https://huggingface.co/vit-base-patch16-224/resolve/main/config.json"}
)

VIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/vit-base-patch16-224"])

VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"google/vit-hybrid-base-bit-384": "https://huggingface.co/vit-hybrid-base-bit-384/resolve/main/config.json"}
)

VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/vit-hybrid-base-bit-384"])

VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/vit-mae-base": "https://huggingface.co/facebook/vit-mae-base/resolve/main/config.json"}
)

VIT_MAE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/vit-mae-base"])

VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"sayakpaul/vit-msn-base": "https://huggingface.co/sayakpaul/vit-msn-base/resolve/main/config.json"}
)

VIT_MSN_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/vit-msn-small"])

VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/vit-det-base": "https://huggingface.co/facebook/vit-det-base/resolve/main/config.json"}
)

VITDET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/vit-det-base"])

VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "hustvl/vitmatte-small-composition-1k": "https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json"
    }
)

VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["hustvl/vitmatte-small-composition-1k"])

VITS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/mms-tts-eng": "https://huggingface.co/facebook/mms-tts-eng/resolve/main/config.json"}
)

VITS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/mms-tts-eng"])

VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "google/vivit-b-16x2-kinetics400": "https://huggingface.co/google/vivit-b-16x2-kinetics400/resolve/main/config.json"
    }
)

VIVIT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["google/vivit-b-16x2-kinetics400"])

WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/wav2vec2-base-960h": "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/config.json"}
)

WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h",
        "facebook/wav2vec2-large-960h-lv60",
        "facebook/wav2vec2-large-960h-lv60-self",
    ]
)

TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "facebook/wav2vec2-base-960h",
        "facebook/wav2vec2-large-960h",
        "facebook/wav2vec2-large-960h-lv60",
        "facebook/wav2vec2-large-960h-lv60-self",
    ]
)

WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/w2v-bert-2.0": "https://huggingface.co/facebook/w2v-bert-2.0/resolve/main/config.json"}
)

WAV2VEC2_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/w2v-bert-2.0"])

WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/wav2vec2-conformer-rel-pos-large": "https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large/resolve/main/config.json"
    }
)

WAV2VEC2_CONFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/wav2vec2-conformer-rel-pos-large"])

WAVLM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/wavlm-base": "https://huggingface.co/microsoft/wavlm-base/resolve/main/config.json"}
)

WAVLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["microsoft/wavlm-base", "microsoft/wavlm-base-plus", "microsoft/wavlm-large"]
)

WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/config.json"}
)

WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai/whisper-base"])

TF_WHISPER_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["openai/whisper-base"])

XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"microsoft/xclip-base-patch32": "https://huggingface.co/microsoft/xclip-base-patch32/resolve/main/config.json"}
)

XCLIP_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/xclip-base-patch32"])

XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"facebook/xglm-564M": "https://huggingface.co/facebook/xglm-564M/resolve/main/config.json"}
)

XGLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/xglm-564M"])

TF_XGLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/xglm-564M"])

XLM_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "FacebookAI/xlm-mlm-en-2048": "https://huggingface.co/FacebookAI/xlm-mlm-en-2048/resolve/main/config.json",
        "FacebookAI/xlm-mlm-ende-1024": "https://huggingface.co/FacebookAI/xlm-mlm-ende-1024/resolve/main/config.json",
        "FacebookAI/xlm-mlm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enfr-1024/resolve/main/config.json",
        "FacebookAI/xlm-mlm-enro-1024": "https://huggingface.co/FacebookAI/xlm-mlm-enro-1024/resolve/main/config.json",
        "FacebookAI/xlm-mlm-tlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-tlm-xnli15-1024/resolve/main/config.json",
        "FacebookAI/xlm-mlm-xnli15-1024": "https://huggingface.co/FacebookAI/xlm-mlm-xnli15-1024/resolve/main/config.json",
        "FacebookAI/xlm-clm-enfr-1024": "https://huggingface.co/FacebookAI/xlm-clm-enfr-1024/resolve/main/config.json",
        "FacebookAI/xlm-clm-ende-1024": "https://huggingface.co/FacebookAI/xlm-clm-ende-1024/resolve/main/config.json",
        "FacebookAI/xlm-mlm-17-1280": "https://huggingface.co/FacebookAI/xlm-mlm-17-1280/resolve/main/config.json",
        "FacebookAI/xlm-mlm-100-1280": "https://huggingface.co/FacebookAI/xlm-mlm-100-1280/resolve/main/config.json",
    }
)

XLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "FacebookAI/xlm-mlm-en-2048",
        "FacebookAI/xlm-mlm-ende-1024",
        "FacebookAI/xlm-mlm-enfr-1024",
        "FacebookAI/xlm-mlm-enro-1024",
        "FacebookAI/xlm-mlm-tlm-xnli15-1024",
        "FacebookAI/xlm-mlm-xnli15-1024",
        "FacebookAI/xlm-clm-enfr-1024",
        "FacebookAI/xlm-clm-ende-1024",
        "FacebookAI/xlm-mlm-17-1280",
        "FacebookAI/xlm-mlm-100-1280",
    ]
)

TF_XLM_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "FacebookAI/xlm-mlm-en-2048",
        "FacebookAI/xlm-mlm-ende-1024",
        "FacebookAI/xlm-mlm-enfr-1024",
        "FacebookAI/xlm-mlm-enro-1024",
        "FacebookAI/xlm-mlm-tlm-xnli15-1024",
        "FacebookAI/xlm-mlm-xnli15-1024",
        "FacebookAI/xlm-clm-enfr-1024",
        "FacebookAI/xlm-clm-ende-1024",
        "FacebookAI/xlm-mlm-17-1280",
        "FacebookAI/xlm-mlm-100-1280",
    ]
)

XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "microsoft/xprophetnet-large-wiki100-cased": "https://huggingface.co/microsoft/xprophetnet-large-wiki100-cased/resolve/main/config.json"
    }
)

XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["microsoft/xprophetnet-large-wiki100-cased"])

XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "FacebookAI/xlm-roberta-base": "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main/config.json",
        "FacebookAI/xlm-roberta-large": "https://huggingface.co/FacebookAI/xlm-roberta-large/resolve/main/config.json",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch": "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-dutch/resolve/main/config.json",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish": "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll02-spanish/resolve/main/config.json",
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english": "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english/resolve/main/config.json",
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german": "https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german/resolve/main/config.json",
    }
)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-dutch",
        "FacebookAI/xlm-roberta-large-finetuned-conll02-spanish",
        "FacebookAI/xlm-roberta-large-finetuned-conll03-english",
        "FacebookAI/xlm-roberta-large-finetuned-conll03-german",
    ]
)

TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "FacebookAI/xlm-roberta-base",
        "FacebookAI/xlm-roberta-large",
        "joeddav/xlm-roberta-large-xnli",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    ]
)

FLAX_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    ["FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-large"]
)

XLM_ROBERTA_XL_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/xlm-roberta-xl": "https://huggingface.co/facebook/xlm-roberta-xl/resolve/main/config.json",
        "facebook/xlm-roberta-xxl": "https://huggingface.co/facebook/xlm-roberta-xxl/resolve/main/config.json",
    }
)

XLM_ROBERTA_XL_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["facebook/xlm-roberta-xl", "facebook/xlm-roberta-xxl"])

XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "xlnet/xlnet-base-cased": "https://huggingface.co/xlnet/xlnet-base-cased/resolve/main/config.json",
        "xlnet/xlnet-large-cased": "https://huggingface.co/xlnet/xlnet-large-cased/resolve/main/config.json",
    }
)

XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["xlnet/xlnet-base-cased", "xlnet/xlnet-large-cased"])

TF_XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["xlnet/xlnet-base-cased", "xlnet/xlnet-large-cased"])

XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {
        "facebook/xmod-base": "https://huggingface.co/facebook/xmod-base/resolve/main/config.json",
        "facebook/xmod-large-prenorm": "https://huggingface.co/facebook/xmod-large-prenorm/resolve/main/config.json",
        "facebook/xmod-base-13-125k": "https://huggingface.co/facebook/xmod-base-13-125k/resolve/main/config.json",
        "facebook/xmod-base-30-125k": "https://huggingface.co/facebook/xmod-base-30-125k/resolve/main/config.json",
        "facebook/xmod-base-30-195k": "https://huggingface.co/facebook/xmod-base-30-195k/resolve/main/config.json",
        "facebook/xmod-base-60-125k": "https://huggingface.co/facebook/xmod-base-60-125k/resolve/main/config.json",
        "facebook/xmod-base-60-265k": "https://huggingface.co/facebook/xmod-base-60-265k/resolve/main/config.json",
        "facebook/xmod-base-75-125k": "https://huggingface.co/facebook/xmod-base-75-125k/resolve/main/config.json",
        "facebook/xmod-base-75-269k": "https://huggingface.co/facebook/xmod-base-75-269k/resolve/main/config.json",
    }
)

XMOD_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(
    [
        "facebook/xmod-base",
        "facebook/xmod-large-prenorm",
        "facebook/xmod-base-13-125k",
        "facebook/xmod-base-30-125k",
        "facebook/xmod-base-30-195k",
        "facebook/xmod-base-60-125k",
        "facebook/xmod-base-60-265k",
        "facebook/xmod-base-75-125k",
        "facebook/xmod-base-75-269k",
    ]
)

YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"hustvl/yolos-small": "https://huggingface.co/hustvl/yolos-small/resolve/main/config.json"}
)

YOLOS_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["hustvl/yolos-small"])

YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP = DeprecatedDict(
    {"uw-madison/yoso-4096": "https://huggingface.co/uw-madison/yoso-4096/resolve/main/config.json"}
)

YOSO_PRETRAINED_MODEL_ARCHIVE_LIST = DeprecatedList(["uw-madison/yoso-4096"])


CONFIG_ARCHIVE_MAP_MAPPING_NAMES = OrderedDict(
    [
        # Add archive maps here)
        ("albert", "ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("align", "ALIGN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("altclip", "ALTCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("audio-spectrogram-transformer", "AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("autoformer", "AUTOFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bark", "BARK_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bart", "BART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("beit", "BEIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bert", "BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("big_bird", "BIG_BIRD_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bigbird_pegasus", "BIGBIRD_PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("biogpt", "BIOGPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bit", "BIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blenderbot", "BLENDERBOT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blenderbot-small", "BLENDERBOT_SMALL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blip", "BLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("blip-2", "BLIP_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bloom", "BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bridgetower", "BRIDGETOWER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("bros", "BROS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("camembert", "CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("canine", "CANINE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("chinese_clip", "CHINESE_CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("clap", "CLAP_PRETRAINED_MODEL_ARCHIVE_LIST"),
        ("clip", "CLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("clipseg", "CLIPSEG_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("clvp", "CLVP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("codegen", "CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("conditional_detr", "CONDITIONAL_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convbert", "CONVBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convnext", "CONVNEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("convnextv2", "CONVNEXTV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("cpmant", "CPMANT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ctrl", "CTRL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("cvt", "CVT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-audio", "DATA2VEC_AUDIO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-text", "DATA2VEC_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("data2vec-vision", "DATA2VEC_VISION_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deberta", "DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deberta-v2", "DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deformable_detr", "DEFORMABLE_DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deit", "DEIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("depth_anything", "DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("deta", "DETA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("detr", "DETR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dinat", "DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dinov2", "DINOV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("distilbert", "DISTILBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("donut-swin", "DONUT_SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dpr", "DPR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("dpt", "DPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("efficientformer", "EFFICIENTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("efficientnet", "EFFICIENTNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("electra", "ELECTRA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("encodec", "ENCODEC_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ernie", "ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ernie_m", "ERNIE_M_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("esm", "ESM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("falcon", "FALCON_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fastspeech2_conformer", "FASTSPEECH2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("flaubert", "FLAUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("flava", "FLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fnet", "FNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("focalnet", "FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fsmt", "FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("funnel", "FUNNEL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("fuyu", "FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gemma", "GEMMA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("git", "GIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("glpn", "GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt2", "GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_bigcode", "GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neo", "GPT_NEO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neox", "GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gpt_neox_japanese", "GPT_NEOX_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gptj", "GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("gptsan-japanese", "GPTSAN_JAPANESE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("graphormer", "GRAPHORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("groupvit", "GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("hubert", "HUBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("ibert", "IBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("idefics", "IDEFICS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("imagegpt", "IMAGEGPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("informer", "INFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("instructblip", "INSTRUCTBLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("jukebox", "JUKEBOX_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("kosmos-2", "KOSMOS2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlm", "LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlmv2", "LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("layoutlmv3", "LAYOUTLMV3_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("led", "LED_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("levit", "LEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("lilt", "LILT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("llama", "LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("llava", "LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("longformer", "LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("longt5", "LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("luke", "LUKE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("lxmert", "LXMERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("m2m_100", "M2M_100_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mamba", "MAMBA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("markuplm", "MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mask2former", "MASK2FORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("maskformer", "MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mbart", "MBART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mctct", "MCTCT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mega", "MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("megatron-bert", "MEGATRON_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mgp-str", "MGP_STR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mistral", "MISTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mixtral", "MIXTRAL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilenet_v1", "MOBILENET_V1_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilenet_v2", "MOBILENET_V2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilevit", "MOBILEVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mobilevitv2", "MOBILEVITV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mpnet", "MPNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mpt", "MPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mra", "MRA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("musicgen", "MUSICGEN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("mvp", "MVP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nat", "NAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nezha", "NEZHA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nllb-moe", "NLLB_MOE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("nystromformer", "NYSTROMFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("oneformer", "ONEFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("olmo", "OLMO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("open-llama", "OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("openai-gpt", "OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("opt", "OPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("owlv2", "OWLV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("owlvit", "OWLVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("patchtsmixer", "PATCHTSMIXER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("patchtst", "PATCHTST_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pegasus", "PEGASUS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pegasus_x", "PEGASUS_X_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("perceiver", "PERCEIVER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("persimmon", "PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("phi", "PHI_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pix2struct", "PIX2STRUCT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("plbart", "PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("poolformer", "POOLFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pop2piano", "POP2PIANO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("prophetnet", "PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("pvt", "PVT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("qdqbert", "QDQBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("qwen2", "QWEN2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("realm", "REALM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("regnet", "REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("rembert", "REMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("resnet", "RESNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("retribert", "RETRIBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roberta", "ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roberta-prelayernorm", "ROBERTA_PRELAYERNORM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roc_bert", "ROC_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("roformer", "ROFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("rwkv", "RWKV_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sam", "SAM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("seamless_m4t", "SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("seamless_m4t_v2", "SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("segformer", "SEGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("seggpt", "SEGGPT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sew", "SEW_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("sew-d", "SEW_D_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("siglip", "SIGLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speech_to_text", "SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speech_to_text_2", "SPEECH_TO_TEXT_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("speecht5", "SPEECHT5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("splinter", "SPLINTER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("squeezebert", "SQUEEZEBERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("stablelm", "STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("starcoder2", "STARCODER2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swiftformer", "SWIFTFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swin", "SWIN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swin2sr", "SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("swinv2", "SWINV2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("switch_transformers", "SWITCH_TRANSFORMERS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("t5", "T5_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("table-transformer", "TABLE_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("tapas", "TAPAS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("time_series_transformer", "TIME_SERIES_TRANSFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("timesformer", "TIMESFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("transfo-xl", "TRANSFO_XL_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("tvlt", "TVLT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("tvp", "TVP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("udop", "UDOP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("unispeech", "UNISPEECH_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("unispeech-sat", "UNISPEECH_SAT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("univnet", "UNIVNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("van", "VAN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("videomae", "VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vilt", "VILT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vipllava", "VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("visual_bert", "VISUAL_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit", "VIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_hybrid", "VIT_HYBRID_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_mae", "VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vit_msn", "VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vitdet", "VITDET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vitmatte", "VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vits", "VITS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("vivit", "VIVIT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("wav2vec2", "WAV_2_VEC_2_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("wav2vec2-bert", "WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("wav2vec2-conformer", "WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("whisper", "WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xclip", "XCLIP_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xglm", "XGLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm", "XLM_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-prophetnet", "XLM_PROPHETNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlm-roberta", "XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xlnet", "XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("xmod", "XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("yolos", "YOLOS_PRETRAINED_CONFIG_ARCHIVE_MAP"),
        ("yoso", "YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP"),
    ]
)
