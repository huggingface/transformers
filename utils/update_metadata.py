# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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

import argparse
import collections
import importlib.util
import os
import re
import tempfile

import pandas as pd
from datasets import Dataset

from huggingface_hub import Repository


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/update_metadata.py
TRANSFORMERS_PATH = "src/transformers"


# This is to make sure the transformers module imported is the one in the repo.
spec = importlib.util.spec_from_file_location(
    "transformers",
    os.path.join(TRANSFORMERS_PATH, "__init__.py"),
    submodule_search_locations=[TRANSFORMERS_PATH],
)
transformers_module = spec.loader.load_module()


# Regexes that match TF/Flax/PT model names.
_re_tf_models = re.compile(r"TF(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
_re_flax_models = re.compile(r"Flax(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")
# Will match any TF or Flax model too so need to be in an else branch afterthe two previous regexes.
_re_pt_models = re.compile(r"(.*)(?:Model|Encoder|Decoder|ForConditionalGeneration)")


# Fill this with tuples (pipeline_tag, model_mapping, auto_model)
PIPELINE_TAGS_AND_AUTO_MODELS = [
    ("pretraining", "MODEL_FOR_PRETRAINING_MAPPING_NAMES", "AutoModelForPreTraining"),
    ("feature-extraction", "MODEL_MAPPING_NAMES", "AutoModel"),
    ("audio-classification", "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES", "AutoModelForAudioClassification"),
    ("text-generation", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", "AutoModelForCausalLM"),
    ("automatic-speech-recognition", "MODEL_FOR_CTC_MAPPING_NAMES", "AutoModelForCTC"),
    ("image-classification", "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForImageClassification"),
    ("image-segmentation", "MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES", "AutoModelForImageSegmentation"),
    ("fill-mask", "MODEL_FOR_MASKED_LM_MAPPING_NAMES", "AutoModelForMaskedLM"),
    ("object-detection", "MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES", "AutoModelForObjectDetection"),
    ("question-answering", "MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES", "AutoModelForQuestionAnswering"),
    ("text2text-generation", "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES", "AutoModelForSeq2SeqLM"),
    ("text-classification", "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES", "AutoModelForSequenceClassification"),
    ("automatic-speech-recognition", "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES", "AutoModelForSpeechSeq2Seq"),
    (
        "table-question-answering",
        "MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES",
        "AutoModelForTableQuestionAnswering",
    ),
    ("token-classification", "MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES", "AutoModelForTokenClassification"),
    ("multiple-choice", "MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES", "AutoModelForMultipleChoice"),
    (
        "next-sentence-prediction",
        "MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES",
        "AutoModelForNextSentencePrediction",
    ),
    (
        "audio-frame-classification",
        "MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES",
        "AutoModelForAudioFrameClassification",
    ),
    ("audio-xvector", "MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES", "AutoModelForAudioXVector"),
]


# Thanks to https://stackoverflow.com/questions/29916065/how-to-do-camelcase-split-in-python
def camel_case_split(identifier):
    "Split a camelcased `identifier` into words."
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", identifier)
    return [m.group(0) for m in matches]


def get_frameworks_table():
    """
    Generates a dataframe containing the supported auto classes for each model type, using the content of the auto
    modules.
    """
    # Dictionary model names to config.
    config_maping_names = transformers_module.models.auto.configuration_auto.CONFIG_MAPPING_NAMES
    model_prefix_to_model_type = {
        config.replace("Config", ""): model_type for model_type, config in config_maping_names.items()
    }

    # Dictionaries flagging if each model prefix has a backend in PT/TF/Flax.
    pt_models = collections.defaultdict(bool)
    tf_models = collections.defaultdict(bool)
    flax_models = collections.defaultdict(bool)

    # Let's lookup through all transformers object (once) and find if models are supported by a given backend.
    for attr_name in dir(transformers_module):
        lookup_dict = None
        if _re_tf_models.match(attr_name) is not None:
            lookup_dict = tf_models
            attr_name = _re_tf_models.match(attr_name).groups()[0]
        elif _re_flax_models.match(attr_name) is not None:
            lookup_dict = flax_models
            attr_name = _re_flax_models.match(attr_name).groups()[0]
        elif _re_pt_models.match(attr_name) is not None:
            lookup_dict = pt_models
            attr_name = _re_pt_models.match(attr_name).groups()[0]

        if lookup_dict is not None:
            while len(attr_name) > 0:
                if attr_name in model_prefix_to_model_type:
                    lookup_dict[model_prefix_to_model_type[attr_name]] = True
                    break
                # Try again after removing the last word in the name
                attr_name = "".join(camel_case_split(attr_name)[:-1])

    all_models = set(list(pt_models.keys()) + list(tf_models.keys()) + list(flax_models.keys()))
    all_models = list(all_models)
    all_models.sort()

    data = {"model_type": all_models}
    data["pytorch"] = [pt_models[t] for t in all_models]
    data["tensorflow"] = [tf_models[t] for t in all_models]
    data["flax"] = [flax_models[t] for t in all_models]

    # Now let's use the auto-mapping names to make sure
    processors = {}
    for t in all_models:
        if t in transformers_module.models.auto.processing_auto.PROCESSOR_MAPPING_NAMES:
            processors[t] = "AutoProcessor"
        elif t in transformers_module.models.auto.tokenization_auto.TOKENIZER_MAPPING_NAMES:
            processors[t] = "AutoTokenizer"
        elif t in transformers_module.models.auto.feature_extraction_auto.FEATURE_EXTRACTOR_MAPPING_NAMES:
            processors[t] = "AutoFeatureExtractor"
        else:
            # Default to AutoTokenizer if a model has nothing, for backward compatibility.
            processors[t] = "AutoTokenizer"

    data["processor"] = [processors[t] for t in all_models]

    return pd.DataFrame(data)


def update_pipeline_and_auto_class_table(table):
    """
    Update the table of model class to (pipeline_tag, auto_class) without removing old keys if they don't exist
    anymore.
    """
    auto_modules = [
        transformers_module.models.auto.modeling_auto,
        transformers_module.models.auto.modeling_tf_auto,
        transformers_module.models.auto.modeling_flax_auto,
    ]
    for pipeline_tag, model_mapping, auto_class in PIPELINE_TAGS_AND_AUTO_MODELS:
        model_mappings = [model_mapping, f"TF_{model_mapping}", f"FLAX_{model_mapping}"]
        auto_classes = [auto_class, f"TF_{auto_class}", f"Flax_{auto_class}"]
        # Loop through all three frameworks
        for module, cls, mapping in zip(auto_modules, auto_classes, model_mappings):
            # The type of pipeline may not exist in this framework
            if not hasattr(module, mapping):
                continue
            # First extract all model_names
            model_names = []
            for name in getattr(module, mapping).values():
                if isinstance(name, str):
                    model_names.append(name)
                else:
                    model_names.extend(list(name))

            # Add pipeline tag and auto model class for those models
            table.update({model_name: (pipeline_tag, cls) for model_name in model_names})

    return table


def update_metadata(token, commit_sha):
    """
    Update the metada for the Transformers repo.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Repository(
            tmp_dir, clone_from="huggingface/transformers-metadata", repo_type="dataset", use_auth_token=token
        )

        frameworks_table = get_frameworks_table()
        frameworks_dataset = Dataset.from_pandas(frameworks_table)
        frameworks_dataset.to_json(os.path.join(tmp_dir, "frameworks.json"))

        tags_dataset = Dataset.from_json(os.path.join(tmp_dir, "pipeline_tags.json"))
        table = {
            tags_dataset[i]["model_class"]: (tags_dataset[i]["pipeline_tag"], tags_dataset[i]["auto_class"])
            for i in range(len(tags_dataset))
        }
        table = update_pipeline_and_auto_class_table(table)

        # Sort the model classes to avoid some nondeterministic updates to create false update commits.
        model_classes = sorted(list(table.keys()))
        tags_table = pd.DataFrame(
            {
                "model_class": model_classes,
                "pipeline_tag": [table[m][0] for m in model_classes],
                "auto_class": [table[m][1] for m in model_classes],
            }
        )
        tags_dataset = Dataset.from_pandas(tags_table)
        tags_dataset.to_json(os.path.join(tmp_dir, "pipeline_tags.json"))

        if repo.is_repo_clean():
            print("Nothing to commit!")
        else:
            commit_message = f"Update with commit {commit_sha}" if commit_sha is not None else "Update"
            repo.push_to_hub(commit_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, help="The token to use to push to the transformers-metadata dataset.")
    parser.add_argument("--commit_sha", type=str, help="The sha of the commit going with this update.")
    args = parser.parse_args()

    update_metadata(args.token, args.commit_sha)
