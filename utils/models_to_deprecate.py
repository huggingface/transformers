# Copyright 2024 The HuggingFace Team. All rights reserved.
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
Script to find a candidate list of models to deprecate based on the number of downloads and the date of the last
commit.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from huggingface_hub import HfApi
from tqdm import tqdm

from transformers.models.auto.configuration_auto import DEPRECATED_MODELS, MODEL_NAMES_MAPPING


api = HfApi()

PATH_TO_REPO = Path(__file__).parent.parent.resolve()
repo = Repo(PATH_TO_REPO)


# Used when the folder name on the hub does not match the folder name in `transformers/models`
# format = {folder name in `transformers/models`: expected tag on the hub}
MODEL_FOLDER_NAME_TO_TAG_MAPPING = {
    "audio_spectrogram_transformer": "audio-spectrogram-transformer",
    "bert_generation": "bert-generation",
    "blenderbot_small": "blenderbot-small",
    "blip_2": "blip-2",
    "dab_detr": "dab-detr",
    "data2vec": "data2vec-audio",  # actually, the base model is never used as a tag, but the sub models are
    "deberta_v2": "deberta-v2",
    "donut": "donut-swin",
    "encoder_decoder": "encoder-decoder",
    "grounding_dino": "grounding-dino",
    "kosmos2": "kosmos-2",
    "kosmos2_5": "kosmos-2.5",
    "megatron_bert": "megatron-bert",
    "mgp_str": "mgp-str",
    "mm_grounding_dino": "mm-grounding-dino",
    "modernbert_decoder": "modernbert-decoder",
    "nllb_moe": "nllb-moe",
    "omdet_turbo": "omdet-turbo",
    "openai": "openai-gpt",
    "roberta_prelayernorm": "roberta-prelayernorm",
    "sew_d": "sew-d",
    "speech_encoder_decoder": "speech-encoder-decoder",
    "table_transformer": "table-transformer",
    "unispeech_sat": "unispeech-sat",
    "vision_encoder_decoder": "vision-encoder-decoder",
    "vision_text_dual_encoder": "vision-text-dual-encoder",
    "wav2vec2_bert": "wav2vec2-bert",
    "wav2vec2_conformer": "wav2vec2-conformer",
    "x_clip": "xclip",
    "xlm_roberta": "xlm-roberta",
    "xlm_roberta_xl": "xlm-roberta-xl",
}

# Used on model architectures with multiple tags on the hub (e.g. on VLMs, we often support a text-only model).
# Applied after the model folder name mapping. format = {base model tag: [extra tags]}
EXTRA_TAGS_MAPPING = {
    "aimv2": ["aimv2_vision_model"],
    "aria": ["aria_text"],
    "bart": ["barthez", "bartpho"],
    "bert": ["bert-japanese", "bertweet", "herbert", "phobert"],
    "beit": ["dit"],
    "blip-2": ["blip_2_qformer"],
    "chinese_clip": ["chinese_clip_vision_model"],
    "clip": ["clip_text_model", "clip_vision_model"],
    "data2vec-audio": ["data2vec-text", "data2vec-vision"],
    "depth_anything": ["depth_anything_v2"],
    "donut-swin": ["nougat"],
    "edgetam": ["edgetam_vision_model"],
    "fastspeech2_conformer": ["fastspeech2_conformer_with_hifigan"],
    "gemma3": ["gemma3_text"],
    "gemma3n": ["gemma3n_audio", "gemma3n_text", "gemma3n_vision"],
    "gpt2": ["cpm", "dialogpt", "gpt-sw3", "megatron_gpt2"],
    "glm4v_moe": ["glm4v_moe_text", "glm4v_moe_vision"],
    "glm4v": ["glm4v_text", "glm4v_vision"],
    "idefics3": ["idefics3_vision"],
    "internvl": ["internvl_vision"],
    "layoutlmv2": ["layoutxlm"],
    "llama": ["code_llama", "falcon3", "llama2", "llama3"],
    "llama4": ["llama4_text"],
    "llava_next": ["granitevision"],
    "luke": ["mluke"],
    "m2m_100": ["nllb"],
    "maskformer": ["maskformer-swin"],
    "mbart": ["mbart50"],
    "parakeet": ["parakeet_ctc", "parakeet_encoder"],
    "lasr": ["lasr_ctc", "lasr_encoder"],
    "perception_lm": ["perception_encoder"],
    "pix2struct": ["deplot", "matcha"],
    "qwen2_5_vl": ["qwen2_5_vl_text"],
    "qwen2_audio": ["qwen2_audio_encoder"],
    "qwen2_vl": ["qwen2_vl_text"],
    "qwen3_vl_moe": ["qwen3_vl_moe_text"],
    "qwen3_vl": ["qwen3_vl_text"],
    "rt_detr": ["rt_detr_resnet"],
    "sam2": ["sam2_hiera_det_model", "sam2_vision_model"],
    "sam": ["sam_hq_vision_model", "sam_vision_model"],
    "siglip2": ["siglip2_vision_model"],
    "siglip": ["siglip_vision_model"],
    "smolvlm": ["smolvlm_vision"],
    "t5": ["byt5", "flan-t5", "flan-ul2", "madlad-400", "myt5", "t5v1.1", "ul2"],
    "voxtral": ["voxtral_encoder"],
    "wav2vec2": ["mms", "wav2vec2_phoneme", "xls_r", "xlsr_wav2vec2"],
    "xlm-roberta": ["xlm-v"],
}

# Similar to `DEPRECATED_MODELS`, but containing the tags when the model tag does not match the model folder name :'(
DEPRECATED_MODELS_TAGS = {"gptsan-japanese", "open-llama", "transfo-xl", "xlm-prophetnet"}


class HubModelLister:
    """
    Utility for getting models from the hub based on tags. Handles errors without crashing the script.
    """

    def __init__(self, tags):
        self.tags = tags
        self.model_list = api.list_models(filter=tags)

    def __iter__(self):
        try:
            yield from self.model_list
        except Exception as e:
            print(f"Error: {e}")
            return


def _extract_commit_hash(commits):
    for commit in commits:
        if commit.startswith("commit "):
            return commit.split(" ")[1]
    return ""


def get_list_of_repo_model_paths(models_dir):
    # Get list of all models in the library
    models = glob.glob(os.path.join(models_dir, "*/modeling_*.py"))

    # Get list of all deprecated models in the library
    deprecated_models = glob.glob(os.path.join(models_dir, "deprecated", "*"))
    # For each deprecated model, remove the deprecated models from the list of all models as well as the symlink path
    for deprecated_model in deprecated_models:
        deprecated_model_name = "/" + deprecated_model.split("/")[-1] + "/"
        models = [model for model in models if deprecated_model_name not in model]
    # Remove deprecated models
    models = [model for model in models if "/deprecated" not in model]
    # Remove auto
    models = [model for model in models if "/auto/" not in model]
    return models


def get_list_of_models_to_deprecate(
    thresh_num_downloads=5_000,
    thresh_date=None,
    use_cache=False,
    save_model_info=False,
    max_num_models=-1,
):
    if thresh_date is None:
        thresh_date = datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 1)
    else:
        thresh_date = datetime.strptime(thresh_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    models_dir = PATH_TO_REPO / "src/transformers/models"
    model_paths = get_list_of_repo_model_paths(models_dir=models_dir)

    if use_cache and os.path.exists("models_info.json"):
        with open("models_info.json", "r") as f:
            models_info = json.load(f)
        # Convert datetimes back to datetime objects
        for model, info in models_info.items():
            info["first_commit_datetime"] = datetime.fromisoformat(info["first_commit_datetime"])

    else:
        print("Building a dictionary of basic model info...")
        models_info = defaultdict(dict)
        for i, model_path in enumerate(tqdm(sorted(model_paths))):
            if max_num_models != -1 and i > max_num_models:
                break
            model = model_path.split("/")[-2]
            if model in models_info:
                continue
            commits = repo.git.log("--diff-filter=A", "--", model_path).split("\n")
            commit_hash = _extract_commit_hash(commits)
            commit_obj = repo.commit(commit_hash)
            committed_datetime = commit_obj.committed_datetime
            models_info[model]["commit_hash"] = commit_hash
            models_info[model]["first_commit_datetime"] = committed_datetime
            models_info[model]["model_path"] = model_path
            models_info[model]["downloads"] = 0
            models_info[model]["tags"] = [model]

        # The keys in the dictionary above are the model folder names. In some cases, the model tag on the hub does not
        # match the model folder name. We replace the key and append the expected tag.
        for folder_name, expected_tag in MODEL_FOLDER_NAME_TO_TAG_MAPPING.items():
            if folder_name in models_info:
                models_info[expected_tag] = models_info[folder_name]
                models_info[expected_tag]["tags"] = [expected_tag]
                del models_info[folder_name]

        # Some models have multiple tags on the hub. We add the expected tag to the list of tags.
        for model_name, extra_tags in EXTRA_TAGS_MAPPING.items():
            if model_name in models_info:
                models_info[model_name]["tags"].extend(extra_tags)

        # Sanity check for the case with all models: the model tags must match the keys in the MODEL_NAMES_MAPPING
        # (= actual model tags on the hub)
        if max_num_models == -1:
            all_model_tags = set()
            for model_name in models_info:
                all_model_tags.update(models_info[model_name]["tags"])

            non_deprecated_model_tags = (
                set(MODEL_NAMES_MAPPING.keys()) - set(DEPRECATED_MODELS_TAGS) - set(DEPRECATED_MODELS)
            )
            if all_model_tags != non_deprecated_model_tags:
                raise ValueError(
                    "The tags of the `models_info` dictionary must match the keys in the `MODEL_NAMES_MAPPING`!"
                    "\nMissing tags in `model_info`: "
                    + str(sorted(non_deprecated_model_tags - all_model_tags))
                    + "\nExtra tags in `model_info`: "
                    + str(sorted(all_model_tags - non_deprecated_model_tags))
                    + "\n\nYou need to update one or more of the following: `MODEL_NAMES_MAPPING`, "
                    "`EXTRA_TAGS_MAPPING` or `DEPRECATED_MODELS_TAGS`."
                )

        # Filter out models which were added less than a year ago
        models_info = {
            model: info for model, info in models_info.items() if info["first_commit_datetime"] < thresh_date
        }

        # We make successive calls to the hub, filtering based on the model tags
        print("Making calls to the hub to find models below the threshold number of downloads...")
        num_models = len(models_info)
        for i, (model, model_info) in enumerate(models_info.items()):
            print(f"{i + 1}/{num_models}: getting hub downloads for model='{model}' (tags={model_info['tags']})")
            for model_tag in model_info["tags"]:
                if model_info["downloads"] > thresh_num_downloads:
                    break
                model_list = HubModelLister(tags=model_tag)
                for hub_model in model_list:
                    if hub_model.private:
                        continue
                    model_info["downloads"] += hub_model.downloads
                    # No need to make further hub calls, it's above the set threshold
                    if model_info["downloads"] > thresh_num_downloads:
                        break

    if save_model_info and not (use_cache and os.path.exists("models_info.json")):
        # Make datetimes serializable
        for model, info in models_info.items():
            info["first_commit_datetime"] = info["first_commit_datetime"].isoformat()
        with open("models_info.json", "w") as f:
            json.dump(models_info, f, indent=4)

    print("\nFinding models to deprecate:")
    n_models_to_deprecate = 0
    models_to_deprecate = {}
    for model, info in models_info.items():
        n_downloads = info["downloads"]
        if n_downloads < thresh_num_downloads:
            n_models_to_deprecate += 1
            models_to_deprecate[model] = info
            print(f"\nModel: {model}")
            print(f"Downloads: {n_downloads}")
            print(f"Date: {info['first_commit_datetime']}")

    # sort models to deprecate by downloads (lowest downloads first)
    models_to_deprecate = sorted(models_to_deprecate.items(), key=lambda x: x[1]["downloads"])

    print("\nModels to deprecate: ", "\n" + "\n".join([model[0] for model in models_to_deprecate]))
    print(f"\nNumber of models to deprecate: {n_models_to_deprecate}")
    print("Before deprecating make sure to verify the models, including if they're used as a module in other models.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_info", action="store_true", help="Save the retrieved model info to a json file.")
    parser.add_argument(
        "--use_cache", action="store_true", help="Use the cached model info instead of calling the hub."
    )
    parser.add_argument(
        "--thresh_num_downloads",
        type=int,
        default=5_000,
        help=(
            "Threshold number of downloads below which a model should be deprecated. Default is 5,000. If you are "
            "considering a sweep and using a cache, set this to the highest number of the sweep."
        ),
    )
    parser.add_argument(
        "--thresh_date",
        type=str,
        default=None,
        help=(
            "Date to consider the first commit from. Format: YYYY-MM-DD. If unset, defaults to one year ago from "
            "today."
        ),
    )
    parser.add_argument(
        "--max_num_models",
        type=int,
        default=-1,
        help="Maximum number of models architectures to consider. -1 means all models. Useful for testing.",
    )
    args = parser.parse_args()

    models_to_deprecate = get_list_of_models_to_deprecate(
        thresh_num_downloads=args.thresh_num_downloads,
        thresh_date=args.thresh_date,
        use_cache=args.use_cache,
        save_model_info=args.save_model_info,
        max_num_models=args.max_num_models,
    )
