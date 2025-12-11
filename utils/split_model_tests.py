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
This script is used to get the list of folders under `tests/models` and split the list into `NUM_SLICES` splits.
The main use case is a GitHub Actions workflow file calling this script to get the (nested) list of folders allowing it
to split the list of jobs to run into multiple slices each containing a smaller number of jobs. This way, we can bypass
the maximum of 256 jobs in a matrix.

See the `setup` and `run_models_gpu` jobs defined in the workflow file `.github/workflows/self-scheduled.yml` for more
details.

Usage:

This script is required to be run under `tests` folder of `transformers` root directory.

Assume we are under `transformers` root directory:
```bash
cd tests
python ../utils/split_model_tests.py --num_splits 64
```
"""

import argparse
import ast
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subdirs",
        type=str,
        default="",
        help="the list of pre-computed model names (directory names under `tests/models`) or directory names under `tests` (except `models`).",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="the number of splits into which the (flat) list of folders will be split.",
    )
    args = parser.parse_args()

    tests = os.getcwd()
    model_tests = os.listdir(os.path.join(tests, "models"))
    d1 = sorted(filter(os.path.isdir, os.listdir(tests)))
    d2 = sorted(filter(os.path.isdir, [f"models/{x}" for x in model_tests]))
    d1.remove("models")
    d = d2 + d1

    if args.subdirs != "":
        model_tests = ast.literal_eval(args.subdirs)
        # We handle both cases with and without prefix because `push-important-models.yml` returns the list without
        # the prefix (i.e. `models`) but `utils/pr_slow_ci_models.py` (called by `self-comment-ci.yml`) returns the
        # list with the prefix (`models`) and some directory names under `tests`.
        d = []
        for x in model_tests:
            if os.path.isdir(x):
                d.append(x)
            if os.path.isdir(f"models/{x}"):
                d.append(f"models/{x}")
        d = sorted(d)

    num_jobs = len(d)
    num_jobs_per_splits = num_jobs // args.num_splits

    model_splits = []
    end = 0
    for idx in range(args.num_splits):
        start = end
        end = start + num_jobs_per_splits + (1 if idx < num_jobs % args.num_splits else 0)
        # Only add the slice if it is not an empty list
        if len(d[start:end]) > 0:
            model_splits.append(d[start:end])

    model_splits = [
        ["models/aimv2"],
        ["models/altclip"],
        ["models/apertus"],
        ["models/arcee"],
        ["models/aya_vision"],
        ["models/beit"],
        ["models/bit"],
        ["models/blip_2"],
        ["models/blt"],
        ["models/chameleon"],
        ["models/chinese_clip"],
        ["models/clip"],
        ["models/clipseg"],
        ["models/clvp"],
        ["models/codegen"],
        ["models/conditional_detr"],
        ["models/convnext"],
        ["models/convnextv2"],
        ["models/csm"],
        ["models/ctrl"],
        ["models/cvt"],
        ["models/cwm"],
        ["models/d_fine"],
        ["models/dab_detr"],
        ["models/data2vec"],
        ["models/dbrx"],
        ["models/deepseek_v2"],
        ["models/deformable_detr"],
        ["models/deit"],
        ["models/depth_anything"],
        ["models/depth_pro"],
        ["models/detr"],
        ["models/dia"],
        ["models/diffllama"],
        ["models/dinov2"],
        ["models/dinov2_with_registers"],
        ["models/dinov3_vit"],
        ["models/dots1"],
        ["models/dpt"],
        ["models/edgetam"],
        ["models/efficientnet"],
        ["models/electra"],
        ["models/emu3"],
        ["models/encoder_decoder"],
        ["models/ernie4_5"],
        ["models/ernie4_5_moe"],
        ["models/exaone4"],
        ["models/falcon"],
        ["models/flava"],
        ["models/flex_olmo"],
        ["models/florence2"],
        ["models/focalnet"],
        ["models/fsmt"],
        ["models/fuyu"],
        ["models/gemma"],
        ["models/gemma3"],
        ["models/gemma3n"],
        ["models/git"],
        ["models/glm"],
        ["models/glm4"],
        ["models/glm4_moe"],
        ["models/glm4v"],
        ["models/gpt_neo"],
        ["models/grounding_dino"],
        ["models/groupvit"],
        ["models/helium"],
        ["models/hiera"],
        ["models/hunyuan_v1_dense"],
        ["models/ijepa"],
        ["models/instructblip"],
        ["models/jamba"],
        ["models/jetmoe"],
        ["models/kosmos2"],
        ["models/kosmos2_5"],
        ["models/led"],
        ["models/levit"],
        ["models/lfm2"],
        ["models/lfm2_moe"],
        ["models/lfm2_vl"],
        ["models/llava"],
        ["models/llava_next"],
        ["models/longcat_flash"],
        ["models/longformer"],
        ["models/longt5"],
        ["models/luke"],
        ["models/mamba2"],
        ["models/mask2former"],
        ["models/maskformer"],
        ["models/mbart"],
        ["models/metaclip_2"],
        ["models/minimax"],
        ["models/ministral"],
        ["models/ministral3"],
        ["models/mistral"],
        ["models/mistral3"],
        ["models/mixtral"],
        ["models/mlcd"],
        ["models/mllama"],
        ["models/mluke"],
        ["models/mm_grounding_dino"],
        ["models/mobilenet_v1"],
        ["models/mobilenet_v2"],
        ["models/mobilevit"],
        ["models/mobilevitv2"],
        ["models/modernbert_decoder"],
        ["models/moonshine"],
        ["models/moshi"],
        ["models/mpt"],
        ["models/musicgen"],
        ["models/musicgen_melody"],
        ["models/mvp"],
        ["models/nanochat"],
        ["models/nemotron"],
        ["models/nllb_moe"],
        ["models/olmo3"],
        ["models/oneformer"],
        ["models/openai"],
        ["models/owlvit"],
        ["models/paligemma"],
        ["models/persimmon"],
        ["models/phi"],
        ["models/phi3"],
        ["models/pix2struct"],
        ["models/pixtral"],
        ["models/plbart"],
        ["models/poolformer"],
        ["models/pvt"],
        ["models/pvt_v2"],
        ["models/qwen2"],
        ["models/qwen2_5_omni"],
        ["models/qwen2_5_vl"],
        ["models/qwen2_moe"],
        ["models/qwen3"],
        ["models/qwen3_moe"],
        ["models/qwen3_next"],
        ["models/qwen3_vl_moe"],
        ["models/rag"],
        ["models/reformer"],
        ["models/regnet"],
        ["models/resnet"],
        ["models/rt_detr"],
        ["models/rt_detr_v2"],
        ["models/sam"],
        ["models/sam2"],
        ["models/sam_hq"],
        ["models/seed_oss"],
        ["models/sew"],
        ["models/sew_d"],
        ["models/siglip"],
        ["models/smollm3"],
        ["models/smolvlm"],
        ["models/stablelm"],
        ["models/starcoder2"],
        ["models/swiftformer"],
        ["models/swin"],
        ["models/swinv2"],
        ["models/t5"],
        ["models/table_transformer"],
        ["models/timesformer"],
        ["models/univnet"],
        ["models/upernet"],
        ["models/vaultgemma"],
        ["models/video_llama_3"],
        ["models/videomae"],
        ["models/vilt"],
        ["models/vision_encoder_decoder"],
        ["models/vit"],
        ["models/vit_msn"],
        ["models/vitdet"],
        ["models/vits"],
        ["models/vivit"],
        ["models/wav2vec2_conformer"],
        ["models/wav2vec2_phoneme"],
        ["models/wav2vec2_with_lm"],
        ["models/whisper"],
        ["models/xglm"],
        ["models/xlm"],
        ["models/yolos"],
        ["models/zamba"],
    ]
    model_splits = [[x[0] for x in model_splits]]

    # Total: 179 models
    print(model_splits)
