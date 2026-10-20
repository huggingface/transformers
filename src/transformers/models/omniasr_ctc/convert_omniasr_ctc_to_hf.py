# Copyright 2026 The HuggingFace Inc. team.
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
import os

import torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

from transformers import OmniASRCTCConfig, OmniASRCTCForCTC, OmniASRCTCProcessor, ParakeetTokenizer, logging

# The speech encoder, its weights and the tokenizer are converted as for the ALM variant, which defines the encoder.
from transformers.models.omniasr.convert_omniasr_to_hf import (
    apply_weight_norm,
    convert_audio_config,
    convert_state_dict,
    convert_tokenizer,
    download_tokenizer,
    get_device_and_dtype,
    get_encoder_key_mapping,
    get_feature_extractor,
    get_original_encoder_config,
    load_state_dict,
    param_count,
    remove_weight_norm,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


CTC_KEY_MAPPING = {
    r"^final_proj\.": "ctc_head.",
}


@torch.no_grad()
def convert_omniasr_ctc_checkpoint(model_card, repo_id=None, bfloat16=False):
    device, dtype = get_device_and_dtype(bfloat16)

    if model_card is None or "CTC" not in model_card:
        raise ValueError(f"Only the CTC variants are supported, got `model_card={model_card!r}`.")

    # 1) Load original model
    pipeline = ASRInferencePipeline(model_card=model_card, device=device, dtype=dtype)
    original_model = pipeline.model
    original_config = get_original_encoder_config(model_card, pipeline.tokenizer.vocab_info.size)

    # 2) Initialize Transformers model
    config = OmniASRCTCConfig(
        audio_config=convert_audio_config(original_config),
        vocab_size=original_config.target_vocab_size,
        pad_token_id=pipeline.tokenizer.vocab_info.pad_idx,
        bos_token_id=pipeline.tokenizer.vocab_info.bos_idx,
        eos_token_id=pipeline.tokenizer.vocab_info.eos_idx,
    )
    hf_model = OmniASRCTCForCTC(config)
    hf_model.to(device).to(dtype)

    # 3) Convert weights
    apply_weight_norm(hf_model)
    print(f"Total parameters (original): {param_count(original_model)}")
    print(f"Total parameters (HF)      : {param_count(hf_model)}")

    state_dict = original_model.state_dict()
    print("Number of keys in original model :", len(state_dict))
    print("Number of keys in HF model       : ", len(hf_model.state_dict()))
    key_mapping = {**get_encoder_key_mapping("model."), **CTC_KEY_MAPPING}
    hf_model = load_state_dict(hf_model, convert_state_dict(state_dict, key_mapping))
    remove_weight_norm(hf_model)

    # 4) Prepare processor (feature extraction and tokenizer)
    tokenizer_path = download_tokenizer(model_card)
    # The CTC loss does not depend on the padding side. `ParakeetTokenizer` decodes CTC outputs, collapsing repeats and
    # dropping the pad token, which is the CTC blank.
    tokenizer = convert_tokenizer(tokenizer_path, padding_side="right", tokenizer_class=ParakeetTokenizer)
    processor = OmniASRCTCProcessor(feature_extractor=get_feature_extractor(), tokenizer=tokenizer)

    # 5) Upload to hub
    if repo_id:
        logger.info("Pushing model to the Hub ...")
        hf_model.push_to_hub(repo_id)
        processor.push_to_hub(repo_id)

    # 6) Cleanup
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)


"""
Setup: see `convert_omniasr_to_hf.py`.

Example conversion:
```python
# -- release v2
python src/transformers/models/omniasr_ctc/convert_omniasr_ctc_to_hf.py \
    --model_card omniASR_CTC_300M_v2 \
    --repo_id bezzam/omniasr-ctc-300m-v2

# -- release v1
python src/transformers/models/omniasr_ctc/convert_omniasr_ctc_to_hf.py \
    --model_card omniASR_CTC_300M \
    --repo_id bezzam/omniasr-ctc-300m
```

See here for available models: https://github.com/facebookresearch/omnilingual-asr?tab=readme-ov-file#model-architectures
Original model checkpoints are saved under: ~/.cache/fairseq2/assets/
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_card", default=None, type=str, help="Name of original model in omnilingual-asr")
    parser.add_argument("--repo_id", default=None, type=str, help="The repository ID for pushing the model to the Hub")
    parser.add_argument("--bfloat16", action="store_true", help="Whether to do bfloat16, otherwise default is float32")
    # Original defaults to bfloat16: https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/inference/pipeline.py#L157
    args = parser.parse_args()

    convert_omniasr_ctc_checkpoint(args.model_card, args.repo_id, args.bfloat16)
