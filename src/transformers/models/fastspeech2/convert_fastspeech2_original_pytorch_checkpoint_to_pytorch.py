# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert FastSpeech2 checkpoint."""


import argparse

import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface

from transformers import FastSpeech2Config, FastSpeech2Model


def convert_wav2vec2_checkpoint(args):
    if "fastspeech2" not in args.fairseq_model_id:
        raise ValueError("`fairseq_model_id` must be a FastSpeech2 checkpoint")
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(args.fairseq_model_id)
    fairseq_model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(models, cfg)

    model_cfg = cfg["model"]
    config = FastSpeech2Config(
        pitch_min=model_cfg.pitch_min,
        pitch_max=model_cfg.pitch_min,
        energy_min=model_cfg.energy_min,
        energy_max=model_cfg.energy_max,
        add_postnet=model_cfg.add_postnet,
        vocab_size=len(vars(task.src_dict)["indices"]),
        num_speakers=1 if model_cfg.speaker_to_id is None else len(model_cfg.speaker_to_id),
    )
    hf_model = FastSpeech2Model(config)
    state_dict = fairseq_model.state_dict()
    _ = state_dict.pop("encoder.embed_positions._float_tensor", None)
    state_dict["encoder.embed_positions.weights"] = fairseq_model.encoder.embed_positions.weights
    state_dict["encoder.mean"] = torch.from_numpy(generator.gcmvn_stats["mean"])
    state_dict["encoder.std"] = torch.from_numpy(generator.gcmvn_stats["std"])
    hf_model.load_state_dict(state_dict)
    hf_model.save_pretrained(args.pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", required=True, type=str, help="Path to the output PyTorch model")
    parser.add_argument("--fairseq_model_id", required=True, type=str, help="fairseq model id")
    args = parser.parse_args()
    convert_wav2vec2_checkpoint(args)
