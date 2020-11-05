# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Convert RoBERTa checkpoint."""


import argparse

import pytorch_lightning as pl
import torch

from transformers.modeling_longformer import LongformerForQuestionAnswering, LongformerModel


class LightningModel(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_labels = 2
        self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)

    # implement only because lightning requires to do so
    def forward(self):
        pass


def convert_longformer_qa_checkpoint_to_pytorch(
    longformer_model: str, longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str
):

    # load longformer model from model identifier
    longformer = LongformerModel.from_pretrained(longformer_model)
    lightning_model = LightningModel(longformer)

    ckpt = torch.load(longformer_question_answering_ckpt_path, map_location=torch.device("cpu"))
    lightning_model.load_state_dict(ckpt["state_dict"])

    # init longformer question answering model
    longformer_for_qa = LongformerForQuestionAnswering.from_pretrained(longformer_model)

    # transfer weights
    longformer_for_qa.longformer.load_state_dict(lightning_model.model.state_dict())
    longformer_for_qa.qa_outputs.load_state_dict(lightning_model.qa_outputs.state_dict())
    longformer_for_qa.eval()

    # save model
    longformer_for_qa.save_pretrained(pytorch_dump_folder_path)

    print("Conversion successful. Model saved under {}".format(pytorch_dump_folder_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--longformer_model",
        default=None,
        type=str,
        required=True,
        help="model identifier of longformer. Should be either `longformer-base-4096` or `longformer-large-4096`.",
    )
    parser.add_argument(
        "--longformer_question_answering_ckpt_path",
        default=None,
        type=str,
        required=True,
        help="Path the official PyTorch Lightning Checkpoint.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    args = parser.parse_args()
    convert_longformer_qa_checkpoint_to_pytorch(
        args.longformer_model, args.longformer_question_answering_ckpt_path, args.pytorch_dump_folder_path
    )
