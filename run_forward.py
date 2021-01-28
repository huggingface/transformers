#!/usr/bin/env python3
from itertools import groupby

import datasets
import fairseq
import numpy as np
import torch

from transformers import Wav2Vec2ForMaskedLM


wav2vec_path = "../add_wav2vec/data/wav2vec_small_960h.pt"

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [wav2vec_path], arg_overrides={"data": "../add_wav2vec/data/"}
)

hf_model = Wav2Vec2ForMaskedLM.from_pretrained("../add_wav2vec/hf/wav2vec2")
model = model[0]
model.eval()

hf_model = Wav2Vec2ForMaskedLM.from_pretrained("../add_wav2vec/hf/wav2vec2")


class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Identity()

    def forward(self, x, padding_mask=None):
        return self.dummy(x)


def test_feature_extractor(hf_feat_extractor, fsq_feat_extract, example_wav):
    # set hf_feat_extractor.output to dummy
    fsq_output = fsq_feat_extract(example_wav)
    hf_output = hf_feat_extractor(example_wav)

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    torch.allclose(hf_output, fsq_output, atol=1e-3)


def test_full_feature_extractor(hf_model, fsq_model, example_wav):
    # set encoder to dummy for hf and fsq
    temp_fsq = fsq_model.encoder
    fsq_model.encoder = DummyEncoder()
    temp_hf = hf_model.encoder
    hf_model.encoder = DummyEncoder()

    fsq_output = fsq_model(example_wav, mask=False, features_only=True)["x"]
    hf_output = hf_model(example_wav)

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    torch.allclose(hf_output, fsq_output, atol=1e-3)

    fsq_model.encoder = temp_fsq
    hf_model.encoder = temp_hf


def test_full_encoder(hf_model, fsq_model, example_wav):
    fsq_output = fsq_model(example_wav, mask=False, features_only=True)["x"]
    hf_output = hf_model(example_wav)

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    torch.allclose(hf_output, fsq_output, atol=1e-3)


def test_full_model(hf_model, fsq_model, example_wav):
    fsq_output = fsq_model(source=example_wav, padding_mask=None)["encoder_out"]
    hf_output = hf_model(example_wav).transpose(0, 1)

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    torch.allclose(hf_output, fsq_output, atol=1e-3)


def test_all(example_wav):
    with torch.no_grad():
        test_feature_extractor(
            hf_model.wav2vec2.feature_extractor, model.w2v_encoder.w2v_model.feature_extractor, example_wav
        )
    print("Succeded 1st part feature extractor Test")

    with torch.no_grad():
        test_full_feature_extractor(hf_model.wav2vec2, model.w2v_encoder.w2v_model, example_wav)
    print("Succeded full feature extractor Test")

    with torch.no_grad():
        # IMPORTANT: It is assumed that layer_norm_first is FALSE
        # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
        # Adapt if necessary
        test_full_encoder(hf_model.wav2vec2, model.w2v_encoder.w2v_model, example_wav)
    print("Succeded full encoder test")

    with torch.no_grad():
        # IMPORTANT: It is assumed that layer_norm_first is FALSE
        # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
        # Adapt if necessary
        test_full_model(hf_model, model, example_wav)
    print("Succeded full model test")


dummy_speech_data = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
input_wav = torch.tensor(dummy_speech_data[0]["speech"])[None, :]

test_all(input_wav)

json_dict = {
    "<s>": 0,
    "<pad>": 1,
    "</s>": 2,
    "<unk>": 3,
    "|": 4,
    "E": 5,
    "T": 6,
    "A": 7,
    "O": 8,
    "N": 9,
    "I": 10,
    "H": 11,
    "S": 12,
    "R": 13,
    "D": 14,
    "L": 15,
    "U": 16,
    "M": 17,
    "W": 18,
    "C": 19,
    "F": 20,
    "G": 21,
    "Y": 22,
    "P": 23,
    "B": 24,
    "V": 25,
    "K": 26,
    "'": 27,
    "X": 28,
    "J": 29,
    "Q": 30,
    "Z": 31,
}


class Decoder:
    def __init__(self, json_dict):
        self.dict = json_dict
        self.look_up = np.asarray(list(self.dict.keys()))

    def decode(self, ids):
        converted_tokens = self.look_up[ids]
        fused_tokens = [tok[0] for tok in groupby(converted_tokens)]
        output = " ".join("".join("".join(fused_tokens).split("<s>")).split("|"))
        return output


fsq_output = model(source=input_wav, padding_mask=None)["encoder_out"]
hf_output = hf_model(input_wav)
argmax_logits = torch.argmax(hf_output[0], axis=-1)

decoder = Decoder(json_dict)
prediction = decoder.decode(argmax_logits)

print(prediction)
