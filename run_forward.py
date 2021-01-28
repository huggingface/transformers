#!/usr/bin/env python3
import datasets
import fairseq
import torch
import soundfile as sf

from transformers import Wav2Vec2ForMaskedLM, Wav2Vec2Tokenizer


wav2vec_path = "../add_wav2vec/data/wav2vec_small_960h.pt"

model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [wav2vec_path], arg_overrides={"data": "../add_wav2vec/data/"}
)
model = model[0]
model.eval()

hf_model = Wav2Vec2ForMaskedLM.from_pretrained("patrickvonplaten/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("patrickvonplaten/wav2vec2-base-960h")


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


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch


def test_real_example(input_wav):
    hf_output = hf_model(input_wav)
    argmax_logits = torch.argmax(hf_output[0], axis=-1)
    prediction = tokenizer.batch_decode(argmax_logits)
    print(prediction)


dummy_speech_data = dummy_speech_data.map(map_to_array, remove_columns=["file"])
input_wav = tokenizer(dummy_speech_data[0]["speech"], return_tensors="pt").input_ids

test_all(input_wav)
test_real_example(input_wav)
