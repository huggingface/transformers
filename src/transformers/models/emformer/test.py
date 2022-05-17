import torch

from transformers import EmformerConfig, EmformerFeatureExtractor, EmformerModel


feature_extractor = EmformerFeatureExtractor.from_pretrained("./emformer-base-librispeech")
feature_extractor.save_pretrained("./emformer-base-librispeech")

waveform = torch.load("../../../../../audio/examples/asr/emformer_rnnt/librispeech_waveform_0.pt")


config = EmformerConfig()
model = EmformerModel(config)
features = feature_extractor(waveform, return_tensors="pt")
with torch.no_grad():
    outputs = model(**features)

