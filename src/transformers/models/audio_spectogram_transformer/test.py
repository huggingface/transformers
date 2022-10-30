import torch
import torchaudio
from huggingface_hub import hf_hub_download

from transformers import (
    AudioSpectogramTransformerConfig,
    AudioSpectogramTransformerFeatureExtractor,
    AudioSpectogramTransformerForSequenceClassification,
)

# define feature extractor and model
feature_extractor = AudioSpectogramTransformerFeatureExtractor()
# config = AudioSpectogramTransformerConfig(num_labels=527)
# model = AudioSpectogramTransformerForSequenceClassification(config)

# read audio
filepath = hf_hub_download(repo_id="nielsr/audio-spectogram-transformer-checkpoint",
                           filename="sample_audio.flac",
                           repo_type="dataset")

raw_speech, _ = torchaudio.load(filepath)

raw_speech = raw_speech.squeeze().numpy()

# prepare audio for the model
inputs = feature_extractor(raw_speech, padding="max_length", return_tensors="pt")

for k,v in inputs.items():
    print(k,v.shape)

# dummy_inputs = torch.randn(1, 1024, 128)

# outputs = model(dummy_inputs)

# print("Shape of logits:", outputs.logits.shape)

# for name, param in model.named_parameters():
#     print(name, param.shape)
