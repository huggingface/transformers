from huggingface_hub import hf_hub_download
import torchaudio

from transformers import ASTFeatureExtractor

filepath = hf_hub_download(
    repo_id="nielsr/audio-spectogram-transformer-checkpoint",
    filename="sample_audio.flac",
    repo_type="dataset",
)

waveform, _ = torchaudio.load(filepath)
waveform = waveform.squeeze().numpy()

max_length = 24
feature_extractor = ASTFeatureExtractor(num_mel_bins=16)
inputs = feature_extractor(waveform, sampling_rate=16000, max_length=max_length, return_tensors="pt")

for k,v in inputs.items():
    print(k,v.shape)