import torch
from transformers import VibeVoiceFeatureExtractor, VibeVoiceAcousticTokenizerModel
from transformers.audio_utils import load_audio_librosa
from scipy.io import wavfile


model_path = "bezzam/VibeVoice-AcousticTokenizer"
fe_path = "bezzam/VibeVoice-1.5B"
sampling_rate = 24000

# load audio
audio = load_audio_librosa(
    "https://hf.co/datasets/bezzam/vibevoice_samples/resolve/main/voices/en-Alice_woman.wav", 
    sampling_rate=sampling_rate
)

# load model
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
feature_extractor = VibeVoiceFeatureExtractor.from_pretrained(fe_path)
model = VibeVoiceAcousticTokenizerModel.from_pretrained(
    model_path, device_map=torch_device,
).to(torch_device).eval()

# encode
inputs = feature_extractor(
    audio, 
    sampling_rate=sampling_rate,
    padding=True,
    pad_to_multiple_of=3200,
    return_attention_mask=False,
    return_tensors="pt"
).to(torch_device)
print("Input audio shape:", inputs.input_features.shape)
with torch.no_grad():
    encoded_outputs = model.encode(inputs.input_features)
print("Latent shape:", encoded_outputs.latents.shape)

# decode
with torch.no_grad():
    decoded_outputs = model.decode(**encoded_outputs)
print("Reconstructed audio shape:", decoded_outputs.audio.shape)

# Save audio
output_fp = "vibevoice_acoustic_tokenizer_reconstructed.wav"
wavfile.write(output_fp, sampling_rate, decoded_outputs.audio.squeeze().cpu().numpy())
