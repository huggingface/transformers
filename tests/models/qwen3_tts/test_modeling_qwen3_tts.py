# Download through Hugging Face
#pip install -U "huggingface_hub[cli]"
#huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
#huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
#huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
#huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
#huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
#huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
import torch
import requests
import soundfile as sf
import io
import torchaudio
from transformers.models.qwen3_tts.modeling_qwen3_tts import (
    Qwen3TTSTokenizerV2EncoderOutput,
    Qwen3TTSTokenizerV2Encoder,
)
from transformers.models.mimi.configuration_mimi import MimiConfig



def load_audio_from_url(url: str, target_sr: int = 24000):
    # Download
    response = requests.get(url)
    response.raise_for_status()

    # Decode audio
    audio, sr = sf.read(io.BytesIO(response.content), dtype="float32")

    # Shape → (channels, length)
    if audio.ndim == 1:
        audio = audio[None, :]   # (1, T)
    else:
        audio = audio.T          # (C, T)

    audio = torch.from_numpy(audio)

    # Resample if needed
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)

    return audio  # (channels, length)

def main():
    encoder = Qwen3TTSTokenizerV2Encoder(config=MimiConfig(None))#**encoder_config)
    ref_audio_path_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone_2.wav"
    audio = load_audio_from_url(ref_audio_path_1, target_sr=24000)
    # Add batch dimension
    input_values = audio.unsqueeze(0)
    # shape: (1, channels, length)    
    padding_mask = torch.ones_like(input_values, dtype=torch.bool)
    encoded_frames = encoder.encode(input_values=input_values.unsqueeze(1), return_dict=True)
    audio_codes = encoded_frames.audio_codes[:, :encoder.encoder_valid_num_quantizers]
    audio_codes = [code[..., :-(-mask.sum() // encoder.encode_downsample_rate)].transpose(0, 1) for code, mask in zip(audio_codes, padding_mask)]
    return Qwen3TTSTokenizerV2EncoderOutput(audio_codes)

if __name__ == "__main__":
    main()